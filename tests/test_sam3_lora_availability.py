"""The probe must explain WHY it is unusable, and never import sam3 or spawn conda.

Covers env.py (pure string/dict construction, no subprocess) and the
sidecar-probe inversion in availability.py (faked ``run_conda`` / ``_run_probe``
so no test requires sam3, conda, or a GPU).
"""

import json
import subprocess
import sys

import pytest

from hydra_suite.training.sam3_lora import availability as av
from hydra_suite.training.sam3_lora import env as sam3_env


@pytest.fixture(autouse=True)
def _cuda_host(monkeypatch):
    """The probe tests below model a CUDA host; the non-CUDA gate has its own."""
    monkeypatch.setattr(av, "_host_has_cuda", lambda: True)


# --------------------------------------------------------------------------
# env.py
# --------------------------------------------------------------------------


def test_resolve_sam3_env_uses_configured_value():
    assert sam3_env.resolve_sam3_env("my-env") == "my-env"


def test_resolve_sam3_env_falls_back_to_env_var(monkeypatch):
    monkeypatch.delenv("HYDRA_SAM3_ENV", raising=False)
    monkeypatch.setenv("HYDRA_SAM3_ENV", "env-from-var")
    assert sam3_env.resolve_sam3_env(None) == "env-from-var"
    assert sam3_env.resolve_sam3_env("") == "env-from-var"


def test_resolve_sam3_env_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("HYDRA_SAM3_ENV", raising=False)
    assert sam3_env.resolve_sam3_env(None) == sam3_env.DEFAULT_SAM3_ENV
    assert sam3_env.resolve_sam3_env() == sam3_env.DEFAULT_SAM3_ENV


def test_sam3_env_command_builds_conda_run_python_module():
    got = sam3_env.sam3_env_command("hydra-sam3", ["pkg.module", "--flag", "x"])
    assert got == [
        "conda",
        "run",
        "-n",
        "hydra-sam3",
        "--no-capture-output",
        "python",
        "-u",
        "-m",
        "pkg.module",
        "--flag",
        "x",
    ]


def test_sam3_env_environ_sets_kmp_duplicate_lib_ok():
    got = sam3_env.sam3_env_environ()
    assert got["KMP_DUPLICATE_LIB_OK"] == "TRUE"


def test_sam3_env_environ_sets_expandable_segments():
    """The allocator flag is not cosmetic: without it 45% of peak VRAM is
    caching-allocator fragmentation (measured 12.99 GiB vs 7.13 GiB on two
    matched full runs), and no short probe can bound the peak, which is what
    measured auto batch sizing depends on.  Both the training and publish
    sidecars source their environment here, so this is the single place that
    guarantees the child gets it regardless of the caller's shell.
    """
    got = sam3_env.sam3_env_environ()
    assert got["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


# --------------------------------------------------------------------------
# probe inversion
# --------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _healthy_payload():
    return {
        "ok": True,
        "missing": [],
        "cuda_available": True,
        "cuda_compute_capability": [8, 9],
        "cuda_bf16_supported": True,
    }


def test_usable_when_child_reports_ok(monkeypatch):
    monkeypatch.setattr(
        av,
        "_run_probe",
        lambda env, timeout: _FakeCompleted(0, json.dumps(_healthy_payload())),
    )
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: True)
    got = av.probe_sam3_training_availability(env="hydra-sam3")
    assert got.usable
    assert got.reason == ""


def test_unusable_surfaces_child_reason_verbatim(monkeypatch):
    monkeypatch.setattr(
        av,
        "_run_probe",
        lambda env, timeout: _FakeCompleted(
            0,
            json.dumps(
                {
                    "ok": False,
                    "missing": [
                        {"package": "triton", "error": "No module named 'triton'"}
                    ],
                }
            ),
        ),
    )
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: True)
    got = av.probe_sam3_training_availability(env="hydra-sam3")
    assert not got.usable
    assert "triton" in got.reason
    assert "No module named 'triton'" in got.reason


def test_missing_checkpoint_is_reported_not_downloaded(monkeypatch):
    monkeypatch.setattr(
        av,
        "_run_probe",
        lambda env, timeout: _FakeCompleted(0, json.dumps(_healthy_payload())),
    )
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: False)
    got = av.probe_sam3_training_availability()
    assert not got.usable
    assert "checkpoint" in got.reason.lower()


def test_timeout_is_reported_distinctly(monkeypatch):
    def _raise(env, timeout):
        raise subprocess.TimeoutExpired(cmd=["conda"], timeout=timeout)

    monkeypatch.setattr(av, "_run_probe", _raise)
    got = av.probe_sam3_training_availability(env="hydra-sam3", timeout=5)
    assert not got.usable
    assert "timed out" in got.reason.lower()


def test_conda_missing_from_path_is_reported_distinctly(monkeypatch):
    def _raise(env, timeout):
        raise FileNotFoundError("conda")

    monkeypatch.setattr(av, "_run_probe", _raise)
    got = av.probe_sam3_training_availability()
    assert not got.usable
    assert "conda" in got.reason.lower()
    assert "path" in got.reason.lower()


def test_env_missing_or_nonzero_exit_is_reported(monkeypatch):
    monkeypatch.setattr(
        av,
        "_run_probe",
        lambda env, timeout: _FakeCompleted(1, "", "EnvironmentLocationNotFound"),
    )
    got = av.probe_sam3_training_availability(env="does-not-exist")
    assert not got.usable
    assert "does-not-exist" in got.reason
    assert "EnvironmentLocationNotFound" in got.reason


def test_malformed_child_output_is_reported(monkeypatch):
    monkeypatch.setattr(
        av, "_run_probe", lambda env, timeout: _FakeCompleted(0, "not json")
    )
    got = av.probe_sam3_training_availability()
    assert not got.usable
    assert "could not be parsed" in got.reason.lower()


def test_probe_script_checklist_matches_host_checklist():
    """The standalone child script duplicates the checklist (it cannot import
    hydra_suite -- see its docstring); keep the two lists in sync."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_sam3_probe_script_under_test", av._PROBE_SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TRAINING_PACKAGES == av.TRAINING_PACKAGES


def test_probe_does_not_import_sam3(monkeypatch):
    monkeypatch.setattr(
        av,
        "_run_probe",
        lambda env, timeout: _FakeCompleted(0, json.dumps(_healthy_payload())),
    )
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: True)
    sys.modules.pop("sam3", None)
    av.probe_sam3_training_availability()
    assert "sam3" not in sys.modules


def test_no_cuda_and_pre_ampere_are_unavailable(monkeypatch):
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: True)
    for payload, expected in (
        ({"ok": True, "missing": [], "cuda_available": False}, "CUDA"),
        (
            {
                "ok": True,
                "missing": [],
                "cuda_available": True,
                "cuda_compute_capability": [7, 5],
            },
            "8.0",
        ),
        (
            {
                "ok": True,
                "missing": [],
                "cuda_available": True,
                "cuda_compute_capability": [8, 9],
                "cuda_bf16_supported": False,
            },
            "BF16",
        ),
    ):
        monkeypatch.setattr(
            av,
            "_run_probe",
            lambda env, timeout, payload=payload: _FakeCompleted(
                0, json.dumps(payload)
            ),
        )
        got = av.probe_sam3_training_availability()
        assert not got.usable
        assert expected in got.reason


def test_sam3_env_command_disables_conda_output_capture():
    """Without --no-capture-output, conda run buffers the child's stdout and
    stderr in full and releases them only at exit.

    A 6-hour training run printed not one progress line, and a run killed
    before exiting loses its diagnostics entirely. `-u` does not help: it
    unbuffers Python, while the capture happens one level above it.
    """
    got = sam3_env.sam3_env_command("hydra-sam3", ["pkg.module"])
    assert "--no-capture-output" in got
    assert got.index("--no-capture-output") < got.index(
        "python"
    ), "the flag is a conda run option and must precede the child command"
    assert "-u" in got


def test_non_cuda_host_is_refused_without_probing_the_sidecar(monkeypatch):
    monkeypatch.setattr(av, "_host_has_cuda", lambda: False)
    monkeypatch.setattr(
        av, "_run_probe", lambda *a, **k: pytest.fail("must not probe the sidecar")
    )
    result = av.probe_sam3_training_availability()
    assert not result.usable
    assert "requires an NVIDIA GPU" in result.reason
    assert "inference" in result.reason  # tells the user what still works


def test_install_hint_points_at_the_installer_not_an_internal_spec():
    assert "install.py --with-sam3-train" in av.DEFAULT_INSTALL_HINT
    assert "superpowers" not in av.DEFAULT_INSTALL_HINT


_EDITABLE = {
    "version": "1.0.0",
    "direct_url": {"url": "file:///src/hydra", "dir_info": {"editable": True}},
}


def test_origin_skew_accepts_the_same_source(monkeypatch):
    monkeypatch.setattr(av, "_host_origin", lambda: dict(_EDITABLE))
    assert av._origin_skew(dict(_EDITABLE)) == ""


def test_origin_skew_flags_a_different_checkout(monkeypatch):
    monkeypatch.setattr(av, "_host_origin", lambda: dict(_EDITABLE))
    other = {
        "version": "1.0.0",
        "direct_url": {"url": "file:///old/hydra", "dir_info": {"editable": True}},
    }
    assert "different hydra-suite" in av._origin_skew(other)


def test_origin_skew_flags_missing_hydra_in_sidecar(monkeypatch):
    monkeypatch.setattr(av, "_host_origin", lambda: dict(_EDITABLE))
    assert "not installed in the SAM3 sidecar" in av._origin_skew(None)


def test_origin_skew_tolerates_old_probe_payloads():
    assert av._origin_skew("absent") == ""


def test_origin_skew_treats_symlinked_spellings_of_one_path_as_equal(
    monkeypatch, tmp_path
):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    host = {
        "version": "1.0.0",
        "direct_url": {"url": real.as_uri(), "dir_info": {"editable": True}},
    }
    side = {
        "version": "1.0.0",
        "direct_url": {
            "url": link.as_uri().replace("/link", "/%6Cink"),
            "dir_info": {"editable": True},
        },
    }
    monkeypatch.setattr(av, "_host_origin", lambda: host)
    assert av._origin_skew(side) == ""
