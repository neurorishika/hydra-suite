"""The probe must explain WHY it is unusable, and never import sam3 or spawn conda.

Covers env.py (pure string/dict construction, no subprocess) and the
sidecar-probe inversion in availability.py (faked ``run_conda`` / ``_run_probe``
so no test requires sam3, conda, or a GPU).
"""

import json
import subprocess
import sys

from hydra_suite.training.sam3_lora import availability as av
from hydra_suite.training.sam3_lora import env as sam3_env

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
