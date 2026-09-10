"""The sidecar a packed job ships must resolve to the SAME engine params the
staging machine built -- this is the unit-level proof of Goal 3 (spec §15.4:
"a packed job resolves identically on the compute box")."""

import json
import os

from hydra_suite.trackerkit.cli_config import (
    TrackerCliVideoProbe,
    load_tracker_cli_session,
)

# Fix B6: `load_tracker_cli_session(None, ...)` CANNOT work.
#   * `video_path: str` is a REQUIRED POSITIONAL (cli_config.py:304-306), and
#   * when `video_probe` is not supplied it calls `probe_video(video_path)`
#     (`:317`), which raises `RuntimeError(f"Failed to open video: ...")` on an
#     unopenable file (`:253-254`).
# The `staging` fixture's video is 2 KiB of zeros -- cv2 cannot open it -- so
# even passing the real path without a probe would raise. Pass BOTH the real
# job-relative video path (read from the sidecar's own `file_path`, which pack
# rewrote to "videos/<name>") AND an injected probe.
_PROBE = TrackerCliVideoProbe(fps=30.0, total_frames=1, width=64, height=64)


def _staging_params(staging, monkeypatch):
    """The STAGING-side engine params: same config, resolved against the
    staging models root (never the packed job's). This is the "expected"
    side of the fix X7 equality check below.
    """
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(staging["models"]))
    monkeypatch.delenv("HYDRA_CONFIG_DIR", raising=False)
    config = {
        "file_path": str(staging["video"]),
        "yolo_obb_direct_model_path": "obb/x.pt",
        "pose_skeleton_file": str(staging["skeleton"]),
    }
    session = load_tracker_cli_session(
        str(staging["video"]), config_data=config, video_probe=_PROBE
    )
    return session.params


def test_packed_sidecar_resolves_identically_to_staging(
    packed_job, staging, monkeypatch
):
    # Fix X7 (round-6): the original assertion only checked CONTAINMENT
    # (resolved path lies somewhere inside <job>/models) -- that passes even
    # when the sidecar resolves to the WRONG model that happens to live
    # inside the job root (a legacy `yolo_model_path` alias, or a pose
    # backend mix-up that ships model B but the sidecar's role still points
    # at model A's job-relative slot). Spec §15.4 requires the sidecar's
    # resolved params equal the staging-side params KEY FOR KEY. Since the
    # two sides resolve against DIFFERENT absolute roots (staging models dir
    # vs. <job>/models), "equal" means: for every role key, the path
    # RELATIVE TO ITS OWN MODELS ROOT is identical on both sides -- that is
    # the actual portable invariant (same model, same role, same relative
    # slot), not merely "somewhere under models/".
    staging_params = _staging_params(staging, monkeypatch)

    monkeypatch.setenv("HYDRA_MODELS_DIR", str(packed_job / "models"))
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(packed_job / "config"))
    monkeypatch.chdir(packed_job)

    def _relative_to_root(value, root):
        return os.path.relpath(os.path.abspath(value), os.path.abspath(str(root)))

    sidecars = list(packed_job.glob("videos/*_config.json"))
    assert sidecars, "packed job produced no sidecar to test against"

    for sidecar in sidecars:
        video_relpath = json.loads(sidecar.read_text())["file_path"]
        session = load_tracker_cli_session(
            video_relpath, config_path=str(sidecar), video_probe=_PROBE
        )
        params = session.params
        for role_key in ("YOLO_OBB_DIRECT_MODEL_PATH", "POSE_MODEL_DIR"):
            staged_value = staging_params.get(role_key, "")
            packed_value = params.get(role_key, "")
            assert bool(staged_value) == bool(packed_value), (
                f"{role_key}: staging has {staged_value!r}, packed sidecar has "
                f"{packed_value!r} -- presence must match"
            )
            if not staged_value:
                continue
            assert os.path.commonpath(
                [
                    os.path.abspath(packed_value),
                    os.path.abspath(str(packed_job / "models")),
                ]
            ) == os.path.abspath(
                str(packed_job / "models")
            ), f"{role_key} resolved outside <job>/models: {packed_value}"
            assert _relative_to_root(
                staged_value, staging["models"]
            ) == _relative_to_root(packed_value, packed_job / "models"), (
                f"{role_key} resolved to a DIFFERENT model: staging picked "
                f"{_relative_to_root(staged_value, staging['models'])!r}, packed sidecar "
                f"picked {_relative_to_root(packed_value, packed_job / 'models')!r}"
            )
        staged_cnn = {
            entry.get("factor_name", i): _relative_to_root(
                entry.get("model_path", ""), staging["models"]
            )
            for i, entry in enumerate(staging_params.get("CNN_CLASSIFIERS", []) or [])
            if entry.get("model_path")
        }
        packed_cnn = {
            entry.get("factor_name", i): _relative_to_root(
                entry.get("model_path", ""), packed_job / "models"
            )
            for i, entry in enumerate(params.get("CNN_CLASSIFIERS", []) or [])
            if entry.get("model_path")
        }
        assert (
            staged_cnn == packed_cnn
        ), f"CNN_CLASSIFIERS resolved differently: staging={staged_cnn} packed={packed_cnn}"


def test_equality_check_has_teeth_when_sidecar_points_at_the_wrong_model(
    packed_job, staging, monkeypatch
):
    """Proves the KEY-FOR-KEY equality assertion above is not vacuous: a
    sidecar deliberately mutated to point at a DIFFERENT (but still
    job-relative, still inside <job>/models) model for the same role must
    fail the same relative-path comparison the previous test relies on."""
    staging_params = _staging_params(staging, monkeypatch)
    staged_value = staging_params["YOLO_OBB_DIRECT_MODEL_PATH"]

    monkeypatch.setenv("HYDRA_MODELS_DIR", str(packed_job / "models"))
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(packed_job / "config"))
    monkeypatch.chdir(packed_job)

    # Plant a second, decoy model inside the job root, under a different
    # models-relative key, so a mutated sidecar still resolves inside
    # <job>/models but to the WRONG model for its role.
    decoy = packed_job / "models" / "obb" / "decoy.pt"
    decoy.write_bytes(b"decoy")

    sidecar = next(packed_job.glob("videos/*_config.json"))
    payload = json.loads(sidecar.read_text())
    payload["yolo_obb_direct_model_path"] = "obb/decoy.pt"
    mutated_sidecar = sidecar.with_name("mutated_config.json")
    mutated_sidecar.write_text(json.dumps(payload))

    session = load_tracker_cli_session(
        payload["file_path"], config_path=str(mutated_sidecar), video_probe=_PROBE
    )
    mutated_value = session.params["YOLO_OBB_DIRECT_MODEL_PATH"]

    def _relative_to_root(value, root):
        return os.path.relpath(os.path.abspath(value), os.path.abspath(str(root)))

    staged_rel = _relative_to_root(staged_value, staging["models"])
    mutated_rel = _relative_to_root(mutated_value, packed_job / "models")

    # This is the exact comparison the equality assertion above performs --
    # asserting it here shows a wrong-model sidecar does NOT compare equal,
    # i.e. the check would have caught it.
    assert staged_rel != mutated_rel
