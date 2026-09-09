import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# Add both src and repo root to path for imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Fix B8: `packed_job` and its siblings below call `pack_job` bare. The
# package exists by Task 10, so a module-level import is safe and removes
# four duplicated lazy imports (one per fixture).
from hydra_suite.data.tracking_job.manifest import JobManifest  # noqa: E402
from hydra_suite.data.tracking_job.pack import pack_job  # noqa: E402
from tests.helpers.tracking_job import _planned  # noqa: E402

# Fixture helpers for classifier backend tests
pytest_plugins = ["tests.test_classifier_fixtures"]


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite characterization goldens instead of asserting against them.",
    )


import json

import numpy as np
import pytest

_FLY_OBB_CONFIG = (
    REPO_ROOT / "tools" / "equivalence" / "fixtures" / "configs" / "fly_obb.json"
)


def _fixture_obb_checkpoint() -> Path | None:
    """Resolve the equivalence fixture's OBB checkpoint, or None when absent.

    The fixture bundle extracts models into ``get_models_dir()``; the clip
    config names the file (key ``yolo_obb_direct_model_path`` in the on-disk
    JSON). There is no ``fixtures/models/`` directory.
    """
    if not _FLY_OBB_CONFIG.exists():
        return None
    from hydra_suite.paths import get_models_dir

    params = json.loads(_FLY_OBB_CONFIG.read_text())
    raw = str(
        params.get("YOLO_OBB_DIRECT_MODEL_PATH")
        or params.get("yolo_obb_direct_model_path")
        or ""
    )
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    resolved = Path(get_models_dir()) / raw
    return resolved if resolved.is_file() else None


@pytest.fixture
def direct_obb_fixture():
    """Real sliced direct-OBB config + loaded models, or skip."""
    checkpoint = _fixture_obb_checkpoint()
    if checkpoint is None:
        pytest.skip("equivalence fixture OBB checkpoint not present")
    from hydra_suite.core.inference.direct_calibration_sweep import (
        build_calibration_config,
    )
    from hydra_suite.core.inference.runtime import RuntimeContext
    from hydra_suite.core.inference.stages.obb import load_obb_models

    config = build_calibration_config(
        str(checkpoint),
        slice_params={
            "SLICE_ENABLED": True,
            "SLICE_GEOMETRY_MODE": "auto_object",
            "SLICE_OBJECT_TILE_FRACTION": 0.4,
            "SLICE_OVERLAP": 0.2,
            "SLICE_TRAINED_BODY_PX": 120.0,
        },
        max_targets=64,
        confidence=0.25,
        runtime_tier="cpu",
    )
    runtime = RuntimeContext.from_config(config)
    models = load_obb_models(config.obb, runtime)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(2)]
    return frames, models, config, runtime


@pytest.fixture(autouse=True)
def _neutralize_leaked_training_flags():
    """Never let a leaked "training is running" widget hang the whole suite.

    Several GUI tests drive ``_resume_training``/``_start_training`` with a fake
    worker and leave the dialog with ``_training_running = True``. That dialog
    stays alive as a top-level widget for the rest of the pytest process, and
    DetectKit's ``TrainingDialog.closeEvent`` refuses to close while training by
    raising a modal ``QMessageBox``. Any later test that closes every top-level
    widget (a fixture several kits copy) then blocks forever -- pytest-timeout
    dumps stacks and ``os._exit``s, so every remaining test silently never runs.

    Clearing the flag is enough to make ``closeEvent`` non-blocking; widget
    lifetimes are deliberately left alone.
    """
    yield
    qtwidgets = sys.modules.get("PySide6.QtWidgets")
    if qtwidgets is None:
        return
    app = qtwidgets.QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        if getattr(widget, "_training_running", False):
            widget._training_running = False


# --- Portable tracking-job fixtures (shared by pack/verify/preflight tests) ---


@pytest.fixture()
def staging(tmp_path):
    """A models root, a video, a skeleton and an advanced config."""
    models = tmp_path / "models"
    (models / "obb").mkdir(parents=True)
    (models / "obb" / "x.pt").write_bytes(b"w")
    videos = tmp_path / "data"
    videos.mkdir()
    video = videos / "colony.mp4"
    video.write_bytes(b"\x00" * 2048)
    skeleton = tmp_path / "skel" / "ant.json"
    skeleton.parent.mkdir()
    skeleton.write_text('{"nodes": []}')
    advanced = tmp_path / "advanced_config.json"
    advanced.write_text('{"adv": true}')
    return {
        "models": models,
        "video": video,
        "skeleton": skeleton,
        "advanced": advanced,
    }


@pytest.fixture()
def _isolated_host_config(tmp_path, monkeypatch):
    """Every preflight test MUST use this (directly or via packed_job*) so a
    real labnas alias configured on the developer's machine (Task 13 Step 5
    configures exactly one) cannot flip a preflight test's outcome."""
    host_cfg = tmp_path / "host_config"
    host_cfg.mkdir()
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(host_cfg))
    monkeypatch.setenv("HYDRA_HOST_CONFIG_DIR", str(host_cfg))
    return host_cfg


@pytest.fixture()
def packed_job(tmp_path, staging, _isolated_host_config):
    """A minimal, valid, CPU-tier packed job. runtime_tier is pinned
    explicitly to "cpu" — build_engine_params defaults an absent tier to
    "gpu" (engine_params.py:805), so leaving it unset would make this
    fixture silently GPU-tier and mask tier-related preflight bugs."""
    planned = _planned(staging, config={"runtime_tier": "cpu"})
    job_dir = tmp_path / "job"
    pack_job(
        job_dir,
        [planned],
        registry_entries=[
            ("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})
        ],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt", "runtime_tier": "cpu"},
        shared_table={},
    )
    return job_dir


@pytest.fixture()
def packed_job_needing_sleap(tmp_path, staging, _isolated_host_config):
    """A job whose keystone config selects the SLEAP pose backend, so
    preflight's conda_envs check must name "sleap" as required.

    Fix W12: `enable_pose_extractor` + `pose_model_type` ALONE do not
    satisfy `is_pose_inference_enabled` (core/tracking/session_policy.py:
    29-32) -- that also requires `detection_method == "yolo_obb"` (via
    `is_individual_pipeline_enabled`) AND a non-empty `pose_model_dir`.
    Without both, `is_pose_export_enabled`/`is_pose_inference_enabled` are
    False, pose inference never actually runs, and pack_job's fix-W12 rule
    correctly (per its own definition) computes an EMPTY conda_envs -- this
    fixture would then not "need" sleap at all, and
    test_a_missing_conda_env_fails_naming_it would be testing nothing."""
    planned = _planned(
        staging,
        config={
            "runtime_tier": "cpu",
            "detection_method": "yolo_obb",
            "enable_pose_extractor": True,
            "pose_model_type": "SLEAP",
            "pose_model_dir": "pose/SLEAP/run",
            "pose_sleap_env": "sleap",
        },
    )
    job_dir = tmp_path / "job"
    pack_job(
        job_dir,
        [planned],
        registry_entries=[
            ("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})
        ],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt", "runtime_tier": "cpu"},
        shared_table={},
    )
    return job_dir


@pytest.fixture()
def packed_job_gpu_tier(tmp_path, staging, _isolated_host_config):
    """A job whose keystone config explicitly requests the "gpu" tier, so
    the runtime_tier check has something real to fail against on a
    CPU-only available_tiers set."""
    planned = _planned(staging, config={"runtime_tier": "gpu"})
    job_dir = tmp_path / "job"
    pack_job(
        job_dir,
        [planned],
        registry_entries=[
            ("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})
        ],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt", "runtime_tier": "gpu"},
        shared_table={},
    )
    return job_dir


@pytest.fixture()
def packed_job_shared(tmp_path, staging, _isolated_host_config):
    """A job whose video is referenced via the "labnas" shared-root alias
    rather than copied/symlinked in directly."""
    planned = _planned(staging, config={"runtime_tier": "cpu"})
    job_dir = tmp_path / "job"
    # Fix B9: the alias root is the video's OWN parent, so
    # match_shared_root yields relpath "colony.mp4" -- NOT "2026-09/colony.mp4".
    # The three shared tests below therefore place their fake mount's copy at
    # "<mount>/colony.mp4". An earlier draft had the tests writing to
    # "<mount>/2026-09/colony.mp4", which preflight never looked at, so
    # test_a_re_encoded_shared_video_fails_the_signature_check passed for the
    # WRONG REASON (file absent, not signature mismatch) and would have kept
    # passing even if the signature check were deleted entirely.
    table = {"labnas": str(staging["video"].parent)}
    pack_job(
        job_dir,
        [planned],
        registry_entries=[
            ("obb/x.pt", {"species": "ant", "source_path": "/host/a.pt"})
        ],
        advanced_config_path=str(staging["advanced"]),
        track_args={"video_list": "videos.txt", "runtime_tier": "cpu"},
        shared_table=table,
    )
    assert JobManifest.read(job_dir / "hydra_job.json").videos[0].shared == {
        "alias": "labnas",
        "relpath": "colony.mp4",
    }, "fixture and tests must agree on the shared relpath (fix B9)"
    return job_dir
