import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# Add both src and repo root to path for imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
