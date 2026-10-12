"""SAHI-ENABLED byte-identity guard: no equivalence fixture enables slicing."""

from dataclasses import asdict

from hydra_suite.core.inference.cache.keys import _slice_config_hash
from hydra_suite.core.inference.config import _slice_config_from_params

# Captured from main BEFORE Task 10 by running this module's _snapshot() on the
# unmodified tree and pasting the printed values below.
PARAMS = {
    "SLICE_ENABLED": True,
    "SLICE_GEOMETRY_MODE": "auto_object",
    "REFERENCE_BODY_SIZE": 40.0,
    "RESIZE_FACTOR": 1.0,
}

EXPECTED_CFG = {
    "enabled": True,
    "geometry_mode": "auto_object",
    "slice_height": 0,
    "slice_width": 0,
    "overlap_height_ratio": 0.2,
    "overlap_width_ratio": 0.2,
    "object_tile_fraction": 0.15,
    "reference_body_px": 40.0,
    "merge_policy": "greedy_nmm",
    "merge_metric": "ios",
    "merge_threshold": 0.5,
    "merge_backend": "cv2",
    "perform_standard_pred": False,
    "tile_batch_size": 16,
    "tile_memory_budget_bytes": 268435456,
}
EXPECTED_HASH = "8f030f5f2f4e1b98f041ebbaf8715df599f49d292a3cd2990332ba9b9ceb354b"


def _snapshot():
    cfg = _slice_config_from_params(PARAMS, "SLICE_", reference_body_px=40.0)
    return asdict(cfg), _slice_config_hash(cfg)


def test_sahi_enabled_slice_config_is_unchanged():
    cfg, digest = _snapshot()
    assert cfg == EXPECTED_CFG
    assert digest == EXPECTED_HASH
