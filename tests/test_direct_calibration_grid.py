import pytest

from hydra_suite.core.inference.direct_calibration_grid import (
    build_candidate_grid,
    estimate_grid_work,
)

TRAINING = {
    "geometry_mode": "auto_object",
    "imgsz": 640,
    "object_tile_fraction": 0.4,
    "overlap": 0.2,
    "reference_body_px": 560.0,
}


def test_grid_always_contains_full_frame_and_training_geometry():
    grid = build_candidate_grid(TRAINING)
    assert any(not c.enabled for c in grid), "full-frame baseline missing"
    training = [
        c
        for c in grid
        if c.enabled
        and c.object_tile_fraction == pytest.approx(0.4)
        and c.overlap == pytest.approx(0.2)
    ]
    assert len(training) == 1 and training[0].label == "Training geometry"


def test_grid_is_deduplicated_and_every_row_is_labelled():
    grid = build_candidate_grid(TRAINING)
    keys = [
        (
            c.enabled,
            c.geometry_mode,
            c.slice_width,
            c.slice_height,
            round(c.overlap, 4),
            round(c.object_tile_fraction, 4),
        )
        for c in grid
    ]
    assert len(keys) == len(set(keys))
    assert all(c.label for c in grid)


def test_slice_params_use_the_real_param_keys():
    """These keys are read verbatim by config._slice_config_from_params."""
    candidate = build_candidate_grid(TRAINING)[1]
    params = candidate.slice_params()
    assert params["SLICE_ENABLED"] is True
    assert params["SLICE_GEOMETRY_MODE"] == "auto_object"
    assert params["SLICE_OVERLAP"] == pytest.approx(candidate.overlap)
    assert params["SLICE_OBJECT_TILE_FRACTION"] == pytest.approx(
        candidate.object_tile_fraction
    )
    assert params["SLICE_TRAINED_BODY_PX"] == pytest.approx(candidate.trained_body_px)
    assert set(params) <= {
        "SLICE_ENABLED",
        "SLICE_GEOMETRY_MODE",
        "SLICE_WIDTH",
        "SLICE_HEIGHT",
        "SLICE_OVERLAP",
        "SLICE_OBJECT_TILE_FRACTION",
        "SLICE_TRAINED_BODY_PX",
    }


def test_full_frame_candidate_disables_slicing():
    full = next(c for c in build_candidate_grid(TRAINING) if not c.enabled)
    assert full.slice_params()["SLICE_ENABLED"] is False


def test_custom_geometry_is_appended_when_requested():
    grid = build_candidate_grid(TRAINING, custom=(1024, 768))
    custom = [c for c in grid if c.geometry_mode == "custom"]
    assert custom and custom[0].slice_width == 1024 and custom[0].slice_height == 768
    assert custom[0].slice_params()["SLICE_WIDTH"] == 1024


def test_work_estimate_reports_tiles_and_flags_over_budget():
    grid = build_candidate_grid(TRAINING)
    estimates = estimate_grid_work(grid, frame_hw=(2160, 3840), imgsz=640, frames=80)
    assert len(estimates) == len(grid)
    full = next(e for e in estimates if not e.candidate.enabled)
    assert full.tiles_per_frame == 1
    assert all(e.total_tiles == e.tiles_per_frame * 80 for e in estimates)
    huge = estimate_grid_work(grid, frame_hw=(2160, 3840), imgsz=640, frames=10**6)
    assert any(e.failed_reason for e in huge)


def test_unplannable_candidate_is_flagged_not_dropped():
    grid = build_candidate_grid(TRAINING, custom=(1, 1))
    estimates = estimate_grid_work(grid, frame_hw=(2160, 3840), imgsz=640, frames=10)
    assert len(estimates) == len(grid), "a failed candidate must still get a row"
    custom = next(e for e in estimates if e.candidate.geometry_mode == "custom")
    assert custom.failed_reason


def _checkpoint(tmp_path, payload=b"weights"):
    path = tmp_path / "m.pt"
    path.write_bytes(payload)
    return path


def test_fingerprint_distinguishes_every_geometry(tmp_path):
    from hydra_suite.core.inference.direct_calibration_grid import (
        build_candidate_grid,
        candidate_cache_fingerprint,
    )

    args = dict(
        checkpoint_path=_checkpoint(tmp_path),
        task="obb",
        image_paths=[tmp_path / "a.png"],
        imgsz=640,
        executor_key="torch:cpu",
        max_detections=64,
        confidence_floor=1e-3,
    )
    keys = {
        candidate_cache_fingerprint(candidate=c, **args)
        for c in build_candidate_grid(TRAINING)
    }
    assert len(keys) == len(build_candidate_grid(TRAINING))


def test_fingerprint_changes_when_weights_or_cap_change(tmp_path):
    from hydra_suite.core.inference.direct_calibration_grid import (
        build_candidate_grid,
        candidate_cache_fingerprint,
    )

    checkpoint = _checkpoint(tmp_path)
    candidate = build_candidate_grid(TRAINING)[1]
    args = dict(
        checkpoint_path=checkpoint,
        task="obb",
        image_paths=[tmp_path / "a.png"],
        candidate=candidate,
        imgsz=640,
        executor_key="torch:cpu",
        max_detections=64,
        confidence_floor=1e-3,
    )
    before = candidate_cache_fingerprint(**args)
    assert candidate_cache_fingerprint(**{**args, "max_detections": 8}) != before
    checkpoint.write_bytes(b"retrained")
    assert candidate_cache_fingerprint(**args) != before


def test_checkpoint_fingerprint_is_prefixed_and_stable(tmp_path):
    from hydra_suite.core.inference.direct_calibration_grid import (
        checkpoint_fingerprint,
    )

    checkpoint = _checkpoint(tmp_path)
    digest = checkpoint_fingerprint(checkpoint)
    assert digest.startswith("sha256:") and len(digest) == 71
    assert digest == checkpoint_fingerprint(checkpoint)
