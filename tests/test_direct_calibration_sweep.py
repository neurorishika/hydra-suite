import numpy as np
import pytest

from hydra_suite.core.inference.direct_calibration_sweep import (
    MergeSettings,
    build_calibration_config,
    config_for_point,
    detections_from_result,
    rescore_parts,
)
from hydra_suite.core.inference.stages import obb as obb_stage
from hydra_suite.core.inference.stages.filtering import filter_for_source

SLICE_PARAMS = {
    "SLICE_ENABLED": True,
    "SLICE_GEOMETRY_MODE": "auto_object",
    "SLICE_OBJECT_TILE_FRACTION": 0.4,
    "SLICE_OVERLAP": 0.2,
    "SLICE_TRAINED_BODY_PX": 120.0,
}


def test_config_is_built_from_params_and_carries_every_claimed_field(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    config = build_calibration_config(
        str(model),
        slice_params=SLICE_PARAMS,
        max_targets=64,
        confidence=0.35,
        runtime_tier="cpu",
    )
    slice_cfg = config.obb.direct.slice
    assert slice_cfg.enabled is True
    assert slice_cfg.geometry_mode == "auto_object"
    assert slice_cfg.object_tile_fraction == pytest.approx(0.4)
    assert slice_cfg.overlap_width_ratio == pytest.approx(0.2)
    assert slice_cfg.reference_body_px == pytest.approx(120.0)
    assert config.obb.confidence_threshold == pytest.approx(0.35)
    assert config.obb.max_detections == 64
    assert config.obb.direct.model_path == str(model)


def test_slice_disabled_is_carried_through(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    params = dict(SLICE_PARAMS)
    params["SLICE_ENABLED"] = False
    config = build_calibration_config(
        str(model),
        slice_params=params,
        max_targets=64,
        confidence=0.35,
        runtime_tier="cpu",
    )
    assert config.obb.direct.slice.enabled is False


def test_merge_settings_reach_the_slice_config(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    config = config_for_point(
        str(model),
        slice_params=SLICE_PARAMS,
        merge=MergeSettings("nmm", "iou", 0.65),
        confidence=0.35,
        max_targets=64,
        runtime_tier="cpu",
        model_task="obb",
    )
    assert config.obb.direct.slice.merge_policy == "nmm"
    assert config.obb.direct.slice.merge_metric == "iou"
    assert config.obb.direct.slice.merge_threshold == pytest.approx(0.65)


def test_offline_rescore_matches_a_fresh_production_run(direct_obb_fixture):
    frames, models, config, runtime = direct_obb_fixture
    model_path = config.obb.direct.model_path
    parts_by_frame, source = obb_stage.collect_obb_parts_by_frame(
        frames, models, config.obb, runtime
    )
    merge = MergeSettings("greedy_nmm", "ios", 0.5)
    for confidence in (0.10, 0.35, 0.60):
        point_config = config_for_point(
            model_path,
            slice_params=SLICE_PARAMS,
            merge=merge,
            confidence=confidence,
            max_targets=config.obb.max_detections,
            runtime_tier="cpu",
            model_task="obb",
        )
        fresh = []
        for raw in obb_stage.run_obb(frames, models, point_config.obb, runtime):
            if isinstance(raw, obb_stage._RawOBBTensors):
                raw = obb_stage.materialize_tensors(
                    raw, point_config.obb.raw_detection_cap
                )
            fresh.append(filter_for_source(point_config, raw, None)[0])
        cached = [
            rescore_parts(parts_by_frame[i], source, point_config, runtime, frame_idx=i)
            for i in range(len(frames))
        ]
        for want, got in zip(fresh, cached):
            assert want.num_detections == got.num_detections
            np.testing.assert_array_equal(want.centroids, got.centroids)
            np.testing.assert_array_equal(want.confidences, got.confidences)


def test_max_detections_truncation_is_visible_in_rescoring(direct_obb_fixture):
    """A too-small max_detections silently caps recall -- prove it bites."""
    frames, models, config, runtime = direct_obb_fixture
    model_path = config.obb.direct.model_path
    parts_by_frame, source = obb_stage.collect_obb_parts_by_frame(
        frames, models, config.obb, runtime
    )
    merge = MergeSettings("greedy_nmm", "ios", 0.5)
    counts = []
    for max_targets in (2, 64):
        point_config = config_for_point(
            model_path,
            slice_params=SLICE_PARAMS,
            merge=merge,
            confidence=0.05,
            max_targets=max_targets,
            runtime_tier="cpu",
            model_task="obb",
        )
        counts.append(
            rescore_parts(
                parts_by_frame[0], source, point_config, runtime, frame_idx=0
            ).num_detections
        )
    assert counts[0] <= 2
    assert counts[0] <= counts[1]


def test_detections_carry_frame_space_polygons_and_class_ids(direct_obb_fixture):
    frames, models, config, runtime = direct_obb_fixture
    result = obb_stage.run_obb(frames, models, config.obb, runtime)[0]
    if isinstance(result, obb_stage._RawOBBTensors):
        result = obb_stage.materialize_tensors(result, config.obb.raw_detection_cap)
    detections = detections_from_result(result)
    assert len(detections) == result.num_detections
    for detection in detections:
        assert detection.polygon_px.ndim == 2 and detection.polygon_px.shape[1] == 2
        assert detection.polygon_px.shape[0] >= 3
        assert isinstance(detection.class_id, int)
