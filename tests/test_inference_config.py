import json
import tempfile

import pytest

from hydra_suite.core.inference.config import (
    CNNConfig,
    HeadTailConfig,
    InferenceConfig,
    InferenceConfigError,
    OBBConfig,
    OBBDirectConfig,
    OBBSequentialConfig,
    SliceConfig,
    build_inference_config_from_params,
)


def _minimal_cpu_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/tmp/obb.pt"),
        ),
        runtime_tier="cpu",
    )


def _minimal_cuda_config() -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/tmp/obb.pt"),
        ),
        runtime_tier="gpu",
    )


def test_from_json_round_trip():
    config = _minimal_cpu_config()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.obb.mode == "direct"
    assert loaded.obb.direct.model_path == "/tmp/obb.pt"
    assert loaded.runtime_tier == "cpu"


def test_round_trip_with_headtail():
    config = InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt")),
        headtail=HeadTailConfig(model_path="/ht.pt"),
        detection_batch_size=4,
        runtime_tier="cpu",
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.headtail.model_path == "/ht.pt"
    assert loaded.detection_batch_size == 4


def test_round_trip_with_cnn_phases():
    config = InferenceConfig(
        obb=OBBConfig(mode="direct", direct=OBBDirectConfig(model_path="/m.pt")),
        cnn_phases=[
            CNNConfig(label="identity", model_path="/cnn.pt"),
        ],
        runtime_tier="cpu",
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert len(loaded.cnn_phases) == 1
    assert loaded.cnn_phases[0].label == "identity"


def test_sequential_config_round_trip():
    config = InferenceConfig(
        obb=OBBConfig(
            mode="sequential",
            sequential=OBBSequentialConfig(
                detect_model_path="/detect.pt",
                obb_model_path="/obb.pt",
                detect_confidence_threshold=0.1,
                obb_confidence_threshold=0.05,
            ),
        ),
        runtime_tier="gpu_fast",
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        config.to_json(f.name)
        path = f.name
    loaded = InferenceConfig.from_json(path)
    assert loaded.obb.sequential.detect_model_path == "/detect.pt"
    assert loaded.obb.sequential.obb_model_path == "/obb.pt"
    assert loaded.obb.sequential.detect_confidence_threshold == pytest.approx(0.1)
    assert loaded.runtime_tier == "gpu_fast"


def test_obb_direct_config_model_task_round_trips(tmp_path):
    config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                model_path="yolo26s-seg.pt",
                model_task="segment",
                fixed_angle_deg=0.0,
            ),
        )
    )
    path = tmp_path / "cfg.json"
    config.to_json(str(path))
    loaded = InferenceConfig.from_json(str(path))

    assert loaded.obb.direct.model_task == "segment"
    assert loaded.obb.direct.fixed_angle_deg == 0.0


def test_obb_direct_config_model_task_defaults_to_obb():
    direct = OBBDirectConfig(model_path="yolo26s-obb.pt")
    assert direct.model_task == "obb"
    assert direct.fixed_angle_deg == 0.0


def test_obb_direct_config_seg_kernel_params_round_trip(tmp_path):
    config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                model_path="yolo26s-seg.pt",
                model_task="segment",
                seg_num_angles=48,
                seg_crop_size=128,
                seg_pad_ratio=0.25,
                seg_mask_threshold=0.35,
            ),
        )
    )
    path = tmp_path / "cfg.json"
    config.to_json(str(path))
    loaded = InferenceConfig.from_json(str(path))

    assert loaded.obb.direct.seg_num_angles == 48
    assert loaded.obb.direct.seg_crop_size == 128
    assert loaded.obb.direct.seg_pad_ratio == pytest.approx(0.25)
    assert loaded.obb.direct.seg_mask_threshold == pytest.approx(0.35)


def test_obb_direct_config_seg_kernel_params_default_to_kernel_defaults():
    direct = OBBDirectConfig(model_path="yolo26s-seg.pt")
    assert direct.seg_num_angles == 24
    assert direct.seg_crop_size == 64
    assert direct.seg_pad_ratio == pytest.approx(0.15)
    assert direct.seg_mask_threshold == pytest.approx(0.5)


def test_slice_config_defaults_off():
    s = SliceConfig()
    assert s.enabled is False
    assert s.geometry_mode == "auto_model"
    assert s.overlap_height_ratio == 0.2 and s.overlap_width_ratio == 0.2
    assert s.merge_policy == "greedy_nmm"
    assert s.merge_metric == "ios"
    assert s.merge_threshold == 0.5
    assert s.merge_backend == "cv2"
    assert s.perform_standard_pred is False


def test_obb_direct_config_has_slice_default():
    d = OBBDirectConfig(model_path="m.pt")
    assert isinstance(d.slice, SliceConfig)
    assert d.slice.enabled is False


def test_obb_direct_from_dict_parses_nested_slice():
    obb = OBBConfig.from_dict(
        {
            "mode": "direct",
            "direct": {
                "model_path": "m.pt",
                "slice": {
                    "enabled": True,
                    "geometry_mode": "custom",
                    "slice_height": 640,
                    "slice_width": 640,
                },
            },
        }
    )
    assert obb.direct.slice.enabled is True
    assert obb.direct.slice.geometry_mode == "custom"
    assert obb.direct.slice.slice_height == 640


def test_obb_iou_threshold_defaults_to_legacy_value_through_production_path(
    tmp_path,
):
    """`_dict_to_config` (the production `InferenceConfig.from_json` path) must
    default a missing `iou_threshold` to the legacy YOLO_IOU_THRESHOLD value
    (0.45), not `OBBConfig`'s own dataclass default (0.7). `OBBConfig.from_dict`
    must match this exactly since `_dict_to_config` now delegates to it — a
    regression here would silently change production filtering behavior."""
    raw = {
        "obb": {
            "mode": "direct",
            "direct": {"model_path": "m.pt"},
            # iou_threshold intentionally omitted.
        },
        "runtime_tier": "cpu",
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(raw))

    loaded = InferenceConfig.from_json(str(path))
    assert loaded.obb.iou_threshold == pytest.approx(0.45)

    # Same dict shape through OBBConfig.from_dict directly must agree.
    obb = OBBConfig.from_dict(dict(raw["obb"]))
    assert obb.iou_threshold == pytest.approx(0.45)


def test_obb_config_full_round_trip_preserves_iou_threshold(tmp_path):
    config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="m.pt"),
            iou_threshold=0.6,
        ),
        runtime_tier="cpu",
    )
    path = tmp_path / "cfg.json"
    config.to_json(str(path))
    loaded = InferenceConfig.from_json(str(path))
    assert loaded.obb.iou_threshold == pytest.approx(0.6)
    assert loaded.obb.mode == "direct"
    assert loaded.obb.direct.model_path == "m.pt"


def test_build_config_reads_slice_params():
    params = {
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "m.pt",
        "SLICE_ENABLED": True,
        "SLICE_GEOMETRY_MODE": "custom",
        "SLICE_HEIGHT": 512,
        "SLICE_WIDTH": 512,
        "SLICE_OVERLAP": 0.25,
        "SLICE_MERGE_POLICY": "nms",
        "SLICE_MERGE_METRIC": "iou",
        "SLICE_MERGE_THRESHOLD": 0.4,
        "SLICE_MERGE_BACKEND": "gpu",
        "SLICE_OBJECT_TILE_FRACTION": 0.2,
        "SLICE_PERFORM_STANDARD_PRED": True,
    }
    cfg = build_inference_config_from_params(params)
    s = cfg.obb.direct.slice
    assert s.enabled is True
    assert s.geometry_mode == "custom"
    assert s.slice_height == 512 and s.slice_width == 512
    assert s.overlap_height_ratio == 0.25 and s.overlap_width_ratio == 0.25
    assert s.merge_policy == "nms" and s.merge_metric == "iou"
    assert s.merge_threshold == 0.4 and s.merge_backend == "gpu"
    assert s.object_tile_fraction == 0.2
    assert s.perform_standard_pred is True


def test_build_config_slice_defaults_when_absent():
    params = {"YOLO_OBB_MODE": "direct", "YOLO_OBB_DIRECT_MODEL_PATH": "m.pt"}
    cfg = build_inference_config_from_params(params)
    assert cfg.obb.direct.slice.enabled is False


def test_slice_config_rejects_bad_geometry_mode():
    with pytest.raises(InferenceConfigError, match="geometry_mode"):
        SliceConfig(geometry_mode="bogus")


def test_slice_config_rejects_bad_merge_policy():
    """A hand-edited advanced_config.json typo (e.g. 'nsm' for 'nms') must be
    rejected loudly instead of silently taking the union branch in
    stages/merge.py's ``if policy == "nms" or len(group) == 1:`` check."""
    with pytest.raises(InferenceConfigError, match="merge_policy"):
        SliceConfig(merge_policy="nsm")


def test_slice_config_rejects_bad_merge_metric():
    with pytest.raises(InferenceConfigError, match="merge_metric"):
        SliceConfig(merge_metric="bogus")


def test_slice_config_rejects_bad_merge_backend():
    with pytest.raises(InferenceConfigError, match="merge_backend"):
        SliceConfig(merge_backend="bogus")


def test_slice_config_defaults_construct_without_raising():
    SliceConfig()


def test_slice_config_valid_non_default_combination_constructs():
    s = SliceConfig(
        enabled=True,
        geometry_mode="custom",
        slice_height=512,
        slice_width=512,
        merge_policy="nms",
        merge_metric="iou",
        merge_backend="gpu",
    )
    assert s.geometry_mode == "custom"
    assert s.merge_policy == "nms"
    assert s.merge_metric == "iou"
    assert s.merge_backend == "gpu"


def test_obb_direct_config_construction_paths_all_valid():
    """SliceConfig is built via SliceConfig() defaults, SliceConfig(**slice_d)
    in OBBConfig.from_dict, and the param-builder -- confirm none of them now
    raise from the new __post_init__ validation."""
    assert OBBDirectConfig(model_path="m.pt").slice.enabled is False

    obb = OBBConfig.from_dict(
        {
            "mode": "direct",
            "direct": {
                "model_path": "m.pt",
                "slice": {"enabled": True, "merge_policy": "nms"},
            },
        }
    )
    assert obb.direct.slice.merge_policy == "nms"

    cfg = build_inference_config_from_params(
        {"YOLO_OBB_MODE": "direct", "YOLO_OBB_DIRECT_MODEL_PATH": "m.pt"}
    )
    assert cfg.obb.direct.slice.enabled is False


def test_reference_body_px_sourced_and_resize_scaled():
    """auto_object needs a real object scale; it comes from REFERENCE_BODY_SIZE
    * RESIZE_FACTOR, the same source/scaling worker.py uses (worker.py:921)."""
    params = {
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "m.pt",
        "SLICE_ENABLED": True,
        "REFERENCE_BODY_SIZE": 30.0,
        "RESIZE_FACTOR": 2.0,
    }
    cfg = build_inference_config_from_params(params)
    assert cfg.obb.direct.slice.reference_body_px == 60.0
