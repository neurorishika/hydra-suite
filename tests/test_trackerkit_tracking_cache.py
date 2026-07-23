from __future__ import annotations

import json

from hydra_suite.trackerkit.tracking_cache import (
    get_tracking_cache_model_ids,
    normalize_tracking_cache_value,
    plan_tracking_cache,
)
from hydra_suite.utils.video_artifacts import build_detection_cache_path


def _base_params():
    return {
        "DETECTION_METHOD": "background_subtraction",
        "RESIZE_FACTOR": 1.0,
        "MAX_TARGETS": 4,
        "COMPUTE_RUNTIME": "cpu",
    }


def test_plan_tracking_cache_reuses_existing_cache_only_when_enabled(tmp_path):
    video_path = tmp_path / "subject.mp4"
    preferred_dir = tmp_path / "preferred"
    legacy_dir = tmp_path / "legacy"
    preferred_dir.mkdir()
    legacy_dir.mkdir()

    params = _base_params()
    model_ids = get_tracking_cache_model_ids(params, "background_subtraction")
    reusable_cache = build_detection_cache_path(
        str(video_path),
        model_ids["inference"],
        artifact_base_dir=preferred_dir,
    )
    reusable_cache.parent.mkdir(parents=True, exist_ok=True)
    reusable_cache.write_text("cached")

    ignored_cache = build_detection_cache_path(
        str(video_path),
        model_ids["inference"],
        artifact_base_dir=legacy_dir,
    )
    ignored_cache.parent.mkdir(parents=True, exist_ok=True)
    ignored_cache.write_text("ignored")

    planned_without_reuse = plan_tracking_cache(
        str(video_path),
        params=dict(params),
        preferred_output_dir=str(preferred_dir),
        use_cached_detections=False,
    )
    planned_with_reuse = plan_tracking_cache(
        str(video_path),
        params=dict(params),
        preferred_output_dir=str(preferred_dir),
        use_cached_detections=True,
    )

    expected_fresh_cache = build_detection_cache_path(
        str(video_path),
        model_ids["inference"],
        artifact_base_dir=tmp_path,
    )

    assert planned_without_reuse.detection_cache_path == str(expected_fresh_cache)
    assert planned_with_reuse.detection_cache_path == str(reusable_cache)


def test_plan_tracking_cache_populates_model_ids(tmp_path):
    video_path = tmp_path / "subject.mp4"
    plan = plan_tracking_cache(
        str(video_path),
        params=_base_params(),
        preferred_output_dir=str(tmp_path),
        use_cached_detections=False,
    )

    assert plan.inference_model_id.startswith("bgsub_")
    assert plan.engine_model_id is None
    assert plan.detection_cache_path.endswith(f"{plan.inference_model_id}.npz")


def test_get_tracking_cache_model_ids_varies_with_yolo_obb_direct_task():
    base_params = {
        "DETECTION_METHOD": "yolo_obb",
        "RESIZE_FACTOR": 1.0,
        "MAX_TARGETS": 4,
        "COMPUTE_RUNTIME": "cpu",
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "yolo26s-obb.pt",
        "YOLO_OBB_DIRECT_TASK": "obb",
        "YOLO_OBB_FIXED_ANGLE_DEG": 0.0,
    }

    obb_task_ids = get_tracking_cache_model_ids(dict(base_params), "yolo_obb")
    detect_task_params = dict(base_params, YOLO_OBB_DIRECT_TASK="detect")
    detect_task_ids = get_tracking_cache_model_ids(detect_task_params, "yolo_obb")

    assert obb_task_ids["inference"] != detect_task_ids["inference"]

    angle_params = dict(
        base_params, YOLO_OBB_DIRECT_TASK="detect", YOLO_OBB_FIXED_ANGLE_DEG=90.0
    )
    angle_ids = get_tracking_cache_model_ids(angle_params, "yolo_obb")

    assert detect_task_ids["inference"] != angle_ids["inference"]


def test_get_tracking_cache_model_ids_varies_with_seg_kernel_params():
    base_params = {
        "DETECTION_METHOD": "yolo_obb",
        "RESIZE_FACTOR": 1.0,
        "MAX_TARGETS": 4,
        "COMPUTE_RUNTIME": "cpu",
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "yolo26s-seg.pt",
        "YOLO_OBB_DIRECT_TASK": "segment",
        "YOLO_OBB_SEG_NUM_ANGLES": 24,
        "YOLO_OBB_SEG_CROP_SIZE": 64,
        "YOLO_OBB_SEG_PAD_RATIO": 0.15,
        "YOLO_OBB_SEG_MASK_THRESHOLD": 0.5,
    }
    base_ids = get_tracking_cache_model_ids(dict(base_params), "yolo_obb")

    for key, changed_value in (
        ("YOLO_OBB_SEG_NUM_ANGLES", 48),
        ("YOLO_OBB_SEG_CROP_SIZE", 128),
        ("YOLO_OBB_SEG_PAD_RATIO", 0.25),
        ("YOLO_OBB_SEG_MASK_THRESHOLD", 0.35),
    ):
        varied_params = dict(base_params, **{key: changed_value})
        varied_ids = get_tracking_cache_model_ids(varied_params, "yolo_obb")
        assert (
            varied_ids["inference"] != base_ids["inference"]
        ), f"{key} did not change the cache fingerprint"


def test_normalize_tracking_cache_value_handles_native_nonfinite_floats():
    normalized = normalize_tracking_cache_value(
        {
            "nan": float("nan"),
            "pos_inf": float("inf"),
            "neg_inf": float("-inf"),
            "finite": 1.25,
        }
    )

    assert normalized == {
        "finite": 1.25,
        "nan": "NaN",
        "neg_inf": "-Infinity",
        "pos_inf": "Infinity",
    }
    assert json.dumps(normalized, sort_keys=True, allow_nan=False)
