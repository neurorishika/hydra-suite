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
