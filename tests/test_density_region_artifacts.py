"""Regression coverage for confidence-density artifact provenance."""

import numpy as np

from hydra_suite.core.tracking.confidence.density_artifacts import (
    density_regions_cache_key,
    density_regions_cache_path,
)


def _params():
    return {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.7,
        "YOLO_TARGET_CLASSES": [0],
        "MAX_TARGETS": 2,
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 1.0,
        "DENSITY_DOWNSAMPLE_FACTOR": 8,
        "DENSITY_MIN_AREA_BODIES": 0.25,
        "ROI_MASK": np.array([[0, 1], [1, 1]], dtype=np.uint8),
        "ARENA_LABELS": np.array([[0, 1], [1, 1]], dtype=np.uint8),
        "N_ARENAS": 1,
        "ANIMALS_PER_ARENA": 2,
    }


def test_density_regions_are_keyed_by_candidate_filtering_and_range(tmp_path):
    params = _params()
    base = density_regions_cache_path(tmp_path, params, 0, 99)

    changed_filter = _params()
    changed_filter["YOLO_CONFIDENCE_THRESHOLD"] = 0.1
    assert density_regions_cache_path(tmp_path, changed_filter, 0, 99) != base
    assert density_regions_cache_path(tmp_path, params, 10, 99) != base


def test_density_regions_are_bound_to_raw_detection_cache_generation(tmp_path):
    params = _params()
    model_a = density_regions_cache_path(
        tmp_path, params, 0, 99, source_signature="raw-cache-model-a"
    )
    model_b = density_regions_cache_path(
        tmp_path, params, 0, 99, source_signature="raw-cache-model-b"
    )

    assert model_a != model_b


def test_density_regions_key_includes_full_roi_and_arena_contents():
    params = _params()
    baseline = density_regions_cache_key(params, 0, 99)

    changed_roi = _params()
    changed_roi["ROI_MASK"] = changed_roi["ROI_MASK"].copy()
    changed_roi["ROI_MASK"][0, 0] = 1
    assert density_regions_cache_key(changed_roi, 0, 99) != baseline

    changed_arena = _params()
    changed_arena["ARENA_LABELS"] = changed_arena["ARENA_LABELS"].copy()
    changed_arena["ARENA_LABELS"][0, 0] = 2
    assert density_regions_cache_key(changed_arena, 0, 99) != baseline
