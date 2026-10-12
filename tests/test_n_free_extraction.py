from types import SimpleNamespace

import numpy as np

from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION
from hydra_suite.core.inference.cache.keys import detection_cache_key
from hydra_suite.core.inference.config import (
    OBBConfig,
    OBBDirectConfig,
    build_inference_config_from_params,
)
from hydra_suite.core.inference.limits import (
    EXTRACTION_CONFIDENCE_FLOOR,
    MAX_DETECTIONS_PER_FRAME,
    DetectionLimitStats,
)
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.stages.obb import (
    effective_raw_detection_cap,
    rank_and_bound,
)


def _params(n):
    return {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": "some.pt",
        "COMPUTE_RUNTIME": "cpu",
        "MAX_TARGETS": n,
    }


def _obb(confs):
    n = len(confs)
    return OBBResult(
        frame_idx=3,
        centroids=np.zeros((n, 2), np.float32),
        angles=np.zeros(n, np.float32),
        sizes=np.ones(n, np.float32),
        shapes=np.ones((n, 2), np.float32),
        confidences=np.asarray(confs, np.float32),
        corners=np.zeros((n, 4, 2), np.float32),
        detection_ids=OBBResult.make_detection_ids(3, n),
    )


def test_schema_is_v6():
    assert CACHE_SCHEMA_VERSION == 6


def test_extraction_cap_ignores_n():
    for n in (1, 10, 500):
        cfg = build_inference_config_from_params(_params(n))
        assert cfg.obb.raw_detection_cap == 0
        # Extraction collects one past the limit so a real truncation is
        # observable (rank_and_bound then cuts to the limit and records it).
        assert effective_raw_detection_cap(cfg.obb) == MAX_DETECTIONS_PER_FRAME + 1
        assert cfg.obb.max_detections == n


def test_explicit_raw_cap_still_honoured_and_bounded():
    assert effective_raw_detection_cap(SimpleNamespace(raw_detection_cap=7)) == 7
    assert (
        effective_raw_detection_cap(SimpleNamespace(raw_detection_cap=5000))
        == MAX_DETECTIONS_PER_FRAME
    )


def test_builder_uses_extraction_floor():
    cfg = build_inference_config_from_params(_params(8))
    assert cfg.obb.direct.confidence_floor == EXTRACTION_CONFIDENCE_FLOOR
    assert (
        OBBDirectConfig(model_path="x.pt").confidence_floor
        == EXTRACTION_CONFIDENCE_FLOOR
    )
    assert OBBConfig().raw_detection_cap == 0


def test_detection_key_independent_of_n():
    k5 = detection_cache_key(build_inference_config_from_params(_params(5)).obb)
    k50 = detection_cache_key(build_inference_config_from_params(_params(50)).obb)
    assert k5 == k50


def test_rank_and_bound_sorts_by_confidence():
    out, count = rank_and_bound(_obb([0.2, 0.9, 0.5]))
    assert count == 3
    assert out.confidences.tolist() == [
        np.float32(0.9),
        np.float32(0.5),
        np.float32(0.2),
    ]
    assert out.detection_ids.tolist() == OBBResult.make_detection_ids(3, 3).tolist()


def test_rank_and_bound_truncates_and_records():
    confs = np.linspace(0.02, 0.99, MAX_DETECTIONS_PER_FRAME + 5)
    out, count = rank_and_bound(_obb(confs))
    assert count == MAX_DETECTIONS_PER_FRAME + 5
    assert out.num_detections == MAX_DETECTIONS_PER_FRAME
    assert out.confidences.min() > np.float32(0.02)
    stats = DetectionLimitStats()
    stats.record(3, count)
    assert "1 frame(s)" in stats.summary()


def test_exactly_the_limit_is_stored_whole_and_not_a_hit():
    confs = np.linspace(0.02, 0.99, MAX_DETECTIONS_PER_FRAME)
    out, count = rank_and_bound(_obb(confs))
    assert count == MAX_DETECTIONS_PER_FRAME
    assert out.num_detections == MAX_DETECTIONS_PER_FRAME
    assert not count > MAX_DETECTIONS_PER_FRAME


def test_probe_row_is_cut_and_is_a_hit():
    confs = np.linspace(0.02, 0.99, MAX_DETECTIONS_PER_FRAME + 1)
    out, count = rank_and_bound(_obb(confs))
    assert count == MAX_DETECTIONS_PER_FRAME + 1
    assert count > MAX_DETECTIONS_PER_FRAME
    assert out.num_detections == MAX_DETECTIONS_PER_FRAME
    assert out.confidences.min() > np.float32(0.02)


def test_runner_records_only_true_truncations(caplog):
    from hydra_suite.core.inference.runner import InferenceRunner

    runner = object.__new__(InferenceRunner)
    runner.detection_limit_stats = DetectionLimitStats()
    exact = runner._rank_and_bound(
        _obb(np.linspace(0.02, 0.99, MAX_DETECTIONS_PER_FRAME)), 40
    )
    assert exact.num_detections == MAX_DETECTIONS_PER_FRAME
    assert runner.detection_limit_stats.frames == []

    with caplog.at_level("WARNING"):
        cut = runner._rank_and_bound(
            _obb(np.linspace(0.02, 0.99, MAX_DETECTIONS_PER_FRAME + 1)), 41
        )
    assert cut.num_detections == MAX_DETECTIONS_PER_FRAME
    assert runner.detection_limit_stats.frames == [(41, MAX_DETECTIONS_PER_FRAME + 1)]
    assert "Frame 41" in caplog.text
