"""Tests for the unified precompute pipeline."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from hydra_suite.core.tracking.precompute import CropConfig, UnifiedPrecompute

# ---------------------------------------------------------------------------
# CropConfig
# ---------------------------------------------------------------------------


def test_crop_config_defaults():
    cfg = CropConfig()
    assert cfg.padding_fraction == 0.1
    assert cfg.suppress_foreign is True
    assert cfg.bg_color == (0, 0, 0)
    assert cfg.reference_aspect_ratio == 2.0


def test_crop_config_custom():
    cfg = CropConfig(padding_fraction=0.2, suppress_foreign=False, bg_color=(255, 0, 0))
    assert cfg.padding_fraction == 0.2
    assert cfg.suppress_foreign is False
    assert cfg.bg_color == (255, 0, 0)


# ---------------------------------------------------------------------------
# UnifiedPrecompute helpers
# ---------------------------------------------------------------------------


def _make_phase(name, is_fatal=False, cache_hit=False, finalize_return=None):
    """Build a mock PrecomputePhase."""
    p = Mock()
    p.name = name
    p.is_fatal = is_fatal
    p.has_cache_hit.return_value = cache_hit
    p.finalize.return_value = finalize_return
    return p


def _make_cap(frame=None):
    cap = Mock()
    if frame is None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    return cap


def _make_det_cache():
    dc = Mock()
    dc.get_frame.return_value = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        None,
        None,
        None,
    )
    return dc


def _make_detector():
    det = Mock()
    det.filter_raw_detections.return_value = ([], [], [], [], [], [], [], [], [])
    return det


# ---------------------------------------------------------------------------
# Empty phase list
# ---------------------------------------------------------------------------


def test_empty_phases_returns_empty_dict_without_reading_cap():
    cap = _make_cap()
    up = UnifiedPrecompute([], CropConfig())
    result = up.run(
        cap=cap,
        detection_cache=_make_det_cache(),
        detector=_make_detector(),
        start_frame=0,
        end_frame=5,
        resize_factor=1.0,
        roi_mask=None,
    )
    assert result == {}
    cap.read.assert_not_called()


# ---------------------------------------------------------------------------
# Dispatch — process_frame called once per frame per phase
# ---------------------------------------------------------------------------


def test_process_frame_called_once_per_frame_per_phase():
    p1 = _make_phase("a", finalize_return="/a")
    p2 = _make_phase("b", finalize_return="/b")

    up = UnifiedPrecompute([p1, p2], CropConfig())
    up.run(_make_cap(), _make_det_cache(), _make_detector(), 0, 2, 1.0, None)

    assert p1.process_frame.call_count == 3  # frames 0, 1, 2
    assert p2.process_frame.call_count == 3


# ---------------------------------------------------------------------------
# Empty detections → process_frame called with empty lists
# ---------------------------------------------------------------------------


def test_process_frame_called_with_empty_crops_when_no_detections():
    p = _make_phase("x", finalize_return=None)
    up = UnifiedPrecompute([p], CropConfig())
    up.run(_make_cap(), _make_det_cache(), _make_detector(), 0, 0, 1.0, None)

    args = p.process_frame.call_args[0]  # positional args
    # args: (frame_idx, crops, det_ids, crop_det_indices, all_obb, crop_offsets)
    frame_idx, crops, det_ids, crop_det_indices, all_obb, crop_offsets = args
    assert frame_idx == 0
    assert crops == []
    assert det_ids == []
    assert all_obb == []
    assert crop_offsets == []


# ---------------------------------------------------------------------------
# All-cache-hit: video read skipped
# ---------------------------------------------------------------------------


def test_all_cache_hit_skips_video_read():
    p = _make_phase("x", cache_hit=True, finalize_return="/cached")
    cap = _make_cap()
    up = UnifiedPrecompute([p], CropConfig())
    result = up.run(cap, _make_det_cache(), _make_detector(), 0, 5, 1.0, None)
    cap.read.assert_not_called()
    assert result == {"x": "/cached"}


# ---------------------------------------------------------------------------
# Partial cache hit: video runs; hit phase process_frame no-op, miss phase runs
# ---------------------------------------------------------------------------


def test_partial_cache_hit_video_runs_miss_phase_gets_frames():
    hit = _make_phase("hit", cache_hit=True, finalize_return="/old")
    miss = _make_phase("miss", cache_hit=False, finalize_return="/new")

    up = UnifiedPrecompute([hit, miss], CropConfig())
    result = up.run(_make_cap(), _make_det_cache(), _make_detector(), 0, 1, 1.0, None)

    assert miss.process_frame.call_count == 2  # both frames processed
    assert (
        hit.process_frame.call_count == 2
    )  # hit phase is still called (it's a no-op internally)
    assert result == {"hit": "/old", "miss": "/new"}


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_stop_check_exits_loop_calls_close_not_finalize():
    p = _make_phase("y")
    cap = _make_cap()

    call_count = [0]

    def stop_after_first():
        call_count[0] += 1
        return call_count[0] > 1

    up = UnifiedPrecompute([p], CropConfig())
    result = up.run(
        cap,
        _make_det_cache(),
        _make_detector(),
        0,
        10,
        1.0,
        None,
        stop_check=stop_after_first,
    )
    p.finalize.assert_not_called()
    p.close.assert_called_once()
    assert result == {"y": None}


# ---------------------------------------------------------------------------
# Fatal phase raises in finalize → propagates; close still called
# ---------------------------------------------------------------------------


def test_fatal_phase_finalize_raises_propagates_and_close_called():
    p = _make_phase("pose", is_fatal=True)
    p.finalize.side_effect = RuntimeError("inference failed")

    up = UnifiedPrecompute([p], CropConfig())
    with pytest.raises(RuntimeError, match="inference failed"):
        up.run(_make_cap(), _make_det_cache(), _make_detector(), 0, 0, 1.0, None)

    p.close.assert_called_once()


# ---------------------------------------------------------------------------
# Non-fatal phase raises in finalize → None in result; warning_cb called; close called
# ---------------------------------------------------------------------------


def test_nonfatal_phase_finalize_raises_returns_none_and_warns():
    p = _make_phase("apriltag", is_fatal=False)
    p.finalize.side_effect = RuntimeError("at failed")
    warnings = []

    up = UnifiedPrecompute([p], CropConfig())
    result = up.run(
        _make_cap(),
        _make_det_cache(),
        _make_detector(),
        0,
        0,
        1.0,
        None,
        warning_cb=lambda t, m: warnings.append((t, m)),
    )
    assert result == {"apriltag": None}
    assert len(warnings) == 1
    p.close.assert_called_once()


# ---------------------------------------------------------------------------
# finalize + close called on ALL phases even when one raises
# ---------------------------------------------------------------------------


def test_close_called_on_all_phases_when_one_finalize_raises():
    p1 = _make_phase("ok", finalize_return="/ok")
    p2 = _make_phase("bad", is_fatal=False)
    p2.finalize.side_effect = RuntimeError("oops")

    up = UnifiedPrecompute([p1, p2], CropConfig())
    up.run(_make_cap(), _make_det_cache(), _make_detector(), 0, 0, 1.0, None)

    p1.close.assert_called_once()
    p2.close.assert_called_once()


# ---------------------------------------------------------------------------
# crop_offsets int-cast and passed through to process_frame
# ---------------------------------------------------------------------------


def test_crop_offsets_passed_to_process_frame():
    """crop_offsets from extract_one_crop are passed to process_frame correctly."""
    p = _make_phase("x", finalize_return=None)

    corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    fake_crop = np.zeros((10, 10, 3), dtype=np.uint8)
    fake_offset = (5, 7)

    # Provide one detection via the detection cache/detector mocks
    det_cache = _make_det_cache()
    det_cache.get_frame.return_value = (
        [[5.0, 5.0, 0.0]],
        [[10, 10]],
        [["rect"]],
        [[0.9]],
        [corners.tolist()],
        [0],
        [0.0],
        [0.0],
        [False],
        None,
        None,
        None,
    )
    detector = _make_detector()
    detector.filter_raw_detections.return_value = (
        [[5.0, 5.0, 0.0]],
        [[10, 10]],
        [["rect"]],
        [[0.9]],
        [corners],
        [0],
        [0.0],
        [0.0],
        [False],
    )

    with patch(
        "hydra_suite.core.tracking.precompute.extract_one_crop",
        return_value=(fake_crop, fake_offset, None),
    ):
        up = UnifiedPrecompute([p], CropConfig())
        up.run(_make_cap(), det_cache, detector, 0, 0, 1.0, None)

    args = p.process_frame.call_args[0]
    frame_idx, crops, det_ids, crop_det_indices, all_obb, crop_offsets = args
    assert len(crop_offsets) == 1
    assert crop_offsets[0] == (5, 7)  # int tuple


# ---------------------------------------------------------------------------
# cap.read() failure mid-sequence → stops early
# ---------------------------------------------------------------------------


def test_cap_read_failure_mid_sequence_stops_early():
    p = _make_phase("x", finalize_return="/x")
    cap = Mock()
    cap.read.side_effect = [
        (True, np.zeros((10, 10, 3), dtype=np.uint8)),
        (True, np.zeros((10, 10, 3), dtype=np.uint8)),
        (False, None),
    ]
    up = UnifiedPrecompute([p], CropConfig())
    result = up.run(cap, _make_det_cache(), _make_detector(), 0, 4, 1.0, None)
    # stopped at frame 2, frames 3-4 not read
    assert cap.read.call_count == 3
    assert result == {"x": "/x"}


# ---------------------------------------------------------------------------
# All-cache-hit path: non-fatal finalize raises → None in result; warning_cb called
# ---------------------------------------------------------------------------


def test_all_cache_hit_nonfatal_finalize_raises_returns_none_and_warns():
    p = _make_phase("x", cache_hit=True, is_fatal=False)
    p.finalize.side_effect = RuntimeError("cache corrupt")
    warnings = []

    up = UnifiedPrecompute([p], CropConfig())
    result = up.run(
        _make_cap(),
        _make_det_cache(),
        _make_detector(),
        0,
        5,
        1.0,
        None,
        warning_cb=lambda t, m: warnings.append((t, m)),
    )
    assert result == {"x": None}
    assert len(warnings) == 1
    p.close.assert_called_once()


# ---------------------------------------------------------------------------
# AprilTagPrecomputePhase
# ---------------------------------------------------------------------------


def test_apriltag_phase_cache_miss_creates_detector(tmp_path):
    cache_path = tmp_path / "tags_0_9.npz"  # does not exist
    with patch("hydra_suite.core.tracking.precompute.AprilTagDetector") as MockDet:
        from hydra_suite.core.tracking.precompute import AprilTagPrecomputePhase

        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=9,
            video_path="",
        )
        assert phase.has_cache_hit() is False
        MockDet.assert_called_once()


def test_apriltag_phase_process_frame_empty_crops_adds_empty_frame(tmp_path):
    cache_path = tmp_path / "tags_0_9.npz"
    with patch("hydra_suite.core.tracking.precompute.AprilTagDetector") as MockDet:
        mock_detector = MockDet.return_value
        mock_detector.detect_in_crops.return_value = []

        from hydra_suite.core.tracking.precompute import AprilTagPrecomputePhase

        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=0,
            video_path="",
        )
        phase.process_frame(0, [], [], [], [], [])
        mock_detector.detect_in_crops.assert_not_called()
        # finalize should still succeed (empty frame was recorded)
        path = phase.finalize()
        assert path == str(cache_path)


def test_apriltag_phase_process_frame_calls_detect_in_crops(tmp_path):
    cache_path = tmp_path / "tags_0_0.npz"
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    with patch("hydra_suite.core.tracking.precompute.AprilTagDetector") as MockDet:
        mock_detector = MockDet.return_value
        mock_detector.detect_in_crops.return_value = []

        from hydra_suite.core.tracking.precompute import AprilTagPrecomputePhase

        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=0,
            video_path="",
        )
        phase.process_frame(0, [crop], [0], [0], [], [(5, 10)])
        mock_detector.detect_in_crops.assert_called_once_with(
            [crop], [(5, 10)], det_indices=[0]
        )
        phase.finalize()


def test_apriltag_phase_process_frame_noop_on_cache_hit(tmp_path):
    """process_frame does not call detect_in_crops when the cache was hit."""
    from hydra_suite.core.tracking.precompute import AprilTagPrecomputePhase
    from hydra_suite.data.tag_observation_cache import TagObservationCache as TOC

    cache_path = tmp_path / "tags_0_2.npz"
    # Write a valid compatible cache covering frames 0-2
    writer = TOC(str(cache_path), mode="w", start_frame=0, end_frame=2)
    for fid in range(3):
        writer.add_frame(fid, [], [], [], [])
    writer.save(
        metadata={
            "family": "tag36h11",
            "start_frame": 0,
            "end_frame": 2,
            "video_path": "",
            "detection_cache_hash": "",
        }
    )
    writer.close()

    with patch("hydra_suite.core.tracking.precompute.AprilTagDetector") as MockDet:
        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=2,
            video_path="",
        )
        assert phase.has_cache_hit() is True
        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        phase.process_frame(0, [crop], [0], [0], [], [(0, 0)])
        # Detector was never created (cache hit) and detect_in_crops never called
        MockDet.assert_not_called()


def test_apriltag_phase_stale_cache_triggers_miss(tmp_path):
    """Cache exists for frames 0-2 but request is for 0-9 → cache miss."""
    from hydra_suite.core.tracking.precompute import AprilTagPrecomputePhase
    from hydra_suite.data.tag_observation_cache import TagObservationCache as TOC

    cache_path = tmp_path / "tags_0_2.npz"
    # Write a valid cache covering only frames 0-2
    writer = TOC(str(cache_path), mode="w", start_frame=0, end_frame=2)
    for fid in range(3):
        writer.add_frame(fid, [], [], [], [])
    writer.save(
        metadata={
            "family": "tag36h11",
            "start_frame": 0,
            "end_frame": 2,
            "video_path": "",
            "detection_cache_hash": "",
        }
    )
    writer.close()

    with patch("hydra_suite.core.tracking.precompute.AprilTagDetector") as MockDet:
        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=9,  # larger range than the cached 0-2
            video_path="",
        )
        assert phase.has_cache_hit() is False
        MockDet.assert_called_once()  # detector created for the miss


def test_apriltag_phase_finalize_returns_path_on_hit(tmp_path):
    from hydra_suite.core.tracking.precompute import AprilTagPrecomputePhase
    from hydra_suite.data.tag_observation_cache import TagObservationCache as TOC

    cache_path = tmp_path / "tags_0_0.npz"
    writer = TOC(str(cache_path), mode="w", start_frame=0, end_frame=0)
    writer.add_frame(0, [], [], [], [])
    writer.save(
        metadata={
            "family": "tag36h11",
            "start_frame": 0,
            "end_frame": 0,
            "video_path": "",
            "detection_cache_hash": "",
        }
    )
    writer.close()

    with patch("hydra_suite.core.tracking.precompute.AprilTagDetector"):
        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=0,
            video_path="",
        )
        assert phase.has_cache_hit() is True
        result = phase.finalize()
        assert result == str(cache_path)


# ---------------------------------------------------------------------------
# CNNPrecomputePhase
# ---------------------------------------------------------------------------


def test_cnn_phase_has_cache_hit_false_when_no_file(tmp_path):
    from hydra_suite.core.identity.classification.cnn import CNNIdentityConfig
    from hydra_suite.core.tracking.precompute import CNNPrecomputePhase

    cache_path = tmp_path / "cnn_0_9.npz"
    with patch("hydra_suite.core.tracking.precompute.CNNIdentityBackend"):
        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake/model.pth"),
            model_path="/fake/model.pth",
            cache_path=cache_path,
            name="cnn_identity",
        )
        assert phase.has_cache_hit() is False


def test_cnn_phase_has_cache_hit_true_when_file_exists(tmp_path):
    from hydra_suite.core.identity.classification.cnn import (
        ClassPrediction,
        CNNIdentityCache,
        CNNIdentityConfig,
    )
    from hydra_suite.core.tracking.precompute import CNNPrecomputePhase

    cache_path = tmp_path / "cnn.npz"
    cache = CNNIdentityCache(cache_path)
    cache.save(
        0,
        [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("cached",),
                confidences=(0.8,),
            )
        ],
    )
    cache.flush()
    phase = CNNPrecomputePhase(
        config=CNNIdentityConfig(model_path="/fake/model.pth"),
        model_path="/fake/model.pth",
        cache_path=cache_path,
        name="cnn_identity",
    )
    assert phase.has_cache_hit() is True


def test_cnn_phase_process_frame_batches_crops(tmp_path):
    from hydra_suite.core.identity.classification.cnn import (
        ClassPrediction,
        CNNIdentityConfig,
    )
    from hydra_suite.core.tracking.precompute import CNNPrecomputePhase

    cache_path = tmp_path / "cnn.npz"
    with patch(
        "hydra_suite.core.tracking.precompute.CNNIdentityBackend"
    ) as MockBackend:
        mock_backend = MockBackend.return_value
        mock_backend.predict_batch.return_value = [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("tag_0",),
                confidences=(0.9,),
            ),
            ClassPrediction(
                det_index=1,
                factor_names=("flat",),
                class_names=("tag_0",),
                confidences=(0.85,),
            ),
        ]

        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake.pth", batch_size=2),
            model_path="/fake.pth",
            cache_path=cache_path,
            name="cnn_identity",
        )

        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        # Two frames each with one crop — batch_size=2, so flush happens after frame 1
        phase.process_frame(0, [crop], [0], [0], [], [(0, 0)])
        phase.process_frame(1, [crop], [0], [0], [], [(0, 0)])
        # One batch of 2 should have been flushed
        assert mock_backend.predict_batch.call_count == 1

        phase.finalize()
        assert cache_path.exists()


def test_cnn_phase_finalize_flushes_partial_batch(tmp_path):
    from hydra_suite.core.identity.classification.cnn import (
        ClassPrediction,
        CNNIdentityConfig,
    )
    from hydra_suite.core.tracking.precompute import CNNPrecomputePhase

    cache_path = tmp_path / "cnn_partial.npz"
    with patch(
        "hydra_suite.core.tracking.precompute.CNNIdentityBackend"
    ) as MockBackend:
        mock_backend = MockBackend.return_value
        mock_backend.predict_batch.return_value = [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("tag_1",),
                confidences=(0.8,),
            )
        ]

        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake.pth", batch_size=10),
            model_path="/fake.pth",
            cache_path=cache_path,
            name="cnn_identity",
        )

        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        phase.process_frame(
            0, [crop], [0], [0], [], [(0, 0)]
        )  # 1 crop, batch=10 → not flushed yet
        assert mock_backend.predict_batch.call_count == 0

        phase.finalize()
        assert mock_backend.predict_batch.call_count == 1  # flushed in finalize
        assert cache_path.exists()


def test_cnn_phase_writes_placeholders_for_skipped_detection_slots(tmp_path):
    from hydra_suite.core.identity.classification.cnn import (
        ClassPrediction,
        CNNIdentityCache,
        CNNIdentityConfig,
    )
    from hydra_suite.core.tracking.precompute import CNNPrecomputePhase

    cache_path = tmp_path / "cnn_skipped_slots.npz"
    with patch(
        "hydra_suite.core.tracking.precompute.CNNIdentityBackend"
    ) as MockBackend:
        mock_backend = MockBackend.return_value
        mock_backend.factor_names = ("flat",)
        mock_backend.predict_batch.return_value = [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("tag_0",),
                confidences=(0.9,),
            )
        ]

        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake.pth", batch_size=8),
            model_path="/fake.pth",
            cache_path=cache_path,
            name="cnn_identity",
        )

        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        phase.process_frame(
            0,
            [crop],
            [0, 1],
            [0],
            [
                np.zeros((4, 2), dtype=np.float32),
                np.zeros((4, 2), dtype=np.float32),
            ],
            [(0, 0)],
        )
        phase.finalize()

        cache = CNNIdentityCache(str(cache_path))
        preds = cache.load(0)
        assert [pred.det_index for pred in preds] == [0, 1]
        assert preds[0].class_names == ("tag_0",)
        assert preds[0].confidences == pytest.approx((0.9,))
        assert preds[1].class_names == (None,)
        assert preds[1].confidences == pytest.approx((0.0,))


def test_cnn_phase_invalidates_flat_cache_for_multihead_model(tmp_path):
    from hydra_suite.core.identity.classification.cnn import (
        ClassPrediction,
        CNNIdentityCache,
        CNNIdentityConfig,
    )
    from hydra_suite.core.tracking.precompute import CNNPrecomputePhase

    cache_path = tmp_path / "cnn_multihead_stale.npz"
    stale_cache = CNNIdentityCache(cache_path)
    stale_cache.save(
        0,
        [
            ClassPrediction(
                det_index=0,
                factor_names=("flat",),
                class_names=("old",),
                confidences=(0.3,),
            )
        ],
    )
    stale_cache.flush()

    model_path = tmp_path / "multihead.fake"
    model_path.write_bytes(b"fake")

    with patch(
        "hydra_suite.core.tracking.precompute.CNNIdentityBackend"
    ) as MockBackend:
        mock_backend = MockBackend.return_value
        mock_backend.factor_names = ("color", "shape")
        mock_backend.predict_batch.return_value = [
            ClassPrediction(
                det_index=0,
                factor_names=("color", "shape"),
                class_names=("red", "circle"),
                confidences=(0.9, 0.8),
            )
        ]

        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path=str(model_path), batch_size=4),
            model_path=str(model_path),
            cache_path=cache_path,
            name="cnn_identity",
        )

        assert phase.has_cache_hit() is False

        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        phase.process_frame(0, [crop], [0], [0], [], [(0, 0)])
        phase.finalize()

        rewritten_cache = CNNIdentityCache(str(cache_path))
        assert rewritten_cache.factor_names == ("color", "shape")
        preds = rewritten_cache.load(0)
        assert len(preds) == 1
        assert preds[0].class_names == ("red", "circle")
        assert preds[0].confidences == pytest.approx((0.9, 0.8))


def test_cnn_phase_realtime_callback_flushes_per_frame(tmp_path):
    from hydra_suite.core.identity.classification.cnn import (
        ClassPrediction,
        CNNIdentityConfig,
    )
    from hydra_suite.core.tracking.precompute import CNNPrecomputePhase

    cache_path = tmp_path / "cnn_realtime.npz"
    with patch(
        "hydra_suite.core.tracking.precompute.CNNIdentityBackend"
    ) as MockBackend:
        mock_backend = MockBackend.return_value
        mock_backend.predict_batch.return_value = [
            ClassPrediction(
                det_index=-1,
                factor_names=("flat",),
                class_names=("tag_0",),
                confidences=(0.9,),
            ),
            ClassPrediction(
                det_index=-1,
                factor_names=("flat",),
                class_names=("tag_1",),
                confidences=(0.85,),
            ),
        ]

        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake.pth", batch_size=8),
            model_path="/fake.pth",
            cache_path=cache_path,
            name="cnn_identity",
        )

        live_updates = []
        phase.set_frame_result_callback(
            lambda frame_idx, preds: live_updates.append(
                (frame_idx, [pred.det_index for pred in preds])
            )
        )

        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        phase.process_frame(0, [crop, crop], [0, 1], [0, 1], [], [(0, 0), (0, 0)])

        assert mock_backend.predict_batch.call_count == 1
        assert live_updates == [(0, [0, 1])]

        phase.finalize()
        assert mock_backend.predict_batch.call_count == 1
        assert cache_path.exists()


def test_cnn_phase_realtime_callback_requests_calibrated_posteriors(tmp_path):
    from hydra_suite.core.identity.calibration import CalibrationModel
    from hydra_suite.core.identity.classification.cnn import (
        ClassPrediction,
        CNNIdentityConfig,
    )
    from hydra_suite.core.tracking.precompute import CNNPrecomputePhase

    cache_path = tmp_path / "cnn_realtime_calibrated.npz"
    calibration = CalibrationModel(temperature=1.7)
    captured = {}
    live_updates = []

    with patch(
        "hydra_suite.core.tracking.precompute.CNNIdentityBackend"
    ) as MockBackend:
        mock_backend = MockBackend.return_value

        def _predict_batch_posteriors(crops, calibration=None):
            captured["calibration"] = calibration
            return (
                [
                    ClassPrediction(
                        det_index=-1,
                        factor_names=("flat",),
                        class_names=("tag_0",),
                        confidences=(0.9,),
                    )
                ],
                [[np.array([0.2, 0.8], dtype=np.float64)]],
            )

        mock_backend.predict_batch_posteriors.side_effect = _predict_batch_posteriors

        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake.pth", batch_size=8),
            model_path="/fake.pth",
            cache_path=cache_path,
            name="cnn_identity",
            calibration_model=calibration,
        )

        def _callback(frame_idx, preds, posteriors, detection_ids=None):
            live_updates.append((frame_idx, preds, posteriors, detection_ids))

        phase.set_frame_result_callback(_callback)

        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        phase.process_frame(
            0,
            [crop],
            [101],
            [0],
            [np.zeros((4, 2), dtype=np.float32)],
            [(0, 0)],
        )

        assert captured["calibration"] is calibration
        assert len(live_updates) == 1
        assert live_updates[0][0] == 0
        assert live_updates[0][3] == [101]
        np.testing.assert_allclose(live_updates[0][2][0][0], [0.2, 0.8])


def test_cnn_phase_process_frame_empty_crops_does_not_add_to_batch(tmp_path):
    from hydra_suite.core.identity.classification.cnn import CNNIdentityConfig
    from hydra_suite.core.tracking.precompute import CNNPrecomputePhase

    cache_path = tmp_path / "cnn_empty.npz"
    with patch(
        "hydra_suite.core.tracking.precompute.CNNIdentityBackend"
    ) as MockBackend:
        mock_backend = MockBackend.return_value

        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake.pth", batch_size=2),
            model_path="/fake.pth",
            cache_path=cache_path,
            name="cnn_identity",
        )
        phase.process_frame(0, [], [], [], [], [])
        mock_backend.predict_batch.assert_not_called()
        phase.finalize()
        # cache still flushed (empty frame recorded)
        assert cache_path.exists()


def test_cnn_phase_close_is_idempotent(tmp_path):
    from hydra_suite.core.identity.classification.cnn import CNNIdentityConfig
    from hydra_suite.core.tracking.precompute import CNNPrecomputePhase

    cache_path = tmp_path / "cnn.npz"
    with patch("hydra_suite.core.tracking.precompute.CNNIdentityBackend"):
        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake.pth"),
            model_path="/fake.pth",
            cache_path=cache_path,
        )
        phase.close()
        phase.close()  # must not raise


# ---------------------------------------------------------------------------
# PosePipeline as PrecomputePhase
# ---------------------------------------------------------------------------


def test_pose_pipeline_has_cache_hit_false_when_not_set():
    from hydra_suite.core.tracking.pose.pose_pipeline import PosePipeline

    pipeline = PosePipeline(
        pose_backend=None,
        cache_writer=None,
        cache_hit=False,
        cache_path="/some/path.hdf5",
        finalize_metadata={},
    )
    assert pipeline.has_cache_hit() is False


def test_pose_pipeline_has_cache_hit_true_when_set():
    from hydra_suite.core.tracking.pose.pose_pipeline import PosePipeline

    pipeline = PosePipeline(
        pose_backend=None,
        cache_writer=None,
        cache_hit=True,
        cache_path="/some/path.hdf5",
        finalize_metadata={},
    )
    assert pipeline.has_cache_hit() is True


def test_pose_pipeline_finalize_returns_cache_path_on_hit():
    from hydra_suite.core.tracking.pose.pose_pipeline import PosePipeline

    pipeline = PosePipeline(
        pose_backend=None,
        cache_writer=None,
        cache_hit=True,
        cache_path="/cached/path.hdf5",
        finalize_metadata={},
    )
    result = pipeline.finalize()
    assert result == "/cached/path.hdf5"


def test_pose_pipeline_process_frame_noop_on_hit():
    from hydra_suite.core.tracking.pose.pose_pipeline import PosePipeline

    pipeline = PosePipeline(
        pose_backend=None,
        cache_writer=None,
        cache_hit=True,
        cache_path="/hit.hdf5",
        finalize_metadata={},
    )
    # Should not raise even with no backend
    pipeline.process_frame(0, [], [], [], [], [])


def test_pose_pipeline_close_idempotent():
    from hydra_suite.core.tracking.pose.pose_pipeline import PosePipeline

    pipeline = PosePipeline(
        pose_backend=None,
        cache_writer=None,
        cache_hit=True,
        cache_path="/hit.hdf5",
        finalize_metadata={},
    )
    pipeline.close()
    pipeline.close()  # must not raise
