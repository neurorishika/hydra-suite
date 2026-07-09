from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_preview_build_yolo_params_includes_headtail_runtime() -> None:
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )

    params = preview_worker._preview_build_yolo_params(
        {
            "compute_runtime": "cpu",
            "headtail_runtime": "onnx_coreml",
            "headtail_batch_size": 24,
            "yolo_headtail_detect_conf_threshold": 0.67,
        },
        1.0,
        False,
    )

    assert params["HEADTAIL_COMPUTE_RUNTIME"] == "onnx_coreml"
    assert params["HEADTAIL_BATCH_SIZE"] == 24
    assert params["YOLO_HEADTAIL_DETECT_CONF_THRESHOLD"] == 0.67


def test_preview_run_yolo_branch_uses_load_obb_executor_not_legacy_detector(
    monkeypatch,
) -> None:
    """Task 2: the YOLO preview branch must call the production
    ``load_obb_executor`` factory instead of constructing a legacy
    ``YOLOOBBDetector`` instance."""
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    runtime_artifacts = importlib.import_module(
        "hydra_suite.core.inference.runtime_artifacts"
    )
    detectors_pkg = importlib.import_module("hydra_suite.core.detectors")

    captured: dict[str, object] = {}

    def _boom(*args, **kwargs):
        raise AssertionError(
            "preview should not construct a legacy YOLOOBBDetector instance"
        )

    monkeypatch.setattr(detectors_pkg, "YOLOOBBDetector", _boom)

    class FakeExecutor:
        names = {0: "ant"}

        def predict(self, *args, **kwargs):
            captured["predict_called"] = True
            return []

    def _fake_load_obb_executor(model_path, compute_runtime, **kwargs):
        captured["model_path"] = model_path
        captured["compute_runtime"] = compute_runtime
        return FakeExecutor()

    monkeypatch.setattr(runtime_artifacts, "load_obb_executor", _fake_load_obb_executor)
    monkeypatch.setattr(
        preview_worker,
        "_preview_resize_frame",
        lambda frame_bgr, test_frame, resize_f: (frame_bgr, test_frame),
    )
    monkeypatch.setattr(
        preview_worker, "_preview_load_headtail_model", lambda yolo_params: None
    )
    monkeypatch.setattr(
        preview_worker,
        "_preview_compute_canonical_crops",
        lambda filtered_corners, frame_to_process, context: (
            [],
            [],
            0.1,
            (0, 0, 0),
            False,
        ),
    )
    monkeypatch.setattr(
        preview_worker, "_preview_run_pose_overlay", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        preview_worker, "_preview_run_cnn_overlay", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        preview_worker, "_preview_run_apriltag_overlay", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        preview_worker, "_preview_cleanup_backends", lambda *args, **kwargs: None
    )

    test_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    context = {
        "yolo_model_path": "detector.pt",
        "yolo_obb_mode": "direct",
        "compute_runtime": "cpu",
        "obb_compute_runtime": "cpu",
    }

    detected_dimensions, out_frame = preview_worker._preview_run_yolo_branch(
        test_frame,
        test_frame.copy(),
        context,
        1.0,
        False,
    )

    assert captured["predict_called"] is True
    assert captured["compute_runtime"] == "cpu"
    assert detected_dimensions == []
    assert out_frame.shape == test_frame.shape


def test_preview_run_yolo_branch_uses_filtered_headtail_hints(monkeypatch) -> None:
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    runtime_artifacts = importlib.import_module(
        "hydra_suite.core.inference.runtime_artifacts"
    )

    corners = np.array(
        [[10.0, 10.0], [22.0, 10.0], [22.0, 16.0], [10.0, 16.0]],
        dtype=np.float32,
    )
    captured: dict[str, object] = {}

    class FakeExecutor:
        names = {0: "ant"}

        def predict(self, *args, **kwargs):
            return []

    monkeypatch.setattr(
        runtime_artifacts,
        "load_obb_executor",
        lambda model_path, compute_runtime, **kwargs: FakeExecutor(),
    )
    monkeypatch.setattr(
        preview_worker,
        "_preview_resize_frame",
        lambda frame_bgr, test_frame, resize_f: (frame_bgr, test_frame),
    )
    monkeypatch.setattr(
        preview_worker, "_preview_load_headtail_model", lambda yolo_params: None
    )
    monkeypatch.setattr(
        preview_worker,
        "_preview_run_yolo_raw_detection",
        lambda executors, frame_to_process, yolo_params, headtail_state=None, extractor=None: (
            [np.array([16.0, 13.0, 0.0], dtype=np.float32)],
            [1.0],
            [(1.0, 2.0)],
            [0.91],
            [corners],
            [0],
            [1.25],
            [0.88],
            [1],
            None,
        ),
    )
    monkeypatch.setattr(
        preview_worker,
        "_preview_compute_canonical_crops",
        lambda filtered_corners, frame_to_process, context: (
            [None] * len(filtered_corners),
            [None] * len(filtered_corners),
            0.1,
            (0, 0, 0),
            False,
        ),
    )
    monkeypatch.setattr(
        preview_worker, "_preview_run_pose_overlay", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        preview_worker, "_preview_run_cnn_overlay", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        preview_worker, "_preview_run_apriltag_overlay", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        preview_worker, "_preview_cleanup_backends", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        preview_worker, "_preview_draw_yolo_footer", lambda *args, **kwargs: None
    )

    def _capture_annotations(
        test_frame,
        filtered_corners,
        detection_confidences,
        filtered_class_labels,
        label_stacks,
        label_anchors,
        pose_keypoints_by_det,
        filtered_headtail,
        context,
    ):
        captured["filtered_headtail"] = list(filtered_headtail)

    monkeypatch.setattr(
        preview_worker, "_preview_draw_obb_annotations", _capture_annotations
    )

    test_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    context = {
        "yolo_model_path": "detector.pt",
        "yolo_headtail_model_path": "headtail.onnx",
        "compute_runtime": "cpu",
        "headtail_runtime": "onnx_coreml",
    }

    preview_worker._preview_run_yolo_branch(
        test_frame,
        test_frame.copy(),
        context,
        1.0,
        False,
    )

    assert len(captured["filtered_headtail"]) == 1
    heading, conf, directed = captured["filtered_headtail"][0]
    assert heading == 1.25
    assert conf == pytest.approx(0.88, abs=1e-4)
    assert directed == 1


class _ArrayWrap:
    """Minimal ``torch.Tensor``-like wrapper exposing ``.cpu().numpy()``."""

    def __init__(self, arr) -> None:
        self._arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeStage1Boxes:
    """Stand-in for ultralytics ``Results.boxes`` (stage-1 plain detector)."""

    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = _ArrayWrap(xyxy)
        self.conf = _ArrayWrap(conf)
        self.cls = _ArrayWrap(cls)

    def __len__(self) -> int:
        return self.xyxy._arr.shape[0]


class _FakeOBB:
    """Stand-in for ultralytics ``Results.obb`` (stage-2 crop-OBB output)."""

    def __init__(self, xywhr, conf, cls) -> None:
        self.xywhr = _ArrayWrap(xywhr)
        self.conf = _ArrayWrap(conf)
        self.cls = _ArrayWrap(cls)

    def __len__(self) -> int:
        return self.xywhr._arr.shape[0]


def _make_sequential_fake_executors():
    """Build fake stage-1 (detect) / stage-2 (crop-OBB) executors.

    Stage-1 returns four candidate boxes (mirrors a real detector emitting
    several distinct crops). Stage-2 returns one detection each for three of
    the four crops, at three different confidences, plus a genuinely empty
    result for the fourth crop -- this exercises
    ``_preview_accumulate_crop_detections``'s empty-result skip,
    ``_preview_sort_merged_detections``'s confidence-descending sort, and its
    ``max_det`` truncation (when the caller caps below 3), all with
    non-trivial, order-scrambled input.
    """
    import types

    stage1_boxes = _FakeStage1Boxes(
        xyxy=[
            [5.0, 5.0, 25.0, 25.0],
            [60.0, 60.0, 80.0, 80.0],
            [120.0, 120.0, 140.0, 140.0],
            [160.0, 160.0, 180.0, 180.0],
        ],
        conf=[0.7, 0.7, 0.7, 0.7],
        cls=[9, 9, 9, 9],
    )
    stage1_result = types.SimpleNamespace(boxes=stage1_boxes, names={9: "blob"})

    # Per-crop stage-2 detections, deliberately *not* pre-sorted by
    # confidence, so a correct implementation must sort them. Box sizes are
    # chosen to correlate with confidence (higher conf -> larger box) so that
    # this fixture is unambiguous under either of the two truncation policies
    # a caller might apply downstream (raw-merge truncates by confidence via
    # ``_preview_sort_merged_detections``; the GUI's final
    # ``filter_raw_detections`` separately truncates to ``MAX_TARGETS`` by
    # detection *size*, largest-first).
    stage2_results = [
        types.SimpleNamespace(
            obb=_FakeOBB(xywhr=[[10.0, 10.0, 6.0, 10.0, 0.3]], conf=[0.4], cls=[1])
        ),
        types.SimpleNamespace(
            obb=_FakeOBB(xywhr=[[15.0, 15.0, 20.0, 20.0, 0.0]], conf=[0.9], cls=[0])
        ),
        types.SimpleNamespace(
            obb=_FakeOBB(xywhr=[[20.0, 20.0, 15.0, 15.0, 0.5]], conf=[0.6], cls=[2])
        ),
        types.SimpleNamespace(obb=None),  # 4th crop: no stage-2 detections at all.
    ]

    class FakeStage1Executor:
        names = {9: "blob"}

        def predict(self, *args, **kwargs):
            return [stage1_result]

    class FakeStage2Executor:
        names = {0: "ant", 1: "ant", 2: "ant"}

        def predict(self, chunk, **kwargs):
            return stage2_results[: len(chunk)]

    return FakeStage1Executor(), FakeStage2Executor()


def test_preview_run_sequential_raw_detection_merges_sorts_and_truncates() -> None:
    """Task 2 finding: exercise the sequential-mode crop merge/sort/truncate path.

    Directly drives ``_preview_run_sequential_raw_detection`` (bypassing the
    GUI wiring) with fake stage-1/stage-2 executors returning multiple
    candidate boxes so ``_preview_accumulate_crop_detections`` and
    ``_preview_sort_merged_detections`` do genuine, non-trivial merge/sort/
    truncate work, not just a pass-through.
    """
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    from hydra_suite.core.detectors import DetectionFilter

    detect_executor, obb_executor = _make_sequential_fake_executors()
    executors = {"mode": "sequential", "detect": detect_executor, "obb": obb_executor}
    yolo_params = {
        "MAX_TARGETS": 1,  # max_det = 2 * MAX_TARGETS = 2 -> forces truncation
        "YOLO_SEQ_CROP_PAD_RATIO": 0.15,
        "YOLO_SEQ_MIN_CROP_SIZE_PX": 64,
        "YOLO_SEQ_ENFORCE_SQUARE_CROP": True,
        "YOLO_SEQ_STAGE2_IMGSZ": 0,  # skip stage-2 resize -> merge scale is 1.0
        "YOLO_SEQ_INDIVIDUAL_BATCH_SIZE": 16,
        "YOLO_SEQ_DETECT_CONF_THRESHOLD": 0.25,
    }
    extractor = DetectionFilter(yolo_params)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    (
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        raw_class_ids,
        stage1_result,
    ) = preview_worker._preview_run_sequential_raw_detection(
        extractor,
        executors,
        frame,
        yolo_params,
        raw_conf_floor=1e-3,
        target_classes=None,
        max_det=2,
    )

    # 4 stage-1 boxes -> 4 crops; 1 crop yields no stage-2 detections (skipped
    # by _preview_accumulate_crop_detections), leaving 3 candidates
    # (confidences 0.4, 0.9, 0.6); max_det=2 truncates to the top 2.
    assert len(raw_meas) == 2
    assert raw_confidences == sorted(raw_confidences, reverse=True)
    assert raw_confidences[0] == pytest.approx(0.9, abs=1e-4)
    assert raw_confidences[1] == pytest.approx(0.6, abs=1e-4)
    # Class ids travel with their detection through the sort/truncate.
    assert raw_class_ids == [0, 2]
    # Corners are well-formed (4, 2) oriented-box corner sets.
    assert len(raw_obb_corners) == 2
    for corners in raw_obb_corners:
        assert np.asarray(corners).shape == (4, 2)
    # The stage-1 Results object is returned (for downstream viz), not lost.
    assert getattr(stage1_result, "boxes", None) is not None


def test_preview_run_yolo_branch_sequential_mode_uses_two_executors(
    monkeypatch,
) -> None:
    """Task 2: end-to-end sequential-mode preview branch, off ``load_obb_executor``.

    Mirrors ``test_preview_run_yolo_branch_uses_load_obb_executor_not_legacy_detector``'s
    monkeypatching pattern for direct mode, but supplies *two* distinct fake
    executors (stage-1 detect + stage-2 crop-OBB) keyed off the ``task=``
    kwarg ``load_obb_executor`` is called with, and asserts the branch reaches
    the final annotation step with correctly-shaped, correctly-ordered
    output -- without ever touching the legacy ``YOLOOBBDetector``.
    """
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    runtime_artifacts = importlib.import_module(
        "hydra_suite.core.inference.runtime_artifacts"
    )
    detectors_pkg = importlib.import_module("hydra_suite.core.detectors")

    def _boom(*args, **kwargs):
        raise AssertionError(
            "preview should not construct a legacy YOLOOBBDetector instance"
        )

    monkeypatch.setattr(detectors_pkg, "YOLOOBBDetector", _boom)

    detect_executor, obb_executor = _make_sequential_fake_executors()
    load_calls: list[dict] = []

    def _fake_load_obb_executor(model_path, compute_runtime, **kwargs):
        load_calls.append({"model_path": model_path, **kwargs})
        if kwargs.get("task") == "detect":
            return detect_executor
        return obb_executor

    monkeypatch.setattr(runtime_artifacts, "load_obb_executor", _fake_load_obb_executor)
    monkeypatch.setattr(
        preview_worker,
        "_preview_resize_frame",
        lambda frame_bgr, test_frame, resize_f: (frame_bgr, test_frame),
    )
    monkeypatch.setattr(
        preview_worker, "_preview_load_headtail_model", lambda yolo_params: None
    )
    monkeypatch.setattr(
        preview_worker,
        "_preview_compute_canonical_crops",
        lambda filtered_corners, frame_to_process, context: (
            [None] * len(filtered_corners),
            [None] * len(filtered_corners),
            0.1,
            (0, 0, 0),
            False,
        ),
    )
    monkeypatch.setattr(
        preview_worker, "_preview_run_pose_overlay", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        preview_worker, "_preview_run_cnn_overlay", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        preview_worker, "_preview_run_apriltag_overlay", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        preview_worker, "_preview_cleanup_backends", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        preview_worker, "_preview_draw_yolo_footer", lambda *args, **kwargs: None
    )

    captured: dict[str, object] = {}

    def _capture_annotations(
        test_frame,
        filtered_corners,
        detection_confidences,
        filtered_class_labels,
        label_stacks,
        label_anchors,
        pose_keypoints_by_det,
        filtered_headtail,
        context,
    ):
        captured["filtered_corners"] = list(filtered_corners)
        captured["detection_confidences"] = list(detection_confidences)
        captured["filtered_class_labels"] = list(filtered_class_labels)

    monkeypatch.setattr(
        preview_worker, "_preview_draw_obb_annotations", _capture_annotations
    )

    test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    context = {
        "yolo_obb_mode": "sequential",
        "yolo_detect_model_path": "detect.pt",
        "yolo_crop_obb_model_path": "crop_obb.pt",
        "compute_runtime": "cpu",
        "obb_compute_runtime": "cpu",
        "yolo_confidence": 0.1,
        "yolo_seq_stage2_imgsz": 0,
        "max_targets": 2,
    }

    detected_dimensions, out_frame = preview_worker._preview_run_yolo_branch(
        test_frame,
        test_frame.copy(),
        context,
        1.0,
        False,
    )

    detect_calls = [c for c in load_calls if c.get("task") == "detect"]
    obb_calls = [c for c in load_calls if c.get("task") == "obb"]
    assert len(detect_calls) == 1
    assert len(obb_calls) == 1
    assert detect_calls[0]["model_path"].endswith("detect.pt")
    assert obb_calls[0]["model_path"].endswith("crop_obb.pt")

    # 3 non-empty crop detections (0.4, 0.9, 0.6) all pass the raw stage
    # (max_det=2*MAX_TARGETS=4), then the final ``MAX_TARGETS=2`` cap in
    # ``filter_raw_detections`` keeps only the top-2 by confidence (0.9,
    # 0.6); both pass the 0.1 confidence-filter threshold and are far enough
    # apart to survive OBB-IOU NMS untouched.
    assert len(captured["filtered_corners"]) == 2
    for corners in captured["filtered_corners"]:
        assert np.asarray(corners).shape == (4, 2)
    assert captured["detection_confidences"] == sorted(
        captured["detection_confidences"], reverse=True
    )
    assert captured["detection_confidences"][0] == pytest.approx(0.9, abs=1e-4)
    assert captured["detection_confidences"][1] == pytest.approx(0.6, abs=1e-4)
    assert out_frame.shape == test_frame.shape


def test_preview_raw_detection_prefilters_headtail_candidates(monkeypatch) -> None:
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )

    captured: dict[str, object] = {}

    corners = [
        np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]], dtype=np.float32),
        np.array(
            [[10.0, 0.0], [14.0, 0.0], [14.0, 2.0], [10.0, 2.0]], dtype=np.float32
        ),
        np.array(
            [[20.0, 0.0], [24.0, 0.0], [24.0, 2.0], [20.0, 2.0]], dtype=np.float32
        ),
    ]

    class FakeExecutor:
        names = {0: "ant"}

        def predict(self, *args, **kwargs):
            return []

    monkeypatch.setattr(
        preview_worker,
        "_preview_run_direct_raw_detection",
        lambda extractor, executor, frame, target_classes, raw_conf_floor, max_det: (
            [np.array([1.0, 1.0, 0.0], dtype=np.float32)] * 3,
            [100.0, 80.0, 60.0],
            [(100.0, 2.0)] * 3,
            [0.9, 0.3, 0.8],
            corners,
            [0, 1, 2],
            None,
        ),
    )

    def _fake_select_candidates(
        params,
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        roi_mask=None,
    ):
        return [2]

    def _fake_run_headtail(
        headtail_state,
        frame_to_process,
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        yolo_params,
        roi_mask=None,
    ):
        candidate_indices = _fake_select_candidates(
            yolo_params,
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
        )
        captured["candidate_indices"] = list(candidate_indices)
        captured["corner_count"] = len(raw_obb_corners)
        hints = [float("nan")] * len(raw_obb_corners)
        confidences = [0.0] * len(raw_obb_corners)
        directed = [0] * len(raw_obb_corners)
        for idx in candidate_indices:
            hints[idx] = 1.25
            confidences[idx] = 0.88
            directed[idx] = 1
        return hints, confidences, directed

    monkeypatch.setattr(
        preview_worker,
        "_preview_select_headtail_candidate_indices",
        _fake_select_candidates,
    )
    monkeypatch.setattr(preview_worker, "_preview_run_headtail", _fake_run_headtail)

    raw = preview_worker._preview_run_yolo_raw_detection(
        {"mode": "direct", "obb": FakeExecutor()},
        np.zeros((32, 32, 3), dtype=np.uint8),
        {"YOLO_OBB_MODE": "direct", "MAX_TARGETS": 2},
        headtail_state=object(),
    )

    assert captured["candidate_indices"] == [2]
    assert captured["corner_count"] == 3
    assert np.isnan(raw[6][0])
    assert np.isnan(raw[6][1])
    assert raw[6][2] == 1.25
    assert raw[7] == [0.0, 0.0, 0.88]
    assert raw[8] == [0, 0, 1]


def test_preview_draw_yolo_footer_reports_disabled_headtail(monkeypatch) -> None:
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )

    captured: dict[str, object] = {}

    def _fake_put_text(*args, **kwargs):
        captured["text"] = args[1]

    monkeypatch.setattr(preview_worker.cv2, "putText", _fake_put_text)

    preview_worker._preview_draw_yolo_footer(
        np.zeros((32, 32, 3), dtype=np.uint8),
        [object()],
        {"YOLO_IOU_THRESHOLD": 0.45},
        {
            "configured_headtail_model_path": "classification/orientation/model.pth",
            "yolo_headtail_model_path": "",
            "headtail_enabled": False,
        },
    )

    assert "head-tail disabled" in str(captured["text"])


def test_preview_draw_obb_annotations_labels_headtail_abstain(monkeypatch) -> None:
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )

    captured_labels: list[list[str]] = []

    monkeypatch.setattr(
        preview_worker,
        "_draw_preview_label_stack",
        lambda _frame, _anchor, lines, _color, font_scale=0.5: captured_labels.append(
            list(lines)
        ),
    )
    monkeypatch.setattr(
        preview_worker, "_draw_preview_pose_points", lambda *args, **kwargs: None
    )

    preview_worker._preview_draw_obb_annotations(
        np.zeros((64, 64, 3), dtype=np.uint8),
        [
            np.array(
                [[10.0, 10.0], [20.0, 10.0], [20.0, 16.0], [10.0, 16.0]],
                dtype=np.float32,
            )
        ],
        [0.95],
        ["ant"],
        [[]],
        [(12, 12)],
        {},
        [(float("nan"), 0.77, 0)],
        {},
    )

    assert any("head abstain 0.77" in " ".join(lines) for lines in captured_labels)


def test_preview_run_cnn_overlay_formats_multihead_predictions(monkeypatch) -> None:
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    cnn_mod = importlib.import_module("hydra_suite.core.identity.classification.cnn")

    class FakeBackend:
        def __init__(self, config, model_path=None, compute_runtime="cpu") -> None:
            self.config = config
            self.model_path = model_path
            self.compute_runtime = compute_runtime

        def predict_batch(self, crops):
            return [
                cnn_mod.ClassPrediction(
                    det_index=0,
                    factor_names=("color", "side"),
                    class_names=("red", None),
                    confidences=(0.93, 0.42),
                )
            ]

        def close(self) -> None:
            return None

    monkeypatch.setattr(cnn_mod, "CNNIdentityBackend", FakeBackend)
    monkeypatch.setattr(
        preview_worker, "resolve_model_path", lambda path: "/tmp/multihead.json"
    )
    monkeypatch.setattr(preview_worker.os.path, "exists", lambda path: True)

    label_stacks = [[]]
    backends = preview_worker._preview_run_cnn_overlay(
        [np.zeros((4, 2), dtype=np.float32)],
        [np.zeros((16, 16, 3), dtype=np.uint8)],
        {
            "cnn_classifiers": [
                {
                    "model_path": "classifier.multihead.json",
                    "label": "cnn_identity",
                    "confidence": 0.5,
                    "batch_size": 8,
                    "scoring_mode": "per_head_average",
                }
            ],
            "cnn_runtime": "cpu",
        },
        label_stacks,
    )

    assert len(backends) == 1
    assert label_stacks == [["cnn_identity: color=red 0.93 | side=unknown 0.42"]]


def test_preview_run_pose_no_build_runtime_config(monkeypatch, caplog) -> None:
    """Task 3: the pose-preview branch must construct ``PoseRuntimeConfig``
    directly (mirroring ``core/inference/stages/pose.py::load_pose_model``)
    instead of calling the legacy ``build_runtime_config`` translation step."""
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    pose_api = importlib.import_module("hydra_suite.core.identity.pose.api")
    pose_types = importlib.import_module("hydra_suite.core.identity.pose.types")
    resolver_mod = importlib.import_module("hydra_suite.runtime.resolver")

    def _boom(*args, **kwargs):
        raise AssertionError(
            "preview pose overlay should not call the legacy build_runtime_config"
        )

    monkeypatch.setattr(pose_api, "build_runtime_config", _boom)

    captured: dict[str, object] = {}

    class FakeBackend:
        def predict_batch(self, crops):
            captured["predict_batch_called"] = True
            return []

    def _fake_create_pose_backend_from_config(config):
        captured["pose_config"] = config
        return FakeBackend()

    monkeypatch.setattr(
        pose_api,
        "create_pose_backend_from_config",
        _fake_create_pose_backend_from_config,
    )

    # Pin the platform so tier -> compute_runtime resolution is deterministic:
    # gpu_fast + CUDA available -> "tensorrt" (matches load_pose_model's SLEAP
    # branch: runtime_flavor="tensorrt", device="cuda").
    monkeypatch.setattr(
        resolver_mod,
        "detect_platform",
        lambda: resolver_mod.PlatformInfo(has_cuda=True, has_mps=False),
    )

    context = {
        "enable_pose_extractor": True,
        "pose_model_type": "sleap",
        "pose_model_dir": "/models/sleap_model",
        "pose_skeleton_file": "",
        "pose_min_kpt_conf_valid": 0.3,
        "pose_batch_size": 2,
        "pose_sleap_env": "sleap_env_x",
        "runtime_tier": "gpu_fast",
    }

    canonical_crops = [np.zeros((8, 8, 3), dtype=np.uint8)]
    canonical_inverses = [np.eye(2, 3, dtype=np.float32)]
    label_stacks = [[]]
    pose_keypoints_by_det: dict[int, np.ndarray] = {}

    with caplog.at_level("WARNING"):
        result = preview_worker._preview_run_pose_overlay(
            [np.zeros((4, 2), dtype=np.float32)],
            canonical_crops,
            canonical_inverses,
            context,
            label_stacks,
            pose_keypoints_by_det,
        )

    assert not any(
        "Preview pose overlay disabled" in rec.message for rec in caplog.records
    )
    assert result is not None
    assert captured["predict_batch_called"] is True

    expected_runtime = resolver_mod.resolve_compute_runtime(
        "gpu_fast",
        resolver_mod.PlatformInfo(has_cuda=True, has_mps=False),
        stage="sleap_pose",
    )
    assert expected_runtime == "tensorrt"

    pose_config = captured["pose_config"]
    assert isinstance(pose_config, pose_types.PoseRuntimeConfig)
    assert pose_config.backend_family == "sleap"
    assert pose_config.runtime_flavor == "tensorrt"
    assert pose_config.device == "cuda"
    assert pose_config.sleap_device == "cuda"
    assert pose_config.sleap_env == "sleap_env_x"


def test_detection_panel_context_runtime_tier_drives_pose_preview_resolution(
    monkeypatch, caplog
) -> None:
    """Critical-finding regression test: `_collect_preview_detection_context`
    (the only place that builds the preview context dict) must populate
    ``runtime_tier`` so that `_preview_run_pose_overlay`'s tier-based
    ``resolve_compute_runtime`` branch is actually reached in production,
    instead of always falling back to the legacy ``compute_runtime`` string.

    This exercises the *full* path: a real ``DetectionPanel`` (via a real
    ``MainWindow``) builds the context, and that context is fed unmodified
    into ``_preview_run_pose_overlay``.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    pose_api = importlib.import_module("hydra_suite.core.identity.pose.api")
    pose_types = importlib.import_module("hydra_suite.core.identity.pose.types")
    resolver_mod = importlib.import_module("hydra_suite.runtime.resolver")

    # Pin the platform so tier -> compute_runtime resolution is deterministic.
    monkeypatch.setattr(
        resolver_mod,
        "detect_platform",
        lambda: resolver_mod.PlatformInfo(has_cuda=True, has_mps=False),
    )

    mw = MainWindow()
    try:
        # Force the tier selector to "gpu_fast" regardless of what tiers the
        # test host's real hardware happens to expose in the combo box.
        monkeypatch.setattr(mw, "_selected_runtime_tier", lambda: "gpu_fast")
        monkeypatch.setattr(
            mw, "_get_resolved_pose_model_dir", lambda backend: "/models/sleap_model"
        )
        monkeypatch.setattr(mw, "_is_pose_inference_enabled", lambda: True)
        monkeypatch.setattr(mw, "_selected_pose_sleap_env", lambda: "sleap_env_x")
        mw._identity_panel.combo_pose_model_type.setCurrentText("SLEAP")

        context = mw._detection_panel._collect_preview_detection_context()

        # The bug: this key was never populated, so preview pose overlay's
        # resolution always fell back to the legacy `compute_runtime` string.
        assert context["runtime_tier"] == "gpu_fast"
        assert context["pose_model_type"] == "sleap"
        assert context["pose_model_dir"] == "/models/sleap_model"
        assert context["enable_pose_extractor"] is True

        def _boom(*args, **kwargs):
            raise AssertionError(
                "preview pose overlay should not call the legacy build_runtime_config"
            )

        monkeypatch.setattr(pose_api, "build_runtime_config", _boom)

        captured: dict[str, object] = {}

        class FakeBackend:
            def predict_batch(self, crops):
                captured["predict_batch_called"] = True
                return []

        def _fake_create_pose_backend_from_config(config):
            captured["pose_config"] = config
            return FakeBackend()

        monkeypatch.setattr(
            pose_api,
            "create_pose_backend_from_config",
            _fake_create_pose_backend_from_config,
        )

        canonical_crops = [np.zeros((8, 8, 3), dtype=np.uint8)]
        canonical_inverses = [np.eye(2, 3, dtype=np.float32)]
        label_stacks = [[]]
        pose_keypoints_by_det: dict[int, np.ndarray] = {}

        with caplog.at_level("WARNING"):
            result = preview_worker._preview_run_pose_overlay(
                [np.zeros((4, 2), dtype=np.float32)],
                canonical_crops,
                canonical_inverses,
                context,
                label_stacks,
                pose_keypoints_by_det,
            )

        assert not any(
            "Preview pose overlay disabled" in rec.message for rec in caplog.records
        )
        assert result is not None
        assert captured["predict_batch_called"] is True

        # The tier ("gpu_fast") that _collect_preview_detection_context
        # threaded through must be what drove the resolved runtime — not the
        # legacy `compute_runtime` fallback string.
        expected_runtime = resolver_mod.resolve_compute_runtime(
            "gpu_fast",
            resolver_mod.PlatformInfo(has_cuda=True, has_mps=False),
            stage="sleap_pose",
        )
        assert expected_runtime == "tensorrt"

        pose_config = captured["pose_config"]
        assert isinstance(pose_config, pose_types.PoseRuntimeConfig)
        assert pose_config.backend_family == "sleap"
        assert pose_config.runtime_flavor == "tensorrt"
        assert pose_config.device == "cuda"
        assert pose_config.sleap_device == "cuda"
    finally:
        mw.close()
