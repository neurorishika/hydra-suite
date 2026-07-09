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
        lambda executors, frame_to_process, yolo_params, headtail_state=None: (
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
