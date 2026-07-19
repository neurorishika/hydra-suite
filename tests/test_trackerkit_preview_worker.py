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


def test_preview_build_inference_params_maps_overlay_and_runtime_keys() -> None:
    """The preview params assembled for ``build_inference_config_from_params``
    must carry the CNN / pose / AprilTag / runtime keys the structured config
    builder reads, mapped off the lowercase preview context."""
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )

    params = preview_worker._preview_build_inference_params(
        {
            "compute_runtime": "cpu",
            "runtime_tier": "gpu_fast",
            "cnn_runtime": "cuda",
            "cnn_classifiers": [{"model_path": "cls.json", "label": "x"}],
            "enable_pose_extractor": True,
            "pose_model_type": "sleap",
            "pose_model_dir": "/models/sleap",
            "use_apriltags": True,
            "apriltag_family": "tag25h9",
            "individual_crop_padding": 0.2,
        },
        1.0,
        False,
    )

    assert params["COMPUTE_RUNTIME"] == "cpu"
    assert params["RUNTIME_TIER"] == "gpu_fast"
    assert params["CNN_COMPUTE_RUNTIME"] == "cuda"
    assert len(params["CNN_CLASSIFIERS"]) == 1
    assert params["ENABLE_POSE_EXTRACTOR"] is True
    assert params["POSE_MODEL_TYPE"] == "sleap"
    assert params["USE_APRILTAGS"] is True
    assert params["APRILTAG_FAMILY"] == "tag25h9"
    assert params["INDIVIDUAL_CROP_PADDING"] == 0.2


class _FakeOBBResult:
    """Minimal stand-in for ``core.inference.result.OBBResult``."""

    def __init__(self, corners, confidences, class_ids=None) -> None:
        self.corners = np.asarray(corners, dtype=np.float32)
        self.confidences = np.asarray(confidences, dtype=np.float32)
        self.class_ids = (
            np.asarray(class_ids, dtype=np.int32) if class_ids is not None else None
        )

    @property
    def num_detections(self) -> int:
        return int(self.corners.shape[0])

    @property
    def class_ids_or_zeros(self) -> np.ndarray:
        if self.class_ids is None:
            return np.zeros(self.num_detections, dtype=np.int32)
        return self.class_ids


class _FakeHeadTail:
    def __init__(self, hints, confs, directed) -> None:
        self.heading_hints = np.asarray(hints, dtype=np.float32)
        self.heading_confidences = np.asarray(confs, dtype=np.float32)
        self.directed_mask = np.asarray(directed, dtype=np.uint8)


class _FakeFrameResult:
    """Minimal ``FrameResult`` exposing exactly the fields the drawing reads."""

    def __init__(
        self, obb=None, headtail=None, cnn=None, pose=None, apriltag=None
    ) -> None:
        self.obb = obb
        self.headtail = headtail
        self.cnn = cnn if cnn is not None else []
        self.pose = pose
        self.apriltag = apriltag


def _install_fake_runner(
    monkeypatch, preview_worker, frame_result, obb_class_names=None
):
    """Replace the module-level ``InferenceRunner`` with a fake returning
    ``frame_result``; returns the shared capture dict."""
    built: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, cfg, **kwargs) -> None:
            built["cfg"] = cfg

        @property
        def obb_class_names(self):
            return obb_class_names

        def run_realtime(self, frame, frame_idx=0, roi_mask=None):
            built["ran"] = True
            built["roi_mask"] = roi_mask
            return frame_result

        def close(self) -> None:
            built["closed"] = True

    monkeypatch.setattr(preview_worker, "InferenceRunner", _FakeRunner, raising=False)
    return built


def test_preview_yolo_branch_drives_inference_runner(monkeypatch) -> None:
    """The YOLO preview branch must drive ``InferenceRunner.run_realtime`` and
    must never construct a legacy ``YOLOOBBDetector``."""
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    detectors_pkg = importlib.import_module("hydra_suite.core.detectors")

    def _boom(*args, **kwargs):
        raise AssertionError(
            "preview should not construct a legacy YOLOOBBDetector instance"
        )

    monkeypatch.setattr(detectors_pkg, "YOLOOBBDetector", _boom, raising=False)

    corners = np.array(
        [[10.0, 10.0], [22.0, 10.0], [22.0, 16.0], [10.0, 16.0]], dtype=np.float32
    )
    obb = _FakeOBBResult([corners], [0.91])
    fr = _FakeFrameResult(obb=obb)
    built = _install_fake_runner(monkeypatch, preview_worker, fr)

    monkeypatch.setattr(
        preview_worker,
        "_preview_resize_frame",
        lambda frame_bgr, test_frame, resize_f: (frame_bgr, test_frame),
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

    assert built["ran"] is True
    assert built["closed"] is True
    assert len(detected_dimensions) == 1
    assert out_frame.shape == test_frame.shape


def test_preview_yolo_branch_empty_frame_result_returns_no_dimensions(
    monkeypatch,
) -> None:
    """A zero-detection ``FrameResult`` yields an empty detected-dimensions list
    and a frame of unchanged shape."""
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )

    obb = _FakeOBBResult(np.zeros((0, 4, 2), dtype=np.float32), np.zeros((0,)))
    fr = _FakeFrameResult(obb=obb)
    _install_fake_runner(monkeypatch, preview_worker, fr)
    monkeypatch.setattr(
        preview_worker,
        "_preview_resize_frame",
        lambda frame_bgr, test_frame, resize_f: (frame_bgr, test_frame),
    )

    test_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detected_dimensions, out_frame = preview_worker._preview_run_yolo_branch(
        test_frame, test_frame.copy(), {"yolo_model_path": "d.pt"}, 1.0, False
    )
    assert detected_dimensions == []
    assert out_frame.shape == test_frame.shape


def test_preview_yolo_branch_forwards_headtail_hints_to_draw(monkeypatch) -> None:
    """Head-tail hints from ``FrameResult.headtail`` reach the OBB annotation
    drawing as per-detection ``(heading, conf, directed)`` tuples."""
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )

    corners = np.array(
        [[10.0, 10.0], [22.0, 10.0], [22.0, 16.0], [10.0, 16.0]], dtype=np.float32
    )
    obb = _FakeOBBResult([corners], [0.91])
    ht = _FakeHeadTail([1.25], [0.88], [1])
    fr = _FakeFrameResult(obb=obb, headtail=ht)
    _install_fake_runner(monkeypatch, preview_worker, fr)

    monkeypatch.setattr(
        preview_worker,
        "_preview_resize_frame",
        lambda frame_bgr, test_frame, resize_f: (frame_bgr, test_frame),
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
        captured["filtered_headtail"] = list(filtered_headtail)
        captured["detection_confidences"] = list(detection_confidences)

    monkeypatch.setattr(
        preview_worker, "_preview_draw_obb_annotations", _capture_annotations
    )

    test_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    preview_worker._preview_run_yolo_branch(
        test_frame, test_frame.copy(), {"yolo_model_path": "d.pt"}, 1.0, False
    )

    assert len(captured["filtered_headtail"]) == 1
    heading, conf, directed = captured["filtered_headtail"][0]
    assert heading == pytest.approx(1.25, abs=1e-4)
    assert conf == pytest.approx(0.88, abs=1e-4)
    assert directed == 1
    assert captured["detection_confidences"][0] == pytest.approx(0.91, abs=1e-4)


def test_preview_yolo_branch_uses_real_class_labels(monkeypatch) -> None:
    """When the OBB model exposes class names, the preview must draw the real
    class label per detection instead of the generic "obj" fallback."""
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )

    corners = np.array(
        [[10.0, 10.0], [22.0, 10.0], [22.0, 16.0], [10.0, 16.0]], dtype=np.float32
    )
    two_corners = np.stack([corners, corners + 30.0])
    obb = _FakeOBBResult(two_corners, [0.91, 0.80], class_ids=[1, 0])
    fr = _FakeFrameResult(obb=obb)
    _install_fake_runner(
        monkeypatch, preview_worker, fr, obb_class_names={0: "ant", 1: "queen"}
    )

    monkeypatch.setattr(
        preview_worker,
        "_preview_resize_frame",
        lambda frame_bgr, test_frame, resize_f: (frame_bgr, test_frame),
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
        captured["filtered_class_labels"] = list(filtered_class_labels)

    monkeypatch.setattr(
        preview_worker, "_preview_draw_obb_annotations", _capture_annotations
    )

    test_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    preview_worker._preview_run_yolo_branch(
        test_frame, test_frame.copy(), {"yolo_model_path": "d.pt"}, 1.0, False
    )

    assert captured["filtered_class_labels"] == ["queen", "ant"]


def test_preview_run_cnn_overlay_formats_multihead_predictions() -> None:
    """CNN overlay must format multi-head ``FrameResult`` CNN predictions,
    taking the arg-max class per factor, into the per-detection label stack."""
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    from hydra_suite.core.inference.result import (
        CNNDetectionPrediction,
        CNNFactorPrediction,
        CNNResult,
    )

    det_pred = CNNDetectionPrediction(
        det_index=0,
        factors=[
            CNNFactorPrediction(
                factor_name="color",
                class_names=["blue", "red"],
                raw_probabilities=np.array([0.07, 0.93], dtype=np.float32),
            ),
            CNNFactorPrediction(
                factor_name="side",
                class_names=["left", "right"],
                raw_probabilities=np.array([0.58, 0.42], dtype=np.float32),
            ),
        ],
    )
    cnn_results = [CNNResult(label="cnn_identity", predictions=[det_pred])]

    label_stacks: list[list[str]] = [[]]
    preview_worker._preview_run_cnn_overlay(cnn_results, label_stacks)

    assert label_stacks == [["cnn_identity: color=red 0.93 | side=left 0.58"]]


def test_preview_run_pose_overlay_labels_and_keypoints() -> None:
    """Pose overlay must add a ``pose <mean> <valid>/<total>`` label line and
    stash frame-space keypoints for the detection."""
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    from hydra_suite.core.inference.result import PoseResult

    keypoints = np.array(
        [[[5.0, 5.0, 0.9], [6.0, 6.0, 0.8], [7.0, 7.0, 0.05]]], dtype=np.float32
    )
    pose = PoseResult(keypoints=keypoints, valid_mask=np.array([True]))

    label_stacks: list[list[str]] = [[]]
    pose_keypoints_by_det: dict[int, np.ndarray] = {}
    preview_worker._preview_run_pose_overlay(
        pose, {"pose_min_kpt_conf_valid": 0.2}, label_stacks, pose_keypoints_by_det
    )

    assert label_stacks[0][0].startswith("pose ")
    assert label_stacks[0][0].endswith("2/3")  # two of three keypoints valid
    assert 0 in pose_keypoints_by_det
    assert pose_keypoints_by_det[0].shape == (3, 3)


def test_preview_run_apriltag_overlay_offsets_corners_to_frame_space(
    monkeypatch,
) -> None:
    """AprilTag overlay must offset crop-local tag corners back into frame
    space using the detection's AABB-crop origin and add a per-detection tag
    label line."""
    preview_worker = importlib.import_module(
        "hydra_suite.trackerkit.gui.workers.preview_worker"
    )
    from hydra_suite.core.inference.result import AprilTagResult

    # Detection AABB at x in [100,140], y in [100,140]; padding 0 -> origin (100,100).
    det_corners = np.array(
        [[100.0, 100.0], [140.0, 100.0], [140.0, 140.0], [100.0, 140.0]],
        dtype=np.float32,
    )
    obb = _FakeOBBResult([det_corners], [0.9])

    tag_corners = np.array(
        [[[2.0, 2.0], [10.0, 2.0], [10.0, 10.0], [2.0, 10.0]]], dtype=np.float32
    )
    apriltag = AprilTagResult(
        tag_ids=[7],
        det_indices=[0],
        centers=np.array([[6.0, 6.0]], dtype=np.float32),
        corners=tag_corners,
    )

    drawn: dict[str, object] = {}

    def _fake_polylines(img, pts, isClosed, color, thickness):
        drawn["pts"] = np.asarray(pts[0])

    monkeypatch.setattr(preview_worker.cv2, "polylines", _fake_polylines)
    monkeypatch.setattr(
        preview_worker, "_draw_preview_label_stack", lambda *a, **k: None
    )

    test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    label_stacks: list[list[str]] = [[]]
    preview_worker._preview_run_apriltag_overlay(
        apriltag, obb, {"individual_crop_padding": 0.0}, label_stacks, test_frame
    )

    # crop-local (2,2) + origin (100,100) -> frame-space (102,102)
    assert drawn["pts"][0].tolist() == [102, 102]
    assert label_stacks[0] == ["tag 7"]


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


def test_detection_panel_context_populates_runtime_tier(monkeypatch) -> None:
    """Regression: ``_collect_preview_detection_context`` must populate
    ``runtime_tier`` so the preview InferenceConfig is built on the selected
    pipeline tier (rather than defaulting)."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])

    from hydra_suite.trackerkit.gui.main_window import MainWindow

    resolver_mod = importlib.import_module("hydra_suite.runtime.resolver")
    monkeypatch.setattr(
        resolver_mod,
        "detect_platform",
        lambda: resolver_mod.PlatformInfo(has_cuda=True, has_mps=False),
    )

    mw = MainWindow()
    try:
        monkeypatch.setattr(mw, "_selected_runtime_tier", lambda: "gpu_fast")
        monkeypatch.setattr(
            mw, "_get_resolved_pose_model_dir", lambda backend: "/models/sleap_model"
        )
        monkeypatch.setattr(mw, "_is_pose_inference_enabled", lambda: True)
        monkeypatch.setattr(mw, "_selected_pose_sleap_env", lambda: "sleap_env_x")
        mw._identity_panel.combo_pose_model_type.setCurrentText("SLEAP")

        context = mw._detection_panel._collect_preview_detection_context()

        assert context["runtime_tier"] == "gpu_fast"
        assert context["pose_model_type"] == "sleap"
        assert context["pose_model_dir"] == "/models/sleap_model"
        assert context["enable_pose_extractor"] is True
    finally:
        mw.close()
