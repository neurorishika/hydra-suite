"""Tests for ModelTestDialog parameter construction."""

from __future__ import annotations

from hydra_suite.trackerkit.gui.dialogs.model_test_dialog import build_test_params


def test_build_test_params_direct_mode():
    params = build_test_params(
        model_path="/models/best.pt",
        role="obb_direct",
        device="cpu",
        imgsz=640,
    )
    assert params["YOLO_OBB_MODE"] == "direct"
    assert params["YOLO_OBB_DIRECT_MODEL_PATH"] == "/models/best.pt"
    assert params["YOLO_MODEL_PATH"] == "/models/best.pt"
    assert params["YOLO_DEVICE"] == "cpu"


def test_build_test_params_seq_crop_obb():
    params = build_test_params(
        model_path="/models/crop.pt",
        role="seq_crop_obb",
        device="cpu",
        imgsz=160,
        crop_pad_ratio=0.15,
        min_crop_size_px=64,
        enforce_square=True,
        detect_model_path="/models/detect.pt",
    )
    assert params["YOLO_OBB_MODE"] == "sequential"
    assert params["YOLO_DETECT_MODEL_PATH"] == "/models/detect.pt"
    assert params["YOLO_CROP_OBB_MODEL_PATH"] == "/models/crop.pt"
    assert params["YOLO_SEQ_STAGE2_IMGSZ"] == 160
    assert params["YOLO_SEQ_CROP_PAD_RATIO"] == 0.15


def test_build_test_params_seq_detect():
    params = build_test_params(
        model_path="/models/detect.pt",
        role="seq_detect",
        device="cuda",
        imgsz=640,
    )
    assert params["YOLO_OBB_MODE"] == "sequential"
    assert params["YOLO_MODEL_PATH"] == "/models/detect.pt"
    assert params["YOLO_DETECT_MODEL_PATH"] == "/models/detect.pt"
    assert "YOLO_CROP_OBB_MODEL_PATH" not in params
    assert params["YOLO_DEVICE"] == "cuda"


def test_build_test_params_defaults():
    params = build_test_params(
        model_path="/models/best.pt",
        role="obb_direct",
        device="cpu",
        imgsz=640,
    )
    assert params["YOLO_CONFIDENCE_THRESHOLD"] == 0.25
    assert params["YOLO_IOU_THRESHOLD"] == 0.45
    assert params["YOLO_MAX_TARGETS"] == 100
    assert params["USE_TENSORRT"] is False
    assert params["USE_ONNX"] is False
