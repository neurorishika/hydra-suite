"""Compare CUDA, ONNX, and TensorRT OBB detections on random video frames.

Usage:
    conda run -n hydra-cuda python tools/compare_runtimes.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─── Load config ──────────────────────────────────────────────────────────────
CFG_PATH = Path("/home/rutalab/libby/v1_config.json")
with open(CFG_PATH) as f:
    cfg = json.load(f)

VIDEO_PATH = cfg["file_path"]
MODEL_KEY = "obb/20260423-172251_26x_obiroi_train36.pt"

from hydra_suite.paths import get_models_dir

MODELS_DIR = get_models_dir()
PT_PATH = MODELS_DIR / MODEL_KEY

CONF_THRES = float(cfg["yolo_confidence_threshold"])  # 0.19
IOU_THRES = float(cfg["yolo_iou_threshold"])  # 0.36
MAX_DET = int(cfg.get("max_targets", 25)) * 4  # raw cap
RAW_FLOOR = 0.001
N_FRAMES = 5
RANDOM_SEED = 42
BODY_AREA = math.pi * (cfg["reference_body_size"] / 2) ** 2
MIN_SIZE = cfg["min_object_size_multiplier"] * BODY_AREA
MAX_SIZE = cfg["max_object_size_multiplier"] * BODY_AREA
REF_AR = cfg["reference_aspect_ratio"]
MIN_AR = REF_AR * cfg["min_aspect_ratio_multiplier"]
MAX_AR = REF_AR * cfg["max_aspect_ratio_multiplier"]

# ─── Pick random frames ────────────────────────────────────────────────────────
rng = random.Random(RANDOM_SEED)
start_f, end_f = int(cfg["start_frame"]), int(cfg["end_frame"])
frame_indices = sorted(rng.sample(range(start_f, end_f + 1), N_FRAMES))
print(f"Testing frames: {frame_indices}")
print(f"CONF_THRES={CONF_THRES}  IOU_THRES={IOU_THRES}  MAX_DET={MAX_DET}")
print(f"Size filter: ellipse_area in [{MIN_SIZE:.1f}, {MAX_SIZE:.1f}]")
print(f"AR filter: [{MIN_AR:.3f}, {MAX_AR:.3f}]")

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Cannot open video: {VIDEO_PATH}"


def read_frame(idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    assert ok, f"Could not read frame {idx}"
    return frame  # BGR HxWxC uint8


from ultralytics import YOLO

# ─── Build the three direct executors ─────────────────────────────────────────
from hydra_suite.core.inference.direct_executors import (
    DirectONNXOBBExecutor,
    DirectPyTorchCUDAOBBExecutor,
    DirectTensorRTOBBExecutor,
)

print("\nLoading PyTorch model …")
pt_model = YOLO(str(PT_PATH))
IMGSZ = 1024

ONNX_PATH = str(next(MODELS_DIR.glob("obb/20260423-172251_26x_obiroi_train36_b1.onnx")))
TRT_PATH = str(
    next(MODELS_DIR.glob("obb/20260423-172251_26x_obiroi_train36_b1.engine"))
)

class_names = {0: "fish"}
nc = 1

print("Creating CUDA executor …")
cuda_exec = DirectPyTorchCUDAOBBExecutor(
    pt_model, IMGSZ, class_names=class_names, class_count=nc
)
print("Creating ONNX executor …")
onnx_exec = DirectONNXOBBExecutor(
    ONNX_PATH, IMGSZ, class_names=class_names, class_count=nc
)
print("Creating TensorRT executor …")
trt_exec = DirectTensorRTOBBExecutor(
    TRT_PATH, IMGSZ, class_names=class_names, class_count=nc
)

print("\nExecutor info:")
print(f"  CUDA: _end2end={cuda_exec._end2end}")
print(
    f"  ONNX: _end2end={getattr(onnx_exec, '_end2end', False)}, fp16={onnx_exec._fp16}, "
    f"static_out={onnx_exec._static_out_shape}"
)
print(
    f"  TRT:  _end2end={trt_exec._end2end}, static_out={getattr(trt_exec, '_static_out_shape', 'N/A')}"
)


# ─── Helper: extract (cx, cy, w, h, angle_rad, conf) from predict() results ──
def run_exec(executor, frame: np.ndarray, conf_thres: float) -> np.ndarray:
    """Return (N,6) array [cx,cy,w,h,angle_rad,conf] sorted by conf desc."""
    results = executor.predict(
        [frame], conf_thres=conf_thres, classes=None, max_det=MAX_DET
    )
    r = results[0]
    if r.obb is None or len(r.obb) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xywhr = r.obb.xywhr.cpu().numpy()  # (N, 5): cx cy w h angle_rad
    conf = r.obb.conf.cpu().numpy()  # (N,)
    data = np.hstack([xywhr, conf[:, None]]).astype(np.float32)
    order = np.argsort(-data[:, 5])
    return data[order]


# ─── Size + AR filter (no additional NMS, just masking) ───────────────────────
def apply_size_ar_filter(data: np.ndarray) -> np.ndarray:
    """Apply ellipse-area and aspect-ratio gates; no NMS."""
    if len(data) == 0:
        return data
    w = data[:, 2]
    h = data[:, 3]
    major = np.maximum(w, h)
    minor = np.minimum(w, h)
    ellipse_area = math.pi * (major / 2.0) * (minor / 2.0)
    ar = np.where(minor > 0, major / minor, 0.0)
    keep = (
        (ellipse_area >= MIN_SIZE)
        & (ellipse_area <= MAX_SIZE)
        & (ar >= MIN_AR)
        & (ar <= MAX_AR)
    )
    return data[keep]


# ─── Comparison helper ────────────────────────────────────────────────────────
def compare(
    a_name, a_data, b_name, b_data, tol_px=1.5, tol_conf=0.02, tol_angle=0.05, top_n=5
):
    if len(a_data) != len(b_data):
        print(f"  ✗ count mismatch: {a_name}={len(a_data)}  {b_name}={len(b_data)}")
        if len(a_data) > 0 and len(b_data) > 0:
            n_show = min(top_n, len(a_data), len(b_data))
            print(f"     {a_name} top-{n_show}:")
            for row in a_data[:n_show]:
                print(
                    f"       cx={row[0]:.1f} cy={row[1]:.1f} w={row[2]:.1f} h={row[3]:.1f} "
                    f"ang={row[4]:.4f} conf={row[5]:.4f}"
                )
            print(f"     {b_name} top-{n_show}:")
            for row in b_data[:n_show]:
                print(
                    f"       cx={row[0]:.1f} cy={row[1]:.1f} w={row[2]:.1f} h={row[3]:.1f} "
                    f"ang={row[4]:.4f} conf={row[5]:.4f}"
                )
        return False

    if len(a_data) == 0:
        print("  ✓ both empty")
        return True

    from scipy.spatial.distance import cdist

    centres_a = a_data[:, :2]
    centres_b = b_data[:, :2]
    D = cdist(centres_a, centres_b)
    col_ind = D.argmin(axis=1)
    matched_b = b_data[col_ind]

    max_pos = np.abs(a_data[:, :2] - matched_b[:, :2]).max()
    max_wh = np.abs(a_data[:, 2:4] - matched_b[:, 2:4]).max()
    max_ang = np.abs(a_data[:, 4] - matched_b[:, 4]).max()
    max_conf = np.abs(a_data[:, 5] - matched_b[:, 5]).max()

    ok = (
        max_pos <= tol_px
        and max_wh <= tol_px
        and max_ang <= tol_angle
        and max_conf <= tol_conf
    )
    sym = "✓" if ok else "✗"
    print(
        f"  {sym} {a_name} vs {b_name}: "
        f"Δpos={max_pos:.3f}px  Δwh={max_wh:.3f}px  "
        f"Δangle={max_ang:.4f}rad  Δconf={max_conf:.4f}"
    )
    if not ok:
        n_show = min(top_n, len(a_data))
        print(f"     {a_name} top-{n_show}:")
        for row in a_data[:n_show]:
            print(
                f"       cx={row[0]:.1f} cy={row[1]:.1f} w={row[2]:.1f} h={row[3]:.1f} "
                f"ang={row[4]:.4f} conf={row[5]:.4f}"
            )
        print(f"     {b_name} top-{n_show} (matched):")
        for row in matched_b[:n_show]:
            print(
                f"       cx={row[0]:.1f} cy={row[1]:.1f} w={row[2]:.1f} h={row[3]:.1f} "
                f"ang={row[4]:.4f} conf={row[5]:.4f}"
            )
    return ok


# ─── Main comparison loop ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 1: Raw pre-NMS detections (conf_thres=RAW_FLOOR=0.001)")
print("=" * 70)

all_ok_raw = True
frames_data = []
for fi in frame_indices:
    frame = read_frame(fi)
    cuda_raw = run_exec(cuda_exec, frame, RAW_FLOOR)
    onnx_raw = run_exec(onnx_exec, frame, RAW_FLOOR)
    trt_raw = run_exec(trt_exec, frame, RAW_FLOOR)
    frames_data.append((fi, frame, cuda_raw, onnx_raw, trt_raw))

    print(
        f"\nFrame {fi}: CUDA={len(cuda_raw)}  ONNX={len(onnx_raw)}  TRT={len(trt_raw)} raw dets"
    )
    ok1 = compare("CUDA", cuda_raw, "ONNX", onnx_raw, tol_px=3.0, tol_conf=0.05)
    ok2 = compare("CUDA", cuda_raw, "TRT", trt_raw, tol_px=3.0, tol_conf=0.05)
    ok3 = compare("ONNX", onnx_raw, "TRT", trt_raw, tol_px=3.0, tol_conf=0.05)
    all_ok_raw = all_ok_raw and ok1 and ok2 and ok3

print("\n" + "=" * 70)
print(
    f"PHASE 2: Detections at conf_thres={CONF_THRES} (working threshold, no size/AR filter)"
)
print("=" * 70)

all_ok_conf = True
frames_conf = []
for fi, frame, cuda_raw, onnx_raw, trt_raw in frames_data:
    cuda_conf = run_exec(cuda_exec, frame, CONF_THRES)
    onnx_conf = run_exec(onnx_exec, frame, CONF_THRES)
    trt_conf = run_exec(trt_exec, frame, CONF_THRES)
    frames_conf.append((fi, cuda_conf, onnx_conf, trt_conf))

    print(
        f"\nFrame {fi}: CUDA={len(cuda_conf)}  ONNX={len(onnx_conf)}  TRT={len(trt_conf)} dets@conf"
    )
    ok1 = compare("CUDA", cuda_conf, "ONNX", onnx_conf)
    ok2 = compare("CUDA", cuda_conf, "TRT", trt_conf)
    ok3 = compare("ONNX", onnx_conf, "TRT", trt_conf)
    all_ok_conf = all_ok_conf and ok1 and ok2 and ok3

print("\n" + "=" * 70)
print(f"PHASE 3: Detections at conf_thres={CONF_THRES} + size/AR filter")
print("=" * 70)

all_ok_filt = True
for fi, cuda_conf, onnx_conf, trt_conf in frames_conf:
    cuda_filt = apply_size_ar_filter(cuda_conf)
    onnx_filt = apply_size_ar_filter(onnx_conf)
    trt_filt = apply_size_ar_filter(trt_conf)

    print(
        f"\nFrame {fi}: CUDA={len(cuda_filt)}  ONNX={len(onnx_filt)}  TRT={len(trt_filt)} filt dets"
    )
    ok1 = compare("CUDA", cuda_filt, "ONNX", onnx_filt)
    ok2 = compare("CUDA", cuda_filt, "TRT", trt_filt)
    ok3 = compare("ONNX", onnx_filt, "TRT", trt_filt)
    all_ok_filt = all_ok_filt and ok1 and ok2 and ok3

cap.release()
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Phase 1 raw dets match (RAW_FLOOR):        {'✓ YES' if all_ok_raw else '✗ NO'}")
print(
    f"Phase 2 conf-thresh dets match (no filter): {'✓ YES' if all_ok_conf else '✗ NO'}"
)
print(
    f"Phase 3 filtered dets match:                {'✓ YES' if all_ok_filt else '✗ NO'}"
)
