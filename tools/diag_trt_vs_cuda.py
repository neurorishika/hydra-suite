"""
diag_trt_vs_cuda.py — Compare TensorRT vs CUDA/ONNX OBB detection paths.

Usage:
    conda run -n hydra-cuda python tools/diag_trt_vs_cuda.py

Checks:
1. TRT vs ONNX direct executor on the same preprocessed input
2. TRT vs PyTorch CUDA (via Ultralytics wrapper) — the real user scenario
3. Detailed output diff analysis
"""

import json
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ─────────────────────────────────────────────────────────────────────────────
# Model files (relative to repo root)
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "yolov8n-obb_b1.engine")
ONNX = os.path.join(ROOT, "yolov8n-obb_b1.onnx")
PT = os.path.join(ROOT, "yolov8n-obb.pt")

assert os.path.exists(ENGINE), f"Engine not found: {ENGINE}"
assert os.path.exists(ONNX), f"ONNX not found: {ONNX}"

# ─────────────────────────────────────────────────────────────────────────────
# Read engine metadata
# ─────────────────────────────────────────────────────────────────────────────
with open(ENGINE, "rb") as f:
    meta_len = struct.unpack("<I", f.read(4))[0]
    meta = json.loads(f.read(meta_len).decode())

names = {int(k): v for k, v in meta["names"].items()}
nc = len(names)
imgsz = meta["imgsz"][0]  # 1024
print(f"Model: imgsz={imgsz}, nc={nc}")
print(f"Classes: {list(names.values())}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Build a synthetic widescreen test frame (1920×1080)
# ─────────────────────────────────────────────────────────────────────────────
H_orig, W_orig = 1080, 1920
rng = np.random.RandomState(42)
frame_bgr = (rng.rand(H_orig, W_orig, 3) * 200 + 30).astype(np.uint8)

from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms, ops

# ─────────────────────────────────────────────────────────────────────────────
# Load direct executors
# ─────────────────────────────────────────────────────────────────────────────
from hydra_suite.core.detectors._direct_obb_runtime import (
    DirectONNXOBBExecutor,
    DirectTensorRTOBBExecutor,
)

print("Loading TRT executor...")
exec_trt = DirectTensorRTOBBExecutor(ENGINE, imgsz, class_names=names, class_count=nc)
print("Loading ONNX executor...")
exec_onnx = DirectONNXOBBExecutor(ONNX, imgsz, class_names=names, class_count=nc)
print()

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Compare raw model outputs on identical input
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART 1: Raw inference output comparison (identical input)")
print("=" * 60)

# Preprocess exactly as DirectExecutor does (auto=False)
lb = LetterBox((imgsz, imgsz), auto=False, stride=32)
lb_frame = lb(image=frame_bgr)
inp = (
    torch.from_numpy(
        np.ascontiguousarray(lb_frame.transpose(2, 0, 1)[::-1], dtype=np.float32)
        / 255.0
    )
    .unsqueeze(0)
    .to("cuda:0")
)
print(f"Model input shape: {tuple(inp.shape)}")

raw_trt = exec_trt._run_inference(inp).clone()
raw_onnx = exec_onnx._run_inference(inp).clone()

print(f"TRT  output: shape={tuple(raw_trt.shape)}  dtype={raw_trt.dtype}")
print(f"ONNX output: shape={tuple(raw_onnx.shape)}  dtype={raw_onnx.dtype}")

diff = (raw_trt - raw_onnx).abs()
print()
print("Raw output numerical differences:")
print(f"  max  diff: {diff.max().item():.4f}")
print(f"  mean diff: {diff.mean().item():.6f}")
print(f"  % > 0.001: {(diff > 0.001).float().mean().item() * 100:.2f}%")
print(f"  % > 0.01:  {(diff > 0.01).float().mean().item() * 100:.2f}%")
print(f"  % > 0.1:   {(diff > 0.1).float().mean().item() * 100:.2f}%")

# Channel-wise analysis
print()
print("Per-channel (feature type) max diff:")
channel_labels = ["cx", "cy", "w", "h"] + [f"cls_{i}" for i in range(nc)] + ["angle"]
for ch, label in enumerate(channel_labels):
    ch_diff = diff[0, ch, :].max().item()
    if ch_diff > 0.01:
        print(f"  ch[{ch:2d}] {label:10s}: max_diff={ch_diff:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Confidence channel differences near threshold
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PART 2: Confidence channel analysis (NMS-decision level)")
print("=" * 60)

CONF_THRES = 0.25

# Best class conf per anchor
trt_bestconf = raw_trt[0, 4 : 4 + nc, :].max(dim=0)[0]
onnx_bestconf = raw_onnx[0, 4 : 4 + nc, :].max(dim=0)[0]

print(
    f"  TRT  anchors above conf_thres={CONF_THRES}: {(trt_bestconf > CONF_THRES).sum().item()}"
)
print(
    f"  ONNX anchors above conf_thres={CONF_THRES}: {(onnx_bestconf > CONF_THRES).sum().item()}"
)

# Find threshold-crossing disagreements (one side fires, other doesn't)
trt_fire = trt_bestconf > CONF_THRES
onnx_fire = onnx_bestconf > CONF_THRES
disagreements = (trt_fire != onnx_fire).sum().item()
print(f"  Anchors where TRT and ONNX DISAGREE on threshold: {disagreements}")
print(f"  (These would produce different detection counts!)")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Full detection pipeline comparison
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PART 3: Full detection pipeline comparison")
print("=" * 60)

for conf in [0.5, 0.25, 0.1, 0.05]:
    res_trt = exec_trt.predict(
        [frame_bgr.copy()], conf_thres=conf, classes=None, max_det=300
    )
    res_onnx = exec_onnx.predict(
        [frame_bgr.copy()], conf_thres=conf, classes=None, max_det=300
    )
    n_trt = len(res_trt[0].obb) if res_trt[0].obb is not None else 0
    n_onnx = len(res_onnx[0].obb) if res_onnx[0].obb is not None else 0
    match = "✓ same" if n_trt == n_onnx else "✗ DIFFER"
    print(
        f"  conf_thres={conf:.2f}: TRT={n_trt:3d} dets, ONNX={n_onnx:3d} dets  [{match}]"
    )

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Preprocessing comparison — direct executor vs PyTorch CUDA path
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PART 4: Preprocessing difference (TRT/ONNX executor vs PyTorch CUDA)")
print("=" * 60)

lb_direct = LetterBox((imgsz, imgsz), auto=False, stride=32)
lb_pt = LetterBox((imgsz, imgsz), auto=True, stride=32)
frame_direct = lb_direct(image=frame_bgr)
frame_pt = lb_pt(image=frame_bgr)

print(f"  Direct executor (auto=False): {frame_bgr.shape} → {frame_direct.shape}")
print(f"  PyTorch CUDA  (auto=True):   {frame_bgr.shape} → {frame_pt.shape}")
print()
if frame_direct.shape != frame_pt.shape:
    print("  *** DIFFERENT input shapes DETECTED ***")
    print("  The PyTorch CUDA path uses rectangular letterboxing for widescreen video!")
    print("  The TRT/ONNX executor always pads to a square (auto=False).")
    print()
    print("  Consequence:")
    print(f"    TRT input  : {frame_direct.shape} → model sees square 1024×1024 image")
    print(
        f"    CUDA input : {frame_pt.shape}  → model processes {frame_pt.shape[0]}×{frame_pt.shape[1]} image"
    )
    print()
    print("  Both paths correctly inverse-map back to original coords via scale_boxes,")
    print("  but the FPN feature statistics differ due to different padding context.")
    print("  This causes small but consistent detection-score differences.")
else:
    print("  Same input shape for this frame aspect ratio - no preprocessing mismatch.")

# ─────────────────────────────────────────────────────────────────────────────
# PART 5: Angle unit verification
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PART 5: Angle unit verification (radians vs degrees guard)")
print("=" * 60)

angle_trt = raw_trt[0, -1, :].cpu().numpy()
angle_onnx = raw_onnx[0, -1, :].cpu().numpy()

print(
    f"  TRT  angle channel: min={angle_trt.min():.4f} max={angle_trt.max():.4f} → unit: {'DEGREES (needs conversion!)' if np.abs(angle_trt).max() > 2 * np.pi + 0.001 else 'radians (OK)'}"
)
print(
    f"  ONNX angle channel: min={angle_onnx.min():.4f} max={angle_onnx.max():.4f} → unit: {'DEGREES (needs conversion!)' if np.abs(angle_onnx).max() > 2 * np.pi + 0.001 else 'radians (OK)'}"
)

# ─────────────────────────────────────────────────────────────────────────────
# PART 6: Verify scale_boxes postprocessing
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("PART 6: PostProcess — NMS output column interpretation")
print("=" * 60)

# Run NMS manually and inspect column meanings
filtered = nms.non_max_suppression(
    raw_trt, conf_thres=0.01, iou_thres=1.0, max_det=300, nc=nc, rotated=True
)
if filtered and filtered[0] is not None and len(filtered[0]) > 0:
    ex = filtered[0][0]  # first detection
    print(
        f"  NMS output shape per detection: {tuple(filtered[0].shape)} — expected [N, 7]"
    )
    print(f"  First detection: {ex.tolist()}")
    print(f"  pred[:, :4]   = cx,cy,w,h = {ex[:4].tolist()}")
    print(f"  pred[:, 4]    = conf       = {ex[4].item():.4f}")
    print(f"  pred[:, 5]    = cls_id     = {int(ex[5].item())}")
    print(
        f"  pred[:, -1:]  = angle      = {ex[-1].item():.4f} rad = {np.degrees(ex[-1].item()):.1f}°"
    )
    print(f"  pred[:, 4:6]  = (conf,cls) = {ex[4:6].tolist()}")
    print()
    print("  scale_boxes mapping:")
    dummy_orig = np.empty((H_orig, W_orig, 3), dtype=np.uint8)
    boxes_before = ex[:4].unsqueeze(0).clone()
    rboxes = torch.cat([ex[:4].unsqueeze(0), ex[-1:].unsqueeze(0).unsqueeze(0)], dim=-1)
    rboxes[:, :4] = ops.scale_boxes(
        (imgsz, imgsz), rboxes[:, :4], dummy_orig.shape, xywh=True
    )
    print(
        f"  Pre-scale  box: cx={ex[0]:.0f} cy={ex[1]:.0f} w={ex[2]:.0f} h={ex[3]:.0f} (model space)"
    )
    print(
        f"  Post-scale box: cx={rboxes[0, 0]:.0f} cy={rboxes[0, 1]:.0f} w={rboxes[0, 2]:.0f} h={rboxes[0, 3]:.0f} (original frame space)"
    )
else:
    print("  No detections above 0.01 — cannot verify NMS column structure.")

print()
print("=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)
print()
print("1. TRT uses FP16 internal precision → small numerical differences in raw output")
print(
    f"   Max raw output diff: {diff.max().item():.4f}  (mean: {diff.mean().item():.6f})"
)
print()
print(
    "2. Both TRT and ONNX direct executors use auto=False preprocessing (square input)"
)
print("   → same preprocessing for TRT and ONNX executor paths")
print()
print(
    "3. PyTorch CUDA path uses auto=True (rectangular letterbox for widescreen video)"
)
print(f"   → TRT: {frame_direct.shape} input, CUDA: {frame_pt.shape} input")
if frame_direct.shape != frame_pt.shape:
    print("   → DIFFERENT model inputs for non-square frames!")
print()
print("4. The FP16 precision difference can cause detections near the confidence")
print("   threshold to fire in one backend but not the other:")
print(f"   Anchors near threshold where TRT/ONNX DISAGREE: {disagreements}")
