#!/usr/bin/env python
"""Crop-preparation experiment for SLEAP pose quality (run in the SLEAP env).

The new pipeline's SLEAP keypoints are poor even though the crops look clean,
while legacy keypoints are coherent. Hypothesis: SLEAP (single-instance) is
sensitive to crop ORIENTATION, and the new pipeline builds the canonical crop
from ultralytics' xyxyxyxy corner order (which can come out 180-rotated /
reflected vs legacy's xywhr-derived order).

This builds several crop-orientation VARIANTS per detection, runs the warm SLEAP
predictor on each (same path the service uses), and dumps crops + predicted
keypoints + confidences to an npz. Overlay/scoring happens offline.

Run (SLEAP env):
  python tools/equivalence/pose_crop_experiment.py \
    --video tools/equivalence/fixtures/clips/ant_pose_headtail.mp4 \
    --detection /tmp/equiv/mps/ant_pose_headtail/new_a/.inference_cache_ant_pose_headtail/detection.npz \
    --model-dir /path/to/models/pose/SLEAP/<unet_single_instance> \
    --device mps --frames 0 --max-dets 8 --out /tmp/pose_exp/result.npz

Only deps: numpy, cv2, torch, sleap_nn, yaml/omegaconf. No hydra_suite import.
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

AR, MARGIN = 2.0, 1.3
PAD = max(0.0, MARGIN - 1.0)


# --- inlined from hydra_suite.core.canonicalization.crop (no hydra dep) ---
def native_dims(corners, ar, pad):
    c = np.asarray(corners, np.float32).reshape(4, 2)
    e01 = float(np.linalg.norm(c[1] - c[0]))
    e12 = float(np.linalg.norm(c[2] - c[1]))
    major = max(e01, e12)
    margin = 1.0 + max(0.0, pad)
    ar = max(1.0, ar)
    cw = max(8, int(math.ceil(major * margin / 2.0) * 2))
    ch = max(8, int(round(cw / ar / 2.0) * 2))
    return cw, ch


def alignment_affine(corners, cw, ch, pad):
    c = np.asarray(corners, np.float32).reshape(4, 2)
    e01 = float(np.linalg.norm(c[1] - c[0]))
    e12 = float(np.linalg.norm(c[2] - c[1]))
    major_vec = c[1] - c[0] if e01 >= e12 else c[2] - c[1]
    cx, cy = float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))
    angle = math.atan2(float(major_vec[1]), float(major_vec[0]))
    major, minor = max(e01, e12), min(e01, e12)
    margin = 1.0 + pad
    hw, hh = major * margin * 0.5, minor * margin * 0.5
    ca, sa = math.cos(angle), math.sin(angle)
    src = np.array([
        [cx - hw * ca + hh * sa, cy - hw * sa - hh * ca],
        [cx + hw * ca + hh * sa, cy + hw * sa - hh * ca],
        [cx - hw * ca - hh * sa, cy - hw * sa + hh * ca],
    ], np.float32)
    dst = np.array([[0, 0], [cw, 0], [0, ch]], np.float32)
    return cv2.getAffineTransform(src, dst)


def corners_from_xywhr(cx, cy, w, h, ang):
    major, minor = max(w, h), min(w, h)
    # angle of the MAJOR axis (mirror _normalize_obb_geometry's swap)
    a = ang if w >= h else ang + math.pi / 2.0
    hw, hh = major / 2.0, minor / 2.0
    xo = np.array([-hw, hw, hw, -hw]); yo = np.array([-hh, -hh, hh, hh])
    ca, sa = math.cos(a), math.sin(a)
    xs = cx + xo * ca - yo * sa
    ys = cy + xo * sa + yo * ca
    return np.stack([xs, ys], axis=1).astype(np.float32)


def warp(frame_bgr, corners, cw, ch):
    M = alignment_affine(corners, cw, ch, PAD)
    return cv2.warpAffine(frame_bgr, M, (cw, ch), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def build_variants(frame_bgr, corners, cx, cy, w, h, ang):
    cw, ch = native_dims(corners, AR, PAD)
    native = warp(frame_bgr, corners, cw, ch)
    xy_corners = corners_from_xywhr(cx, cy, w, h, ang)
    cw2, ch2 = native_dims(xy_corners, AR, PAD)
    xywhr = warp(frame_bgr, xy_corners, cw2, ch2)
    return {
        "native": native,
        "native_180": cv2.rotate(native, cv2.ROTATE_180),
        "xywhr": xywhr,
        "xywhr_180": cv2.rotate(xywhr, cv2.ROTATE_180),
        "xywhr_flipv": cv2.flip(xywhr, 0),
    }


def load_preprocess_config(model_dir):
    import yaml
    cfg = Path(model_dir) / "training_config.yaml"
    if not cfg.exists():
        return None
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    return (raw.get("data_config") or {}).get("preprocessing") or None


def build_predictor(model_dir, device, batch, pc):
    from sleap_nn.inference.predictors import Predictor
    kw = dict(model_paths=[model_dir], device=device, batch_size=batch)
    if pc is not None:
        try:
            from omegaconf import OmegaConf
            kw["preprocess_config"] = OmegaConf.create(pc)
        except Exception:
            kw["preprocess_config"] = pc
    return Predictor.from_model_paths(**kw)


def prepared_batch(crops_bgr, pred):
    import torch
    from sleap_nn.data.resizing import apply_sizematcher
    pc = pred.preprocess_config
    mh, mw = pc["max_height"], pc["max_width"]
    ensure_rgb = bool(pc.get("ensure_rgb", True))
    ensure_gray = bool(pc.get("ensure_grayscale", False))
    imgs, fidxs, vidxs, org_szs, eff = [], [], [], [], []
    for i, crop in enumerate(crops_bgr):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1))
        img = torch.from_numpy(np.expand_dims(chw, 0).copy())
        H, W = crop.shape[:2]
        img, e = apply_sizematcher(img, mh, mw)
        if ensure_rgb and img.shape[-3] != 3:
            img = img.repeat(1, 3, 1, 1)
        elif ensure_gray and img.shape[-3] != 1:
            import torchvision.transforms.functional as TF
            img = TF.rgb_to_grayscale(img, num_output_channels=1)
        imgs.append(img.unsqueeze(0))
        org_szs.append(torch.Tensor([H, W]).unsqueeze(0).unsqueeze(0))
        eff.append(torch.tensor(e)); fidxs.append(i); vidxs.append(0)
    return imgs, fidxs, vidxs, org_szs, [], eff


def predict(pred, crops_bgr):
    if getattr(pred, "inference_model", None) is None:
        pred._initialize_inference_model()
    pred.preprocess = True
    raw = list(pred._run_inference_on_batch(*prepared_batch(crops_bgr, pred)))
    peaks, vals = [], []
    for ex in raw:
        pk = np.asarray(ex.get("pred_instance_peaks"))
        pv = ex.get("pred_peak_values")
        if pk.ndim == 2:
            pk = pk[None]; pv = None if pv is None else np.asarray(pv)[None]
        for b in range(pk.shape[0]):
            peaks.append(np.asarray(pk[b], np.float32))
            vals.append(None if pv is None else np.asarray(pv[b], np.float32))
    return peaks, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--detection", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--frames", default="0")
    ap.add_argument("--max-dets", type=int, default=8)
    ap.add_argument("--out", default="/tmp/pose_exp/result.npz")
    args = ap.parse_args()

    det = np.load(args.detection, allow_pickle=True)  # trusted local npz
    ids = det["detection_ids"]; corners = det["corners"]; cents = det["centroids"]
    angles = det["angles"]; sizes = det["sizes"]; shapes = det["shapes"]
    # recover w,h from sizes(=major*minor) and aspect(shapes[:,1]=major/minor)
    asp = shapes[:, 1]
    major = np.sqrt(sizes * asp); minor = np.sqrt(sizes / asp)
    by_id = {int(ids[i]): i for i in range(len(ids))}

    pc = load_preprocess_config(args.model_dir)
    pred = build_predictor(args.model_dir, args.device, args.batch, pc)

    cap = cv2.VideoCapture(args.video)
    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    variant_names = ["native", "native_180", "xywhr", "xywhr_180", "xywhr_flipv"]
    rec = {v: {"crops": [], "kpts": [], "vals": [], "detids": []} for v in variant_names}

    for fno in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = sorted([int(d) for d in ids if int(d) // 10000 == fno])[: args.max_dets]
        per_variant = {v: [] for v in variant_names}
        order = []
        for did in dets:
            i = by_id[did]
            variants = build_variants(frame, corners[i].reshape(4, 2),
                                      float(cents[i, 0]), float(cents[i, 1]),
                                      float(major[i]), float(minor[i]), float(angles[i]))
            for v in variant_names:
                per_variant[v].append(variants[v])
            order.append(did)
        for v in variant_names:
            if not per_variant[v]:
                continue
            peaks, vals = predict(pred, per_variant[v])
            for n, did in enumerate(order):
                rec[v]["crops"].append(per_variant[v][n])
                rec[v]["kpts"].append(peaks[n] if n < len(peaks) else np.zeros((0, 2), np.float32))
                rec[v]["vals"].append(vals[n] if n < len(vals) and vals[n] is not None else np.zeros(0, np.float32))
                rec[v]["detids"].append(did)
        print(f"frame {fno}: {len(dets)} dets x {len(variant_names)} variants done")
    cap.release()

    out = {}
    for v in variant_names:
        out[f"{v}__crops"] = np.array(rec[v]["crops"], dtype=object)
        out[f"{v}__kpts"] = np.array(rec[v]["kpts"], dtype=object)
        out[f"{v}__vals"] = np.array(rec[v]["vals"], dtype=object)
        out[f"{v}__detids"] = np.array(rec[v]["detids"], dtype=np.int64)
        # quick confidence summary
        allv = np.concatenate([np.asarray(x).ravel() for x in rec[v]["vals"] if np.asarray(x).size]) \
            if any(np.asarray(x).size for x in rec[v]["vals"]) else np.zeros(0)
        print(f"  {v}: mean_conf={allv.mean():.3f}" if allv.size else f"  {v}: no vals")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, variant_names=np.array(variant_names), **out)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
