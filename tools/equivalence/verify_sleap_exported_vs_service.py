#!/usr/bin/env python
"""Verify SleapExportedBackend (direct ONNX/TensorRT) against SleapServiceBackend
(native, via the conda-env HTTP/shared-memory service) on real crops.

Both backends are constructed through the production selector
`create_pose_backend_from_config` (same code path the benchmark dialog and
tracking pipeline use), fed the exact same native crops extracted from real
detections, and their keypoints are compared pixel-for-pixel.

Run in the HYDRA env that has onnxruntime-gpu / tensorrt installed (it also
spawns the SLEAP service as a subprocess, so the `sleap` conda env must exist):

  PYTHONPATH=src python tools/equivalence/verify_sleap_exported_vs_service.py \
    --video tools/equivalence/fixtures/clips/ant_pose_headtail.mp4 \
    --detection-cache /path/to/..._detection_cache_....npz \
    --model-dir "$HOME/.local/share/hydra-suite/models/pose/SLEAP/20260214-224154_unet_ant_single_instance" \
    --skeleton tools/equivalence/fixtures/ooceraea_biroi.json \
    --runtime-flavor tensorrt --device cuda \
    --frames 0,50,100,150,200 --max-dets 8 \
    --out /tmp/sleap_export_verify.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.core.canonicalization.crop import (
    compute_alignment_affine,
    compute_native_crop_dimensions,
)
from hydra_suite.core.identity.pose.api import create_pose_backend_from_config
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig

AR, MARGIN = 2.0, 1.3
PAD = max(0.0, MARGIN - 1.0)


def native_crop(frame_bgr, corners):
    cw, ch = compute_native_crop_dimensions(corners, AR, PAD)
    M, _ = compute_alignment_affine(corners, cw, ch, PAD)
    return cv2.warpAffine(
        frame_bgr, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def collect_crops(video_path, cache_path, frames, max_dets):
    det = np.load(cache_path, allow_pickle=True)  # trusted local detection cache
    cap = cv2.VideoCapture(video_path)
    crops, labels = [], []
    for fno in frames:
        obb_key = f"frame_{fno:06d}_obb"
        id_key = f"frame_{fno:06d}_detection_ids"
        if obb_key not in det or id_key not in det:
            print(f"  frame {fno}: no cached detections, skipping")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {fno}: could not read frame, skipping")
            continue
        obb = det[obb_key]
        ids = det[id_key]
        n = min(max_dets, len(ids))
        for i in range(n):
            crops.append(native_crop(frame, obb[i].reshape(4, 2)))
            labels.append((int(fno), int(ids[i])))
    cap.release()
    return crops, labels


def build_backend(runtime_flavor, args, out_root, keypoint_names):
    cfg = PoseRuntimeConfig(
        backend_family="sleap",
        runtime_flavor=runtime_flavor,
        device=args.device,
        model_path=args.model_dir,
        out_root=out_root,
        min_valid_conf=args.min_valid_conf,
        sleap_env=args.sleap_env,
        sleap_device=args.device,
        sleap_batch=args.batch_size,
        sleap_max_instances=1,
        keypoint_names=keypoint_names,
    )
    return create_pose_backend_from_config(cfg)


def keypoints_array(result, n_kpts):
    kp = getattr(result, "keypoints", None)
    if kp is None:
        return np.zeros((n_kpts, 3), dtype=np.float32)
    arr = np.asarray(kp, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((n_kpts, 3), dtype=np.float32)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--detection-cache", required=True)
    ap.add_argument("--model-dir", required=True, help="Native SLEAP model directory")
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--runtime-flavor", required=True, choices=["onnx", "tensorrt"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sleap-env", default="sleap")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-valid-conf", type=float, default=0.2)
    ap.add_argument("--frames", default="0,50,100,150,200")
    ap.add_argument("--max-dets", type=int, default=8)
    ap.add_argument(
        "--conf-tol",
        type=float,
        default=0.2,
        help="Valid-keypoint threshold used for comparison, independent of backend min_valid_conf",
    )
    ap.add_argument(
        "--perf-reps",
        type=int,
        default=5,
        help="Repeat predict_batch this many times (post-warmup) for stable throughput numbers",
    )
    ap.add_argument("--out", default="/tmp/sleap_export_verify.json")
    args = ap.parse_args()

    keypoint_names = json.loads(Path(args.skeleton).read_text()).get(
        "keypoint_names", []
    )
    n_kpts = len(keypoint_names)
    frames = [int(x) for x in args.frames.split(",") if x.strip() != ""]

    print(
        f"Extracting native crops for frames {frames} (max {args.max_dets} dets/frame)..."
    )
    crops, labels = collect_crops(
        args.video, args.detection_cache, frames, args.max_dets
    )
    print(f"Collected {len(crops)} crops.")
    if not crops:
        raise SystemExit("No crops collected; check --frames / --detection-cache.")

    out_root = str(Path(args.out).parent / "sleap_verify_out")
    Path(out_root).mkdir(parents=True, exist_ok=True)

    def _timed_predict(backend, label):
        backend.warmup()
        # Discard the first (cold) call, then repeat for stable throughput.
        results = backend.predict_batch(crops)
        durations = []
        for _ in range(max(0, args.perf_reps)):
            t0 = time.perf_counter()
            results = backend.predict_batch(crops)
            durations.append(time.perf_counter() - t0)
        if durations:
            mean_s = sum(durations) / len(durations)
            print(
                f"  {label}: {len(durations)} reps, mean {mean_s * 1000:.1f} ms/batch "
                f"({mean_s / len(crops) * 1000:.2f} ms/crop, {len(crops) / mean_s:.1f} crops/s)"
            )
        return results, durations

    print(f"\n[1/2] Running SleapServiceBackend (native, device={args.device})...")
    service_backend = build_backend("native", args, out_root, keypoint_names)
    service_results, service_durations = _timed_predict(
        service_backend, "SleapServiceBackend (gpu tier, native)"
    )
    close = getattr(service_backend, "close", None)
    if callable(close):
        close()

    print(
        f"\n[2/2] Running SleapExportedBackend (runtime_flavor={args.runtime_flavor}, device={args.device})..."
    )
    exported_backend = build_backend(
        args.runtime_flavor, args, out_root, keypoint_names
    )
    backend_type = type(exported_backend).__name__
    print(f"  -> factory returned: {backend_type}")
    exported_results, exported_durations = _timed_predict(
        exported_backend, f"SleapExportedBackend (gpu_fast tier, {args.runtime_flavor})"
    )
    close = getattr(exported_backend, "close", None)
    if callable(close):
        close()

    svc_kp = np.stack([keypoints_array(r, n_kpts) for r in service_results])
    exp_kp = np.stack([keypoints_array(r, n_kpts) for r in exported_results])

    both_valid = (svc_kp[:, :, 2] > args.conf_tol) & (exp_kp[:, :, 2] > args.conf_tol)
    dist = np.linalg.norm(svc_kp[:, :, :2] - exp_kp[:, :, :2], axis=-1)
    dist_valid = dist[both_valid]

    svc_valid_count = int((svc_kp[:, :, 2] > args.conf_tol).sum())
    exp_valid_count = int((exp_kp[:, :, 2] > args.conf_tol).sum())
    both_valid_count = int(both_valid.sum())
    total_kpts = svc_kp.shape[0] * svc_kp.shape[1]

    summary = {
        "backend_factory_returned": backend_type,
        "runtime_flavor": args.runtime_flavor,
        "n_crops": len(crops),
        "n_keypoints_per_crop": n_kpts,
        "total_kpt_slots": int(total_kpts),
        "service_valid_kpts": svc_valid_count,
        "exported_valid_kpts": exp_valid_count,
        "both_valid_kpts": both_valid_count,
        "pos_dist_px": {
            "mean": float(dist_valid.mean()) if dist_valid.size else None,
            "median": float(np.median(dist_valid)) if dist_valid.size else None,
            "p95": float(np.percentile(dist_valid, 95)) if dist_valid.size else None,
            "max": float(dist_valid.max()) if dist_valid.size else None,
        },
        "perf": {
            "service_ms_per_batch": (
                sum(service_durations) / len(service_durations) * 1000
                if service_durations
                else None
            ),
            "exported_ms_per_batch": (
                sum(exported_durations) / len(exported_durations) * 1000
                if exported_durations
                else None
            ),
            "service_crops_per_s": (
                len(crops) * len(service_durations) / sum(service_durations)
                if service_durations
                else None
            ),
            "exported_crops_per_s": (
                len(crops) * len(exported_durations) / sum(exported_durations)
                if exported_durations
                else None
            ),
        },
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "summary": summary,
                "labels": labels,
                "service_keypoints": svc_kp.tolist(),
                "exported_keypoints": exp_kp.tolist(),
            },
            f,
        )
    print(f"\nSaved full detail -> {args.out}")


if __name__ == "__main__":
    main()
