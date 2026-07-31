#!/usr/bin/env python
"""Route identical native crops through the production SleapServiceBackend.

Isolates whether the bad pipeline pose keypoints come from the SERVICE path
(shared-memory + _run_inference_direct) or from the batch-padding the pipeline
applies upstream. We extract per-detection NATIVE crops (no uniform padding) and
call the real SleapServiceBackend.predict_batch on them, then save keypoints.

Compare offline to:
  * the direct experiment (result.npz, native crops, GOOD), and
  * the pipeline cache (pose.npz, padded batch, BAD).

If THIS (native crops via service) == GOOD  -> the bug is the padded batch.
If THIS == BAD                              -> the bug is the service path.

Run in the HYDRA env (it spawns the SLEAP service):
  PYTHONPATH=src python tools/equivalence/pose_service_vs_direct.py \
    --video tools/equivalence/fixtures/clips/ant_pose_headtail.mp4 \
    --detection /tmp/.../detection.npz \
    --model-dir "/abs/path/to/pose/SLEAP/<unet_single_instance>" \
    --skeleton tools/equivalence/fixtures/ooceraea_biroi.json \
    --device mps --frames 0 --max-dets 8 --out /tmp/pose_exp/service_native.npz
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.core.canonicalization.crop import (
    compute_alignment_affine,
    compute_native_crop_dimensions,
)
from hydra_suite.core.identity.pose.backends.sleap import SleapServiceBackend

AR, MARGIN = 2.0, 1.3
PAD = max(0.0, MARGIN - 1.0)


def native_crop(frame_bgr, corners):
    cw, ch = compute_native_crop_dimensions(corners, AR, PAD)
    M, _ = compute_alignment_affine(corners, cw, ch, PAD)
    return cv2.warpAffine(
        frame_bgr, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--detection", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--frames", default="0")
    ap.add_argument("--max-dets", type=int, default=8)
    ap.add_argument("--out", default="/tmp/pose_exp/service_native.npz")
    args = ap.parse_args()

    names = json.loads(Path(args.skeleton).read_text()).get("keypoint_names", [])
    det = np.load(args.detection, allow_pickle=True)  # trusted local npz
    ids = det["detection_ids"]
    corners = det["corners"]
    by_id = {int(ids[i]): i for i in range(len(ids))}

    out_root = tempfile.mkdtemp(prefix="pose_svc_test_")
    backend = SleapServiceBackend(
        model_dir=args.model_dir,
        out_root=out_root,
        keypoint_names=names,
        min_valid_conf=0.2,
        sleap_env="sleap",
        sleap_device=args.device,
        sleap_batch=25,
        sleap_max_instances=1,
        runtime_flavor="native",
    )
    backend.warmup()

    cap = cv2.VideoCapture(args.video)
    rec_kpts, rec_ids = [], []
    for fno in [int(x) for x in args.frames.split(",") if x.strip()]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = sorted([int(d) for d in ids if int(d) // 10000 == fno])[: args.max_dets]
        crops = [native_crop(frame, corners[by_id[d]].reshape(4, 2)) for d in dets]
        results = backend.predict_batch(crops)  # production service+shm path
        for n, d in enumerate(dets):
            r = results[n]
            kp = getattr(r, "keypoints", None)
            rec_kpts.append(
                np.asarray(kp, np.float32)
                if kp is not None
                else np.zeros((len(names), 3), np.float32)
            )
            rec_ids.append(d)
        print(f"frame {fno}: {len(dets)} native crops -> service")
        # quick per-det validity
        for n, d in enumerate(dets):
            k = rec_kpts[len(rec_kpts) - len(dets) + n]
            nz = (
                int((np.asarray(k)[:, 2] > 0.2).sum()) if np.asarray(k).ndim == 2 else 0
            )
            print(f"  det {d}: {nz} kpts>0.2")
    cap.release()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        kpts=np.array(rec_kpts, dtype=object),
        detids=np.array(rec_ids, np.int64),
    )
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
