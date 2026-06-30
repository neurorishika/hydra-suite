#!/usr/bin/env python
"""Pose-extraction overlay diagnostic (offline, no SLEAP re-run).

Uses cached inference outputs to visually validate pose-keypoint quality and the
crop->image coordinate transform:

  * crop panels   : the exact canonical crop SLEAP saw, with its predicted
                    keypoints overlaid in CROP coordinates -> is the prediction
                    itself good?
  * frame panel   : the same keypoints mapped to image coordinates via the crop
                    affine inverse, overlaid on the full frame, alongside the
                    LEGACY image-space keypoints -> is the transform/placement
                    correct, and does it agree with legacy?

Inputs are the npz caches written by a tracking run plus the source video.
Run in the hydra env (cv2 + matplotlib); no SLEAP / GPU needed.
"""
import argparse
import os
import sys

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from hydra_suite.core.canonicalization.crop import (  # noqa: E402
    compute_alignment_affine,
    compute_native_crop_dimensions,
    invert_keypoints,
)

AR, MARGIN = 2.0, 1.3
PAD = max(0.0, MARGIN - 1.0)
ANT = [0, 1, 2, 3, 4]
POST = [6, 7]


def detid_frame(did):
    return int(did) // 10000


def load_legacy_kpts(path):
    d = np.load(path, allow_pickle=True)  # trusted local npz
    out = {}
    for k in d.keys():
        if not k.endswith("_pose_keypoints"):
            continue
        f = k.split("_")[1]
        ids = d[f"frame_{f}_detection_ids"]
        arr = np.atleast_1d(d[k])
        for i, did in enumerate(ids):
            if i < len(arr) and arr[i] is not None:
                out[int(did)] = np.asarray(arr[i], dtype=np.float32)
    return out


def kpt_color(j):
    if j in ANT:
        return (1.0, 0.0, 0.0)  # anterior red
    if j in POST:
        return (0.0, 0.4, 1.0)  # posterior blue
    return (0.0, 0.9, 0.0)  # other green


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--detection", required=True, help="detection.npz")
    ap.add_argument("--pose", required=True, help="pose.npz")
    ap.add_argument("--legacy-pose", default=None, help="legacy pose_cache npz (optional)")
    ap.add_argument("--frames", default="0,100,250")
    ap.add_argument("--max-dets", type=int, default=8)
    ap.add_argument("--min-conf", type=float, default=0.2)
    ap.add_argument("--outdir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "pose_diag"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    det = np.load(args.detection, allow_pickle=True)  # trusted local npz
    pose = np.load(args.pose, allow_pickle=True)
    corners_by_id = {int(det["detection_ids"][i]): det["corners"][i].reshape(4, 2)
                     for i in range(len(det["detection_ids"]))}
    # pose rows -> det_id
    pk = pose["keypoints"]
    pf = pose["frame_indices"]
    pdi = pose["det_indices"]
    kpts_by_id = {}
    for r in range(len(pk)):
        did = int(pf[r]) * 10000 + int(pdi[r])
        kpts_by_id[did] = np.asarray(pk[r], dtype=np.float32)
    legacy = load_legacy_kpts(args.legacy_pose) if args.legacy_pose else {}

    cap = cv2.VideoCapture(args.video)
    frames = [int(x) for x in args.frames.split(",") if x.strip() != ""]

    for fno in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"frame {fno}: read failed")
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets = sorted([did for did in corners_by_id
                       if detid_frame(did) == fno and did in kpts_by_id])[: args.max_dets]
        if not dets:
            print(f"frame {fno}: no detections with pose")
            continue

        # ---- Frame-level overlay: mapped-new (x) vs legacy (o) ----
        figF, axF = plt.subplots(1, 1, figsize=(11, 11))
        axF.imshow(frame_rgb)
        axF.set_title(f"frame {fno}: new->image (x, transform) vs legacy (o)  red=anterior blue=posterior")
        for did in dets:
            corners = corners_by_id[did]
            cw, ch = compute_native_crop_dimensions(corners, AR, PAD)
            M, _ = compute_alignment_affine(corners, cw, ch, PAD)
            M_inv = cv2.invertAffineTransform(M)
            k = kpts_by_id[did]
            img_xy = invert_keypoints(k[:, :2].astype(np.float32).copy(), M_inv)
            axF.plot(*zip(*[corners[i] for i in [0, 1, 2, 3, 0]]), "-", color="yellow", lw=0.6)
            for j in range(len(k)):
                if k[j, 2] >= args.min_conf:
                    axF.plot(img_xy[j, 0], img_xy[j, 1], "x", color=kpt_color(j), ms=6, mew=1.5)
            lk = legacy.get(did)
            if lk is not None:
                for j in range(len(lk)):
                    if lk[j, 2] >= args.min_conf:
                        axF.plot(lk[j, 0], lk[j, 1], "o", color=kpt_color(j), ms=7, mfc="none", mew=1.2)
        axF.axis("off")
        fpath = os.path.join(args.outdir, f"frame_{fno:04d}_overlay.png")
        figF.savefig(fpath, dpi=110, bbox_inches="tight")
        plt.close(figF)

        # ---- Crop panels: the crop SLEAP saw + its keypoints in CROP coords ----
        ncol = min(4, len(dets))
        nrow = int(np.ceil(len(dets) / ncol))
        figC, axesC = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False)
        for ax in axesC.ravel():
            ax.axis("off")
        for idx, did in enumerate(dets):
            corners = corners_by_id[did]
            cw, ch = compute_native_crop_dimensions(corners, AR, PAD)
            M, _ = compute_alignment_affine(corners, cw, ch, PAD)
            crop = cv2.warpAffine(frame_bgr, M, (cw, ch), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ax = axesC.ravel()[idx]
            ax.imshow(crop)
            ax.set_title(f"det {did} ({cw}x{ch})", fontsize=8)
            k = kpts_by_id[did]
            for j in range(len(k)):
                if k[j, 2] >= args.min_conf:
                    ax.plot(k[j, 0], k[j, 1], "x", color=kpt_color(j), ms=7, mew=1.6)
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        figC.suptitle(f"frame {fno}: canonical crops + SLEAP keypoints (CROP coords)  red=ant blue=post")
        cpath = os.path.join(args.outdir, f"frame_{fno:04d}_crops.png")
        figC.savefig(cpath, dpi=110, bbox_inches="tight")
        plt.close(figC)

        # ---- Zoomed per-ant frame insets: new->image (x) vs legacy (o) ----
        figZ, axesZ = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow), squeeze=False)
        for ax in axesZ.ravel():
            ax.axis("off")
        for idx, did in enumerate(dets):
            corners = corners_by_id[did]
            cw, ch = compute_native_crop_dimensions(corners, AR, PAD)
            M, _ = compute_alignment_affine(corners, cw, ch, PAD)
            M_inv = cv2.invertAffineTransform(M)
            k = kpts_by_id[did]
            img_xy = invert_keypoints(k[:, :2].astype(np.float32).copy(), M_inv)
            cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
            half = max(np.ptp(corners[:, 0]), np.ptp(corners[:, 1])) * 0.9 + 10
            x0, x1 = int(cx - half), int(cx + half)
            y0, y1 = int(cy - half), int(cy + half)
            x0, y0 = max(0, x0), max(0, y0)
            sub = frame_rgb[y0:y1, x0:x1]
            ax = axesZ.ravel()[idx]
            ax.imshow(sub)
            ax.set_title(f"det {did}", fontsize=8)
            for j in range(len(k)):
                if k[j, 2] >= args.min_conf:
                    ax.plot(img_xy[j, 0] - x0, img_xy[j, 1] - y0, "x", color=kpt_color(j), ms=8, mew=1.8)
            lk = legacy.get(did)
            if lk is not None:
                for j in range(len(lk)):
                    if lk[j, 2] >= args.min_conf:
                        ax.plot(lk[j, 0] - x0, lk[j, 1] - y0, "o", color=kpt_color(j),
                                ms=9, mfc="none", mew=1.4)
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        figZ.suptitle(f"frame {fno}: ZOOM  new->image (x) vs legacy (o)  red=ant blue=post")
        zpath = os.path.join(args.outdir, f"frame_{fno:04d}_zoom.png")
        figZ.savefig(zpath, dpi=120, bbox_inches="tight")
        plt.close(figZ)
        print(f"frame {fno}: {len(dets)} dets -> {fpath} ; {cpath} ; {zpath}")

    cap.release()
    print(f"\nDone. Open PNGs under: {args.outdir}")


if __name__ == "__main__":
    main()
