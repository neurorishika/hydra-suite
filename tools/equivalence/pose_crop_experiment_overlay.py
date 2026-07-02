#!/usr/bin/env python
"""Overlay + score the pose_crop_experiment output (offline, hydra env).

Reads result.npz from pose_crop_experiment.py and renders a grid: rows =
detections, cols = crop-orientation variants, each cell = crop with predicted
keypoints overlaid (red=anterior, blue=posterior, green=other). Column titles
carry mean keypoint confidence. The variant whose keypoints form a coherent ant
skeleton (and highest confidence) is the correct crop preparation.
"""

import argparse
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ANT = [0, 1, 2, 3, 4]
POST = [6, 7]


def color(j):
    if j in ANT:
        return (1.0, 0.0, 0.0)
    if j in POST:
        return (0.0, 0.4, 1.0)
    return (0.0, 0.9, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument(
        "--outdir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "pose_exp")
    )
    ap.add_argument("--min-conf", type=float, default=0.2)
    ap.add_argument("--max-rows", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)  # trusted local experiment output
    variants = [str(v) for v in d["variant_names"]]
    detids = d[f"{variants[0]}__detids"]
    n = min(len(detids), args.max_rows)
    ncol = len(variants)

    # per-variant mean confidence
    print("variant mean-confidence ranking:")
    scores = []
    for v in variants:
        vals = [np.asarray(x).ravel() for x in d[f"{v}__vals"] if np.asarray(x).size]
        mc = np.concatenate(vals).mean() if vals else 0.0
        scores.append((v, float(mc)))
    for v, mc in sorted(scores, key=lambda t: -t[1]):
        print(f"  {v:14s} mean_conf={mc:.3f}")

    fig, axes = plt.subplots(n, ncol, figsize=(2.7 * ncol, 2.7 * n), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ci, v in enumerate(variants):
        crops = d[f"{v}__crops"]
        kpts = d[f"{v}__kpts"]
        vals = d[f"{v}__vals"]
        mc = dict(scores)[v]
        for ri in range(n):
            ax = axes[ri][ci]
            crop = np.asarray(crops[ri])
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            kp = np.asarray(kpts[ri])
            vv = np.asarray(vals[ri])
            for j in range(len(kp)):
                c = vv[j] if j < len(vv) else 1.0
                if c >= args.min_conf:
                    ax.plot(kp[j, 0], kp[j, 1], "x", color=color(j), ms=7, mew=1.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("on")
            if ri == 0:
                ax.set_title(f"{v}\nmean_conf={mc:.2f}", fontsize=9)
    out = os.path.join(args.outdir, "variant_comparison.png")
    fig.suptitle(
        "crop-orientation variants: SLEAP keypoints (red=anterior blue=posterior)",
        y=1.001,
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
