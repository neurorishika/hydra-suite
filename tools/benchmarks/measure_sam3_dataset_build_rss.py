"""Measure SAM3 COCO preparation RSS as source frame count grows."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import tempfile
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora.dataset_build import build_sam3_coco_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")

    with tempfile.TemporaryDirectory(prefix="hydra-sam3-rss-") as temporary:
        root = Path(temporary)
        images = root / "source" / "images"
        labels = root / "source" / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        (root / "source" / "classes.txt").write_text("ant\n", encoding="utf-8")
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        for index in range(args.frames):
            cv2.imwrite(str(images / f"frame-{index:08d}.jpg"), image)
            (labels / f"frame-{index:08d}.txt").write_text(
                "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n",
                encoding="utf-8",
            )
        stats = build_sam3_coco_dataset(
            root / "source",
            root / "output",
            Sam3LoraParams(
                prompt="ant",
                geometry_mode="custom",
                slice_width=8,
                slice_height=8,
                tile_overlap=0.0,
                keep_empty_tiles=False,
            ),
        )
        maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() != "Darwin":
            maximum_rss *= 1024
        print(
            json.dumps(
                {
                    "frames": args.frames,
                    "max_rss_bytes": maximum_rss,
                    "train_images": stats["train_images"],
                    "val_images": stats["val_images"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
