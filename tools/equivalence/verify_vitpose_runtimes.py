"""Compare ViTPose runtimes against the native torch reference on real crops.

Native torch is the oracle. Prints max per-keypoint pixel deviation for each
requested flavor. fp32 everywhere, so thresholds are tight:
torch/onnx sub-pixel, tensorrt/coreml <= ~1px.

Usage:
    python tools/equivalence/verify_vitpose_runtimes.py CHECKPOINT CROP_DIR \
        --flavors coreml tensorrt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from hydra_suite.core.identity.pose.backends.vitpose import (
    ViTPoseBackend,
    auto_export_vitpose_model,
)
from hydra_suite.core.identity.pose.types import PoseRuntimeConfig


def _native(checkpoint: str, crops: List[np.ndarray]) -> np.ndarray:
    be = ViTPoseBackend(checkpoint, device="cpu", runtime_flavor="native")
    return np.stack([r.keypoints for r in be.predict_batch(crops)])


def compare_runtimes(
    checkpoint: str, crops: List[np.ndarray], flavors: List[str]
) -> Dict[str, float]:
    ref = _native(checkpoint, crops)
    out: Dict[str, float] = {}
    for flavor in flavors:
        device = {"coreml": "mps", "tensorrt": "cuda"}.get(flavor, "cpu")
        cfg = PoseRuntimeConfig(
            backend_family="vitpose",
            model_path=checkpoint,
            runtime_flavor=flavor,
            device=device,
        )
        art = auto_export_vitpose_model(cfg, flavor)
        be = ViTPoseBackend(
            checkpoint, device=device, runtime_flavor=flavor, exported_model_path=art
        )
        got = np.stack([r.keypoints for r in be.predict_batch(crops)])
        out[flavor] = float(np.max(np.abs(got[..., :2] - ref[..., :2])))
    return out


def _load_crops(crop_dir: Path) -> List[np.ndarray]:
    import cv2

    return [cv2.imread(str(p)) for p in sorted(crop_dir.glob("*.png"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("crop_dir")
    ap.add_argument("--flavors", nargs="+", default=["coreml"])
    args = ap.parse_args()
    crops = _load_crops(Path(args.crop_dir))
    devs = compare_runtimes(args.checkpoint, crops, args.flavors)
    for flavor, dev in devs.items():
        print(f"{flavor}: max pixel deviation vs native = {dev:.4f}")


if __name__ == "__main__":
    main()
