"""Phase-1 numeric equivalence probe (native-GPU exact wins).

Phase 1 as shipped only wraps compute in ``inference_mode`` (classifier + crop
paths) and adds pinned/``non_blocking`` H2D for classifier crops — all
determinism-preserving. (An earlier ``channels_last`` attempt was reverted after
CUDA verification showed it was a no-op on the real classifier and crashed OBB
inference.) This script runs the affected production code paths on deterministic,
seeded inputs and dumps the raw outputs to JSON so the SAME script can be run in
two source trees (pre-Phase-1 base vs post-Phase-1 HEAD) on the SAME box and the
outputs diffed.

Equivalence criterion: classifier logits and OBB detection geometry at HEAD must
match base. The shipped changes are transport/graph-only (no compute reorder), so
outputs are expected bit-exact; the harness still tolerates the spec's ~0.006 px
/ 1e-3 rel-tol device-invariance envelope.

Usage (run in each tree, then diff the two JSON files):
    PYTHONPATH=src python tools/equivalence/phase1_equivalence.py \
        --classifier /path/to/efficientnet_b0.pth \
        --obb /path/to/yolo-obb.pt \
        --runtime cuda --out /tmp/eq_head.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Prefer an explicit PYTHONPATH (lets one script file run against two different
# source trees for base-vs-HEAD comparison); fall back to this tree's src.
try:
    import hydra_suite  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def _seeded_crops(n: int, h: int, w: int, seed: int = 1234) -> list:
    rng = np.random.RandomState(seed)
    return [rng.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _seeded_frame(imgsz: int, seed: int = 4321) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)


def classifier_logits(model_path: str, runtime: str, batch: int) -> dict:
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    probe = ClassifierBackend(model_path, compute_runtime="cpu")
    probe._ensure_loaded()
    h, w = probe._metadata.input_size
    probe.close()

    crops = _seeded_crops(batch, h, w)
    be = ClassifierBackend(model_path, compute_runtime=runtime)
    out = be.predict_batch(crops)  # list[N][K] prob vectors
    be.close()
    flat = np.concatenate([np.concatenate(row) for row in out]).astype(np.float64)
    return {
        "n": len(out),
        "sum": float(flat.sum()),
        "mean": float(flat.mean()),
        "first16": [round(float(v), 8) for v in flat[:16]],
    }


def obb_detections(model_path: str, runtime: str, imgsz: int) -> dict:
    from hydra_suite.core.inference.runtime_artifacts import load_obb_executor

    frame = _seeded_frame(imgsz)
    exe = load_obb_executor(
        model_path, compute_runtime=runtime, auto_export=False, max_det=100
    )
    results = exe.predict([frame], conf=0.05, iou=0.5, imgsz=imgsz, verbose=False)
    r = results[0]
    try:
        xywhr = r.obb.xywhr.detach().cpu().numpy().astype(np.float64)
    except Exception:
        xywhr = np.zeros((0, 5))
    xywhr = xywhr[np.lexsort((xywhr[:, 1], xywhr[:, 0]))] if len(xywhr) else xywhr
    return {
        "n_det": int(len(xywhr)),
        "sum": float(xywhr.sum()) if len(xywhr) else 0.0,
        "first_rows": xywhr[:5].round(6).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default=None)
    ap.add_argument("--obb", default=None)
    ap.add_argument("--runtime", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result: dict = {"runtime": args.runtime}
    if args.classifier:
        result["classifier"] = classifier_logits(
            args.classifier, args.runtime, args.batch
        )
    if args.obb:
        result["obb"] = obb_detections(args.obb, args.runtime, args.imgsz)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
