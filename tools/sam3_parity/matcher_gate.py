#!/usr/bin/env python
"""Before/after gate for the `calibration.match_one_to_one` containment fix.

The whole point of this measurement is that ONLY THE SCORING CHANGED, so
predictions must be bit-identical between the two arms. That is enforced
structurally, not by hoping inference is deterministic: SAM3 runs ONCE per
checkpoint, the raw candidates are cached (in memory, and to a pickle so the
scoring can be re-run later on any machine with no GPU), and both matchers
then score that one cache in the same process.

Arm A ("shipped") is a FROZEN, verbatim copy of the defective matcher as it
stood at `d9d9ac31` -- vendored here rather than imported, because the point
of a before/after is that the "before" cannot drift when `src/` is fixed.
Arm B ("fixed") imports the live matcher from `src/`.

`tools/sam3_parity/baseline.json` is a PRE-REGISTERED artifact and is never
read for anything but its run parameters, and never written.

Usage (on a GPU box, in an env with ultralytics -- `hydra-cuda`, NOT the
`hydra-sam3` training sidecar; see README.md):

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=$PWD/src python \\
      tools/sam3_parity/matcher_gate.py --from-baseline tools/sam3_parity/baseline.json \\
      --out tools/sam3_parity/matcher_gate_results.json

Or score a previously written cache with no GPU at all:

    python tools/sam3_parity/matcher_gate.py --score-only --cache /tmp/gate_cache.pkl \\
      --from-baseline tools/sam3_parity/baseline.json --out results.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_models import (  # noqa: E402
    _calibrate_one_model,
    _default_labeler_factory,
    _load_frames_from_coco,
)

# Confidences chosen to line up with the table in
# docs/superpowers/specs/2026-09-05-sam3-spike-parity-measurement-findings.md
DEFAULT_CONFIDENCES = (0.2, 0.5)


# ---------------------------------------------------------------------------
# ARM A -- frozen copy of the defective matcher at d9d9ac31. Do not "improve"
# anything below this line; it is a fossil, and its bugs are the measurement.
# ---------------------------------------------------------------------------


def _frozen_centroid(poly: np.ndarray) -> np.ndarray:
    return np.asarray(poly, dtype=np.float64).reshape(-1, 2).mean(axis=0)


def _frozen_contains(poly: np.ndarray, point: np.ndarray) -> bool:
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def frozen_match_one_to_one(
    pred_polys: Sequence[np.ndarray],
    label_polys: Sequence[np.ndarray],
    *,
    area_band=None,
    min_quality: float | None = None,
) -> list[tuple[int, int]]:
    from hydra_suite.core.inference.semantic.shape_prior import (
        MIN_MATCH_QUALITY,
        in_band,
        match_quality,
    )

    if min_quality is None:
        min_quality = MIN_MATCH_QUALITY
    pred_c = [_frozen_centroid(p) for p in pred_polys]
    label_c = [_frozen_centroid(g) for g in label_polys]
    admissible = [i for i, p in enumerate(pred_polys) if in_band(p, area_band)]
    pairs: list[tuple[float, int, int]] = []
    for pi in admissible:
        pc = pred_c[pi]
        for gi, gc in enumerate(label_c):
            if not (
                _frozen_contains(label_polys[gi], pc)
                or _frozen_contains(pred_polys[pi], gc)
            ):
                continue
            quality = match_quality(pred_polys[pi], label_polys[gi])
            if quality < min_quality:
                continue
            pairs.append((-quality, pi, gi))
    pairs.sort(key=lambda t: (t[0], float(np.hypot(*(pred_c[t[1]] - label_c[t[2]])))))
    used_p: set[int] = set()
    used_g: set[int] = set()
    out: list[tuple[int, int]] = []
    for _neg_q, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        out.append((pi, gi))
    return out


def containment_only_match(
    pred_polys: Sequence[np.ndarray],
    label_polys: Sequence[np.ndarray],
    *,
    area_band=None,
) -> list[tuple[int, int]]:
    """Arm C: the FIXED representative point, containment only, NO IoU route.

    The IoU route was justified by an INDIRECT measurement -- an
    area-centroid-only gate reaching 0.929 where a plain IoU rule reached
    0.962 -- but `representative_point` is strictly stronger than an area
    centroid, so 0.929 is a lower bound on containment-only, not a
    measurement of it. This arm measures it directly, so the project can
    decide whether `ADMISSIBLE_IOU` earns its place as a production knob.

    Implemented by neutralising the ONE `polygon_iou` reference the IoU route
    uses, rather than by copying the matcher: a copy would drift, and
    `match_quality`'s own internal overlap term (imported separately in
    `shape_prior`) must stay live so the ranking and the quality floor are
    unchanged between arms.
    """
    from hydra_suite.core.inference.semantic import calibration

    original = calibration.polygon_iou
    calibration.polygon_iou = lambda _a, _b: 0.0
    try:
        return calibration.match_one_to_one(
            pred_polys, label_polys, area_band=area_band
        )
    finally:
        calibration.polygon_iou = original


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_arm(
    previews: Sequence,
    match_fn,
    *,
    tile_fraction: float | None,
    confidence: float,
    area_band,
    merge_iou: float,
) -> dict:
    """Recall / extras-per-tile / mean match quality for one matcher.

    Every arm consumes the SAME `previews` (one inference pass) and the SAME
    `area_band`; the only thing that varies across arms is `match_fn`.
    """
    from hydra_suite.core.inference.semantic.shape_prior import match_quality
    from hydra_suite.core.inference.semantic.tiling import merge_candidates

    n_labels = n_preds = n_matched = 0
    qualities: list[float] = []
    per_tile: dict[str, list[int]] = {}
    for preview in previews:
        candidates = preview.candidates_by_fraction.get(tile_fraction, ())
        merged = merge_candidates(
            candidates,
            confidence_threshold=confidence,
            iou_threshold=merge_iou,
            area_band=area_band,
        )
        preds = [m.polygon_px for m in merged]
        labels = [g.polygon_px for g in preview.ground_truth]
        pairs = match_fn(preds, labels, area_band=area_band)
        n_labels += len(labels)
        n_preds += len(preds)
        n_matched += len(pairs)
        qualities.extend(match_quality(preds[pi], labels[gi]) for pi, gi in pairs)
        per_tile[str(preview.image_path)] = [
            len(preds) - len(pairs),
            len(labels) - len(pairs),
        ]
    n_tiles = max(len(previews), 1)
    return {
        "confidence": confidence,
        "n_tiles": len(previews),
        "n_labels": n_labels,
        "n_predictions": n_preds,
        "n_matched": n_matched,
        "recall": (n_matched / n_labels) if n_labels else 0.0,
        "extras_per_tile": (n_preds - n_matched) / n_tiles,
        "missed_per_tile": (n_labels - n_matched) / n_tiles,
        "mean_quality": float(np.mean(qualities)) if qualities else 0.0,
        "per_tile_extras_missed": per_tile,
    }


def vertex_mean_outside_rate(previews: Sequence) -> dict:
    """The 15.9 % statistic, recomputed on this run's own labels."""
    from hydra_suite.core.inference.semantic.calibration import (
        _contains,
        representative_point,
    )

    total = outside_mean = outside_fixed = 0
    for preview in previews:
        for g in preview.ground_truth:
            poly = g.polygon_px
            total += 1
            if not _frozen_contains(poly, _frozen_centroid(poly)):
                outside_mean += 1
            if not _contains(poly, representative_point(poly)):
                outside_fixed += 1
    return {
        "n_labels": total,
        "vertex_mean_outside": outside_mean,
        "vertex_mean_outside_fraction": (outside_mean / total) if total else 0.0,
        "representative_point_outside": outside_fixed,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-baseline",
        type=Path,
        help="Read run parameters (checkpoints, frames, prompt, tiling) from a "
        "baseline.json rather than retyping them. READ ONLY -- never written.",
    )
    ap.add_argument("--checkpoint-a", type=Path)
    ap.add_argument("--checkpoint-b", type=Path)
    ap.add_argument("--frames-dir", type=Path)
    ap.add_argument("--coco-json", type=Path)
    ap.add_argument("--prompt")
    ap.add_argument("--reference-body-px", type=float)
    ap.add_argument("--tile-fraction", type=float, default=None)
    ap.add_argument("--seam-margin-px", type=float)
    ap.add_argument("--merge-iou", type=float)
    ap.add_argument("--confidences", type=float, nargs="+", default=None)
    ap.add_argument("--cache", type=Path, default=None, help="Pickle of previews.")
    ap.add_argument("--score-only", action="store_true", help="Require --cache.")
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main(argv: Sequence[str] | None = None, *, labeler_factory=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    factory = labeler_factory or _default_labeler_factory
    cfg: dict = {}
    if args.from_baseline is not None:
        cfg = json.loads(Path(args.from_baseline).read_text())

    def pick(name, key=None, cast=None):
        value = getattr(args, name.replace("-", "_"))
        if value is None and cfg:
            value = cfg.get(key or name.replace("-", "_"))
        return cast(value) if (cast and value is not None) else value

    checkpoints = {
        "a": pick("checkpoint-a", "checkpoint_a", Path),
        "b": pick("checkpoint-b", "checkpoint_b", Path),
    }
    frames_dir = pick("frames-dir", "frames_dir", Path)
    coco_json = pick("coco-json", "coco_json", Path)
    prompt = pick("prompt")
    reference_body_px = pick("reference-body-px", "reference_body_px", float)
    tile_fraction = (
        args.tile_fraction
        if args.tile_fraction is not None
        else cfg.get("tile_fraction")
    )
    seam_margin_px = pick("seam-margin-px", "seam_margin_px", float)
    merge_iou = pick("merge-iou", "merge_iou", float)
    confidences = tuple(args.confidences or DEFAULT_CONFIDENCES)

    import hydra_suite
    from hydra_suite.core.inference.semantic.calibration import match_one_to_one
    from hydra_suite.core.inference.semantic.shape_prior import fit_area_band

    previews_by_model: dict[str, list] = {}
    if args.score_only:
        if args.cache is None:
            raise SystemExit("--score-only requires --cache")
        previews_by_model = pickle.loads(Path(args.cache).read_bytes())
    else:
        frames = _load_frames_from_coco(coco_json, frames_dir)
        if not frames:
            raise SystemExit(f"no frames loaded from {coco_json}")
        for key, checkpoint in checkpoints.items():
            if checkpoint is None:
                continue
            print(f"[gate] inference for checkpoint {key}: {checkpoint}", flush=True)
            _points, previews = _calibrate_one_model(
                factory(checkpoint),
                frames,
                prompt,
                reference_body_px=reference_body_px,
                tile_fraction=tile_fraction,
                seam_margin_px=seam_margin_px,
                merge_iou=merge_iou,
            )
            previews_by_model[key] = previews
        if args.cache is not None:
            Path(args.cache).write_bytes(pickle.dumps(previews_by_model))

    results: dict = {
        "_": "Before/after gate for the calibration matcher containment fix. "
        "NOT a re-run of baseline.json (which is pre-registered and untouched): "
        "same frames, same cached predictions, scoring varied.",
        "hydra_suite_module": hydra_suite.__file__,
        "coco_json": str(coco_json),
        "frames_dir": str(frames_dir),
        "checkpoints": {k: str(v) for k, v in checkpoints.items() if v},
        "prompt": prompt,
        "reference_body_px": reference_body_px,
        "tile_fraction": tile_fraction,
        "seam_margin_px": seam_margin_px,
        "merge_iou": merge_iou,
        "arms": {},
        "label_geometry": {},
    }
    for key, previews in previews_by_model.items():
        area_band = fit_area_band(
            [g.polygon_px for p in previews for g in p.ground_truth]
        )
        results["label_geometry"][key] = vertex_mean_outside_rate(previews)
        results["arms"][key] = {}
        for arm_name, fn in (
            ("shipped_vertex_mean", frozen_match_one_to_one),
            ("fixed_representative_point", match_one_to_one),
            ("fixed_containment_only_no_iou_route", containment_only_match),
        ):
            results["arms"][key][arm_name] = [
                score_arm(
                    previews,
                    fn,
                    tile_fraction=tile_fraction,
                    confidence=c,
                    area_band=area_band,
                    merge_iou=merge_iou,
                )
                for c in confidences
            ]
    Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True))
    for key, arms in results["arms"].items():
        for arm_name, rows in arms.items():
            for row in rows:
                print(
                    f"[gate] ckpt={key} arm={arm_name} conf={row['confidence']} "
                    f"recall={row['recall']:.4f} "
                    f"extras/tile={row['extras_per_tile']:.4f} "
                    f"mean_quality={row['mean_quality']:.4f}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
