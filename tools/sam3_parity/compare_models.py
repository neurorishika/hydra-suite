"""Task 0 (SAM3 spike-parity plan): a measurement that can resolve the effect.

The prior comparison (two independent 16-frame means, ~5-8 extras/frame) had
a noise floor of sd ~= 0.6-0.7 extras/frame against a claimed 1.2-2.0 target
delta -- too small to distinguish real movement from frame-sampling noise.
This module fixes that by:

* Step 1: reporting PAIRED per-frame differences (``extras_a[i] - extras_b[i]``
  for the SAME frame ``i``), so frame difficulty cancels instead of adding
  variance to two independent means.
* Step 2: a stated significance criterion, decided BEFORE running anything --
  a delta counts as real only if the bootstrap CI over the paired
  differences excludes zero. See ``paired_comparison``.
* Step 3: a procedural definition of "matched recall" -- hold one model's
  recall fixed, linearly interpolate the OTHER model's extras/frame onto
  that recall value across its own confidence sweep. See
  ``interpolate_extras_at_recall``.
* Step 4: AP / the full PR curve, which needs no threshold matching at all.
  See ``average_precision`` / ``precision_recall_curve``.
* Step 5: adjudication helpers that split a model's "extra" detections into
  those a human should inspect (not matched to a label AND not matched to
  the other model's own predictions) versus ordinary disagreement, because
  extras are scored against labels that can themselves be incomplete. See
  ``unmatched_predictions`` / ``extras_unique_to_a``.

Everything above is a PURE function over plain arrays/polygons and is unit
tested on synthetic data in ``tests/test_sam3_parity_compare.py`` -- that is
the real gate for this file, independent of whether SAM3 is importable on
this machine.

`_run_live_comparison` wires those pure functions to the live calibration
harness (``core/inference/semantic/calibration.py``) end to end: it loads
frames + labels from a COCO json, runs ``calibrate()`` for each of two
labelers, recomputes per-frame extras/missed at a chosen confidence from the
cached raw candidates (for the Step 1-2 paired comparison), builds
``OperatingPoint`` sweeps per model (for Step 3-4), and writes
``baseline.json`` with the exact calibration arguments and frame list. The
ONLY step that requires ``sam3`` to be importable is constructing the two
``SemanticLabeler``s from their checkpoints (``_default_labeler_factory``,
called lazily); everything else is plain numpy/dataclass plumbing and is
exercised end to end in ``tests/test_sam3_parity_compare.py`` via a fake
labeler (no sam3, no GPU). Running it for real against real checkpoints is
Step 6 of the plan's Task 0 and is deliberately DEFERRED on this machine
(macOS, triton-blocked, no published checkpoints locally) -- see
``tools/sam3_parity/README.md`` for the exact command to run it on a GPU
box. No numbers are fabricated in its place.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:  # pragma: no cover - exercised implicitly by whichever branch runs
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - depends on the environment
    _scipy_stats = None

# ---------------------------------------------------------------------------
# Step 1-2: paired per-frame comparison + significance criterion
# ---------------------------------------------------------------------------


def paired_frame_diffs(
    values_a: Sequence[float], values_b: Sequence[float]
) -> np.ndarray:
    """Per-frame ``a - b``, given two SAME-LENGTH, SAME-ORDER per-frame series.

    Raises ``ValueError`` on a length mismatch -- silently zipping
    mismatched frame lists would pair the wrong frames and understate real
    per-frame variance exactly the way the two-independent-means comparison
    did.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"paired series must have matching per-frame shape, got "
            f"{a.shape} vs {b.shape}"
        )
    if a.size == 0:
        raise ValueError("paired series must not be empty")
    return a - b


def sign_test(diffs: Sequence[float], *, mu0: float = 0.0) -> float:
    """Two-sided exact sign-test p-value that ``diffs`` are drawn from a
    distribution symmetric about ``mu0``.

    Uses ``scipy.stats.binomtest`` when scipy is importable; otherwise hand-
    rolls the exact binomial tail via ``math.comb`` (scipy is not guaranteed
    to be installed in every environment this tool ships to).  Ties (a
    frame with ``diff == mu0``) are dropped, matching the classical sign
    test's treatment of ties.
    """
    d = np.asarray(diffs, dtype=np.float64) - mu0
    nonzero = d[d != 0]
    n = nonzero.size
    if n == 0:
        return 1.0
    k = int(np.sum(nonzero > 0))
    if _scipy_stats is not None:
        return float(_scipy_stats.binomtest(k, n, 0.5).pvalue)
    return _binomial_two_sided_p(k, n)


def _binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial-test p-value for ``k`` successes of ``n``
    trials under p=0.5, without scipy: sum the probability of every outcome
    at least as extreme as ``k`` (small-sample method matching
    ``scipy.stats.binomtest``'s default)."""
    pmf = [math.comb(n, i) * (0.5**n) for i in range(n + 1)]
    p_k = pmf[k]
    # A tolerance guards against floating-point pmf values that should be
    # exactly equal (e.g. k and n-k under a symmetric binomial) comparing
    # as strictly greater/less due to rounding.
    tol = 1e-12
    return float(sum(p for p in pmf if p <= p_k + tol))


def bootstrap_ci(
    diffs: Sequence[float],
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the MEAN of ``diffs``.

    Resamples frames with replacement (the unit of exchangeability here is
    the frame, matching the pairing in ``paired_frame_diffs``), not
    individual detections.
    """
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        raise ValueError("bootstrap_ci requires at least one value")
    rng = np.random.default_rng(seed)
    n = d.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = d[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


@dataclass(frozen=True)
class PairedComparison:
    """The full Step 1/2 result for one paired per-frame metric."""

    n_frames: int
    mean_diff: float
    sign_test_p: float
    ci_low: float
    ci_high: float
    ci_level: float
    significant: bool  # True iff the CI excludes zero


def paired_comparison(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Run the Step 1/2 paired comparison: ``mean(a - b)``, its sign-test
    p-value, and a bootstrap CI. ``significant`` is the plan's pre-registered
    criterion -- the CI must exclude zero -- and is the ONLY thing that may
    be reported as a real effect; anything smaller is null.
    """
    diffs = paired_frame_diffs(values_a, values_b)
    lo, hi = bootstrap_ci(diffs, n_resamples=n_resamples, ci=ci, seed=seed)
    return PairedComparison(
        n_frames=int(diffs.size),
        mean_diff=float(diffs.mean()),
        sign_test_p=sign_test(diffs),
        ci_low=lo,
        ci_high=hi,
        ci_level=ci,
        significant=bool(lo > 0.0 or hi < 0.0),
    )


# ---------------------------------------------------------------------------
# Step 3: matched-recall interpolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatingPoint:
    """One confidence-sweep sample: recall and cost at that confidence."""

    confidence: float
    recall: float
    extra_per_frame: float


def interpolate_extras_at_recall(
    points: Sequence[OperatingPoint], target_recall: float
) -> float | None:
    """Linear interpolation of ``extra_per_frame`` at ``target_recall``,
    procedurally defining "matched recall" (plan Step 3): the REFERENCE
    model's recall is held fixed at ``target_recall``; this interpolates the
    OTHER model's own confidence sweep onto that same recall value.

    Returns ``None`` when ``target_recall`` falls outside the sweep's
    achieved recall range -- extrapolating past the measured points would
    silently invent an answer, exactly the ambiguity this step exists to
    remove. Recall is expected to be monotonically non-increasing as
    confidence rises; points are sorted by recall before interpolating so
    the direction ``np.interp`` traverses is always ascending.
    """
    if not points:
        return None
    ordered = sorted(points, key=lambda p: p.recall)
    recalls = np.array([p.recall for p in ordered], dtype=np.float64)
    extras = np.array([p.extra_per_frame for p in ordered], dtype=np.float64)
    if target_recall < recalls[0] or target_recall > recalls[-1]:
        return None
    return float(np.interp(target_recall, recalls, extras))


# ---------------------------------------------------------------------------
# Step 4: AP / PR curve
# ---------------------------------------------------------------------------


def precision_recall_curve(
    points: Sequence[OperatingPoint], *, missed_per_frame: Sequence[float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Recall/precision arrays sorted by ascending recall, from
    per-confidence ``(recall, extra_per_frame)`` pairs, matched-per-frame.

    ``precision`` at a point is derived as ``matched / (matched + extra)``
    using per-frame-normalised extras and the recall value directly, so it
    needs no re-derivation of the underlying instance counts: with ``r`` the
    recall and ``e`` the mean extras/frame at a fixed mean total-labels/frame
    ``t`` (which cancels out of the ratio), precision = ``r*t / (r*t + e)``.
    ``t`` is arbitrary and fixed at 1.0 when not supplied via
    ``missed_per_frame`` (which lets the caller recover it exactly as
    ``matched_per_frame + missed_per_frame``, with ``matched_per_frame =
    recall`` in those same units) -- callers that only have recall/extra may
    pass ``missed_per_frame=None`` and get a precision curve that is
    monotone-correct but not on an absolute per-frame instance scale.
    """
    if missed_per_frame is not None and len(missed_per_frame) != len(points):
        raise ValueError("missed_per_frame must align 1:1 with points")
    indexed = sorted(enumerate(points), key=lambda ip: ip[1].recall)
    order = [i for i, _p in indexed]
    ordered = [p for _i, p in indexed]
    recalls = np.array([p.recall for p in ordered], dtype=np.float64)
    extras = np.array([p.extra_per_frame for p in ordered], dtype=np.float64)
    if missed_per_frame is not None:
        missed = np.array([missed_per_frame[i] for i in order], dtype=np.float64)
        # matched = recall * total, and total = matched + missed, so
        # total = missed / (1 - recall) when recall < 1.
        with np.errstate(divide="ignore", invalid="ignore"):
            total = np.where(recalls < 1.0, missed / (1.0 - recalls), np.nan)
        matched = recalls * total
    else:
        matched = recalls
    denom = matched + extras
    precisions = np.divide(matched, denom, out=np.ones_like(matched), where=denom > 0)
    return recalls, precisions


def average_precision(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """PASCAL-VOC-style AP: integrate the monotone-decreasing precision
    envelope (each precision replaced by the max precision at that recall or
    higher) over recall via the trapezoidal rule.

    Recall/precision need not be pre-sorted or de-duplicated; this sorts by
    recall internally. NaN precision values (recall == 1.0 with no total
    derivable) are dropped before integrating.
    """
    r = np.asarray(recalls, dtype=np.float64)
    p = np.asarray(precisions, dtype=np.float64)
    mask = ~np.isnan(p)
    r, p = r[mask], p[mask]
    if r.size == 0:
        return 0.0
    order = np.argsort(r)
    r, p = r[order], p[order]
    # Monotone precision envelope: precision(r) := max(precision(r' >= r)).
    envelope = np.maximum.accumulate(p[::-1])[::-1]
    if r.size == 1:
        return float(envelope[0] * r[0])
    return float(np.trapezoid(envelope, r))


# ---------------------------------------------------------------------------
# Step 5: adjudication -- clutter vs. unlabelled ants
# ---------------------------------------------------------------------------


def unmatched_predictions(
    pred_polys: Sequence[np.ndarray],
    label_polys: Sequence[np.ndarray],
    *,
    match_fn,
    **match_kwargs,
) -> list[int]:
    """Indices into ``pred_polys`` NOT claimed by any label under
    ``match_fn`` (expected: ``calibration.match_one_to_one``, injected
    rather than imported so this module stays testable without the
    calibration harness's cv2/shape_prior dependencies).
    """
    pairs = match_fn(pred_polys, label_polys, **match_kwargs)
    matched_pred = {pi for pi, _gi in pairs}
    return [i for i in range(len(pred_polys)) if i not in matched_pred]


def extras_unique_to_a(
    pred_polys_a: Sequence[np.ndarray],
    pred_polys_b: Sequence[np.ndarray],
    label_polys: Sequence[np.ndarray],
    *,
    match_fn,
    iou_fn,
    cross_iou_threshold: float = 0.3,
    **match_kwargs,
) -> list[int]:
    """Indices into ``pred_polys_a`` that are (a) not matched to any label
    and (b) not overlapping (IoU >= ``cross_iou_threshold``) any of model
    B's own predictions.

    These are the detections Step 5 says to look at BY EYE: an extra that
    both models agree on is model-independent evidence about the SCENE
    (probably an unlabelled ant, since two independently-trained models
    hallucinating the identical location is unlikely); an extra unique to
    one model is more likely that model's own clutter. This function only
    prepares the candidate set for inspection -- it does not itself decide
    "clutter" vs. "unlabelled ant", which the plan requires a human to do.
    """
    unmatched_a = unmatched_predictions(
        pred_polys_a, label_polys, match_fn=match_fn, **match_kwargs
    )
    out = []
    for i in unmatched_a:
        poly_a = pred_polys_a[i]
        overlaps_b = any(
            iou_fn(poly_a, poly_b) >= cross_iou_threshold for poly_b in pred_polys_b
        )
        if not overlaps_b:
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Live orchestration (Step 6). Sam3-import-free except for the ONE seam
# (`_default_labeler_factory`) that constructs a model from a checkpoint --
# every other function below is plain numpy/dataclass plumbing over the
# calibration harness and is exercised end to end (with a fake labeler) in
# tests/test_sam3_parity_compare.py.
# ---------------------------------------------------------------------------


def _default_labeler_factory(checkpoint: Path):
    """Build a real SAM3 labeler from *checkpoint*. Requires ``import sam3``
    to succeed -- deferred here so nothing else in this module does.
    """
    from hydra_suite.core.inference.semantic.calibration import CONFIDENCE_GRID
    from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

    # confidence_floor must be at or below the lowest confidence this tool's
    # sweep will ever re-threshold at, or the calibration cache silently
    # excludes candidates a later confidence in the grid needs -- see
    # calibration.py's own docstring on the same hazard.
    return Sam3SemanticLabeler.from_variant(
        checkpoint=checkpoint, confidence_floor=CONFIDENCE_GRID[0]
    )


def _load_frames_from_coco(
    coco_json: Path, frames_dir: Path
) -> list[tuple[Path, list]]:
    """Load ``(image_path, [LabelRecord, ...])`` pairs from a COCO
    instance-segmentation json, the format calibration's ``frames`` argument
    expects (`core/inference/semantic/calibration.py::calibrate`).

    Only the first polygon of a (possibly multi-part) ``segmentation`` is
    used, matching the labelled-frame data this tool targets (single-part
    ant/worm outlines); annotations with no segmentation are skipped.
    """
    from hydra_suite.data.al.escalation import LabelRecord
    from hydra_suite.utils.geometry_levels import GeometryLevel

    data = json.loads(Path(coco_json).read_text())
    images_by_id = {img["id"]: img for img in data.get("images", [])}
    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    cat_id_to_class = {cat["id"]: i for i, cat in enumerate(categories)}
    records_by_image: dict[int, list] = {img_id: [] for img_id in images_by_id}
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id not in images_by_id:
            continue
        seg = ann.get("segmentation")
        if not seg:
            continue
        flat = seg[0] if isinstance(seg[0], (list, tuple)) else seg
        pts = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
        class_id = cat_id_to_class.get(ann.get("category_id"), 0)
        records_by_image[image_id].append(
            LabelRecord(
                class_id=class_id,
                confidence=1.0,
                points=pts,
                level=GeometryLevel.POLYGON,
            )
        )
    frames = [
        (Path(frames_dir) / img["file_name"], records_by_image[image_id])
        for image_id, img in images_by_id.items()
    ]
    frames.sort(key=lambda item: str(item[0]))
    return frames


def _calibrate_one_model(
    labeler,
    frames: Sequence[tuple[Path, list]],
    prompt: str,
    *,
    reference_body_px: float,
    tile_fraction: float | None,
    seam_margin_px: float,
    merge_iou: float,
) -> tuple[list, list]:
    """Run ``calibration.calibrate`` for one labeler at one tile fraction,
    returning ``(CalibrationPoint list, CalibrationPreviewFrame list)``.
    """
    from hydra_suite.core.inference.semantic.calibration import calibrate

    preview_holder: list = []
    points = calibrate(
        labeler,
        frames,
        prompt,
        reference_body_px=reference_body_px,
        tile_fractions=(tile_fraction,),
        seam_margin_px=seam_margin_px,
        merge_iou=merge_iou,
        preview_sink=preview_holder.append,
    )
    previews = preview_holder[0] if preview_holder else []
    return points, previews


def _operating_points_from_calibration(
    points: Sequence, tile_fraction: float | None
) -> tuple[list[OperatingPoint], list[float]]:
    """Project ``CalibrationPoint``s at *tile_fraction* onto the
    ``OperatingPoint`` shape Steps 3-4 consume, plus their aligned
    ``missed_per_frame`` (needed to recover an absolute precision scale in
    ``precision_recall_curve``).
    """
    at_fraction = [p for p in points if p.tile_fraction == tile_fraction]
    operating_points = [
        OperatingPoint(
            confidence=p.confidence, recall=p.recall, extra_per_frame=p.extra_per_frame
        )
        for p in at_fraction
    ]
    missed_per_frame = [p.missed_per_frame for p in at_fraction]
    return operating_points, missed_per_frame


def _per_frame_extras_missed(
    previews: Sequence,
    tile_fraction: float | None,
    confidence: float,
    area_band,
    merge_iou: float,
) -> dict[Path, tuple[int, int]]:
    """Re-threshold each preview frame's cached raw candidates at
    *confidence* and return ``{image_path: (extra_count, missed_count)}`` --
    the per-frame series Steps 1-2's paired comparison needs, which
    ``calibrate()`` itself only exposes pre-aggregated into a mean.
    """
    from hydra_suite.core.inference.semantic.calibration import match_one_to_one
    from hydra_suite.core.inference.semantic.tiling import merge_candidates

    out: dict[Path, tuple[int, int]] = {}
    for preview in previews:
        candidates = preview.candidates_by_fraction.get(tile_fraction, ())
        merged = merge_candidates(
            candidates,
            confidence_threshold=confidence,
            iou_threshold=merge_iou,
            area_band=area_band,
        )
        preds = [m.polygon_px for m in merged]
        label_polys = [g.polygon_px for g in preview.ground_truth]
        pairs = match_one_to_one(preds, label_polys, area_band=area_band)
        out[preview.image_path] = (
            len(preds) - len(pairs),
            len(label_polys) - len(pairs),
        )
    return out


def _run_live_comparison(
    *,
    checkpoint_a: Path,
    checkpoint_b: Path,
    frames_dir: Path,
    coco_json: Path,
    prompt: str,
    reference_body_px: float,
    tile_fraction: float | None,
    seam_margin_px: float,
    merge_iou: float,
    compare_confidence: float,
    target_recall: float,
    out_path: Path,
    labeler_factory: Callable[[Path], object] | None = None,
) -> dict:
    """Run Task 0 Steps 1-5 end to end against two checkpoints and write
    *out_path* as ``baseline.json``.

    *labeler_factory* defaults to ``_default_labeler_factory`` (real SAM3,
    requires ``sam3``); tests inject a fake to exercise every line of this
    function's orchestration without sam3 or a GPU.
    """
    from hydra_suite.core.inference.semantic.shape_prior import fit_area_band

    factory = labeler_factory or _default_labeler_factory
    frames = _load_frames_from_coco(coco_json, frames_dir)
    if not frames:
        raise ValueError(f"No frames resolved from {coco_json} under {frames_dir}")

    # One area band for BOTH models, fitted from the shared ground truth --
    # comparability requires the same admissibility gate on both sides
    # (see calibration.calibrate's own docstring on the same point).
    area_band = fit_area_band(
        [poly for _path, labels in frames for poly in (r.points for r in labels)]
    )

    per_model: dict[str, dict] = {}
    for key, checkpoint in (("a", checkpoint_a), ("b", checkpoint_b)):
        labeler = factory(checkpoint)
        points, previews = _calibrate_one_model(
            labeler,
            frames,
            prompt,
            reference_body_px=reference_body_px,
            tile_fraction=tile_fraction,
            seam_margin_px=seam_margin_px,
            merge_iou=merge_iou,
        )
        operating_points, missed_per_frame = _operating_points_from_calibration(
            points, tile_fraction
        )
        per_frame = _per_frame_extras_missed(
            previews, tile_fraction, compare_confidence, area_band, merge_iou
        )
        per_model[key] = {
            "checkpoint": str(checkpoint),
            "operating_points": operating_points,
            "missed_per_frame": missed_per_frame,
            "per_frame": per_frame,
        }

    # Step 1-2: paired per-frame comparison, restricted to frames BOTH
    # models actually produced a result for -- a model-specific load
    # failure must not silently misalign the pairing.
    common_paths = sorted(
        set(per_model["a"]["per_frame"]) & set(per_model["b"]["per_frame"]),
        key=str,
    )
    if not common_paths:
        raise ValueError(
            "Models produced no frames in common; cannot run a paired comparison."
        )
    extras_a = [per_model["a"]["per_frame"][p][0] for p in common_paths]
    extras_b = [per_model["b"]["per_frame"][p][0] for p in common_paths]
    comparison = paired_comparison(extras_a, extras_b)

    # Step 4: AP / PR curve, no threshold matching required.
    recalls_a, precisions_a = precision_recall_curve(
        per_model["a"]["operating_points"],
        missed_per_frame=per_model["a"]["missed_per_frame"],
    )
    recalls_b, precisions_b = precision_recall_curve(
        per_model["b"]["operating_points"],
        missed_per_frame=per_model["b"]["missed_per_frame"],
    )
    ap_a = average_precision(recalls_a, precisions_a)
    ap_b = average_precision(recalls_b, precisions_b)

    # Step 3: matched-recall interpolation, held at the pre-stated
    # target_recall for both models independently.
    extras_at_target_a = interpolate_extras_at_recall(
        per_model["a"]["operating_points"], target_recall
    )
    extras_at_target_b = interpolate_extras_at_recall(
        per_model["b"]["operating_points"], target_recall
    )

    baseline = {
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "frames_dir": str(frames_dir),
        "coco_json": str(coco_json),
        "prompt": prompt,
        "reference_body_px": reference_body_px,
        "tile_fraction": tile_fraction,
        "seam_margin_px": seam_margin_px,
        "merge_iou": merge_iou,
        "compare_confidence": compare_confidence,
        "target_recall": target_recall,
        "n_frames": len(common_paths),
        "frames": [str(p) for p in common_paths],
        "paired_extras_per_frame": asdict(comparison),
        "average_precision": {"a": ap_a, "b": ap_b},
        "extras_per_frame_at_target_recall": {
            "a": extras_at_target_a,
            "b": extras_at_target_b,
        },
    }
    Path(out_path).write_text(json.dumps(baseline, indent=2, sort_keys=True))
    return baseline


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--reference-body-px", type=float, required=True)
    parser.add_argument("--tile-fraction", type=float, default=None)
    parser.add_argument("--seam-margin-px", type=float, default=8.0)
    parser.add_argument("--merge-iou", type=float, default=0.5)
    parser.add_argument(
        "--compare-confidence",
        type=float,
        default=0.5,
        help="Confidence at which per-frame extras/missed are computed for "
        "the Step 1-2 paired comparison. Fixed here rather than swept, per "
        "the plan's Step 2: state the criterion before running.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.9,
        help="Recall each model's own confidence sweep is interpolated onto "
        "for the Step 3 matched-recall extras/frame figure.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "baseline.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _run_live_comparison(
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        frames_dir=args.frames_dir,
        coco_json=args.coco_json,
        prompt=args.prompt,
        reference_body_px=args.reference_body_px,
        tile_fraction=args.tile_fraction,
        seam_margin_px=args.seam_margin_px,
        merge_iou=args.merge_iou,
        compare_confidence=args.compare_confidence,
        target_recall=args.target_recall,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
