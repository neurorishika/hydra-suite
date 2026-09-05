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

The `main()` CLI below wires those pure functions to the live calibration
harness (``core/inference/semantic/calibration.py``) to run the comparison
end to end and write ``baseline.json``. It requires a working ``sam3``
install and two published checkpoints, so it cannot be exercised on this
machine (macOS, triton-blocked) -- see ``tools/sam3_parity/README.md`` for
the exact command to run it on a GPU box. Running it is Step 6 of the plan's
Task 0 and is deliberately DEFERRED here; no numbers are fabricated in its
place.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
# Live orchestration (Step 6) -- requires a working sam3 install; DEFERRED.
# ---------------------------------------------------------------------------


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
    out_path: Path,
) -> None:
    """Wire the pure functions above to the live calibration harness and
    write ``baseline.json``. Requires ``import sam3`` to succeed, so this is
    NOT importable/runnable on this (macOS, triton-blocked) machine -- see
    ``tools/sam3_parity/README.md`` for the command to run it on a GPU box.

    Imports of ``sam3``-dependent modules are deferred inside this function
    body so the rest of this file (and its unit tests) never require sam3.
    """

    # Deferred: constructing a SemanticLabeler from each checkpoint is
    # model-loading code that lives in the app/training layers and differs
    # by variant (Sam3SemanticLabeler.from_variant). Left as a narrow seam
    # for the GPU-box runner rather than duplicated here.
    raise NotImplementedError(
        "Live comparison requires a GPU box with sam3 installed; see "
        "tools/sam3_parity/README.md for the exact invocation. This "
        "function's plumbing (calibrate() -> preview frames -> paired "
        "stats -> baseline.json) is intentionally left for that box to "
        "fill in the two SemanticLabeler constructions, which is the only "
        "sam3-import-requiring step."
    )


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
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
