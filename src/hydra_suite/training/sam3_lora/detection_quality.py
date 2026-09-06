"""Per-epoch DETECTION-QUALITY (AP) evidence for a SAM3 LoRA run.

WHY THIS EXISTS. The run already records a per-epoch validation LOSS series
(``cli.val_series.jsonl``). A 2026-09-06 checkpoint-ladder study on a
genuinely held-out fold measured every per-query validation signal --
``val_loss_mean``, ``loss_ce``, ``loss_bbox``, ``loss_giou``, ``loss_mask``,
``loss_dice``, ``presence_loss`` -- ANTI-correlating with held-out AP
(Spearman -0.4 to -1.0; all of them ranked the best checkpoint LAST). So the
loss series is the wrong signal to ever select on, while AP was the most
stable signal measured (paired effect size 2.92). This module records AP so
that a future selection decision has a signal worth considering.

WHAT IT IS NOT. It selects nothing and stops nothing. If you are here to
wire best-checkpoint selection or early stopping onto this number: the
evidence for AP is ONE run, ONE seed, ONE corpus, and seed variance is
entirely unmeasured. AP is a *candidate* signal, not a validated selection
criterion. The anti-correlation warnings on the loss path stand unchanged.

FOUR LIMITATIONS, stated where someone would otherwise assume otherwise:

1. WITHIN-RUN ONLY, NOT COMPARABLE ACROSS RUNS. AP moved 0.96 -> 0.61-0.69
   for the SAME models purely by changing tile size (1766 px vs 971 px).
   Differencing two runs' AP is meaningless unless their derived datasets
   are identical.
2. TILE-SPACE, NOT FRAME-MERGED. The eval harness
   (``tools/sam3_parity/compare_models.py``) merges tiles back to frames and
   scores against the original frame labels. A training run cannot: its
   derived COCO split stores only tile-CLIPPED ground truth, and reassembling
   clipped fragments is not the same thing as a frame label. So this number
   is not comparable with the harness's either.
3. NO NMS AND NO CROSS-TILE MERGE. The study's AP came through the
   production ultralytics path (presence multiplication AND NMS at IoU 0.7).
   Here the raw training-head queries are scored directly, so duplicate
   queries inflate ``extras`` and depress precision. The stability the study
   measured therefore does NOT transfer verbatim; this is a trend signal.
4. SCORED ON THE VALIDATION SPLIT, which the run's own checkpointing has
   seen the loss of. It is not the held-out fold the study used.

COST. It piggybacks on the loss pass's forward -- no second inference -- so
its marginal cost is CPU only (contour extraction plus the confidence sweep's
matching). It is measured, not estimated: ``ap_elapsed_s`` in every row. It
also has its OWN cadence knob (``cli.AP_CADENCE_ENV``) so a run that finds it
expensive can back it off without giving up the loss series.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from hydra_suite.core.inference.semantic.calibration import (
    CONFIDENCE_GRID,
    match_one_to_one,
)
from hydra_suite.core.inference.semantic.detection_metrics import (
    SweepCounts,
    average_precision_from_counts,
)
from hydra_suite.core.inference.semantic.shape_prior import fit_area_band

from .datapoints import RES, _scale_polygons_to_res
from .protocol import emit_log

# Keys on SAM3's last-stage output dict. Named here rather than inlined so a
# vendor rename produces one clear failure with a list of what WAS present,
# not a silent zero. Scoring mirrors Meta's own postprocessor
# (`sam3/eval/postprocessors.py`): sigmoid(logits).max(-1) * sigmoid(presence).
PRED_LOGITS_KEY = "pred_logits"
PRED_MASKS_KEY = "pred_masks"
PRESENCE_KEY = "presence_logit_dec"


@dataclass(frozen=True)
class TileCounts:
    """One tile's instance counts at one confidence."""

    matched: int
    extras: int
    missed: int


def _last_stage(query_output: Any) -> Any:
    """The final decoder stage's output dict for one query.

    ``outputs.output`` is one list of stage dicts per query; Meta's
    postprocessor scores the LAST stage, so that is what is scored here.
    """
    if isinstance(query_output, dict):
        return query_output
    stages = list(query_output)
    if not stages:
        raise KeyError("SAM3 query output had no decoder stages")
    return stages[-1]


def extract_predictions(
    output: Any, *, target_size: int = RES
) -> tuple[list[np.ndarray], list[float], bool]:
    """Predicted polygons (in ``target_size`` space), scores, presence flag.

    The third element says whether the presence multiplication was actually
    applied. It is returned rather than swallowed because its absence CHANGES
    THE SCORE DEFINITION -- see the note at the presence lookup below.

    Raises ``KeyError`` naming the keys that WERE present when the expected
    ones are missing. On the CUDA box this error message is the only
    debugging channel available for a vendor rename, so it must be loud and
    specific rather than degrading to an empty prediction list (which would
    silently record AP = 0 and look like a training collapse).
    """
    from hydra_suite.core.inference.masks import mask_to_contour

    stage = _last_stage(output)
    missing = [k for k in (PRED_LOGITS_KEY, PRED_MASKS_KEY) if k not in stage]
    if missing:
        raise KeyError(
            f"SAM3 output is missing {missing}; present keys: {sorted(stage)}"
        )
    logits = stage[PRED_LOGITS_KEY]
    masks = stage[PRED_MASKS_KEY]
    scores_t = logits.sigmoid()[0].max(-1).values
    presence = stage.get(PRESENCE_KEY)
    presence_used = presence is not None
    if presence_used:
        # A per-image scalar: it rescales every query in the tile equally, so
        # within-tile ranking is unchanged and only absolute thresholds move.
        scores_t = scores_t * presence.sigmoid().reshape(-1)[0]
    # If it is ABSENT, the scores are no longer Meta's postprocessor
    # definition: cross-tile ordering and therefore the sweep counts and AP
    # move, while every number still looks perfectly plausible. A vendor
    # rename of THIS key alone would not trip the `missing` check above, so
    # the fact is propagated out and recorded per row as `ap_presence_used`
    # (plus a one-time log). It is deliberately NOT an error: presence is a
    # per-image scalar, so a run without it is still internally consistent
    # and a within-run trend -- the only thing this metric claims to be --
    # and discarding a usable series would cost more than it saves. What is
    # unacceptable is the definition changing with no trace, and the flag is
    # that trace, in the DATA rather than only in the docs.
    scores_np = scores_t.detach().float().cpu().numpy().reshape(-1)
    masks_np = masks[0].detach().float().cpu().numpy() > 0.0

    height, width = masks_np.shape[-2:]
    scale = np.asarray(
        [float(target_size) / float(width), float(target_size) / float(height)],
        dtype=np.float32,
    )
    polygons: list[np.ndarray] = []
    scores: list[float] = []
    for index in range(masks_np.shape[0]):
        contour = mask_to_contour(masks_np[index])
        if contour is None or contour.shape[0] < 3:
            continue
        polygons.append(contour.astype(np.float32) * scale)
        scores.append(float(scores_np[index]))
    return polygons, scores, presence_used


def count_tile(
    pred_polys: Sequence[np.ndarray],
    pred_scores: Sequence[float],
    gt_polys: Sequence[np.ndarray],
    crowd_polys: Sequence[np.ndarray],
    *,
    confidence: float,
    area_band: Any = None,
) -> TileCounts:
    """Match one tile at one confidence, with COCO IGNORE semantics.

    Supervised (non-crowd) ground truth is matched first. Predictions left
    over are then matched against the tile's CROWD instances -- the sub-floor
    fragments ``datapoints.select_output_objects`` excludes from supervision
    -- and those are dropped from ``extras`` entirely: they are neither a
    true positive (the model was never asked to find them) nor a false one
    (there really is an animal there). Recall's denominator is the
    non-crowd ground truth only.

    Uses ``calibration.match_one_to_one`` -- the CORRECTED matcher (b9e92bc7:
    ``representative_point`` plus the containment gate, no IoU route). The
    old vertex-mean centroid vetoed 15.9% of real matches; an AP computed
    with it would be worse than no AP at all.
    """
    keep = [i for i, s in enumerate(pred_scores) if float(s) >= float(confidence)]
    kept = [np.asarray(pred_polys[i], dtype=np.float32) for i in keep]
    gt = [np.asarray(p, dtype=np.float32) for p in gt_polys]
    pairs = match_one_to_one(kept, gt, area_band=area_band)
    matched = len(pairs)
    claimed = {pi for pi, _gi in pairs}
    leftover = [poly for i, poly in enumerate(kept) if i not in claimed]
    ignored = 0
    if crowd_polys and leftover:
        crowd = [np.asarray(p, dtype=np.float32) for p in crowd_polys]
        ignored = len(match_one_to_one(leftover, crowd, area_band=area_band))
    return TileCounts(
        matched=matched,
        extras=len(leftover) - ignored,
        missed=len(gt) - matched,
    )


def positive_query_plan(descriptors: Sequence[Any]) -> list[int | None]:
    """Query index -> descriptor index for POSITIVE queries, ``None`` for negatives.

    ``dataloader.collate_batches`` flattens descriptors in order, each one
    contributing its positive query first and then one query per negative
    prompt (``datapoints.build_tile_datapoint``), then chunks that flat
    sequence by batch size. Rebuilding the same flat sequence here is what
    lets predictions be aligned to their ground truth without threading extra
    state through the collator.
    """
    plan: list[int | None] = []
    for index, descriptor in enumerate(descriptors):
        plan.append(index)
        plan.extend([None] * len(descriptor.negative_prompts))
    return plan


def _tile_ground_truth(descriptor: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Supervised and crowd ground-truth polygons in RES space.

    Scaled with the SAME helper the training transform uses, so the geometry
    the metric scores is the geometry the model was trained on. A descriptor
    without recorded tile dimensions is assumed already RES-sized, which is
    what ``_scale_polygons_to_res`` treats as a no-op anyway.
    """
    width = int(getattr(descriptor, "width", 0) or RES)
    height = int(getattr(descriptor, "height", 0) or RES)
    supervised = [
        np.asarray(i.polygon, dtype=np.float32)
        for i in descriptor.instances
        if not i.is_crowd
    ]
    crowd = [
        np.asarray(i.polygon, dtype=np.float32)
        for i in descriptor.instances
        if i.is_crowd
    ]
    return (
        _scale_polygons_to_res(supervised, width, height),
        _scale_polygons_to_res(crowd, width, height),
    )


class DetectionQualityAccumulator:
    """Collects per-tile predictions during the loss pass, then scores AP.

    Fed the SAME ``outputs`` object the validation loss already computed --
    there is no second forward pass, so this cannot perturb the RNG streams
    or the model's mode beyond what ``cli._record_epoch_validation`` already
    guards.

    Never raises into training. Any failure (a vendor key rename, a
    degenerate polygon, an empty split) is captured and surfaced as
    ``ap_error`` in the epoch's row, because a broken metric must be visible
    in the series rather than silently absent or fatal to a multi-hour run.
    """

    def __init__(self, descriptors: Sequence[Any], *, confidences=CONFIDENCE_GRID):
        self._descriptors = list(descriptors)
        self._plan = positive_query_plan(self._descriptors)
        self._confidences = tuple(float(c) for c in confidences)
        self._cursor = 0
        self._predictions: dict[int, tuple[list[np.ndarray], list[float]]] = {}
        self._error: str | None = None
        self._elapsed = 0.0
        self._presence_used: bool | None = None

    def observe(self, outputs: Any) -> None:
        """Record one batch's per-query outputs, in query order."""
        started = time.perf_counter()
        try:
            for query_output in outputs.output:
                if self._cursor >= len(self._plan):
                    raise RuntimeError(
                        "SAM3 validation produced more queries "
                        f"({self._cursor + 1}) than the descriptor plan "
                        f"({len(self._plan)}); prediction/label alignment "
                        "cannot be trusted."
                    )
                descriptor_index = self._plan[self._cursor]
                self._cursor += 1
                if descriptor_index is None:
                    continue
                polys, scores, presence_used = extract_predictions(query_output)
                self._predictions[descriptor_index] = (polys, scores)
                if self._presence_used is None:
                    self._presence_used = presence_used
                    if not presence_used:
                        emit_log(
                            "SAM3 validation outputs carry no "
                            f"{PRESENCE_KEY!r}: AP scores are NOT Meta's "
                            "postprocessor definition (no presence "
                            "multiplication). Recorded as "
                            "ap_presence_used=false; training unaffected."
                        )
                else:
                    self._presence_used = self._presence_used and presence_used
        except Exception as exc:  # noqa: BLE001 - recording must never kill a run
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            self._elapsed += time.perf_counter() - started

    def result(self) -> dict[str, Any]:
        """The AP block to merge into the epoch's ``val_series.jsonl`` row."""
        started = time.perf_counter()
        try:
            if self._error is not None:
                raise RuntimeError(self._error)
            if not self._predictions:
                raise RuntimeError("no positive-query predictions were collected")
            # An UNDER-run is the dangerous direction and `observe` cannot see
            # it: a batch that yields fewer queries than planned leaves the
            # cursor behind, and every LATER prediction is then scored against
            # the wrong tile's ground truth -- a plausible-looking, silently
            # wrong AP. Checked here, once, where the whole pass is visible.
            if self._cursor != len(self._plan):
                raise RuntimeError(
                    f"SAM3 validation produced {self._cursor} queries but the "
                    f"descriptor plan expects {len(self._plan)}; "
                    "prediction/label alignment cannot be trusted."
                )
            ground_truth = {
                index: _tile_ground_truth(self._descriptors[index])
                for index in self._predictions
            }
            # One area band for the whole pass, fitted from this split's own
            # ground truth -- the same admissibility gate on every tile, as
            # `compare_models._run_live_comparison` does for its two models.
            all_gt = [poly for gt, _crowd in ground_truth.values() for poly in gt]
            area_band = fit_area_band(all_gt) if all_gt else None
            counts: list[SweepCounts] = []
            for confidence in self._confidences:
                matched = extras = missed = 0
                for index, (polys, scores) in self._predictions.items():
                    gt, crowd = ground_truth[index]
                    tile = count_tile(
                        polys,
                        scores,
                        gt,
                        crowd,
                        confidence=confidence,
                        area_band=area_band,
                    )
                    matched += tile.matched
                    extras += tile.extras
                    missed += tile.missed
                counts.append(
                    SweepCounts(
                        confidence=confidence,
                        matched=matched,
                        extras=extras,
                        missed=missed,
                    )
                )
            n_tiles = len(self._predictions)
            ap = average_precision_from_counts(counts, n_tiles)
            elapsed = self._elapsed + (time.perf_counter() - started)
            return {
                "ap": ap,
                "ap_tiles": n_tiles,
                # The raw JSONL should be self-describing: a reader who never
                # opens the docs still learns this AP is TILE-space (not the
                # eval harness's frame-merged score) and therefore a
                # within-run trend, not a cross-corpus comparable number.
                "ap_scope": "tile",
                "ap_presence_used": bool(self._presence_used),
                "ap_elapsed_s": elapsed,
                # The sweep itself, so the row can be re-scored offline (a
                # different AP definition, a recall-first operating point)
                # without a second GPU pass.
                "ap_sweep": [
                    {
                        "confidence": c.confidence,
                        "matched": c.matched,
                        "extras": c.extras,
                        "missed": c.missed,
                    }
                    for c in counts
                ],
            }
        except Exception as exc:  # noqa: BLE001 - recording must never kill a run
            return {
                "ap_error": f"{type(exc).__name__}: {exc}",
                "ap_elapsed_s": self._elapsed + (time.perf_counter() - started),
            }
