"""Per-epoch detection-quality metric (AP) for SAM3 LoRA training.

`import sam3` is unavailable on macOS, so every test here drives the real
code with STUB model outputs shaped exactly like the last-stage SAM3 output
dictionary documented in
`docs/superpowers/plans/2026-08-31-detectkit-sam3-finetune.md` (Task 12,
Step 1). That stub is the gate: it exercises extraction, the ignore
semantics, the sweep, the row schema, and the never-kill-training posture
without a GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


def _square(x, y, size=10.0):
    return np.array(
        [[x, y], [x + size, y], [x + size, y + size], [x, y + size]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# The shared scoring code moved out of tools/ into core/
# ---------------------------------------------------------------------------


def test_average_precision_lives_in_core_not_in_a_tools_script():
    """A training-time metric may not depend on a tools/ script."""
    from hydra_suite.core.inference.semantic import detection_metrics

    assert hasattr(detection_metrics, "average_precision")
    assert hasattr(detection_metrics, "precision_recall_curve")
    assert hasattr(detection_metrics, "OperatingPoint")


def test_the_parity_tool_reuses_the_core_implementation():
    """compare_models must not keep a second copy of the same arithmetic."""
    import sys
    from pathlib import Path

    # NOTE: insert `tools/sam3_parity`, never `tools` -- `tools/sam3/` is a
    # directory that shadows the real `sam3` package and would silently turn
    # every `importorskip("sam3")` elsewhere in the suite into a false pass.
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "tools" / "sam3_parity")
    )
    import compare_models

    from hydra_suite.core.inference.semantic import detection_metrics

    assert compare_models.average_precision is detection_metrics.average_precision
    assert (
        compare_models.precision_recall_curve
        is detection_metrics.precision_recall_curve
    )
    assert compare_models.OperatingPoint is detection_metrics.OperatingPoint


def test_average_precision_from_counts_matches_the_curve_path():
    from hydra_suite.core.inference.semantic.detection_metrics import (
        OperatingPoint,
        SweepCounts,
        average_precision,
        average_precision_from_counts,
        precision_recall_curve,
    )

    counts = [
        SweepCounts(confidence=0.1, matched=90, extras=60, missed=10),
        SweepCounts(confidence=0.5, matched=70, extras=10, missed=30),
        SweepCounts(confidence=0.9, matched=30, extras=1, missed=70),
    ]
    n_frames = 10
    points = [
        OperatingPoint(
            confidence=c.confidence,
            recall=c.matched / (c.matched + c.missed),
            extra_per_frame=c.extras / n_frames,
        )
        for c in counts
    ]
    missed_per_frame = [c.missed / n_frames for c in counts]
    recalls, precisions = precision_recall_curve(
        points, missed_per_frame=missed_per_frame
    )
    assert average_precision_from_counts(counts, n_frames) == pytest.approx(
        average_precision(recalls, precisions)
    )


def test_average_precision_from_counts_is_zero_without_ground_truth():
    from hydra_suite.core.inference.semantic.detection_metrics import (
        SweepCounts,
        average_precision_from_counts,
    )

    counts = [SweepCounts(confidence=0.5, matched=0, extras=3, missed=0)]
    assert average_precision_from_counts(counts, 4) == 0.0


# ---------------------------------------------------------------------------
# Extraction from the SAM3 last-stage output dict
# ---------------------------------------------------------------------------


def _stub_output(mask_bools, logits, presence=None):
    """One SAM3 query output dict: (1, Q, C) logits, (1, Q, H, W) masks."""
    torch = pytest.importorskip("torch")

    masks = torch.where(
        torch.as_tensor(np.asarray(mask_bools)), torch.tensor(5.0), torch.tensor(-5.0)
    ).unsqueeze(0)
    out = {
        "pred_logits": torch.logit(
            torch.as_tensor(logits, dtype=torch.float32)
        ).reshape(1, len(logits), 1),
        "pred_masks": masks,
    }
    if presence is not None:
        out["presence_logit_dec"] = torch.logit(torch.tensor([[float(presence)]]))
    return out


def test_predictions_are_extracted_and_scored_like_metas_postprocessor():
    from hydra_suite.training.sam3_lora import detection_quality as dq

    masks = np.zeros((2, 32, 32), dtype=bool)
    masks[0, 4:16, 4:16] = True
    masks[1, 20:30, 20:30] = True
    output = _stub_output(masks, [0.9, 0.4], presence=0.5)

    polys, scores, presence_used = dq.extract_predictions(output, target_size=64)
    assert presence_used is True

    assert len(polys) == 2 and len(scores) == 2
    # score = sigmoid(logits).max(-1) * sigmoid(presence)
    assert scores == pytest.approx([0.45, 0.20], abs=1e-4)
    # Polygons are scaled from mask resolution into the target (RES) space.
    assert polys[0][:, 0].max() > 16.0


def test_a_missing_output_key_names_the_keys_that_were_present():
    from hydra_suite.training.sam3_lora import detection_quality as dq

    with pytest.raises(KeyError) as excinfo:
        dq.extract_predictions({"nonsense": 1}, target_size=64)
    assert "nonsense" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Counting: COCO ignore semantics for sub-floor tile fragments
# ---------------------------------------------------------------------------


def test_crowd_fragments_are_ignored_rather_than_counted_as_extras():
    from hydra_suite.training.sam3_lora.detection_quality import count_tile

    gt = [_square(0, 0)]
    crowd = [_square(40, 40)]
    preds = [_square(0.5, 0.5), _square(40.5, 40.5), _square(80, 80)]
    scores = [0.9, 0.9, 0.9]

    counts = count_tile(preds, scores, gt, crowd, confidence=0.5, area_band=None)
    assert counts.matched == 1
    assert counts.missed == 0
    # The crowd-matched prediction is neither a TP nor an FP; only the
    # genuinely spurious third prediction is an extra.
    assert counts.extras == 1


def test_confidence_thresholding_moves_recall_and_extras():
    from hydra_suite.training.sam3_lora.detection_quality import count_tile

    gt = [_square(0, 0), _square(40, 40)]
    preds = [_square(0.5, 0.5), _square(40.5, 40.5)]
    scores = [0.9, 0.2]

    high = count_tile(preds, scores, gt, [], confidence=0.5, area_band=None)
    low = count_tile(preds, scores, gt, [], confidence=0.1, area_band=None)
    assert (high.matched, high.missed) == (1, 1)
    assert (low.matched, low.missed) == (2, 0)


# ---------------------------------------------------------------------------
# The accumulator: query alignment and the resulting stats block
# ---------------------------------------------------------------------------


class _Descriptor:
    def __init__(self, polys, n_neg=1, crowd=()):
        from hydra_suite.training.sam3_lora.dataloader import InstanceDescriptor

        self.instances = tuple(
            InstanceDescriptor(
                polygon=tuple((float(x), float(y)) for x, y in poly), is_crowd=False
            )
            for poly in polys
        ) + tuple(
            InstanceDescriptor(
                polygon=tuple((float(x), float(y)) for x, y in poly), is_crowd=True
            )
            for poly in crowd
        )
        self.negative_prompts = tuple(f"neg{i}" for i in range(n_neg))
        self.width = 0
        self.height = 0


def test_only_positive_queries_are_scored_and_alignment_follows_the_batching():
    from hydra_suite.training.sam3_lora import detection_quality as dq

    descriptors = [_Descriptor([_square(0, 0)], n_neg=1) for _ in range(2)]
    plan = dq.positive_query_plan(descriptors)
    # descriptor0 positive, descriptor0 negative, descriptor1 positive, ...
    assert plan == [0, None, 1, None]


def test_accumulator_produces_an_ap_stats_block():
    torch = pytest.importorskip("torch")
    assert torch is not None
    from hydra_suite.training.sam3_lora import detection_quality as dq

    gt = _square(100, 100, size=200)
    descriptors = [_Descriptor([gt], n_neg=1)]
    accumulator = dq.DetectionQualityAccumulator(descriptors)

    masks = np.zeros((1, 64, 64), dtype=bool)
    masks[0, 12:26, 12:26] = True  # ~ the same square at 1008/64 scale
    positive = _stub_output(masks, [0.95])
    negative = _stub_output(np.zeros((1, 64, 64), dtype=bool), [0.01])

    class _Outputs:
        def __init__(self, queries):
            self.output = [[q] for q in queries]

    accumulator.observe(_Outputs([positive, negative]))
    stats = accumulator.result()

    assert stats["ap_tiles"] == 1
    assert 0.0 <= stats["ap"] <= 1.0
    assert stats["ap"] > 0.0, "a near-perfect prediction must score above zero"


def test_accumulator_reports_an_error_instead_of_raising_into_training():
    from hydra_suite.training.sam3_lora import detection_quality as dq

    descriptors = [_Descriptor([_square(0, 0)], n_neg=0)]
    accumulator = dq.DetectionQualityAccumulator(descriptors)

    class _Outputs:
        output = [[{"garbage": True}]]

    accumulator.observe(_Outputs())
    stats = accumulator.result()
    assert "ap_error" in stats
    assert "ap" not in stats


# ---------------------------------------------------------------------------
# CLI wiring: cadence, row schema, and the no-selection guarantee
# ---------------------------------------------------------------------------


def test_ap_has_its_own_cadence_knob_defaulting_to_the_loss_cadence(monkeypatch):
    from hydra_suite.training.sam3_lora import cli

    monkeypatch.delenv(cli.VAL_CADENCE_ENV, raising=False)
    monkeypatch.delenv(cli.AP_CADENCE_ENV, raising=False)
    assert cli.ap_cadence() == cli.val_cadence()
    monkeypatch.setenv(cli.VAL_CADENCE_ENV, "2")
    assert cli.ap_cadence() == 2, "AP defaults to the same cadence as the loss pass"
    monkeypatch.setenv(cli.AP_CADENCE_ENV, "5")
    assert cli.ap_cadence() == 5
    monkeypatch.setenv(cli.AP_CADENCE_ENV, "garbage")
    assert cli.ap_cadence() == 2


def test_ap_keys_land_in_the_same_val_series_row(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from hydra_suite.training.sam3_lora import cli

    monkeypatch.setattr(
        cli,
        "_evaluate_split",
        lambda *a, **k: {
            "val_loss_mean": 1.0,
            "val_batches": 1,
            "val_terms_mean": {},
            "elapsed_s": 0.1,
            "ap": 0.42,
            "ap_tiles": 7,
            "ap_elapsed_s": 0.25,
        },
    )

    class _Model:
        training = True

        def train(self):
            self.training = True

    cli._record_epoch_validation(
        _Model(), None, None, None, None, None, None, True, tmp_path, 3
    )
    row = json.loads((tmp_path / cli.VAL_SERIES_FILENAME).read_text().strip())
    assert row["epoch"] == 3
    assert row["ap"] == 0.42
    assert row["ap_cadence"] == cli.ap_cadence()
    assert row["ap_elapsed_s"] == 0.25
    assert row["val_loss_mean"] == 1.0, "the loss series is untouched"


def test_the_ap_pass_runs_inside_the_existing_rng_guard():
    """No second, unprotected inference pass: AP is collected from the SAME
    forward that the loss pass already runs, inside the same guard."""
    import inspect

    from hydra_suite.training.sam3_lora import cli

    guarded = inspect.getsource(cli._record_epoch_validation)
    assert "get_rng_state" in guarded and "set_rng_state" in guarded
    split = inspect.getsource(cli._evaluate_split)
    assert "DetectionQualityAccumulator" in split or "accumulator" in split


def test_nothing_selects_or_stops_on_the_recorded_metric():
    """AP is recorded and drives NOTHING -- not selection, not stopping.

    This test used to forbid the substrings "early_stop"/"patience" anywhere
    in `cli`. That was a proxy for the real invariant and it stopped being
    usable on 2026-09-08, when a user decision added early stopping on
    `val_loss_mean` (off by default). It is NOT weakened here: the invariant
    it was protecting -- that the detection-quality numbers select and stop
    nothing -- is now asserted directly against the stopping rule's own
    inputs and against the training loop, which is a STRONGER statement than
    a substring ban that a variable rename would have defeated.
    """
    import inspect

    from hydra_suite.training.sam3_lora import cli

    source = inspect.getsource(cli)
    # Checkpoint selection: still nothing, on any metric.
    assert "best_ap" not in source, "recording only, no selection"

    # The stopping rule's entire input is one already-computed loss number.
    # No AP, no sweep, no F1 can reach it.
    tracker_source = inspect.getsource(cli.EarlyStopTracker)
    for forbidden in ("ap", "sweep", "f1", "iou", "recall", "precision"):
        assert (
            f'"{forbidden}"' not in tracker_source
            and f"'{forbidden}'" not in tracker_source
        ), f"{forbidden}: the stopping rule must read val_loss_mean only"
    assert "val_loss_mean" in tracker_source

    # And the loop feeds it val_loss_mean, from the record it already wrote.
    loop = inspect.getsource(cli.run_training)
    assert 'record["val_loss_mean"]' in loop
    assert "early_stop.observe(" in loop


def test_the_within_run_only_limitation_is_documented_where_it_matters():
    import inspect

    from hydra_suite.core.inference.semantic import detection_metrics
    from hydra_suite.training.sam3_lora import detection_quality

    for module in (detection_metrics, detection_quality):
        text = inspect.getsource(module)
        assert "not comparable" in text.lower()


# ---------------------------------------------------------------------------
# Silent-corruption guards (Fable review)
# ---------------------------------------------------------------------------


def test_a_short_query_run_is_loud_rather_than_misaligning_every_later_tile():
    """An UNDER-run is the dangerous direction: the cursor falls behind and
    every later prediction is scored against the wrong tile's ground truth,
    producing a plausible-looking but wrong AP with no error."""
    from hydra_suite.training.sam3_lora import detection_quality as dq

    descriptors = [_Descriptor([_square(0, 0)], n_neg=0) for _ in range(3)]
    accumulator = dq.DetectionQualityAccumulator(descriptors)

    masks = np.zeros((1, 32, 32), dtype=bool)
    masks[0, 4:16, 4:16] = True

    class _ShortOutputs:
        # Only two of the three planned queries came back.
        output = [[_stub_output(masks, [0.9])], [_stub_output(masks, [0.9])]]

    accumulator.observe(_ShortOutputs())
    stats = accumulator.result()
    assert "ap" not in stats
    assert "ap_error" in stats
    assert "2" in stats["ap_error"] and "3" in stats["ap_error"]


def test_an_overrun_is_still_loud():
    from hydra_suite.training.sam3_lora import detection_quality as dq

    descriptors = [_Descriptor([_square(0, 0)], n_neg=0)]
    accumulator = dq.DetectionQualityAccumulator(descriptors)
    masks = np.zeros((1, 32, 32), dtype=bool)
    masks[0, 4:16, 4:16] = True

    class _LongOutputs:
        output = [[_stub_output(masks, [0.9])], [_stub_output(masks, [0.9])]]

    accumulator.observe(_LongOutputs())
    assert "ap_error" in accumulator.result()


def test_a_dropped_presence_key_is_recorded_not_silently_absorbed():
    """Losing `presence_logit_dec` CHANGES the score definition. It stays
    internally consistent (presence is a per-image scalar), so it is recorded
    as a row flag rather than failing the pass -- but it must leave a trace."""
    from hydra_suite.training.sam3_lora import detection_quality as dq

    masks = np.zeros((1, 32, 32), dtype=bool)
    masks[0, 4:16, 4:16] = True

    descriptors = [_Descriptor([_square(100, 100, size=200)], n_neg=0)]

    with_presence = dq.DetectionQualityAccumulator(descriptors)

    class _Outputs:
        def __init__(self, query):
            self.output = [[query]]

    with_presence.observe(_Outputs(_stub_output(masks, [0.9], presence=0.8)))
    assert with_presence.result()["ap_presence_used"] is True

    without = dq.DetectionQualityAccumulator(descriptors)
    without.observe(_Outputs(_stub_output(masks, [0.9])))
    assert without.result()["ap_presence_used"] is False


def test_extract_predictions_reports_whether_presence_was_applied():
    from hydra_suite.training.sam3_lora import detection_quality as dq

    masks = np.zeros((1, 32, 32), dtype=bool)
    masks[0, 4:16, 4:16] = True

    _p, scores, used = dq.extract_predictions(_stub_output(masks, [0.9], presence=0.5))
    assert used is True and scores == pytest.approx([0.45], abs=1e-4)
    _p, scores, used = dq.extract_predictions(_stub_output(masks, [0.9]))
    assert used is False and scores == pytest.approx([0.9], abs=1e-4)


def test_the_row_marks_its_own_scope_so_the_raw_jsonl_is_self_describing():
    from hydra_suite.training.sam3_lora import detection_quality as dq

    masks = np.zeros((1, 32, 32), dtype=bool)
    masks[0, 4:16, 4:16] = True
    descriptors = [_Descriptor([_square(100, 100, size=200)], n_neg=0)]
    accumulator = dq.DetectionQualityAccumulator(descriptors)

    class _Outputs:
        output = [[_stub_output(masks, [0.9])]]

    accumulator.observe(_Outputs())
    assert accumulator.result()["ap_scope"] == "tile"


# ---------------------------------------------------------------------------
# Batch-size alignment: `outputs.output` is per INTERACTIVITY STAGE, not per
# query. Meta's `SAM3Output` is a `List[List[Dict]]` indexed by
# stage-then-step (`sam3/model/model_misc.py`, `class SAM3Output`), and
# `collate_fn_api` builds one `FindStage`/`BatchedFindTarget` per stage --
# NOT one per datapoint. The batch lives INSIDE the tensors as dim 0
# (`pred_logits` [B, Q, C], `pred_masks` [B, Q, H, W]).
#
# Our tiles all share one `query_processing_order`, so `len(outputs.output)`
# is 1 for EVERY forward regardless of batch size. At batch 1 that happens to
# equal the query count, which is exactly why every test above passed while a
# real batch-8 run recorded
#   "produced 288 queries but the descriptor plan expects 2304".
# These tests must therefore be parametric over batch size or they cannot see
# the bug at all.
# ---------------------------------------------------------------------------


_GRID = 64  # stub mask resolution; RES / _GRID is the extraction scale
_CELL = 16  # grid cells between adjacent stub objects
_EXTENT = 13  # object side in grid cells


def _stub_batch_stage(mask_stacks, score_lists, presence=0.9):
    """One SAM3 last-stage dict for a WHOLE batch: [B, Q, ...] tensors."""
    torch = pytest.importorskip("torch")

    masks = torch.stack(
        [
            torch.where(
                torch.as_tensor(np.asarray(stack)),
                torch.tensor(5.0),
                torch.tensor(-5.0),
            )
            for stack in mask_stacks
        ]
    )
    logits = torch.stack(
        [
            torch.logit(torch.as_tensor(scores, dtype=torch.float32)).reshape(
                len(scores), 1
            )
            for scores in score_lists
        ]
    )
    stage = {"pred_logits": logits, "pred_masks": masks}
    if presence is not None:
        stage["presence_logit_dec"] = torch.logit(
            torch.full((len(mask_stacks), 1), float(presence))
        )
    return stage


class _StageOutputs:
    """`SAM3Output`-shaped: one outer entry per stage, each a list of steps."""

    def __init__(self, stage):
        self.output = [[stage]]


def _distinct_object(index):
    """Grid cell and matching RES-space ground-truth square for one query.

    Every query gets a DIFFERENT location, so a batch iterated in the wrong
    order scores each prediction against another tile's ground truth and the
    match fails -- a count-only assertion would not notice.
    """
    from hydra_suite.training.sam3_lora.detection_quality import RES

    column = (index % 4) * _CELL
    row = (index // 4) * _CELL
    scale = float(RES) / float(_GRID)
    gt = _square(column * scale, row * scale, size=_EXTENT * scale)
    mask = np.zeros((1, _GRID, _GRID), dtype=bool)
    mask[0, row : row + _EXTENT, column : column + _EXTENT] = True
    return mask, gt


@pytest.mark.parametrize(
    "n_queries,batch_size",
    [(16, 1), (16, 4), (16, 8), (10, 8)],  # (10, 8) exercises the ragged tail
)
def test_predictions_stay_aligned_at_every_batch_size(n_queries, batch_size):
    pytest.importorskip("torch")
    from hydra_suite.training.sam3_lora import detection_quality as dq

    objects = [_distinct_object(i) for i in range(n_queries)]
    descriptors = [_Descriptor([gt], n_neg=0) for _mask, gt in objects]
    accumulator = dq.DetectionQualityAccumulator(descriptors)

    for start in range(0, n_queries, batch_size):
        chunk = objects[start : start + batch_size]
        stage = _stub_batch_stage(
            [mask for mask, _gt in chunk], [[0.95] for _ in chunk]
        )
        accumulator.observe(_StageOutputs(stage))

    stats = accumulator.result()
    assert "ap_error" not in stats, stats.get("ap_error")
    assert stats["ap_tiles"] == n_queries
    # Each query's prediction must land on ITS OWN descriptor: a perfect
    # one-to-one prediction set can only score ~1.0 if the mapping is right.
    assert stats["ap"] > 0.99


def test_a_short_batch_dimension_is_still_caught_at_batch_eight():
    """The completeness guard must keep firing when the shortfall is inside
    the batch dimension rather than in the number of forwards."""
    pytest.importorskip("torch")
    from hydra_suite.training.sam3_lora import detection_quality as dq

    objects = [_distinct_object(i) for i in range(8)]
    descriptors = [_Descriptor([gt], n_neg=0) for _mask, gt in objects]
    accumulator = dq.DetectionQualityAccumulator(descriptors)

    # Only six of the eight planned queries came back in the one forward.
    short = objects[:6]
    accumulator.observe(
        _StageOutputs(
            _stub_batch_stage([m for m, _g in short], [[0.95] for _ in short])
        )
    )
    stats = accumulator.result()
    assert "ap" not in stats
    assert "6" in stats["ap_error"] and "8" in stats["ap_error"]


# ---------------------------------------------------------------------------
# AP at full recall (regression: the singular missed/(1-recall) inversion)
# ---------------------------------------------------------------------------

# The confidence sweep of a real 10-epoch SAM3 run's epoch 1, reconstructed
# from its recorded endpoints: recall reaches EXACTLY 1.0 at confidence 0.05
# (matched=771, missed=0) while emitting 3360 extras against 771 ground-truth
# instances, and decays to matched=11 at confidence 0.95. The stored AP was
# 0.9857328145265889 == 760/771 == 1 - 11/771 -- i.e. exactly
# max(recall) - min(recall), the signature of a precision curve pinned at 1.0.
_EPOCH1_MATCHED = [
    771,
    771,
    760,
    700,
    640,
    580,
    520,
    460,
    400,
    340,
    290,
    240,
    200,
    160,
    120,
    90,
    60,
    30,
    11,
]
_EPOCH1_EXTRAS = [
    3360,
    2500,
    1800,
    1300,
    950,
    700,
    520,
    390,
    290,
    215,
    160,
    120,
    90,
    66,
    48,
    34,
    22,
    12,
    4,
]
_EPOCH1_GT = 771


def _sweep_counts(matched, extras, gt=_EPOCH1_GT):
    from hydra_suite.core.inference.semantic.detection_metrics import SweepCounts

    return [
        SweepCounts(
            confidence=round(0.05 * (i + 1), 2), matched=m, extras=e, missed=gt - m
        )
        for i, (m, e) in enumerate(zip(matched, extras))
    ]


def test_full_recall_point_does_not_inflate_ap():
    """A sweep containing a ``missed == 0`` point must still score honestly.

    Pre-fix this returned 0.9857328145265889 (== 760/771), because the
    full-recall point's precision silently defaulted to 1.0 and the monotone
    envelope carried that 1.0 across the whole curve.
    """
    from hydra_suite.core.inference.semantic.detection_metrics import (
        average_precision_from_counts,
    )

    ap = average_precision_from_counts(
        _sweep_counts(_EPOCH1_MATCHED, _EPOCH1_EXTRAS), 100
    )
    assert np.isfinite(ap)
    # 3360 extras against 771 labels at the full-recall point: precision there
    # is 771/4131 ~ 0.187, and no point on this sweep exceeds ~0.73.
    assert 0.0 < ap < 0.75
    assert ap != pytest.approx(760 / 771, abs=1e-9)


def test_over_predicting_sweep_scores_worse_than_a_precise_one():
    """The property that actually failed: extras must cost something.

    Both sweeps reach recall 1.0; the sloppy one pays ~10x the extras at
    every confidence. Pre-fix both collapsed to ``max(recall) - min(recall)``
    and scored IDENTICALLY, so a mere finiteness check would not catch it.
    """
    from hydra_suite.core.inference.semantic.detection_metrics import (
        average_precision_from_counts,
    )

    precise = _sweep_counts(_EPOCH1_MATCHED, [e // 10 for e in _EPOCH1_EXTRAS])
    sloppy = _sweep_counts(_EPOCH1_MATCHED, _EPOCH1_EXTRAS)
    ap_precise = average_precision_from_counts(precise, 100)
    ap_sloppy = average_precision_from_counts(sloppy, 100)
    assert ap_precise > ap_sloppy
    # And not merely by a rounding margin.
    assert ap_precise - ap_sloppy > 0.1


def test_counts_path_matches_missed_path_without_full_recall():
    """No behaviour change where the old inversion was non-singular.

    Every sweep whose recall never reaches 1.0 keeps the AP it always had --
    which is what lets a stored series be read as "only the exact-k/GT
    epochs were corrupted".
    """
    from hydra_suite.core.inference.semantic.detection_metrics import (
        OperatingPoint,
        average_precision,
        average_precision_from_counts,
        precision_recall_curve,
    )

    counts = _sweep_counts(_EPOCH1_MATCHED[2:], _EPOCH1_EXTRAS[2:])  # drops recall==1
    frames = 100
    points = [
        OperatingPoint(
            confidence=c.confidence,
            recall=c.matched / (c.matched + c.missed),
            extra_per_frame=c.extras / frames,
        )
        for c in counts
    ]
    recalls, precisions = precision_recall_curve(
        points, missed_per_frame=[c.missed / frames for c in counts]
    )
    assert average_precision_from_counts(counts, frames) == pytest.approx(
        average_precision(recalls, precisions), rel=1e-12
    )


def test_missed_only_path_recovers_the_scale_at_full_recall():
    """The shared curve helper is fixed for the ``missed_per_frame`` caller too.

    ``total`` is a corpus property, so the full-recall point borrows it from
    the rest of the sweep rather than defaulting to precision 1.0.
    """
    from hydra_suite.core.inference.semantic.detection_metrics import (
        OperatingPoint,
        precision_recall_curve,
    )

    points = [
        OperatingPoint(confidence=0.9, recall=0.5, extra_per_frame=5.0),
        OperatingPoint(confidence=0.1, recall=1.0, extra_per_frame=30.0),
    ]
    _recalls, precisions = precision_recall_curve(points, missed_per_frame=[5.0, 0.0])
    # total = 10/frame; at full recall matched=10, extras=30 -> 0.25.
    assert precisions[0] == pytest.approx(0.5)
    assert precisions[1] == pytest.approx(0.25)
