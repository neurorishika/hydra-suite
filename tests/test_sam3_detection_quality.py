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

    polys, scores = dq.extract_predictions(output, target_size=64)

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
    import inspect

    from hydra_suite.training.sam3_lora import cli

    source = inspect.getsource(cli)
    for forbidden in ("best_ap", "early_stop", "patience"):
        assert forbidden not in source, f"{forbidden}: recording only, no selection"


def test_the_within_run_only_limitation_is_documented_where_it_matters():
    import inspect

    from hydra_suite.core.inference.semantic import detection_metrics
    from hydra_suite.training.sam3_lora import detection_quality

    for module in (detection_metrics, detection_quality):
        text = inspect.getsource(module)
        assert "not comparable" in text.lower()
