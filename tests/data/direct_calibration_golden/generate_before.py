"""Score the frozen corpus under the CURRENT (legacy, pre-2026-09-06) direct
calibration rules, and write the result to ``before.json``.

Task 3 of docs/superpowers/plans/2026-09-06-direct-path-scoring-unification.md.

This is a ONE-TIME snapshot generator. It is not a standing test: the plan's
before/after design explicitly forbids asserting these numbers against live
code, because Task 4 changes the scoring rules on purpose. Re-run this
script only to regenerate ``before.json`` from a checkout of the exact
commit recorded in its own ``generated_at_commit`` field (see the report for
that sha) -- running it after Task 4 lands would silently capture the NEW
rules under the OLD label, which is exactly the provenance failure the plan
warns about.

Uses only the pure functions ``match_frame``/``score_frames``/
``recommend_balanced`` from ``core/inference/direct_calibration.py`` --
no model inference, no ultralytics.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hydra_suite.core.inference.direct_calibration import (
    RECOMMENDATION_RULE,
    RECOMMENDATION_RULE_ID,
    CalibrationDetection,
    DirectCalibrationPoint,
    match_frame,
    recommend_balanced,
    score_frames,
)

HERE = Path(__file__).parent
CORPUS_PATH = HERE / "corpus.json"
OUT_PATH = HERE / "before.json"


def _detection(record: dict) -> CalibrationDetection:
    import numpy as np

    return CalibrationDetection(
        class_id=record["class_id"],
        polygon_px=np.asarray(record["polygon_px"], dtype=np.float32),
        confidence=record.get("confidence", 1.0),
    )


def _frame_pair(frame: dict):
    labels = [_detection(r) for r in frame["labels"]]
    predictions = [_detection(r) for r in frame["predictions"]]
    return predictions, labels


def _git_sha() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()
    )


def main():
    corpus = json.loads(CORPUS_PATH.read_text())
    cases = corpus["cases"]

    case_results = []
    for case in cases:
        task = case["task"]
        frame_pairs = [_frame_pair(f) for f in case["frames"]]

        per_frame = [
            match_frame(predictions, labels, iou_threshold=0.5, task=task)
            for predictions, labels in frame_pairs
        ]
        aggregate = score_frames(frame_pairs, iou_threshold=0.5, task=task)

        case_results.append(
            {
                "name": case["name"],
                "task": task,
                "per_frame": [
                    {
                        "matched": s.matched,
                        "missed": s.missed,
                        "extra": s.extra,
                        "duplicate": s.duplicate,
                        "mean_iou": s.mean_iou,
                    }
                    for s in per_frame
                ],
                "aggregate": {
                    "frames": aggregate.frames,
                    "matched": aggregate.matched,
                    "missed": aggregate.missed,
                    "extra": aggregate.extra,
                    "duplicate": aggregate.duplicate,
                    "precision": aggregate.precision,
                    "recall": aggregate.recall,
                    "f1": aggregate.f1,
                    "mean_iou": aggregate.mean_iou,
                },
            }
        )

    # --- recommend_balanced() exercise ----------------------------------
    # recommend_balanced() operates on already-scored DirectCalibrationPoint
    # rows from a confidence sweep, not raw frames. There is no sweep here
    # (no model, no confidence axis), so we build a small synthetic set of
    # operating points DIRECTLY from the three bulk-easy-matches cases
    # (one per task) by perturbing their aggregate scores/timings by hand,
    # labelled clearly as synthetic. This exercises MIN_MATCHED_INSTANCES
    # eligibility and the Pareto/F1-tolerance/fastest selection with a
    # real Pareto frontier, without inventing a second scorer.
    bulk_obb = next(c for c in case_results if c["name"] == "bulk_easy_matches_obb")
    bulk_agg = bulk_obb["aggregate"]

    from hydra_suite.core.inference.direct_calibration import CalibrationScore

    def _variant(label, matched, missed, extra, mean_iou, seconds, failed=""):
        precision = matched / (matched + extra) if matched + extra else 0.0
        recall = matched / (matched + missed) if matched + missed else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        score = CalibrationScore(
            frames=bulk_agg["frames"],
            matched=matched,
            missed=missed,
            extra=extra,
            duplicate=0,
            precision=precision,
            recall=recall,
            f1=f1,
            mean_iou=mean_iou,
        )
        return DirectCalibrationPoint(
            label=label,
            enabled=True,
            geometry_mode="sliced",
            tile_width=640,
            tile_height=640,
            overlap=0.2,
            object_tile_fraction=0.1,
            max_detections=300,
            tiles_per_frame=4,
            seconds_per_frame=seconds,
            confidence=0.25,
            merge_policy="nms",
            merge_metric="iou",
            merge_threshold=0.5,
            merge_backend="numpy",
            score=score,
            failed_reason=failed,
            candidate_index=len(label),
        )

    # Fix-round addition (Finding 2): the original 6-point set never put two
    # points inside F1_TOLERANCE=0.01 of each other at different speeds, so
    # recommend_balanced()'s Pareto-frontier + F1-tolerance + fastest-wins
    # branches were never actually decided by the tie-break -- one point was
    # simply dominant. This set is constructed so that:
    #   - synthetic_best_slow has the best F1 (1.0) but is the SLOWEST.
    #   - synthetic_near_best_faster has F1 ~= 0.994 (within the 0.01
    #     tolerance of 1.0) and is almost twice as fast (0.50s vs 0.90s).
    #     Because it is faster on the "seconds" cost axis, it is NOT
    #     Pareto-dominated by synthetic_best_slow (dominance requires being
    #     beaten or tied on EVERY axis) -- both survive the Pareto filter,
    #     both clear the F1-tolerance band, and the fastest of the two,
    #     synthetic_near_best_faster, is the one recommend_balanced() must
    #     pick. If this variant ever again resolves to the slow point, the
    #     tie-break is not doing what the rule text claims.
    #   - synthetic_sparse_dominated is strictly worse than
    #     synthetic_near_best_faster on every cost axis (more missed, more
    #     extra, AND slower) so it is genuinely Pareto-dominated, not merely
    #     eligibility-excluded.
    synthetic_points = [
        _variant(
            "synthetic_best_slow",
            matched=bulk_agg["matched"],
            missed=bulk_agg["missed"],
            extra=bulk_agg["extra"],
            mean_iou=bulk_agg["mean_iou"],
            seconds=0.90,
        ),
        _variant(
            "synthetic_near_best_faster",
            matched=bulk_agg["matched"] - 1,
            missed=bulk_agg["missed"] + 1,
            extra=bulk_agg["extra"],
            mean_iou=max(0.0, bulk_agg["mean_iou"] - 0.01),
            seconds=0.50,
        ),
        # Strictly worse than synthetic_near_best_faster on every cost axis
        # (missed, extra, AND seconds) -- genuinely Pareto-dominated.
        _variant(
            "synthetic_sparse_dominated",
            matched=bulk_agg["matched"] - 10,
            missed=bulk_agg["missed"] + 10,
            extra=bulk_agg["extra"] + 8,
            mean_iou=max(0.0, bulk_agg["mean_iou"] - 0.05),
            seconds=0.60,
        ),
        # Below MIN_MATCHED_INSTANCES -- must be excluded by eligibility
        # regardless of how good its rates look.
        _variant(
            "synthetic_undersampled",
            matched=10,
            missed=0,
            extra=0,
            mean_iou=0.95,
            seconds=0.05,
        ),
        # Explicitly failed candidate -- must be excluded by eligibility.
        _variant(
            "synthetic_failed",
            matched=0,
            missed=bulk_agg["matched"] + bulk_agg["missed"],
            extra=0,
            mean_iou=0.0,
            seconds=0.10,
            failed="ran out of memory",
        ),
    ]

    chosen, explanation = recommend_balanced(synthetic_points)

    recommendation_result = {
        "rule_id": RECOMMENDATION_RULE_ID,
        "rule_text": RECOMMENDATION_RULE,
        "candidate_labels": [p.label for p in synthetic_points],
        "chosen_label": chosen.label if chosen else None,
        "chosen_f1": chosen.score.f1 if chosen else None,
        "chosen_seconds_per_frame": chosen.seconds_per_frame if chosen else None,
        "explanation": explanation,
    }

    payload = {
        "generated_at_commit": _git_sha(),
        "scoring_rules": {
            "matcher": "hard IoU >= 0.5, class-aware, descending-IoU greedy one-to-one",
            "iou_threshold": 0.5,
            "area_band": None,
            "recommender": "recommend_balanced (F1-tolerance + Pareto + fastest)",
            "recommender_rule_id": RECOMMENDATION_RULE_ID,
        },
        "cases": case_results,
        "recommend_balanced_demo": recommendation_result,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUT_PATH} at commit {payload['generated_at_commit']}")


if __name__ == "__main__":
    main()
