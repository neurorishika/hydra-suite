"""Score the frozen direct-calibration corpus under a GIVEN checkout.

Task 5 of docs/superpowers/plans/2026-09-06-direct-path-scoring-unification.md.

Unlike ``generate_before.py`` (a one-time legacy snapshot), this script is
STAGE-AGNOSTIC on purpose: the same script text is run against four different
``src/`` trees -- the three intermediate ruling commits (D7, then +D9, then
+D8) and HEAD -- so that each ruling's contribution to the numbers is
MEASURED rather than asserted. It is also the scorer the standing
characterization test re-runs against ``after.json``.

Usage::

    PYTHONPATH=<tree>/src python generate_after.py \
        --out /somewhere/stage_a.json [--stage d7] [--no-area-band]

Provenance is recorded IN the payload (resolved ``hydra_suite`` package path
plus the git sha of the tree that package came from) because the failure mode
of this whole exercise is silently scoring HEAD four times: an editable
install shadows ``PYTHONPATH`` and every "stage" comes out identical. If the
sha in a staged payload is not the stage's commit, the attribution is fiction.

Uses only pure functions from ``core/inference/direct_calibration.py``. No
model inference, no ultralytics.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np

import hydra_suite
from hydra_suite.core.inference import direct_calibration as dc

HERE = Path(__file__).parent
CORPUS_PATH = HERE / "corpus.json"


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def _package_root() -> Path:
    return Path(hydra_suite.__file__).resolve().parent


def _git_sha(path: Path) -> str:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(path), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - provenance best effort
        return "unknown"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(path),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - provenance best effort
        # Can't tell if the tree is dirty -- don't silently stamp a clean
        # sha over an unknown working-tree state.
        return f"{sha}-unknown-dirty-state"
    return f"{sha}-dirty" if dirty else sha


# --------------------------------------------------------------------------
# corpus -> detections
# --------------------------------------------------------------------------
def _detection(record: dict):
    return dc.CalibrationDetection(
        class_id=record["class_id"],
        polygon_px=np.asarray(record["polygon_px"], dtype=np.float32),
        confidence=record.get("confidence", 1.0),
    )


def _frame_pair(frame: dict):
    labels = [_detection(r) for r in frame["labels"]]
    predictions = [_detection(r) for r in frame["predictions"]]
    return predictions, labels


def _supports(func, name: str) -> bool:
    return name in inspect.signature(func).parameters


def _band_fields(band):
    """Serialize an ``AreaBand`` (min_px2/max_px2/median_px2/n_labels)."""
    if band is None:
        return None
    return {
        "min": float(band.min_px2),
        "max": float(band.max_px2),
        "median": float(band.median_px2),
        "n": int(band.n_labels),
    }


def _score_dict(score, *, per_frame: bool):
    payload = {
        "matched": score.matched,
        "missed": score.missed,
        "extra": score.extra,
        "duplicate": score.duplicate,
        "mean_iou": score.mean_iou,
        "mean_quality": float(getattr(score, "mean_quality", 0.0)),
    }
    if not per_frame:
        payload.update(
            {
                "frames": score.frames,
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
            }
        )
    return payload


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_payload(corpus: dict, *, use_area_band: bool, stage: str) -> dict:
    cases = corpus["cases"]
    case_results = []

    band_supported = _supports(dc.match_frame, "area_band") and hasattr(
        dc, "fit_calibration_area_band"
    )
    band_active = use_area_band and band_supported

    for case in cases:
        task = case["task"]
        frame_pairs = [_frame_pair(f) for f in case["frames"]]

        # D9: ONE band pooled over this evidence set's whole label set,
        # mirroring the job's single fit at
        # detectkit/jobs/direct_calibration.py:842.
        band = None
        if band_active:
            band = dc.fit_calibration_area_band(
                [labels for _, labels in frame_pairs], task=task
            )

        extra = {"area_band": band} if band_active else {}
        per_frame = [
            dc.match_frame(predictions, labels, task=task, **extra)
            for predictions, labels in frame_pairs
        ]
        aggregate = dc.score_frames(frame_pairs, task=task, **extra)

        case_results.append(
            {
                "name": case["name"],
                "task": task,
                "area_band": _band_fields(band),
                "per_frame": [_score_dict(s, per_frame=True) for s in per_frame],
                "aggregate": _score_dict(aggregate, per_frame=False),
            }
        )

    # --- recommend_balanced() exercise ----------------------------------
    # Same synthetic operating-point construction as generate_before.py, so
    # the recommender comparison is rule-vs-rule on IDENTICAL candidates.
    # ONE deliberate difference: mean_quality is threaded onto the synthetic
    # scores. generate_before.py predates the field, so it defaulted to 0.0 --
    # under D8's quality floor (>=0.35) that would refuse every candidate and
    # the demo would show a construction artifact instead of the rule. The
    # legacy F1 recommender ignores mean_quality entirely, so seeding it does
    # not change what the legacy rule would have chosen; comparability holds.
    bulk_obb = next(c for c in case_results if c["name"] == "bulk_easy_matches_obb")
    bulk_agg = bulk_obb["aggregate"]

    score_field_names = {f.name for f in dataclasses.fields(dc.CalibrationScore)}
    point_field_names = {f.name for f in dataclasses.fields(dc.DirectCalibrationPoint)}

    def _variant(label, matched, missed, extra_count, mean_iou, seconds, failed=""):
        precision = matched / (matched + extra_count) if matched + extra_count else 0.0
        recall = matched / (matched + missed) if matched + missed else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        score_kwargs = dict(
            frames=bulk_agg["frames"],
            matched=matched,
            missed=missed,
            extra=extra_count,
            duplicate=0,
            precision=precision,
            recall=recall,
            f1=f1,
            mean_iou=mean_iou,
        )
        if "mean_quality" in score_field_names:
            # Perturbed in lockstep with mean_iou, same as the before-side
            # construction does for mean_iou.
            delta = bulk_agg["mean_iou"] - mean_iou
            score_kwargs["mean_quality"] = max(
                0.0, float(bulk_agg.get("mean_quality", 0.0)) - delta
            )
        score = dc.CalibrationScore(**score_kwargs)
        point_kwargs = dict(
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
        point_kwargs = {k: v for k, v in point_kwargs.items() if k in point_field_names}
        return dc.DirectCalibrationPoint(**point_kwargs)

    synthetic_points = [
        _variant(
            "synthetic_best_slow",
            matched=bulk_agg["matched"],
            missed=bulk_agg["missed"],
            extra_count=bulk_agg["extra"],
            mean_iou=bulk_agg["mean_iou"],
            seconds=0.90,
        ),
        _variant(
            "synthetic_near_best_faster",
            matched=bulk_agg["matched"] - 1,
            missed=bulk_agg["missed"] + 1,
            extra_count=bulk_agg["extra"],
            mean_iou=max(0.0, bulk_agg["mean_iou"] - 0.01),
            seconds=0.50,
        ),
        _variant(
            "synthetic_sparse_dominated",
            matched=bulk_agg["matched"] - 10,
            missed=bulk_agg["missed"] + 10,
            extra_count=bulk_agg["extra"] + 8,
            mean_iou=max(0.0, bulk_agg["mean_iou"] - 0.05),
            seconds=0.60,
        ),
        _variant(
            "synthetic_undersampled",
            matched=10,
            missed=0,
            extra_count=0,
            mean_iou=0.95,
            seconds=0.05,
        ),
        _variant(
            "synthetic_failed",
            matched=0,
            missed=bulk_agg["matched"] + bulk_agg["missed"],
            extra_count=0,
            mean_iou=0.0,
            seconds=0.10,
            failed="ran out of memory",
        ),
    ]

    chosen, explanation = dc.recommend_balanced(synthetic_points)

    recommendation_result = {
        "rule_id": dc.RECOMMENDATION_RULE_ID,
        "rule_text": dc.RECOMMENDATION_RULE,
        "candidate_labels": [p.label for p in synthetic_points],
        "chosen_label": chosen.label if chosen else None,
        "chosen_f1": chosen.score.f1 if chosen else None,
        "chosen_recall": chosen.score.recall if chosen else None,
        "chosen_mean_quality": (
            float(getattr(chosen.score, "mean_quality", 0.0)) if chosen else None
        ),
        "chosen_seconds_per_frame": chosen.seconds_per_frame if chosen else None,
        "explanation": explanation,
    }

    package_root = _package_root()
    return {
        "stage": stage,
        "generated_at_commit": _git_sha(package_root),
        "hydra_suite_package_root": str(package_root),
        "scoring_rules": {
            "matcher": (
                "shared containment matcher "
                "(representative point + containment, quality ranking, no IoU gate)"
                if _supports(dc.match_frame, "area_band")
                else "hard IoU >= 0.5, class-aware, descending-IoU greedy one-to-one"
            ),
            "area_band": (
                "fitted per evidence set via fit_calibration_area_band"
                if band_active
                else None
            ),
            "recommender": dc.RECOMMENDATION_RULE_ID,
            "recommender_rule_id": dc.RECOMMENDATION_RULE_ID,
            "min_matched_instances": getattr(dc, "MIN_MATCHED_INSTANCES", None),
            "min_recall": getattr(dc, "MIN_RECALL", None),
            "min_mean_quality": getattr(dc, "MIN_MEAN_QUALITY", None),
        },
        "cases": case_results,
        "recommend_balanced_demo": recommendation_result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--out", type=Path, default=HERE / "after.json")
    parser.add_argument("--stage", default="after")
    parser.add_argument(
        "--no-area-band",
        action="store_true",
        help="Score without D9's shape prior (used for the D7-only stage).",
    )
    parser.add_argument(
        "--expect-src",
        type=Path,
        default=None,
        help="Fail unless hydra_suite resolves under this src root. Guards "
        "against an editable install shadowing PYTHONPATH.",
    )
    args = parser.parse_args()

    if args.expect_src is not None:
        resolved = _package_root()
        expected = args.expect_src.resolve()
        if expected not in resolved.parents:
            raise SystemExit(
                f"PROVENANCE FAILURE: hydra_suite resolved to {resolved}, "
                f"which is not under the requested src root {expected}. "
                "An editable install is shadowing PYTHONPATH; the staged "
                "attribution would be fiction."
            )

    corpus = json.loads(args.corpus.read_text())
    payload = build_payload(
        corpus, use_area_band=not args.no_area_band, stage=args.stage
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {args.out} | stage={payload['stage']} "
        f"commit={payload['generated_at_commit']} "
        f"src={payload['hydra_suite_package_root']}"
    )


if __name__ == "__main__":
    main()
