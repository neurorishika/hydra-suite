"""Deterministic generator for the direct-calibration before-gate corpus.

Task 3 of docs/superpowers/plans/2026-09-06-direct-path-scoring-unification.md.

This script builds a small, hand-and-seed-generated prediction+label corpus
that exercises the geometric cases the D7/D8/D9 rulings are about, and writes
it to ``corpus.json`` next to this file. It contains NO model inference and
NO randomness beyond a pinned ``numpy`` seed used only for the "bulk" filler
case (used to get matched-instance counts above ``MIN_MATCHED_INSTANCES``).

Run with: ``python tests/data/direct_calibration_golden/generate_corpus.py``
from the repo root (or worktree root). Re-running reproduces byte-identical
output because every random draw is seeded and geometry is otherwise fixed
Python literals.

Do NOT edit corpus.json by hand -- edit this generator and re-run it, so the
corpus stays reproducible from source.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20260906
OUT_PATH = Path(__file__).with_name("corpus.json")


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _frame(labels, predictions):
    return {"labels": labels, "predictions": predictions}


def _det(class_id, polygon, confidence=1.0):
    return {"class_id": class_id, "polygon_px": polygon, "confidence": confidence}


def build_d7_case():
    """A correct silhouette whose extent convention inflates area ~1.7x.

    Label (body core): 20x20 square, area 400, centroid (110, 110).
    Prediction (traced silhouette incl. legs/antennae): 40x17 rectangle,
    area 680 == 1.7x the label area, positioned so its centroid AND the
    label's centroid each fall inside the other polygon (mutual
    containment), but the rasterized IoU is below the legacy 0.5 gate.

    Verified by hand: intersection = 20 * 16 = 320, union = 400 + 680 - 320
    = 760, IoU = 320 / 760 ~= 0.421 < 0.5.
    """
    label = _rect(100, 100, 120, 120)  # area 400
    prediction = _rect(90, 104, 130, 121)  # area 40*17=680, 1.7x label area
    return _frame(
        [_det(0, label)],
        [_det(0, prediction)],
    )


def build_d9_case():
    """A blob spanning two animals that the legacy IoU-only gate credits as
    a success against one of them (no area band to reject the oversize).

    L1: 20x20 at (200,200)-(220,220). L2: adjacent 20x20 at (222,200)-
    (242,220). Blob prediction: (198,198)-(224,222), 26x24 = 624 px^2,
    almost fully covering L1 and reaching 2 px into L2.

    Verified by hand: blob/L1 intersection = 400 (L1 fully inside blob),
    union = 624 + 400 - 400 = 624, IoU = 400/624 ~= 0.641 >= 0.5 -> matched.
    blob/L2 intersection = 2*20 = 40, union = 624 + 400 - 40 = 984,
    IoU ~= 0.041 -> not matched, L2 counted missed.
    """
    label_a = _rect(200, 200, 220, 220)
    label_b = _rect(222, 200, 242, 220)
    blob = _rect(198, 198, 224, 222)
    return _frame(
        [_det(0, label_a), _det(0, label_b)],
        [_det(0, blob)],
    )


def build_cluster_steal_case():
    """A dense cluster where an oversized prediction steals a neighbour's
    label from the correct, tighter prediction for that same label.

    L1: (200,200)-(220,220). L2 (adjacent, distinct animal):
    (222,200)-(242,220).
    P_correct (tight, slightly offset box for L1): (205,200)-(225,220).
    P_oversized (covers L1 fully + reaches toward L2): (198,198)-(224,222).

    Verified by hand:
    - P_correct/L1: intersection 15*20=300, union 400+400-300=500,
      IoU=0.6.
    - P_oversized/L1: intersection 400 (L1 fully inside), union
      624+400-400=624, IoU ~= 0.641 -- HIGHER than P_correct's IoU, so the
      legacy descending-IoU greedy sort awards L1 to P_oversized first.
      P_correct then has nowhere left to match and becomes an extra.
    - P_oversized/L2 and P_correct/L2 both fall well under 0.5 (~0.04-0.08),
      so L2 is never matched: L2 is a genuine miss (no prediction covers
      it), independent of the steal.
    """
    label1 = _rect(200, 200, 220, 220)
    label2 = _rect(222, 200, 242, 220)
    p_correct = _rect(205, 200, 225, 220)
    p_oversized = _rect(198, 198, 224, 222)
    return _frame(
        [_det(0, label1), _det(0, label2)],
        [_det(0, p_oversized), _det(0, p_correct)],
    )


def build_degenerate_case():
    """Degenerate / NaN / infinite geometry.

    Includes: a label with fewer than 3 points (must be dropped by
    ``_valid_polygon``), a prediction containing NaN coordinates, a
    prediction containing an infinite coordinate, and one normal matched
    pair so the frame is not entirely empty. This case documents current
    behaviour -- it does not assert a "correct" answer to a bug, only
    records what today's code does when it sees this input.
    """
    normal_label = _rect(300, 300, 310, 310)
    normal_pred = _rect(300, 300, 310, 310)
    degenerate_label_2pt = [[400, 400], [410, 410]]  # < 3 points
    nan_pred = [[420, 420], [430.0, float("nan")], [430, 430], [420, 430]]
    inf_pred = [[440, 440], [float("inf"), 440], [450, 450], [440, 450]]
    return _frame(
        [
            _det(0, normal_label),
            _det(0, degenerate_label_2pt),
        ],
        [
            _det(0, normal_pred),
            _det(0, nan_pred),
            _det(0, inf_pred),
        ],
    )


def build_bulk_case(task: str, class_id: int, n_frames: int, per_frame: int, rng):
    """Many easy, well-separated near-perfect matches, seeded.

    Purpose: push the total matched-instance count for this task's bulk
    case comfortably past ``MIN_MATCHED_INSTANCES`` (60) so
    ``recommend_balanced`` eligibility is exercised realistically, without
    hand-writing 60+ literal boxes. Boxes are placed on a fixed-seed jittered
    grid with generous spacing so they never collide with each other or with
    the hand-built cases above (which all live below x=500). This case is
    otherwise geometrically uninteresting -- it is filler, not a ruling case.
    """
    frames = []
    grid_stride = 60
    for frame_index in range(n_frames):
        labels = []
        predictions = []
        for item_index in range(per_frame):
            col = item_index % 8
            row = item_index // 8
            base_x = 1000 + col * grid_stride
            base_y = 1000 + frame_index * (grid_stride * 4) + row * grid_stride
            jitter = rng.integers(-3, 4, size=4)  # x0,y0,x1,y1 jitter for prediction
            x0, y0, x1, y1 = base_x, base_y, base_x + 20, base_y + 20
            label = _rect(x0, y0, x1, y1)
            px0, py0 = x0 + int(jitter[0]), y0 + int(jitter[1])
            px1, py1 = x1 + int(jitter[2]), y1 + int(jitter[3])
            if px1 <= px0:
                px1 = px0 + 18
            if py1 <= py0:
                py1 = py0 + 18
            prediction = _rect(px0, py0, px1, py1)
            labels.append(_det(class_id, label))
            predictions.append(_det(class_id, prediction))
        frames.append(_frame(labels, predictions))
    return frames


def main():
    rng = np.random.default_rng(SEED)

    cases = []

    for task in ("detect", "obb", "segment"):
        cases.append(
            {
                "name": f"d7_extent_convention_inflation_{task}",
                "task": task,
                "description": (
                    "Correct silhouette, ~1.7x labelled body-core area; "
                    "IoU < 0.5 under the legacy gate but the two centroids "
                    "are mutually contained."
                ),
                "frames": [build_d7_case()],
            }
        )
        cases.append(
            {
                "name": f"d9_blob_spans_two_animals_{task}",
                "task": task,
                "description": (
                    "One oversized blob prediction spans two labelled "
                    "animals; legacy IoU-only gate credits it as a match "
                    "against one of them."
                ),
                "frames": [build_d9_case()],
            }
        )
        cases.append(
            {
                "name": f"dense_cluster_label_steal_{task}",
                "task": task,
                "description": (
                    "An oversized prediction out-scores the correct, "
                    "tighter prediction for the same label under "
                    "descending-IoU greedy sort, stealing the label and "
                    "turning the correct prediction into an extra."
                ),
                "frames": [build_cluster_steal_case()],
            }
        )
        cases.append(
            {
                "name": f"degenerate_geometry_{task}",
                "task": task,
                "description": (
                    "Sub-triangle label, NaN-coordinate prediction, and "
                    "infinite-coordinate prediction alongside one normal "
                    "match. Characterizes current behaviour; does not "
                    "assert correctness of a bug."
                ),
                "frames": [build_degenerate_case()],
            }
        )

    # Bulk filler, seeded, per task -- gives score_frames()/recommend_balanced()
    # realistic matched-instance volume (>= MIN_MATCHED_INSTANCES = 60).
    for task, class_id in (("detect", 0), ("obb", 0), ("segment", 0)):
        bulk_frames = build_bulk_case(
            task=task, class_id=class_id, n_frames=5, per_frame=16, rng=rng
        )
        cases.append(
            {
                "name": f"bulk_easy_matches_{task}",
                "task": task,
                "description": (
                    "Seeded filler of well-separated near-perfect matches "
                    "(80 instances/task) so score_frames()/recommend_balanced() "
                    "clear MIN_MATCHED_INSTANCES=60. Not a ruling case."
                ),
                "frames": bulk_frames,
            }
        )

    payload = {"seed": SEED, "cases": cases}
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUT_PATH} with {len(cases)} cases")


if __name__ == "__main__":
    main()
