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


def _rotate(points, angle_deg, center):
    """Rotate a list of [x, y] points about ``center`` by ``angle_deg``.

    Pure geometry helper, no randomness. Used to build non-axis-aligned
    instances so ``detect``'s AABB reduction (``_as_task_polygon``) is
    genuinely exercised -- on axis-aligned input, AABB reduction is the
    identity transform and cannot diverge from full-polygon IoU.
    """
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    pts = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    rotated = np.stack(
        [pts[:, 0] * cos_t - pts[:, 1] * sin_t, pts[:, 0] * sin_t + pts[:, 1] * cos_t],
        axis=1,
    )
    return (rotated + np.asarray(center, dtype=np.float64)).tolist()


def _round2(points):
    return [[round(x, 3), round(y, 3)] for x, y in points]


def _horseshoe(cx, cy, r_outer, r_inner, gap_deg, n=24):
    """A C-shaped (annulus-sector) polygon: a non-convex mask-style
    silhouette whose area centroid AND vertex mean both fall in the empty
    gap/hole, outside the polygon itself. Verified numerically (see
    ``build_nonconvex_centroid_outside_case`` docstring) against
    ``core.inference.match_geometry.representative_point``, which is
    exactly the function this case exists to exercise.
    """
    start = gap_deg / 2
    end = 360 - gap_deg / 2
    theta_outer = np.deg2rad(np.linspace(start, end, n))
    theta_inner = np.deg2rad(np.linspace(end, start, n))
    outer = np.stack(
        [cx + r_outer * np.cos(theta_outer), cy + r_outer * np.sin(theta_outer)],
        axis=1,
    )
    inner = np.stack(
        [cx + r_inner * np.cos(theta_inner), cy + r_inner * np.sin(theta_inner)],
        axis=1,
    )
    return np.concatenate([outer, inner], axis=0).tolist()


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


def build_d7_nonconvex_rotated_case():
    """A rotated, non-convex version of the D7 1.7x-area case.

    Fix-round addition: the plain D7 rectangle proves the thresholding half
    of D7 but nothing else -- it is axis-aligned (so ``detect``'s AABB
    reduction is the identity transform on it) and convex (so it cannot
    exercise ``representative_point``'s non-convex fallback path). This
    case makes the headline case representative of a real traced mask
    silhouette: a body-core rectangle with one non-convex "leg" spike
    protruding from it, at the SAME 1.7x area ratio as the plain case,
    then both polygons rotated 35 degrees about the label's centroid so
    ``detect`` (AABB-reduced) genuinely differs from ``obb``/``segment``
    (full polygon).

    Verified by hand (pre-rotation, since shoelace area and IoU are
    rotation-invariant): label is a 20x20 square, area 400. Prediction is
    a 7-vertex polygon (rectangle (90,104)-(126,121) with a spike apex at
    (160,112)), shoelace area 680.0 == 1.7x the label, and NOT convex
    (two reflex vertices where the spike rejoins the rectangle edge).
    Rotated IoU (both polygons rotated 35 deg about (110,110)) ~= 0.423 <
    0.5 -- still a simultaneous miss+extra under the legacy gate. Both
    representative points (area centroid, since the prediction's centroid
    lands inside its own body -- the spike is thin relative to the body)
    fall inside the other polygon, confirmed against
    ``core.inference.match_geometry.representative_point`` at generation
    time; that mutual-containment property is what Task 4's matcher will
    rely on. This case does NOT test the pole-of-inaccessibility fallback
    -- see ``build_nonconvex_centroid_outside_case`` for that.
    """
    label = _rect(100, 100, 120, 120)  # area 400
    prediction = [
        [90, 104],
        [126, 104],
        [126, 110],
        [160, 112],
        [126, 114],
        [126, 121],
        [90, 121],
    ]  # shoelace area 680.0 == 1.7x, non-convex (reflex at the spike base)
    center = (110, 110)
    label_r = _round2(_rotate(label, 35, center))
    prediction_r = _round2(_rotate(prediction, 35, center))
    return _frame(
        [_det(0, label_r)],
        [_det(0, prediction_r)],
    )


def build_rotated_task_divergence_case():
    """Rotated rectangles chosen so ``detect``'s AABB reduction crosses the
    0.5 IoU gate in the OPPOSITE direction from ``obb``/``segment`` --
    i.e. this single case's match/miss OUTCOME (not just its IoU number)
    differs by task. This is the direct proof that "three-task coverage"
    is not one task's arithmetic three times over.

    Fix-round addition (Finding 1). Both label and prediction are 20x20
    squares, one offset by (-6, -2) from the other, both rotated 15
    degrees about a shared center (700, 700).

    Verified by hand at generation time: full-polygon IoU (what ``obb``/
    ``segment`` score) ~= 0.464 -- BELOW the 0.5 gate, so those two tasks
    score this as a miss + an extra. The same two rotated squares reduced
    to their axis-aligned bounding quads first (what ``detect`` scores,
    per ``_as_task_polygon``) have IoU ~= 0.511 -- AT OR ABOVE the gate,
    so ``detect`` scores this as a match. Recorded once here in
    ``obb``-native form; the per-task loop in ``main`` reuses the same
    geometry for all three tasks and lets ``match_frame``'s own
    ``_as_task_polygon`` perform the ``detect`` reduction, so the
    divergence is measured by the scorer itself, not asserted by the
    generator.
    """
    label = _rect(690, 690, 710, 710)
    prediction = _rect(684, 688, 704, 708)
    center = (700, 700)
    label_r = _round2(_rotate(label, 15, center))
    prediction_r = _round2(_rotate(prediction, 15, center))
    return _frame(
        [_det(0, label_r)],
        [_det(0, prediction_r)],
    )


def build_nonconvex_centroid_outside_case():
    """A mask-style non-convex silhouette (a C-shaped/horseshoe outline,
    standing in for a body-plus-limbs trace with a concave bite) whose
    AREA CENTROID -- not just its vertex mean -- falls OUTSIDE the polygon.

    Fix-round addition (Finding 2). This is the D7 ruling's actual
    containment-robustness subject matter: ``representative_point``
    exists specifically because a real traced silhouette's area centroid
    can land in a concavity outside the shape, which would wrongly fail a
    containment gate that trusted the naive centroid. Nothing in the
    corpus before this fix so much as attempted a shape where the AREA
    centroid (not merely the vertex mean) is outside.

    Verified numerically at generation time against
    ``core.inference.match_geometry.representative_point``/``_contains``:
    for this 48-vertex horseshoe (outer radius 50, inner radius 30, 100
    degree gap, centered at (800, 800)):
    - the naive vertex mean is OUTSIDE the polygon;
    - the ``cv2.moments`` AREA centroid is ALSO outside (both fall in the
      annulus's hollow center);
    - ``representative_point`` falls through moments -> pole of
      inaccessibility and returns (815.0, 763.0), which IS inside the
      polygon (confirmed via ``_contains``).
    The label is a small 12x12 square placed exactly at that verified
    representative point, so a correct future matcher can find it, while
    the legacy IoU-only gate (which never calls ``representative_point``)
    scores this pair at IoU ~= 0.04 -- a miss + an extra, same as the
    other under-threshold cases, since this case predates D7 adoption.
    """
    silhouette = _round2(_horseshoe(800, 800, 50, 30, 100, n=24))
    label = _rect(809, 757, 821, 769)
    return _frame(
        [_det(0, label)],
        [_det(0, silhouette)],
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
                "name": f"d7_nonconvex_rotated_{task}",
                "task": task,
                "description": (
                    "Rotated, non-convex version of the D7 1.7x-area case "
                    "(a body-core rectangle plus one leg-spike, rotated 35 "
                    "degrees) -- makes the headline D7 case representative "
                    "of a real traced silhouette instead of an axis-"
                    "aligned rectangle that is merely too big."
                ),
                "frames": [build_d7_nonconvex_rotated_case()],
            }
        )
        cases.append(
            {
                "name": f"rotated_task_divergence_{task}",
                "task": task,
                "description": (
                    "Rotated squares whose full-polygon IoU (obb/segment, "
                    "~0.464, below the gate) and AABB-reduced IoU (detect, "
                    "~0.511, at/above the gate) fall on OPPOSITE sides of "
                    "the legacy 0.5 threshold -- proves detect/obb/segment "
                    "produce genuinely different match outcomes, not just "
                    "different IoU numbers on the same outcome."
                ),
                "frames": [build_rotated_task_divergence_case()],
            }
        )
        cases.append(
            {
                "name": f"nonconvex_centroid_outside_{task}",
                "task": task,
                "description": (
                    "Non-convex horseshoe/C-shaped silhouette whose AREA "
                    "centroid (not just its vertex mean) falls outside the "
                    "polygon, in the concave gap -- the exact failure mode "
                    "representative_point()'s pole-of-inaccessibility "
                    "fallback exists to handle. Label sits at the verified "
                    "representative point."
                ),
                "frames": [build_nonconvex_centroid_outside_case()],
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
