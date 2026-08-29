import numpy as np

from hydra_suite.core.inference.semantic.calibration import (
    CONFIDENCE_GRID,
    CalibrationPoint,
    match_one_to_one,
    recommend,
)
from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.utils.geometry_levels import GeometryLevel


def _pt(
    *,
    frac=0.05,
    tiles=16,
    conf=0.2,
    missed=1.0,
    extra=5.0,
    recall=0.95,
    matched=70,
    secs=22.0,
):
    return CalibrationPoint(
        tile_fraction=frac,
        tile_px=None if frac is None else int(80 / frac),
        tiles_per_frame=tiles,
        seconds_per_frame=secs,
        confidence=conf,
        missed_per_frame=missed,
        extra_per_frame=extra,
        recall=recall,
        n_matched=matched,
    )


def _sq(cx, cy, side=20.0):
    h = side / 2.0
    return np.array(
        [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]],
        dtype=np.float32,
    )


class _CountingLabeler:
    """Detects nothing; counts one call per tile. Deliberately local to this
    file rather than imported from another test module -- ``tests/`` is not a
    package here."""

    def __init__(self):
        self.calls = 0

    @property
    def name(self):
        return "counting"

    def label_image(
        self, image_bgr, prompt, *, confidence_threshold=0.0, max_instances=0
    ):
        self.calls += 1
        return []


def test_each_label_and_prediction_is_used_at_most_once():
    labels = [_sq(100, 100)]
    # Two predictions both centred near the one label.
    preds = [_sq(101, 100), _sq(99, 101)]
    pairs = match_one_to_one(preds, labels)
    assert len(pairs) == 1
    assert len({p for p, _ in pairs}) == 1 and len({g for _, g in pairs}) == 1


def test_dense_cluster_does_not_let_a_blob_steal_a_neighbours_label():
    # Two labels 30 px apart; one oversized prediction centred between them.
    labels = [_sq(100, 100), _sq(130, 100)]
    blob = _sq(115, 100, side=70.0)
    pairs = match_one_to_one([blob], labels)
    assert len(pairs) == 1  # it matches ONE label, not both


def test_containment_gate_rejects_a_far_away_prediction():
    labels = [_sq(100, 100)]
    preds = [_sq(400, 400)]
    assert match_one_to_one(preds, labels) == []


def test_matching_works_for_aabb_obb_and_polygon_labels():
    label_aabb = _sq(50, 50)
    label_obb = np.array([[80, 78], [96, 82], [92, 98], [76, 94]], dtype=np.float32)
    label_poly = np.array(
        [[150, 150], [162, 148], [168, 158], [158, 168], [148, 162]], dtype=np.float32
    )
    preds = [_sq(51, 50), _sq(86, 88), _sq(157, 157)]
    pairs = match_one_to_one(preds, [label_aabb, label_obb, label_poly])
    assert len(pairs) == 3


def test_confidence_grid_is_ascending_and_bounded():
    assert list(CONFIDENCE_GRID) == sorted(CONFIDENCE_GRID)
    assert 0.0 < CONFIDENCE_GRID[0] and CONFIDENCE_GRID[-1] < 1.0


def test_recommend_refuses_below_the_minimum_matched_count():
    points = [_pt(conf=c, matched=3) for c in (0.2, 0.4)]
    best, reason = recommend(points)
    assert best is None
    assert "insufficient" in reason.lower()


def test_recall_floor_excludes_the_f1_optimal_point():
    # The recall floor -- not a tie-break between comparable points -- is
    # what expresses "recall over F1" here: f1_optimal misses 5/frame with
    # only 2 extra (better F1) but its recall (0.79) fails MIN_RECALL, so it
    # is excluded from eligibility outright, leaving recall_first as the
    # only candidate.
    recall_first = _pt(conf=0.20, missed=1.0, extra=30.0, recall=0.958, matched=70)
    f1_optimal = _pt(conf=0.60, missed=5.0, extra=2.0, recall=0.79, matched=60)
    best, reason = recommend([f1_optimal, recall_first])
    assert best is recall_first
    assert reason == ""


def test_recommend_prefers_the_cheapest_tiling_that_clears_the_recall_floor():
    # 4 tiles/frame is ~4x cheaper over a whole project than 16. If both clear
    # the floor, cost wins -- a full run is hours.
    cheap = _pt(frac=0.10, tiles=4, conf=0.20, recall=0.95, secs=6.0)
    dear = _pt(frac=0.03, tiles=36, conf=0.20, recall=0.97, secs=50.0)
    best, reason = recommend([dear, cheap])
    assert best is cheap and reason == ""


def test_recommend_breaks_tile_ties_on_highest_confidence():
    low = _pt(frac=0.05, tiles=16, conf=0.20, recall=0.96, extra=30.0)
    high = _pt(frac=0.05, tiles=16, conf=0.45, recall=0.93, extra=8.0)
    best, _ = recommend([low, high])
    assert best is high


def test_recommend_ignores_cheap_points_that_miss_the_recall_floor():
    # Full frame is cheapest of all, and on a tiled rig it finds nothing.
    full_frame = _pt(frac=None, tiles=1, conf=0.20, recall=0.05, matched=4, secs=3.0)
    tiled = _pt(frac=0.05, tiles=16, conf=0.20, recall=0.95, matched=70)
    best, _ = recommend([full_frame, tiled])
    assert best is tiled


def test_recommend_excludes_points_below_the_matched_floor():
    # A cheap configuration that finds almost nothing can post a perfect
    # recall on 4 matches. The floor must exclude it from eligibility, not
    # merely veto the frontier's best-matched point.
    thin = _pt(frac=None, tiles=1, conf=0.20, recall=1.0, matched=4, secs=3.0)
    solid = _pt(frac=0.05, tiles=16, conf=0.20, recall=0.95, matched=70)
    best, reason = recommend([thin, solid])
    assert best is solid and reason == ""


def test_calibrate_runs_one_inference_pass_per_tile_fraction(tmp_path):
    import cv2

    from hydra_suite.core.inference.semantic.calibration import calibrate
    from hydra_suite.core.inference.semantic.tiling import candidate_tile_plans

    img = tmp_path / "f0.png"
    cv2.imwrite(str(img), np.zeros((2000, 2000, 3), dtype=np.uint8))
    labeler = _CountingLabeler()
    opts = candidate_tile_plans((2000, 2000), 80.0)
    points = calibrate(
        labeler,
        [(img, [])],
        "ant",
        reference_body_px=80.0,
        seam_margin_px=4,
        merge_iou=0.5,
    )
    assert labeler.calls == sum(o.tiles_per_frame for o in opts)
    assert {p.tile_fraction for p in points} == {o.fraction for o in opts}
    assert len(points) == len(opts) * len(CONFIDENCE_GRID)
    assert all(p.seconds_per_frame >= 0.0 for p in points)


def test_calibrate_averages_tiles_per_frame_across_mixed_frame_sizes(tmp_path):
    # tiles_per_frame depends on FRAME DIMENSIONS, not just reference_body_px:
    # a 2000x2000 frame and a 4000x4000 frame tile differently at the same
    # fraction. Capturing the count from only the first frame that resolves
    # a fraction (rather than averaging across every frame that resolves it)
    # would silently misreport it -- and recommend() sorts on this count.
    import cv2

    from hydra_suite.core.inference.semantic.calibration import calibrate
    from hydra_suite.core.inference.semantic.tiling import candidate_tile_plans

    small = tmp_path / "small.png"
    large = tmp_path / "large.png"
    cv2.imwrite(str(small), np.zeros((2000, 2000, 3), dtype=np.uint8))
    cv2.imwrite(str(large), np.zeros((4000, 4000, 3), dtype=np.uint8))

    reference_body_px = 80.0
    opts_small = {
        o.fraction: o for o in candidate_tile_plans((2000, 2000), reference_body_px)
    }
    opts_large = {
        o.fraction: o for o in candidate_tile_plans((4000, 4000), reference_body_px)
    }
    # A fraction that resolves on BOTH sizes but tiles differently on each --
    # otherwise this test could not distinguish "averaged" from "first frame".
    shared = [
        f
        for f in opts_small
        if f in opts_large
        and opts_small[f].tiles_per_frame != opts_large[f].tiles_per_frame
    ]
    assert shared, "fixture assumption broken: no fraction tiles differently"
    frac = shared[0]
    expected_avg = round(
        (opts_small[frac].tiles_per_frame + opts_large[frac].tiles_per_frame) / 2
    )
    assert expected_avg != opts_small[frac].tiles_per_frame  # first-frame != average

    labeler = _CountingLabeler()
    points = calibrate(
        labeler,
        [(small, []), (large, [])],
        "ant",
        reference_body_px=reference_body_px,
        seam_margin_px=4,
        merge_iou=0.5,
    )
    matches = [p for p in points if p.tile_fraction == frac]
    assert matches
    assert all(p.tiles_per_frame == expected_avg for p in matches)


def test_frontier_rows_are_sorted_and_project_the_run_time():
    from hydra_suite.detectkit.gui.dialogs.calibration_results_dialog import (
        frontier_rows,
    )

    cheap = _pt(frac=0.10, tiles=4, conf=0.30, secs=6.0)
    dear = _pt(frac=0.03, tiles=36, conf=0.20, secs=50.0)
    rows = frontier_rows([dear, cheap], recommended=cheap, project_frames=1000)
    # Cheapest tiling first, then descending confidence within a tiling.
    assert rows[0]["tile"] == "0.10 (4 tiles/frame)"
    assert rows[0]["recommended"] is True
    # 6 s/frame x 1000 frames = 100 minutes, shown as hours:minutes.
    assert rows[0]["projected"] == "1 h 40 m"
    assert rows[1]["projected"] == "13 h 53 m"


def test_frontier_rows_label_the_full_frame_option():
    from hydra_suite.detectkit.gui.dialogs.calibration_results_dialog import (
        frontier_rows,
    )

    rows = frontier_rows([_pt(frac=None, tiles=1)], recommended=None, project_frames=10)
    assert rows[0]["tile"] == "full frame"
    assert rows[0]["recommended"] is False


def test_calibrate_holds_at_most_one_decoded_frame_in_memory(tmp_path):
    """I8: the precompute pass used to retain every decoded frame.

    ``per_frame`` held the decoded ``image`` for every labelled frame before
    a single inference pass ran. At 4512^2 BGR that is ~61 MB each, so a
    50-frame labelled set cost ~3 GB up front. Only the path and the frame
    dimensions are needed to plan the tiles.
    """
    import gc
    import weakref

    import cv2
    import numpy as np

    from hydra_suite.core.inference.semantic import calibration as cal

    n_frames = 5
    paths = []
    for i in range(n_frames):
        path = tmp_path / f"f{i}.png"
        cv2.imwrite(str(path), np.zeros((200, 200, 3), dtype=np.uint8))
        paths.append(path)

    alive: list = []
    real_imread = cv2.imread

    def _tracking_imread(*a, **k):
        arr = real_imread(*a, **k)
        alive.append(weakref.ref(arr))
        return arr

    peak = []

    class _Peaking:
        name = "fake"

        def label_image(self, image_bgr, prompt, **k):
            gc.collect()
            peak.append(sum(1 for r in alive if r() is not None))
            return []

    frames = [
        (
            p,
            [
                LabelRecord(
                    class_id=0,
                    confidence=1.0,
                    points=np.array(
                        [[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32
                    ),
                    level=GeometryLevel.POLYGON,
                )
            ],
        )
        for p in paths
    ]
    original = cal.cv2.imread
    cal.cv2.imread = _tracking_imread
    try:
        cal.calibrate(
            _Peaking(),
            frames,
            "ant",
            reference_body_px=20.0,
            tile_fractions=(0.20, None),
            seam_margin_px=2.0,
            merge_iou=0.5,
        )
    finally:
        cal.cv2.imread = original

    assert peak, "the labeler must have been called"
    # Exactly one frame resident at a time, regardless of how many there are.
    assert max(peak) == 1, f"decoded frames held simultaneously: {max(peak)}"
