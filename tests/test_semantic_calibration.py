import numpy as np

from hydra_suite.core.inference.semantic.base import SemanticInstance
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
    quality=0.6,
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
        area_min_px2=120.0,
        area_max_px2=1400.0,
        mean_quality=quality,
        median_iou=quality,
        median_area_ratio=quality,
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


def test_calibration_reports_each_tile_with_a_running_eta(tmp_path):
    import cv2

    from hydra_suite.core.inference.semantic.calibration import calibrate

    img = tmp_path / "f0.png"
    cv2.imwrite(str(img), np.zeros((400, 400, 3), dtype=np.uint8))
    updates = []

    calibrate(
        _CountingLabeler(),
        [(img, [])],
        "ant",
        reference_body_px=20.0,
        tile_fractions=(0.10,),
        overlap=0.0,
        seam_margin_px=4,
        merge_iou=0.5,
        progress=lambda pct, message: updates.append((pct, message)),
    )

    tile_updates = [(pct, message) for pct, message in updates if ", tile " in message]
    assert len(tile_updates) == 4
    assert "tile 1/4" in tile_updates[0][1]
    assert "ETA" in tile_updates[0][1]
    assert tile_updates[-1][0] == 100


def test_calibrate_exposes_rethresholdable_visual_evidence(tmp_path):
    import cv2

    from hydra_suite.core.inference.semantic.calibration import calibrate

    img = tmp_path / "f0.png"
    cv2.imwrite(str(img), np.zeros((256, 256, 3), dtype=np.uint8))
    records = [
        LabelRecord(
            class_id=3,
            confidence=1.0,
            points=_sq(100, 100),
            level=GeometryLevel.POLYGON,
        )
    ]
    previews = []

    calibrate(
        _CountingLabeler(),
        [(img, records)],
        "ant",
        reference_body_px=80.0,
        tile_fractions=(None,),
        seam_margin_px=4,
        merge_iou=0.5,
        preview_sink=previews.extend,
    )

    assert len(previews) == 1
    assert previews[0].image_path == img
    assert previews[0].ground_truth[0].class_id == 3
    assert None in previews[0].candidates_by_fraction


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


class _ScriptedByCallLabeler:
    """Returns a detection only on the designated call indices (0-based)."""

    def __init__(self, hits):
        self.calls = 0
        self._hits = set(hits)

    @property
    def name(self):
        return "scripted"

    def label_image(
        self, image_bgr, prompt, *, confidence_threshold=0.0, max_instances=0
    ):
        idx = self.calls
        self.calls += 1
        return [SemanticInstance(_sq(30, 30), 0.9)] if idx in self._hits else []


def _calibrate_two_frames(tmp_path, labeler, should_stop=None):
    """One 0.5 fraction over two 400x400 frames = 9 tiles each, 18 calls.

    Frame 0 carries no labels; frame 1 carries one at (230, 230), which only
    the LAST tile (200, 200, 400, 400) can produce -- a tile-local (30, 30)
    detection offset into frame space.
    """
    import cv2

    from hydra_suite.core.inference.semantic.calibration import calibrate

    paths = []
    for i in range(2):
        q = tmp_path / f"f{i}.png"
        cv2.imwrite(str(q), np.zeros((400, 400, 3), dtype=np.uint8))
        paths.append(q)
    label = LabelRecord(
        class_id=0,
        confidence=1.0,
        points=_sq(230, 230),
        level=GeometryLevel.POLYGON,
    )
    return calibrate(
        labeler,
        [(paths[0], []), (paths[1], [label])],
        "ant",
        reference_body_px=100.0,
        tile_fractions=(0.5,),
        seam_margin_px=2,
        merge_iou=0.5,
        should_stop=should_stop,
    )


def test_a_cancel_inside_the_last_frame_does_not_count_that_frame(tmp_path):
    """F6: a cancel on the LAST frame slipped past the completeness filter.

    The inner loop appended the entry and bumped ``entry["n"]`` BEFORE the
    next ``should_stop`` check, so ``entry["n"] < seen_frames`` dropped every
    incomplete fraction EXCEPT one cancelled on the final frame -- that one
    survived with full standing, its per-frame error rates computed over a
    frame whose late tiles were never run. The missed animal in those tiles
    is therefore reported, but averaged as if the frame had been searched.

    Here frame 1's only label is findable solely by tile 8. Cancelling after
    12 calls (frame 1, tile 3) must DISCARD frame 1, not count it as a frame
    on which the animal was missed.
    """

    # Cells above the detection's own 0.9 score legitimately miss it; the
    # comparison is over the cells where the detection is admissible.
    def _admissible(points):
        return [p for p in points if p.confidence <= 0.9]

    complete = _calibrate_two_frames(tmp_path, _ScriptedByCallLabeler({17}))
    assert complete, "the uncancelled sweep must produce a frontier"
    assert all(p.missed_per_frame == 0.0 for p in _admissible(complete))

    labeler = _ScriptedByCallLabeler({17})
    points = _calibrate_two_frames(
        tmp_path, labeler, should_stop=lambda: labeler.calls >= 12
    )
    assert labeler.calls == 12, "the sweep was not cut short inside frame 1"
    assert points, "frame 0 completed, so its measurement must survive"
    # Pre-fix this was 0.5: the un-searched frame 1 was counted, so the label
    # its last tile would have found was booked as a miss.
    assert all(
        p.missed_per_frame == 0.0 for p in _admissible(points)
    ), "a frame cancelled mid-tiling was counted as a searched frame"


def test_calibration_results_dialog_marks_a_cancelled_sweep_as_partial(qtbot=None):
    """F6, other half: a cancelled calibration must SAY so in the results."""
    import pytest

    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication, QLabel

    from hydra_suite.detectkit.gui.dialogs.calibration_results_dialog import (
        CalibrationResultsDialog,
    )

    app = QApplication.instance() or QApplication([])
    pts = [_pt()]

    def _labels(partial):
        dlg = CalibrationResultsDialog(
            pts, pts[0], "", project_frames=100, partial=partial
        )
        text = " ".join(w.text() for w in dlg.findChildren(QLabel))
        dlg.deleteLater()
        return text

    assert "PARTIAL" in _labels(True)
    assert "cancelled" in _labels(True).lower()
    assert "PARTIAL" not in _labels(False)
    del app


def test_calibration_worker_exposes_its_cancelled_state(tmp_path):
    """The dialog reads `worker.cancelled` to decide the partial marker."""
    from hydra_suite.detectkit.jobs.semantic_escalation import CalibrationWorker

    w = CalibrationWorker([], "ant", "sam3", {}, project_dir=tmp_path)
    assert w.cancelled is False
    w.cancel()
    assert w.cancelled is True


def _band_for(side=20.0, n=10):
    from hydra_suite.core.inference.semantic.shape_prior import fit_area_band

    return fit_area_band([_sq(0, 0, side=side) for _ in range(n)])


def test_an_arena_blob_no_longer_earns_recall_credit():
    """The core mistargeting bug.

    Containment alone let one huge region claim a label, and because
    recommend() is recall-first, calibration then SELECTED for whatever
    configuration produced such regions.
    """
    labels = [_sq(100, 100), _sq(130, 100), _sq(160, 100)]
    blob = _sq(130, 100, side=400.0)
    assert match_one_to_one([blob], labels) == []
    assert len(match_one_to_one([blob], labels, area_band=_band_for())) == 0


def test_a_subpart_sized_fragment_is_not_a_find():
    labels = [_sq(100, 100, side=20.0)]
    leg = _sq(103, 103, side=3.0)
    assert match_one_to_one([leg], labels, area_band=_band_for()) == []


def test_a_correct_body_still_matches_under_the_band():
    labels = [_sq(100, 100, side=20.0)]
    traced = _sq(101, 100, side=26.0)  # legs traced, ~1.7x area
    assert len(match_one_to_one([traced], labels, area_band=_band_for())) == 1


def test_pairs_rank_by_quality_not_centroid_distance():
    """In a cluster the nearest centroid is not always the best fit.

    Two labels; the prediction that is a size/shape match for label B sits
    marginally closer to label A. Distance-first pairing hands it to A and
    strands B; quality-first pairs each with its true partner.
    """
    label_a = _sq(100, 100, side=60.0)
    label_b = _sq(118, 100, side=20.0)
    small = _sq(112, 100, side=20.0)  # a body-B-shaped prediction
    big = _sq(100, 100, side=60.0)  # an exact body-A prediction
    pairs = dict(match_one_to_one([small, big], [label_a, label_b]))
    assert pairs == {0: 1, 1: 0}


def test_a_pair_below_the_quality_floor_is_not_counted_as_a_find():
    """Containment admits it; the quality floor still rejects it.

    A prediction ~39x the label's area contains that label's centroid, so
    the old gate scored it as a find. It is not one, and no band is needed
    to say so -- the graded score alone rejects it.
    """
    labels = [_sq(100, 100, side=20.0)]
    assert match_one_to_one([_sq(100, 100, side=125.0)], labels) == []
    # ... while a merely generous mask over the same label still counts.
    assert len(match_one_to_one([_sq(100, 100, side=30.0)], labels)) == 1


def test_calibration_points_carry_the_band_and_quality(tmp_path):
    import cv2

    from hydra_suite.core.inference.semantic.calibration import calibrate

    img = tmp_path / "f.png"
    cv2.imwrite(str(img), np.zeros((400, 400, 3), dtype=np.uint8))
    label = LabelRecord(
        class_id=0,
        confidence=1.0,
        points=_sq(230, 230),
        level=GeometryLevel.POLYGON,
    )

    class _Finder:
        name = "finder"

        def label_image(self, image_bgr, prompt, **k):
            return [SemanticInstance(_sq(30, 30), 0.9)]

    points = calibrate(
        _Finder(),
        [(img, [label])],
        "ant",
        reference_body_px=100.0,
        tile_fractions=(0.5,),
        seam_margin_px=2,
        merge_iou=0.5,
    )
    assert points
    p = points[0]
    assert p.area_min_px2 > 0 and p.area_max_px2 > p.area_min_px2
    hit = [q for q in points if q.n_matched > 0]
    assert hit, "the scripted detection should match the label"
    assert 0.0 < hit[0].mean_quality <= 1.0
    assert 0.0 < hit[0].median_iou <= 1.0
    assert 0.0 < hit[0].median_area_ratio <= 1.0


def test_recommend_refuses_a_point_whose_matches_are_mistargeted():
    """Recall bought with blobs is not recall."""
    sloppy = _pt(conf=0.20, recall=0.99, matched=90, tiles=4, quality=0.12)
    honest = _pt(conf=0.20, recall=0.95, matched=70, tiles=16, quality=0.62)
    best, reason = recommend([sloppy, honest])
    assert best is honest and reason == ""


def test_recommend_explains_a_frontier_that_is_entirely_mistargeted():
    points = [_pt(conf=c, recall=0.99, matched=90, quality=0.05) for c in (0.2, 0.4)]
    best, reason = recommend(points)
    assert best is None
    assert "quality" in reason.lower() or "mistarget" in reason.lower()
