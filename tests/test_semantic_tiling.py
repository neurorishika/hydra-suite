import numpy as np
import pytest

from hydra_suite.core.inference.semantic.base import SemanticInstance, SemanticLabeler
from hydra_suite.core.inference.semantic.tiling import (
    SEMANTIC_TILE_FRACTION_SEED,
    TILE_FRACTION_GRID,
    TileCandidate,
    TileCollectionCancelled,
    candidate_tile_plans,
    collect_candidates,
    merge_candidates,
    plan_for_frame,
    resolve_tile_px,
)


class FakeLabeler:
    """Returns a scripted list of instances per call, in TILE-LOCAL coords."""

    def __init__(self, scripted: list[list[SemanticInstance]]) -> None:
        self._scripted = list(scripted)
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake"

    def label_image(
        self, image_bgr, prompt, *, confidence_threshold=0.0, max_instances=0
    ):
        out = self._scripted[self.calls] if self.calls < len(self._scripted) else []
        self.calls += 1
        return [i for i in out if i.confidence >= confidence_threshold]


def test_fake_labeler_satisfies_the_protocol():
    assert isinstance(FakeLabeler([]), SemanticLabeler)


def test_semantic_instance_is_frozen():
    inst = SemanticInstance(
        polygon_px=np.zeros((4, 2), dtype=np.float32), confidence=0.5
    )
    try:
        inst.confidence = 0.9
    except Exception as exc:
        # dataclasses.FrozenInstanceError's message text is always
        # "cannot assign to field '...'" -- it never contains the word
        # "frozen" -- only the exception TYPE name does. Check the type.
        assert "frozen" in type(exc).__name__.lower()
    else:
        raise AssertionError("SemanticInstance must be frozen")


def _sq(x0, y0, side):
    return np.array(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
        dtype=np.float32,
    )


def test_tile_px_scales_with_body_size_and_ignores_the_training_fraction():
    # The fraction is an ARGUMENT, not a tuned constant. The TRAINING fraction
    # (0.15) would give 533, the config measured to cost 5.7x for no gain.
    assert resolve_tile_px(80.0, 0.05) == 1600
    assert resolve_tile_px(80.0, 0.10) == 800
    assert SEMANTIC_TILE_FRACTION_SEED == 0.05


def test_tile_px_is_none_when_body_size_or_fraction_is_unknown():
    assert resolve_tile_px(0.0, 0.05) is None
    assert resolve_tile_px(-1.0, 0.05) is None
    assert resolve_tile_px(80.0, None) is None  # None fraction = full frame
    assert resolve_tile_px(None, 0.05) is None


def test_candidate_plans_cover_the_grid_and_always_include_full_frame():
    opts = candidate_tile_plans((4512, 4512), 80.0)
    assert None in TILE_FRACTION_GRID
    by_frac = {o.fraction: o for o in opts}
    assert by_frac[None].tile_px is None
    assert by_frac[None].tiles_per_frame == 1
    # Finer fraction -> smaller tile (tile_px = body_px / fraction) -> more tiles.
    assert by_frac[0.10].tiles_per_frame > by_frac[0.03].tiles_per_frame


def test_candidate_plans_skip_fractions_that_degenerate_or_explode():
    # Small frame: a 1600 px tile exceeds it, so that fraction would just be a
    # second full-frame pass -- skip it rather than pay for a duplicate.
    # (0.10 -> 800 px, which fits inside a 900 px frame and is kept.)
    opts = candidate_tile_plans((900, 900), 80.0)
    assert [o.fraction for o in opts if o.fraction is not None] == [0.10]
    # Unknown body size leaves full-frame as the only option.
    assert [o.fraction for o in candidate_tile_plans((4512, 4512), None)] == [None]
    # A fraction that breaches MAX_TILES_PER_FRAME is skipped, not raised.
    assert all(
        o.tiles_per_frame <= 4096 for o in candidate_tile_plans((40000, 40000), 20.0)
    )


def test_full_frame_plan_has_no_interior_seams():
    opt = next(
        o for o in candidate_tile_plans((2000, 2000), None) if o.fraction is None
    )
    at_edge = SemanticInstance(_sq(0, 400, 6), 0.9)
    cands = collect_candidates(
        FakeLabeler([[at_edge]]),
        np.zeros((2000, 2000, 3), dtype=np.uint8),
        opt.plan,
        "ant",
        confidence_threshold=0.0,
        max_instances=0,
        seam_margin_px=4,
    )
    assert len(cands) == 1


def test_plan_covers_the_frame_with_overlap():
    plan = plan_for_frame((4512, 4512), 1504, 0.2)
    assert plan.slice_wh == (1504, 1504)
    assert len(plan.tiles) == 16
    assert plan.tiles[-1][2] == 4512 and plan.tiles[-1][3] == 4512


def test_plan_rejects_pathological_geometry():
    with pytest.raises(ValueError, match="tile ceiling|ceiling"):
        plan_for_frame((10000, 10000), 64, 0.9)


def test_candidates_are_offset_into_frame_space():
    # One tile at (1000, 500); a detection at tile-local (10, 20) must come
    # back at frame (1010, 520).
    plan = plan_for_frame((2000, 2000), 1000, 0.0)
    idx = plan.tiles.index((1000, 0, 2000, 1000))
    scripted = [[] for _ in plan.tiles]
    scripted[idx] = [SemanticInstance(_sq(400, 400, 60), 0.9)]
    labeler = FakeLabeler(scripted)
    cands = collect_candidates(
        labeler,
        np.zeros((2000, 2000, 3), dtype=np.uint8),
        plan,
        "ant",
        confidence_threshold=0.0,
        max_instances=0,
        seam_margin_px=4,
    )
    assert len(cands) == 1
    assert cands[0].polygon_px[:, 0].min() == pytest.approx(1400.0)
    assert cands[0].polygon_px[:, 1].min() == pytest.approx(400.0)


def test_seam_touching_detection_is_dropped_but_frame_edge_is_kept():
    plan = plan_for_frame((2000, 2000), 1000, 0.0)
    # Tile 0 is (0, 0, 1000, 1000): its x=1000 edge is an interior seam,
    # its x=0 edge is the frame edge.
    at_interior_seam = SemanticInstance(_sq(994, 400, 6), 0.9)
    at_frame_edge = SemanticInstance(_sq(0, 400, 6), 0.9)
    scripted = [[at_interior_seam, at_frame_edge]] + [[] for _ in plan.tiles[1:]]
    cands = collect_candidates(
        FakeLabeler(scripted),
        np.zeros((2000, 2000, 3), dtype=np.uint8),
        plan,
        "ant",
        confidence_threshold=0.0,
        max_instances=0,
        seam_margin_px=4,
    )
    assert len(cands) == 1
    assert cands[0].polygon_px[:, 0].min() == pytest.approx(0.0)


def test_merge_collapses_one_object_seen_in_two_overlapping_tiles():
    dup = [
        TileCandidate(_sq(100, 100, 40), 0.8, 0),
        TileCandidate(_sq(102, 101, 40), 0.6, 1),
    ]
    merged = merge_candidates(dup, confidence_threshold=0.0, iou_threshold=0.5)
    assert len(merged) == 1
    assert merged[0].confidence == 0.8  # the higher-scoring survivor wins


def test_merging_at_a_threshold_equals_merging_only_the_kept_subset():
    """The REAL per-threshold invariant.

    The old test here was named
    ``test_merge_is_redone_per_threshold_not_post_filtered`` and claimed a
    high threshold "resurrects" a candidate its suppressor had removed. That
    property does not exist -- greedy NMS runs in descending confidence, so a
    suppressor always outscores its victim -- and the test's second half
    hand-filtered the suppressor out of the INPUT, which no threshold could
    do. It therefore could not fail for the reason its name claimed.

    What merge_candidates does guarantee, and what this pins, is that
    thresholding is applied to the merge INPUT: survivors at T are exactly
    the survivors of merging only the >=T candidates.
    """
    cands = [
        TileCandidate(_sq(0, 0, 50), 0.40, 0),
        TileCandidate(_sq(2, 2, 50), 0.90, 1),
        TileCandidate(_sq(400, 400, 50), 0.60, 2),
        TileCandidate(_sq(402, 401, 50), 0.30, 3),
        TileCandidate(_sq(800, 800, 50), 0.75, 4),
    ]
    for threshold in (0.0, 0.25, 0.35, 0.5, 0.65, 0.8, 0.95):
        got = merge_candidates(cands, confidence_threshold=threshold, iou_threshold=0.5)
        want = merge_candidates(
            [c for c in cands if c.confidence >= threshold],
            confidence_threshold=0.0,
            iou_threshold=0.5,
        )
        assert [round(g.confidence, 4) for g in got] == [
            round(w.confidence, 4) for w in want
        ], f"threshold {threshold}"

    # And concretely: at 0.5 the 0.30 candidate is gone, so the 0.60 one it
    # was suppressed by survives alone rather than as a pair.
    at_low = merge_candidates(cands, confidence_threshold=0.0, iou_threshold=0.5)
    at_high = merge_candidates(cands, confidence_threshold=0.5, iou_threshold=0.5)
    assert sorted(round(c.confidence, 2) for c in at_low) == [0.6, 0.75, 0.9]
    assert sorted(round(c.confidence, 2) for c in at_high) == [0.6, 0.75, 0.9]


def test_should_stop_halts_between_tiles_and_raises_rather_than_truncating():
    """F1: a cancelled frame must be UNMISTAKABLE, not a short list.

    A partial return is indistinguishable from a genuinely sparse frame, and
    the escalation job cached it as complete -- so resume skipped the frame
    forever. TileCollectionCancelled cannot be ignored by accident.
    """
    plan = plan_for_frame((2000, 2000), 1000, 0.0)
    labeler = FakeLabeler([[] for _ in plan.tiles])
    with pytest.raises(TileCollectionCancelled) as exc:
        collect_candidates(
            labeler,
            np.zeros((2000, 2000, 3), dtype=np.uint8),
            plan,
            "ant",
            confidence_threshold=0.0,
            max_instances=0,
            seam_margin_px=4,
            should_stop=lambda: labeler.calls >= 2,
        )
    assert labeler.calls == 2
    assert exc.value.tiles_done == 2
    assert exc.value.tiles_total == len(plan.tiles)


def test_collect_candidates_returns_normally_when_never_cancelled():
    """The completion path must NOT raise: only cancellation does."""
    plan = plan_for_frame((2000, 2000), 1000, 0.0)
    labeler = FakeLabeler([[] for _ in plan.tiles])
    assert (
        collect_candidates(
            labeler,
            np.zeros((2000, 2000, 3), dtype=np.uint8),
            plan,
            "ant",
            confidence_threshold=0.0,
            max_instances=0,
            seam_margin_px=4,
            should_stop=lambda: False,
        )
        == []
    )
    assert labeler.calls == len(plan.tiles)
