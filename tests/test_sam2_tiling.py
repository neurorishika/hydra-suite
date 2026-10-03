"""Shared tile sizing and SAM2 owner-tile segmentation."""

import types

from hydra_suite.core.inference.semantic.tiling import TilingSettings


def test_tiling_settings_default_is_full_frame():
    t = TilingSettings()
    assert t.resolved_tile_px() is None
    assert t.plan_for((400, 600)).tiles == [(0, 0, 600, 400)]


def test_tiling_settings_resolves_and_plans():
    t = TilingSettings(reference_body_px=50.0, tile_fraction=0.25)
    assert t.resolved_tile_px() == 200
    assert len(t.plan_for((400, 600)).tiles) > 1


def test_tile_larger_than_frame_falls_back_to_full_frame():
    t = TilingSettings(tile_px=1000)
    assert t.plan_for((400, 600)).tiles == [(0, 0, 600, 400)]


def test_escalation_request_tiling_defaults_preserve_positional_args():
    from hydra_suite.detectkit.jobs.sam2_escalation import EscalationRequest

    req = EscalationRequest(types.SimpleNamespace(), ["a"], "v", True)
    assert req.overwrite is True and req.tiling == TilingSettings()


def test_semantic_request_exposes_the_same_tiling():
    from hydra_suite.detectkit.jobs.semantic_escalation import SemanticEscalationRequest

    req = SemanticEscalationRequest(
        types.SimpleNamespace(),
        ["a"],
        "sam3",
        "ant",
        reference_body_px=50.0,
        tile_fraction=0.25,
        overlap=0.3,
    )
    assert req.tiling == TilingSettings(50.0, 0.25, None, 0.3)


# -- owner-tile segmentation -------------------------------------------------

from types import SimpleNamespace as NS  # noqa: E402

import numpy as np  # noqa: E402

from hydra_suite.core.inference.sam2.tiling import (  # noqa: E402
    assign_owner_tiles,
    segment_boxes,
)


def test_owner_is_tile_with_largest_margin():
    tiles = [(0, 0, 100, 100), (50, 0, 150, 100)]
    owned, unowned = assign_owner_tiles([(60, 40, 80, 60)], tiles)
    # margins: tile0 -> min(60, 40, 20, 40) = 20; tile1 -> min(10, 40, 70, 40) = 10
    assert owned == {0: [0]} and unowned == []


def test_tie_breaks_to_lowest_tile_index():
    tiles = [(0, 0, 100, 100), (0, 0, 100, 100)]
    owned, _ = assign_owner_tiles([(40, 40, 60, 60)], tiles)
    assert owned == {0: [0]}


def test_box_not_fully_inside_any_tile_is_unowned():
    tiles = [(0, 0, 100, 100), (100, 0, 200, 100)]
    owned, unowned = assign_owner_tiles([(90, 10, 110, 20)], tiles)
    assert owned == {} and unowned == [0]


class _Exec:
    def __init__(self):
        self.log = []
        self.shape = None

    def set_image(self, img):
        self.shape = img.shape[:2]
        self.log.append(("set", img.shape[:2]))

    def segment(self, box, pos, neg):
        self.log.append(
            ("seg", tuple(box), tuple(map(tuple, pos)), tuple(map(tuple, neg)))
        )
        m = np.zeros(self.shape, bool)
        x1, y1, x2, y2 = (int(v) for v in box)
        m[y1:y2, x1:x2] = True
        return m, 0.5


def _p(box, neg=()):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return NS(box_xyxy=box, positive_points=[(cx, cy)], negative_points=list(neg))


def test_tiled_segment_round_trips_coordinates_and_skips_empty_tiles():
    img = np.zeros((100, 300, 3), np.uint8)
    tiles = [(0, 0, 100, 100), (100, 0, 200, 100), (200, 0, 300, 100)]
    ex = _Exec()
    out = segment_boxes(ex, img, [_p((210, 10, 230, 30))], tiles)
    assert ex.log[0] == ("set", (100, 100))
    assert ex.log[1][1] == (10, 10, 30, 30)  # shifted into tile 2
    assert len([e for e in ex.log if e[0] == "set"]) == 1  # empty tiles skipped
    m = out[0].mask
    assert m.shape == (100, 300) and m[10:30, 210:230].all() and m.sum() == 400
    assert out[0].owner_tile == 2


def test_negative_points_outside_owner_tile_are_dropped():
    img = np.zeros((100, 200, 3), np.uint8)
    tiles = [(0, 0, 100, 100), (100, 0, 200, 100)]
    ex = _Exec()
    segment_boxes(ex, img, [_p((10, 10, 30, 30), neg=[(20, 20), (150, 50)])], tiles)
    assert ex.log[1][3] == ((20.0, 20.0),)


def test_unowned_boxes_get_one_full_frame_pass_after_tiles():
    img = np.zeros((100, 200, 3), np.uint8)
    tiles = [(0, 0, 100, 100), (100, 0, 200, 100)]
    prompts = [_p((90, 10, 110, 30)), _p((10, 10, 30, 30)), _p((95, 50, 105, 60))]
    ex = _Exec()
    out = segment_boxes(ex, img, prompts, tiles)
    sets = [e for e in ex.log if e[0] == "set"]
    assert sets == [("set", (100, 100)), ("set", (100, 200))]
    assert [o.owner_tile for o in out] == [None, 0, None]
    assert out[0].mask[10:30, 90:110].all()


def test_single_full_frame_tile_is_the_legacy_call_sequence():
    img = np.zeros((100, 200, 3), np.uint8)
    ex = _Exec()
    prompts = [_p((10, 10, 30, 30)), _p((50, 50, 70, 70))]
    segment_boxes(ex, img, prompts, [(0, 0, 200, 100)])
    assert ex.log == [
        ("set", (100, 200)),
        ("seg", (10, 10, 30, 30), ((20.0, 20.0),), ()),
        ("seg", (50, 50, 70, 70), ((60.0, 60.0),), ()),
    ]


def test_no_prompts_encodes_nothing_when_tiled():
    ex = _Exec()
    out = segment_boxes(
        ex,
        np.zeros((100, 200, 3), np.uint8),
        [],
        [(0, 0, 100, 100), (100, 0, 200, 100)],
    )
    assert out == [] and ex.log == []
