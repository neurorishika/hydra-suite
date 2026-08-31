import numpy as np
import pytest

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.merge import MergeMode, merge_records
from hydra_suite.utils.geometry_levels import GeometryLevel


def _quad(x, y, size, class_id=0, level=GeometryLevel.OBB):
    return LabelRecord(
        class_id=class_id,
        confidence=1.0,
        points=np.array(
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size]],
            dtype=np.float32,
        ),
        level=level,
    )


def test_overwrite_returns_only_staged():
    existing = [_quad(0, 0, 10), _quad(100, 100, 10)]
    staged = [_quad(50, 50, 10)]

    out = merge_records(
        existing,
        staged,
        mode=MergeMode.OVERWRITE,
        iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 1
    np.testing.assert_allclose(out[0].points, staged[0].points)


def test_add_new_appends_a_non_overlapping_staged_record():
    existing = [_quad(0, 0, 10)]
    staged = [_quad(100, 100, 10)]

    out = merge_records(
        existing,
        staged,
        mode=MergeMode.ADD_NEW,
        iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 2
    assert out[0] is existing[0]


def test_add_new_drops_a_staged_record_that_overlaps():
    existing = [_quad(0, 0, 20)]
    staged = [_quad(1, 1, 20)]  # IoU well above 0.5

    out = merge_records(
        existing,
        staged,
        mode=MergeMode.ADD_NEW,
        iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 1
    assert out[0] is existing[0]


def test_add_new_iou_boundary_in_both_directions():
    """A staged record is dropped at IoU >= threshold and kept just below."""
    existing = [_quad(0, 0, 20)]
    overlapping = _quad(2, 0, 20)  # 18/22 columns shared -> IoU ~= 0.818

    dropped = merge_records(
        existing,
        [overlapping],
        mode=MergeMode.ADD_NEW,
        iou_threshold=0.80,
        level=GeometryLevel.OBB,
    )
    kept = merge_records(
        existing,
        [overlapping],
        mode=MergeMode.ADD_NEW,
        iou_threshold=0.85,
        level=GeometryLevel.OBB,
    )

    assert len(dropped) == 1
    assert len(kept) == 2


def test_add_new_compares_against_every_existing_record_not_just_the_first():
    existing = [_quad(0, 0, 10), _quad(100, 100, 20)]
    staged = [_quad(101, 101, 20)]

    out = merge_records(
        existing,
        staged,
        mode=MergeMode.ADD_NEW,
        iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 2


def test_add_new_with_empty_existing_keeps_all_staged():
    staged = [_quad(0, 0, 10), _quad(100, 100, 10)]

    out = merge_records(
        [],
        staged,
        mode=MergeMode.ADD_NEW,
        iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 2


def test_add_new_with_empty_staged_returns_existing_unchanged():
    existing = [_quad(0, 0, 10)]

    out = merge_records(
        existing,
        [],
        mode=MergeMode.ADD_NEW,
        iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert out == existing


def test_add_new_never_modifies_reorders_or_drops_an_existing_record():
    """The invariant that makes immediate application safe."""
    existing = [
        _quad(0, 0, 20, class_id=1),
        _quad(60, 60, 20, class_id=2),
        _quad(200, 200, 20, class_id=3),
    ]
    before = [r.points.copy() for r in existing]
    staged = [_quad(1, 1, 20), _quad(400, 400, 20), _quad(61, 61, 20)]

    out = merge_records(
        existing,
        staged,
        mode=MergeMode.ADD_NEW,
        iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    # id(), not ==: LabelRecord.__eq__ compares ndarrays elementwise and
    # raises on bool(). List __eq__ happens to short-circuit on identity
    # here, but relying on that is a trap the next edit would spring.
    assert [id(r) for r in out[: len(existing)]] == [id(r) for r in existing]
    for rec, pts in zip(existing, before):
        np.testing.assert_array_equal(rec.points, pts)  # unmutated
    assert len(out) == len(existing) + 1  # only the disjoint one survived


def test_survivors_are_the_tail_slice():
    """The positional contract the file-level accept path relies on.

    Identity, via id(), NOT `in`/`==`. LabelRecord is a plain dataclass
    holding an ndarray, so its generated __eq__ compares `points`
    elementwise and bool() on the resulting array raises "truth value of an
    array is ambiguous" the moment class_id and confidence happen to match.
    """
    existing = [_quad(0, 0, 20)]
    staged = [_quad(1, 1, 20), _quad(400, 400, 20)]

    out = merge_records(
        existing,
        staged,
        mode=MergeMode.ADD_NEW,
        iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    head, tail = out[: len(existing)], out[len(existing) :]
    assert [id(r) for r in head] == [id(r) for r in existing]
    assert all(not any(r is e for e in existing) for r in tail)
    assert len(tail) == 1


def test_staged_above_the_target_level_is_derived_down():
    existing = [_quad(0, 0, 10, level=GeometryLevel.OBB)]
    poly = LabelRecord(
        class_id=0,
        confidence=1.0,
        points=np.array(
            [[100, 100], [120, 102], [125, 120], [110, 130], [98, 118]],
            dtype=np.float32,
        ),
        level=GeometryLevel.POLYGON,
    )

    out = merge_records(
        existing,
        [poly],
        mode=MergeMode.ADD_NEW,
        iou_threshold=0.5,
        level=GeometryLevel.OBB,
    )

    assert len(out) == 2
    assert out[1].level is GeometryLevel.OBB
    assert out[1].points.shape == (4, 2)


def test_records_below_the_target_level_are_refused_not_invented():
    """The primitive stays strict; lifting is the CALLER's explicit choice.

    `staged_review.accept_frame` re-tags records before calling this when a
    lift is genuinely wanted (a quad encoded as a 4-point polygon), so the
    primitive never has to guess.
    """
    existing = [_quad(0, 0, 10, level=GeometryLevel.POLYGON)]
    staged = [_quad(100, 100, 10, level=GeometryLevel.OBB)]

    with pytest.raises(ValueError, match="upward"):
        merge_records(
            existing,
            staged,
            mode=MergeMode.ADD_NEW,
            iou_threshold=0.5,
            level=GeometryLevel.POLYGON,
        )
