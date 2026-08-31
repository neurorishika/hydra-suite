import json
from pathlib import Path

import pytest

from hydra_suite.detectkit.gui.models import OBBSource, StagedReview
from hydra_suite.detectkit.jobs import staged_review as sr


def _source(tmp_path, labels: dict[str, str], level="obb", classes="object\n"):
    root = tmp_path / "src"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for rel, text in labels.items():
        p = root / "labels" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        (root / "images" / rel).with_suffix(".png").parent.mkdir(
            parents=True, exist_ok=True
        )
    (root / "classes.txt").write_text(classes)
    return OBBSource(path=str(root), name="src", level=level)


def _staging(tmp_path, labels: dict[str, str], classes="object\n"):
    # MUST live under artifacts/pending_escalations/: with project_dir=None,
    # `_is_safe_to_delete` (sam2_escalation.py:35) accepts a path only if its
    # parent is "pending_escalations" and its grandparent is "artifacts".
    # A staging dir anywhere else is silently NOT deleted, and
    # test_finish_review_removes_staging would fail with nothing to show why.
    root = tmp_path / "artifacts" / "pending_escalations" / "staging"
    (root / "labels").mkdir(parents=True)
    for rel, text in labels.items():
        p = root / "labels" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (root / "classes.txt").write_text(classes)
    return root


def test_decisions_round_trip(tmp_path):
    staged = _staging(tmp_path, {"a.txt": "", "b.txt": ""})

    sr.write_decisions(staged, {"a.txt": sr.ACCEPTED_ADD_NEW})

    assert sr.read_decisions(staged) == {"a.txt": sr.ACCEPTED_ADD_NEW}


def test_decisions_read_as_empty_when_absent(tmp_path):
    assert sr.read_decisions(_staging(tmp_path, {})) == {}


def test_staged_frames_are_sorted_posix_relative_paths(tmp_path):
    staged = _staging(tmp_path, {"b.txt": "", "sub/a.txt": ""})

    assert sr.staged_frames(staged) == ["b.txt", "sub/a.txt"]


def test_review_progress_counts_decided_over_total(tmp_path):
    staged = _staging(tmp_path, {"a.txt": "", "b.txt": "", "c.txt": ""})
    sr.write_decisions(staged, {"a.txt": sr.REJECTED})

    assert sr.review_progress(staged) == (1, 3)


def test_snapshot_captures_labels_level_and_classes(tmp_path):
    source = _source(tmp_path, {"a.txt": "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"})
    staged = _staging(tmp_path, {"a.txt": ""})

    sr.ensure_snapshot(source, staged)

    assert (staged / "labels_before" / "a.txt").read_text().startswith("0 ")
    state = json.loads((staged / "state_before.json").read_text())
    assert state["level"] == "obb"
    assert state["classes_txt"] == "object\n"


def test_snapshot_is_taken_once_and_never_overwritten(tmp_path):
    source = _source(tmp_path, {"a.txt": "original\n"})
    staged = _staging(tmp_path, {"a.txt": ""})

    sr.ensure_snapshot(source, staged)
    (Path(source.path) / "labels" / "a.txt").write_text("changed\n")
    sr.ensure_snapshot(source, staged)

    assert (staged / "labels_before" / "a.txt").read_text() == "original\n"


def test_revert_restores_labels_level_and_classes_and_clears_decisions(tmp_path):
    source = _source(tmp_path, {"a.txt": "original\n"}, level="obb")
    staged = _staging(tmp_path, {"a.txt": ""})
    sr.ensure_snapshot(source, staged)

    (Path(source.path) / "labels" / "a.txt").write_text("accepted\n")
    (Path(source.path) / "labels" / "new.txt").write_text("appeared\n")
    (Path(source.path) / "classes.txt").write_text("object\nant\n")
    source.level = "polygon"
    sr.write_decisions(staged, {"a.txt": sr.ACCEPTED_OVERWRITE})

    sr.revert_review(source, staged)

    assert (Path(source.path) / "labels" / "a.txt").read_text() == "original\n"
    assert not (Path(source.path) / "labels" / "new.txt").exists()
    assert (Path(source.path) / "classes.txt").read_text() == "object\n"
    assert source.level == "obb"
    assert sr.read_decisions(staged) == {}


def test_revert_clears_classes_appended_to_a_source_that_had_none(tmp_path):
    """Restoring the class list must mean restoring it, including to empty."""
    source = _source(tmp_path, {"a.txt": "original\n"}, classes="")
    staged = _staging(tmp_path, {"a.txt": ""})
    sr.ensure_snapshot(source, staged)
    (Path(source.path) / "classes.txt").write_text("larva\n")

    sr.revert_review(source, staged)

    assert (Path(source.path) / "classes.txt").read_text() == ""


def test_revert_without_a_snapshot_is_refused(tmp_path):
    source = _source(tmp_path, {"a.txt": "original\n"})
    staged = _staging(tmp_path, {"a.txt": ""})

    with pytest.raises(RuntimeError, match="no snapshot"):
        sr.revert_review(source, staged)


def test_matching_class_names_map_by_name(tmp_path):
    source = _source(tmp_path, {}, classes="ant\nbeetle\n")
    staged = _staging(tmp_path, {}, classes="beetle\n")

    assert sr.resolve_staged_class_ids(source, staged) == {0: 1}


def test_an_unknown_staged_class_is_appended_to_the_source(tmp_path):
    source = _source(tmp_path, {}, classes="ant\n")
    staged = _staging(tmp_path, {}, classes="ant\nlarva\n")

    mapping = sr.resolve_staged_class_ids(source, staged)

    assert mapping == {0: 0, 1: 1}
    assert (Path(source.path) / "classes.txt").read_text() == "ant\nlarva\n"


def test_appending_never_renumbers_an_existing_class(tmp_path):
    source = _source(tmp_path, {}, classes="ant\nbeetle\n")
    staged = _staging(tmp_path, {}, classes="larva\n")

    mapping = sr.resolve_staged_class_ids(source, staged)

    assert mapping == {0: 2}
    assert (Path(source.path) / "classes.txt").read_text().splitlines()[:2] == [
        "ant",
        "beetle",
    ]


def test_resolution_is_idempotent(tmp_path):
    source = _source(tmp_path, {}, classes="ant\n")
    staged = _staging(tmp_path, {}, classes="larva\n")

    first = sr.resolve_staged_class_ids(source, staged)
    second = sr.resolve_staged_class_ids(source, staged)

    assert first == second
    assert (Path(source.path) / "classes.txt").read_text() == "ant\nlarva\n"


import numpy as np
from PIL import Image

from hydra_suite.data.al.labels import read_label_file
from hydra_suite.data.al.merge import MergeMode


def _image(source: OBBSource, rel_stem: str, size=(100, 200)):
    """A real PNG, because accept must read the frame size off the image."""
    path = Path(source.path) / "images" / f"{rel_stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[0], size[1], 3), dtype=np.uint8)).save(path)
    return path


def _obb_line(x1, y1, x2, y2, w=200, h=100, class_id=0):
    xs = [x1, x2, x2, x1]
    ys = [y1, y1, y2, y2]
    coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in zip(xs, ys))
    return f"{class_id} {coords}\n"


def _wired(
    tmp_path,
    source_labels,
    staged_labels,
    level="obb",
    target_level="obb",
    producer="sam2",
):
    source = _source(tmp_path, source_labels, level=level)
    staged = _staging(tmp_path, staged_labels)
    for rel in set(source_labels) | set(staged_labels):
        _image(source, rel[:-4])
    source.staged_review = StagedReview(
        staged_path=str(staged), target_level=target_level, producer=producer
    )
    return source, staged


def test_accept_overwrite_replaces_the_frames_labels(tmp_path):
    source, staged = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": _obb_line(50, 50, 70, 70)},
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.OVERWRITE)

    out = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))
    assert len(out) == 1
    assert out[0].points[:, 0].min() > 40
    assert sr.read_decisions(staged)["a.txt"] == sr.ACCEPTED_OVERWRITE


def test_accept_add_new_appends_only_the_non_overlapping_staged(tmp_path):
    source, staged = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": _obb_line(0, 0, 20, 20) + _obb_line(60, 60, 80, 80)},
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW, iou_threshold=0.5)

    out = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))
    assert len(out) == 2
    assert sr.read_decisions(staged)["a.txt"] == sr.ACCEPTED_ADD_NEW


def test_add_new_keeps_the_existing_lines_byte_for_byte(tmp_path):
    original = _obb_line(0, 0, 20, 20)
    source, _ = _wired(
        tmp_path, {"a.txt": original}, {"a.txt": _obb_line(60, 60, 80, 80)}
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    text = (Path(source.path) / "labels" / "a.txt").read_text()
    assert text.startswith(original)


def test_add_new_preserves_crlf_line_endings_byte_for_byte(tmp_path):
    """ "Verbatim" means bytes, not lines.

    read_text() applies universal-newline translation, so a label file
    hand-edited on Windows would come back LF-only -- every existing line
    changed, in the one branch whose entire purpose is not to touch them.
    """
    original = _obb_line(0, 0, 20, 20).replace("\n", "\r\n")
    source, _ = _wired(
        tmp_path, {"a.txt": original}, {"a.txt": _obb_line(60, 60, 80, 80)}
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    written = (Path(source.path) / "labels" / "a.txt").read_bytes()
    assert written.startswith(original.encode())


def test_reject_changes_nothing_but_records_the_decision(tmp_path):
    original = _obb_line(0, 0, 20, 20)
    source, staged = _wired(
        tmp_path, {"a.txt": original}, {"a.txt": _obb_line(60, 60, 80, 80)}
    )

    sr.reject_frame(source, "a.txt")

    assert (Path(source.path) / "labels" / "a.txt").read_text() == original
    assert sr.read_decisions(staged)["a.txt"] == sr.REJECTED


def test_accepting_a_frame_with_no_existing_label_creates_one(tmp_path):
    source, _ = _wired(tmp_path, {}, {"a.txt": _obb_line(60, 60, 80, 80)})

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    assert (Path(source.path) / "labels" / "a.txt").is_file()


def test_the_first_accept_snapshots_and_later_ones_do_not(tmp_path):
    source, staged = _wired(
        tmp_path,
        {
            "a.txt": "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",
            "b.txt": "0 0.3 0.3 0.4 0.3 0.4 0.4 0.3 0.4\n",
        },
        {"a.txt": _obb_line(60, 60, 80, 80), "b.txt": _obb_line(60, 60, 80, 80)},
    )
    before_a = (Path(source.path) / "labels" / "a.txt").read_text()

    sr.accept_frame(source, "a.txt", mode=MergeMode.OVERWRITE)
    sr.accept_frame(source, "b.txt", mode=MergeMode.OVERWRITE)

    assert (staged / "labels_before" / "a.txt").read_text() == before_a


def test_accepting_polygons_into_an_obb_source_promotes_it(tmp_path):
    poly = (
        "0 "
        + " ".join(
            f"{x / 200:.6f} {y / 100:.6f}"
            for x, y in [(60, 60), (80, 62), (85, 80), (70, 88), (58, 78)]
        )
        + "\n"
    )
    source, _ = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": poly},
        level="obb",
        target_level="polygon",
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    assert source.level == "polygon"
    out = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))
    assert all(r.level.name == "POLYGON" for r in out)


def test_a_promoted_quad_does_not_read_back_as_an_obb(tmp_path):
    poly = (
        "0 "
        + " ".join(
            f"{x / 200:.6f} {y / 100:.6f}"
            for x, y in [(60, 60), (80, 62), (85, 80), (70, 88), (58, 78)]
        )
        + "\n"
    )
    source, _ = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": poly},
        level="obb",
        target_level="polygon",
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    lifted = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))[0]
    assert lifted.points.shape == (5, 2)


def test_promotion_does_not_drift_the_lifted_coordinates(tmp_path):
    poly = (
        "0 "
        + " ".join(
            f"{x / 200:.6f} {y / 100:.6f}"
            for x, y in [(60, 60), (80, 62), (85, 80), (70, 88), (58, 78)]
        )
        + "\n"
    )
    source, _ = _wired(
        tmp_path,
        {"a.txt": _obb_line(10, 10, 30, 30)},
        {"a.txt": poly},
        level="obb",
        target_level="polygon",
    )
    before = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))[0]

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    after = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))[0]
    np.testing.assert_allclose(after.points[:4], before.points, atol=0.05)


def test_staged_below_the_source_level_is_lifted_to_it(tmp_path):
    source, _ = _wired(
        tmp_path,
        {
            "a.txt": "0 "
            + " ".join(
                f"{v:.6f}" for v in [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.15, 0.25, 0.1, 0.2]
            )
            + "\n"
        },
        {"a.txt": _obb_line(120, 60, 160, 90)},
        level="polygon",
        target_level="obb",
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    assert source.level == "polygon"
    out = read_label_file(Path(source.path) / "labels" / "a.txt", (100, 200))
    assert len(out) == 2


def test_accept_all_only_touches_undecided_frames(tmp_path):
    source, staged = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20), "b.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": _obb_line(60, 60, 80, 80), "b.txt": _obb_line(60, 60, 80, 80)},
    )
    sr.reject_frame(source, "a.txt")

    changed = sr.accept_all(source, mode=MergeMode.OVERWRITE)

    assert changed == 1
    assert sr.read_decisions(staged)["a.txt"] == sr.REJECTED
    assert sr.read_decisions(staged)["b.txt"] == sr.ACCEPTED_OVERWRITE


def test_a_review_is_complete_when_every_frame_is_decided(tmp_path):
    source, _ = _wired(
        tmp_path,
        {},
        {"a.txt": _obb_line(60, 60, 80, 80), "b.txt": _obb_line(60, 60, 80, 80)},
    )

    sr.reject_frame(source, "a.txt")
    assert not sr.is_complete(source)

    sr.reject_frame(source, "b.txt")
    assert sr.is_complete(source)


def test_finish_review_removes_staging_and_clears_the_source(tmp_path):
    source, staged = _wired(tmp_path, {}, {"a.txt": _obb_line(60, 60, 80, 80)})
    sr.reject_frame(source, "a.txt")

    sr.finish_review(source, project_dir=None)

    assert source.staged_review is None
    assert not staged.exists()


def test_rejecting_everything_leaves_the_source_reviewed(tmp_path):
    """Nothing machine-derived landed, so nothing was un-confirmed."""
    source, _ = _wired(tmp_path, {}, {"a.txt": _obb_line(60, 60, 80, 80)})
    source.reviewed = True
    sr.reject_all(source)

    sr.finish_review(source, project_dir=None)

    assert source.reviewed is True


def test_accepting_any_frame_marks_the_source_unreviewed(tmp_path):
    source, _ = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": _obb_line(60, 60, 80, 80)},
    )
    source.reviewed = True
    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    sr.finish_review(source, project_dir=None)

    assert source.reviewed is False


def test_revert_after_a_mixed_review_restores_byte_identical_labels(tmp_path):
    """Spec test 4: accept a mix of frames in BOTH modes, then revert."""
    source, staged = _wired(
        tmp_path,
        {
            "a.txt": _obb_line(0, 0, 20, 20),
            "b.txt": _obb_line(5, 5, 25, 25),
            "c.txt": _obb_line(9, 9, 29, 29),
        },
        {
            "a.txt": _obb_line(60, 60, 80, 80),
            "b.txt": _obb_line(70, 70, 90, 90),
            "c.txt": _obb_line(80, 80, 95, 95),
        },
    )
    before = {
        p.name: p.read_bytes() for p in (Path(source.path) / "labels").rglob("*.txt")
    }

    sr.accept_frame(source, "a.txt", mode=MergeMode.OVERWRITE)
    sr.accept_frame(source, "b.txt", mode=MergeMode.ADD_NEW)
    sr.reject_frame(source, "c.txt")
    sr.revert_review(source, staged)

    after = {
        p.name: p.read_bytes() for p in (Path(source.path) / "labels").rglob("*.txt")
    }
    assert after == before


def test_revert_after_a_promoting_accept_restores_the_level_too(tmp_path):
    poly = (
        "0 "
        + " ".join(
            f"{x / 200:.6f} {y / 100:.6f}"
            for x, y in [(60, 60), (80, 62), (85, 80), (70, 88), (58, 78)]
        )
        + "\n"
    )
    source, staged = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": poly},
        level="obb",
        target_level="polygon",
    )
    before = (Path(source.path) / "labels" / "a.txt").read_bytes()

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)
    sr.revert_review(source, staged)

    assert source.level == "obb"
    assert (Path(source.path) / "labels" / "a.txt").read_bytes() == before


@pytest.mark.parametrize("producer", ["sam2", "sam3", "inference"])
def test_accept_is_producer_agnostic(tmp_path, producer):
    """Spec test 5: identical staged content -> identical outcome, always.

    This is the test that fails if `producer` ever becomes load-bearing again.
    """
    source, _ = _wired(
        tmp_path,
        {"a.txt": _obb_line(0, 0, 20, 20)},
        {"a.txt": _obb_line(60, 60, 80, 80)},
        producer=producer,
    )

    sr.accept_frame(source, "a.txt", mode=MergeMode.ADD_NEW)

    assert (Path(source.path) / "labels" / "a.txt").read_text() == (
        _obb_line(0, 0, 20, 20) + _obb_line(60, 60, 80, 80)
    )


def test_accept_frame_finds_a_nested_image_by_relative_path(tmp_path):
    """`_image_for`'s nested relative-path lookup, pinned through a real accept.

    Every other accept_frame/accept_all test in this suite uses a flat key
    ("a.txt"). A source's images/labels can be nested (images/train/...,
    as source_import.py's materializer can produce), and `_image_for`
    resolves the staged label's rel path directly against the source's own
    images/ tree -- this is the only test that actually exercises that
    nested lookup end to end through accept_frame, rather than only through
    `staged_frames`' sorting (which never reaches `_image_for` at all).
    """
    source, staged = _wired(
        tmp_path,
        {},
        {"train/a.txt": _obb_line(60, 60, 80, 80)},
    )

    sr.accept_frame(source, "train/a.txt", mode=MergeMode.ADD_NEW)

    out = read_label_file(Path(source.path) / "labels" / "train" / "a.txt", (100, 200))
    assert len(out) == 1
    assert sr.read_decisions(staged)["train/a.txt"] == sr.ACCEPTED_ADD_NEW


def test_accept_frame_raises_when_the_staged_frames_image_is_missing(tmp_path):
    """The new behaviour, pinned: `accept_frame` RAISES on a ghost staged
    label (no origin image), rather than silently skipping it as the old
    sibling-source accept used to (it warned and counted the label
    `orphaned`). `accept_frame` is an explicit user action on one named
    frame, so silently doing nothing would be the worse failure here --
    this is deliberate, not a regression to fix.
    """
    original = _obb_line(0, 0, 20, 20)
    source = _source(tmp_path, {"a.txt": original})
    staged = _staging(tmp_path, {"a.txt": _obb_line(60, 60, 80, 80)})
    # No _image(source, "a") call: the staged label's origin image genuinely
    # does not exist.
    source.staged_review = StagedReview(
        staged_path=str(staged), target_level="obb", producer="sam2"
    )

    with pytest.raises(RuntimeError, match="No image found"):
        sr.accept_frame(source, "a.txt", mode=MergeMode.OVERWRITE)

    assert (Path(source.path) / "labels" / "a.txt").read_text() == original
    assert sr.read_decisions(staged) == {}


def test_accept_all_propagates_a_missing_image_error_and_stays_recoverable(tmp_path):
    """accept_all over a staging dir with one ghost frame aborts loudly
    (Finding 2 above) rather than silently skipping it -- a user who hits
    this rejects the ghost frame (`reject_frame`) to unblock the rest of a
    bulk accept.

    Pins the recovery story around that abort: the frame decided before the
    ghost frame keeps its decision and its accepted content on disk, the
    ghost frame itself never wrote anything, and `revert_review` -- using
    the snapshot `accept_frame` took before the abort -- still restores the
    source to its pre-review state.
    """
    original_a = _obb_line(0, 0, 20, 20)
    source = _source(tmp_path, {"a.txt": original_a})
    _image(source, "a")
    # No _image(source, "ghost"): this staged label's image genuinely does
    # not exist. "a.txt" sorts before "ghost.txt", so accept_all reaches and
    # accepts it before hitting the ghost frame.
    staged = _staging(
        tmp_path,
        {
            "a.txt": _obb_line(60, 60, 80, 80),
            "ghost.txt": _obb_line(60, 60, 80, 80),
        },
    )
    source.staged_review = StagedReview(
        staged_path=str(staged), target_level="obb", producer="sam2"
    )

    with pytest.raises(RuntimeError, match="No image found"):
        sr.accept_all(source, mode=MergeMode.OVERWRITE)

    # The frame accepted before the abort kept its decision and its content.
    assert sr.read_decisions(staged)["a.txt"] == sr.ACCEPTED_OVERWRITE
    assert (Path(source.path) / "labels" / "a.txt").read_text() != original_a
    # The ghost frame itself never got a decision or wrote anything.
    assert "ghost.txt" not in sr.read_decisions(staged)
    assert not (Path(source.path) / "labels" / "ghost.txt").exists()

    sr.revert_review(source, staged)

    assert (Path(source.path) / "labels" / "a.txt").read_text() == original_a
    assert sr.read_decisions(staged) == {}
