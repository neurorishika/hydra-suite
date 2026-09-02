import shutil
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.data.al.merge import MergeMode
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs import staged_review as sr
from hydra_suite.detectkit.jobs.sam2_escalation import EscalationRequest, run_escalation


class _FakeExec:
    """Returns a full-object mask for detection 0, empty mask for others."""

    def __init__(self):
        self.calls = 0

    def set_image(self, img):
        pass

    def segment(self, box, pos, neg):
        self.calls += 1
        if self.calls == 1:
            m = np.zeros((100, 100), bool)
            m[10:40, 10:40] = True
            return m, 0.9
        return np.zeros((100, 100), bool), 0.0  # -> fallback


def _make_source(tmp_path):
    root = tmp_path / "sources" / "orig"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
    # two OBB detections
    (root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n" "0 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n"
    )
    (root / "classes.txt").write_text("ant\n")
    return OBBSource(path=str(root), name="orig", level="obb")


def test_escalation_stages_without_touching_canonical_labels(tmp_path):
    src = _make_source(tmp_path)
    original_label_text = (Path(src.path) / "labels" / "a.txt").read_text()
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )

    result = run_escalation(req, _FakeExec())

    assert result.staged == ["orig"]
    assert result.primed == 1 and result.fell_back == 1
    assert result.skipped == []

    # Canonical source untouched.
    assert (Path(src.path) / "labels" / "a.txt").read_text() == original_label_text
    assert src.level == "obb"
    assert src.reviewed is True

    # No new OBBSource registered.
    assert [s.name for s in project.sources] == ["orig"]

    pending = src.staged_review
    assert pending is not None
    assert pending.target_level == "polygon"
    assert pending.producer_variant == "sam2.1-hiera-base_plus"
    staged_label = Path(pending.staged_path) / "labels" / "a.txt"
    assert staged_label.exists() and len(staged_label.read_text().splitlines()) == 2
    assert (Path(pending.staged_path) / "classes.txt").read_text() == "ant\n"


def test_escalation_routes_duplicate_display_names_by_source_path(tmp_path):
    first = _make_source(tmp_path / "first")
    second = _make_source(tmp_path / "second")
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[first, second])
    req = EscalationRequest(
        project=project,
        source_names=[first.name],
        source_paths=[first.path],
        variant="sam2.1-hiera-base_plus",
    )

    result = run_escalation(req, _FakeExec())

    assert result.staged == [first.name]
    assert first.staged_review is not None
    assert second.staged_review is None


def test_escalation_request_preserves_positional_overwrite_argument():
    req = EscalationRequest(object(), ["source"], "variant", True)

    assert req.overwrite is True
    assert req.source_paths == []


def test_no_label_file_is_staged_for_an_image_with_no_source_boxes(tmp_path):
    """`write_label_file([])` creates a zero-byte file, and `staged_frames()`
    would count it as a reviewable frame; `accept_frame(..., OVERWRITE)`
    would then overwrite the source's real labels with nothing. A source
    image with no boxes to escalate (an empty or absent label file) must
    get no staged label file at all, matching the contract
    `inference_stager.py` already documents ("Frames with no detections
    are not staged at all").
    """
    src = _make_source(tmp_path)
    # A second image with an empty label file: read_boxes_from_label
    # returns [] for it, so `records` stays empty.
    cv2.imwrite(
        str(Path(src.path) / "images" / "b.jpg"), np.zeros((100, 100, 3), np.uint8)
    )
    (Path(src.path) / "labels" / "b.txt").write_text("")
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )

    result = run_escalation(req, _FakeExec())

    staged_labels = Path(result.staged and src.staged_review.staged_path) / "labels"
    assert (staged_labels / "a.txt").exists()
    assert not (staged_labels / "b.txt").exists()


class _BleedingExec:
    """Returns a full-frame mask for every box -- simulates SAM2 predicting
    well past the OBB it was prompted with (soft box guidance, not a hard
    crop)."""

    def set_image(self, img):
        self._shape = img.shape[:2]

    def segment(self, box, pos, neg):
        return np.ones(self._shape, bool), 0.9


def test_escalation_polygon_stays_within_original_obb_when_sam2_bleeds(tmp_path):
    """Regression: a full-frame SAM2 mask must be clipped to the source OBB
    before contour extraction, not written out unbounded."""
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )

    result = run_escalation(req, _BleedingExec())

    assert result.primed == 2
    staged_label = Path(src.staged_review.staged_path) / "labels" / "a.txt"
    lines = staged_label.read_text().splitlines()
    assert len(lines) == 2

    # Source OBBs (normalized, 100x100 image): [0.1,0.4]x[0.1,0.4] and
    # [0.6,0.9]x[0.6,0.9] -- the staged polygon for each detection must stay
    # within its own OBB's extent, not the full [0,1] frame.
    expected_bounds = [(0.1, 0.4), (0.6, 0.9)]
    for line, (lo, hi) in zip(lines, expected_bounds):
        vals = [float(v) for v in line.split()[1:]]
        xs, ys = vals[0::2], vals[1::2]
        assert min(xs) >= lo - 0.02 and max(xs) <= hi + 0.02
        assert min(ys) >= lo - 0.02 and max(ys) <= hi + 0.02


def test_rerun_without_overwrite_skips_existing_pending(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    first_staged_path = src.staged_review.staged_path

    result2 = run_escalation(req, _FakeExec(), overwrite=False)

    assert result2.staged == []
    assert len(result2.skipped) == 1 and result2.skipped[0][0] == "orig"
    assert src.staged_review.staged_path == first_staged_path  # untouched


def test_rerun_with_overwrite_restages(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    first_staged_path = src.staged_review.staged_path

    result2 = run_escalation(req, _FakeExec(), overwrite=True)

    assert result2.staged == ["orig"]
    assert result2.skipped == []
    # Same content-hashed staging dir reused, not accumulated.
    assert src.staged_review.staged_path == first_staged_path
    assert Path(first_staged_path).is_dir()


def test_accept_all_promotes_labels_and_finish_review_resets_reviewed(tmp_path):
    """Ported from accept_pending_escalation's whole-source promote: same
    promoted content, same `reviewed` reset, now via frame-granular
    accept_all(mode=OVERWRITE) + finish_review rather than one wholesale
    rmtree+copytree call."""
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    staged_label_text = (
        Path(src.staged_review.staged_path) / "labels" / "a.txt"
    ).read_text()
    staged_path = src.staged_review.staged_path

    accepted = sr.accept_all(src, mode=MergeMode.OVERWRITE)
    sr.finish_review(src, tmp_path)

    assert accepted == 1
    assert src.staged_review is None
    assert src.level == "polygon"
    assert src.reviewed is False
    assert (Path(src.path) / "labels" / "a.txt").read_text() == staged_label_text
    assert not Path(staged_path).exists()


def _make_nested_source(tmp_path):
    """A source whose images/labels use a nested split layout (images/train/...),
    as dataset_inspector.py's directory-layout scan supports and as
    source_import.py's materializer can produce."""
    root = tmp_path / "sources" / "nested"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    cv2.imwrite(
        str(root / "images" / "train" / "a.jpg"), np.zeros((100, 100, 3), np.uint8)
    )
    (root / "labels" / "train" / "a.txt").write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"
    )
    (root / "classes.txt").write_text("ant\n")
    return OBBSource(path=str(root), name="nested", level="obb")


def test_escalation_stages_nested_image_layout_correctly(tmp_path):
    """Regression: staging must mirror the source's directory structure, not
    flatten to top-level images/*.* -- a split layout (images/train/...) has
    zero images at the top level, which used to silently stage nothing and
    made accept() delete every label with no staged replacement."""
    src = _make_nested_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["nested"], variant="sam2.1-hiera-base_plus"
    )

    result = run_escalation(req, _FakeExec())

    assert result.staged == ["nested"]
    staged_label = Path(src.staged_review.staged_path) / "labels" / "train" / "a.txt"
    assert staged_label.exists()
    assert len(staged_label.read_text().splitlines()) == 1


def test_accept_frame_refuses_when_its_staged_label_is_missing(tmp_path):
    """Ported from accept_pending_escalation's whole-source pre-check.

    That whole-source check ("does staging have a label for every image the
    source currently has one for") is gone by design: frame-granular
    accept_frame only ever touches the ONE frame it is given, so it cannot
    delete a label it has nothing staged to replace -- there is no wholesale
    delete step left to guard against. What remains portable is the
    per-frame guard: accepting a frame whose OWN staged label is missing
    must refuse and leave the source untouched, exactly as Task 7 wired it.
    """
    src = _make_source(tmp_path)
    # Add a second image/label pair.
    cv2.imwrite(
        str(Path(src.path) / "images" / "b.jpg"), np.zeros((100, 100, 3), np.uint8)
    )
    (Path(src.path) / "labels" / "b.txt").write_text(
        "0 0.2 0.2 0.5 0.2 0.5 0.5 0.2 0.5\n"
    )
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())

    # Simulate an image that failed to stage (e.g. cv2.imread returned None
    # during escalation): remove its staged label after the fact.
    staged_b = Path(src.staged_review.staged_path) / "labels" / "b.txt"
    staged_b.unlink()

    original_a = (Path(src.path) / "labels" / "a.txt").read_text()
    original_b = (Path(src.path) / "labels" / "b.txt").read_text()

    with pytest.raises(RuntimeError):
        sr.accept_frame(src, "b.txt", mode=MergeMode.OVERWRITE)

    # Nothing was touched -- refusal happens before any write, and the frame
    # with a real staged label ('a.txt') is untouched too, since accept_frame
    # was never asked to accept it.
    assert src.staged_review is not None
    assert (Path(src.path) / "labels" / "a.txt").read_text() == original_a
    assert (Path(src.path) / "labels" / "b.txt").read_text() == original_b


# test_reject_pending_escalation_discards_staging_leaves_source_untouched is
# deleted, not ported: reject_pending_escalation is dead after the review
# dialog's removal (finish_review calls remove_staged_escalation_dir
# directly), and the property it pinned -- rejecting a frame changes nothing
# on disk -- is already covered for the new path by
# test_reject_changes_nothing_but_records_the_decision in
# tests/test_detectkit_staged_review.py.


def _make_multiclass_source(tmp_path):
    """A two-class source: one instance of class 0, one of class 1."""
    root = tmp_path / "sources" / "multi"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
    (root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n" "1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n"
    )
    (root / "classes.txt").write_text("ant\nbee\n")
    return OBBSource(path=str(root), name="multi", level="obb")


def test_escalation_preserves_per_instance_class_ids(tmp_path):
    """Regression: escalation used to write class_id=0 for every staged label.

    Accepting now overwrites the source's OWN labels in place (rather than
    landing in a throwaway `_seg` sibling), so a hardcoded 0 destroyed a
    multi-class source's per-instance class assignments.
    """
    src = _make_multiclass_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["multi"], variant="sam2.1-hiera-base_plus"
    )

    run_escalation(req, _FakeExec())

    staged_lines = (
        (Path(src.staged_review.staged_path) / "labels" / "a.txt")
        .read_text()
        .splitlines()
    )
    assert [line.split()[0] for line in staged_lines] == ["0", "1"]

    sr.accept_frame(src, "a.txt", mode=MergeMode.OVERWRITE)

    promoted_lines = (Path(src.path) / "labels" / "a.txt").read_text().splitlines()
    assert [line.split()[0] for line in promoted_lines] == ["0", "1"]


def test_escalation_stages_non_jpg_png_images(tmp_path):
    """Regression: the image walk hardcoded .jpg/.jpeg/.png, so a source with
    e.g. a .bmp image staged no label for it and accept() then refused
    forever on the missing-labels check. It must use DetectKit's canonical
    IMG_EXTS instead."""
    root = tmp_path / "sources" / "bmp"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "a.bmp"), np.zeros((100, 100, 3), np.uint8))
    (root / "labels" / "a.txt").write_text("0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n")
    (root / "classes.txt").write_text("ant\n")
    src = OBBSource(path=str(root), name="bmp", level="obb")
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["bmp"], variant="sam2.1-hiera-base_plus"
    )

    run_escalation(req, _FakeExec())

    assert (Path(src.staged_review.staged_path) / "labels" / "a.txt").exists()

    sr.accept_frame(src, "a.txt", mode=MergeMode.OVERWRITE)  # must not refuse
    sr.finish_review(src, tmp_path)

    assert src.level == "polygon"
    assert src.staged_review is None


def test_read_boxes_from_label_parses_class_id(tmp_path):
    from hydra_suite.detectkit.jobs.sam2_prompts import read_boxes_from_label

    label = tmp_path / "a.txt"
    label.write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"  # obb, class 0
        "2 0.5 0.5 0.2 0.2\n"  # aabb, class 2
    )

    boxes = read_boxes_from_label(label, 100, 100)

    assert [b.class_id for b in boxes] == [0, 2]


def test_revert_review_fails_loudly_if_clearing_source_labels_fails(
    tmp_path, monkeypatch
):
    """Ported from accept_pending_escalation's clearing-failure guard.

    `revert_review` is now the only rmtree-then-copytree on a source's
    labels (accept_frame's OVERWRITE/ADD_NEW paths write in place, never
    rmtree). A failed pre-copy delete there must still raise BEFORE any copy
    starts, exactly as it had to for the old wholesale accept -- an
    unsuppressed rmtree that fails must not fall through to a copytree that
    then blows up with FileExistsError, leaving labels/ half-deleted.
    """
    import shutil as _shutil

    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    staged_root = src.staged_review.staged_path
    # accept_frame snapshots the pre-review labels/ and then writes the
    # accepted content -- revert_review needs that snapshot to exist.
    sr.accept_frame(src, "a.txt", mode=MergeMode.OVERWRITE)
    post_accept = (Path(src.path) / "labels" / "a.txt").read_text()

    def _boom(*args, **kwargs):
        raise PermissionError("locked")

    monkeypatch.setattr(_shutil, "rmtree", _boom)
    copied = []
    monkeypatch.setattr(
        _shutil, "copytree", lambda *a, **k: copied.append(a)  # must never run
    )

    with pytest.raises(PermissionError):
        sr.revert_review(src, staged_root)

    assert copied == []
    assert (Path(src.path) / "labels" / "a.txt").read_text() == post_accept


def test_accept_frame_works_when_source_has_no_labels_dir(tmp_path):
    """The pre-write mkdir is guarded on existence: a source with no labels/
    at write time must still accept a frame cleanly.

    Boxes still have to come from SOMEWHERE for SAM2 to escalate (it
    prompts from the source's own OBBs), so `_make_source` -- which has
    two boxes -- runs the escalation first; the source's labels/ dir is
    then removed to exercise the guard `accept_frame` needs when it writes
    the accepted content back. (Escalating a source with genuinely zero
    boxes now stages nothing for that frame at all -- see
    `test_no_label_file_is_staged_for_an_image_with_no_source_boxes`.)
    """
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    shutil.rmtree(Path(src.path) / "labels")

    sr.accept_frame(src, "a.txt", mode=MergeMode.OVERWRITE)

    assert src.level == "polygon"
    assert (Path(src.path) / "labels" / "a.txt").exists()


def test_reject_refuses_to_delete_out_of_bounds_staged_path(tmp_path):
    """staged_path round-trips through the saved project file, so it is
    untrusted input from disk. A path outside the project's
    artifacts/pending_escalations/ must be left alone (not recursively
    deleted). Re-pointed at `remove_staged_escalation_dir` directly (rather
    than through the now-deleted `reject_pending_escalation`): that is the
    delete primitive `finish_review` calls, and it alone is what bounds this
    rmtree -- losing this coverage would be a real regression.
    """
    from hydra_suite.detectkit.jobs.sam2_escalation import remove_staged_escalation_dir

    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("do not delete\n")

    attempted = remove_staged_escalation_dir(str(outside), tmp_path)

    assert attempted is False
    assert (outside / "keep.txt").exists()


def test_reject_refuses_to_delete_filesystem_root(tmp_path):
    from hydra_suite.detectkit.jobs.sam2_escalation import (
        _is_safe_to_delete,
        remove_staged_escalation_dir,
    )

    assert _is_safe_to_delete("/", tmp_path) is False
    assert _is_safe_to_delete("/") is False
    assert _is_safe_to_delete("", tmp_path) is False
    # The staging root itself is not deletable either -- only entries in it.
    assert (
        _is_safe_to_delete(tmp_path / "artifacts" / "pending_escalations", tmp_path)
        is False
    )
    assert (
        _is_safe_to_delete(
            tmp_path / "artifacts" / "pending_escalations" / "a-b-c", tmp_path
        )
        is True
    )
    # Without a project_dir, the structural shape is what is checked.
    assert (
        _is_safe_to_delete(tmp_path / "artifacts" / "pending_escalations" / "a-b-c")
        is True
    )
    assert _is_safe_to_delete(tmp_path / "somewhere" / "a-b-c") is False

    # Re-pointed at remove_staged_escalation_dir directly, as above: it must
    # refuse (not raise) on "/" too.
    assert remove_staged_escalation_dir("/", tmp_path) is False


def test_accept_frame_without_pending_review_raises():
    src = OBBSource(name="orig", level="obb")
    with pytest.raises(ValueError):
        sr.accept_frame(src, "a.txt", mode=MergeMode.OVERWRITE)


# test_reject_without_pending_raises is deleted, not ported:
# reject_pending_escalation is dead. Its lower-level replacement,
# remove_staged_escalation_dir, is a delete PRIMITIVE with no "pending
# review" concept to validate against -- it takes a bare path, and its
# refuse-out-of-bounds behaviour is already covered by the two tests above.


def test_worker_runs_with_injected_executor(tmp_path):
    from hydra_suite.detectkit.jobs.sam2_escalation import Sam2EscalationWorker

    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project,
        source_names=["orig"],
        variant="sam2.1-hiera-base_plus",
    )
    worker = Sam2EscalationWorker(req, executor=_FakeExec())
    captured = {}
    worker.result_ready.connect(lambda r: captured.update(staged=r.staged))
    worker.execute()  # call directly (no thread) — BaseWorker pattern
    assert captured["staged"] == ["orig"]


def test_on_mutated_fires_when_the_staged_pointer_is_written(tmp_path):
    """Without it the pointer lives only in memory until the run returns,
    so a failure or an app close orphans the whole staging directory --
    the review bar keys off the pointer, not off the files on disk."""
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )

    seen = []
    run_escalation(req, _FakeExec(), on_mutated=lambda: seen.append(src.staged_review))

    assert len(seen) == 1
    assert seen[0] is not None and seen[0] is src.staged_review


def test_replacing_a_pointer_clears_it_before_the_new_one_is_written(tmp_path):
    """The old staging dir is deleted at that moment, so leaving the
    pointer set makes a failure in between offer a review that cannot be
    opened or dismissed."""
    from hydra_suite.detectkit.gui.models import StagedReview

    src = _make_source(tmp_path)
    stale = tmp_path / "artifacts" / "pending_escalations" / "stale"
    (stale / "labels").mkdir(parents=True)
    src.staged_review = StagedReview(staged_path=str(stale), producer="sam2")
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project,
        source_names=["orig"],
        variant="sam2.1-hiera-base_plus",
        overwrite=True,
    )

    seen = []
    run_escalation(
        req,
        _FakeExec(),
        overwrite=True,
        on_mutated=lambda: seen.append(src.staged_review),
    )

    assert seen[0] is None  # the clear, persisted on its own
    assert not stale.exists()
    assert seen[-1] is not None and seen[-1].staged_path != str(stale)
