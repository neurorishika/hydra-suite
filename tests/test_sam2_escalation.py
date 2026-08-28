import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.sam2_escalation import (
    EscalationRequest,
    accept_pending_escalation,
    reject_pending_escalation,
    run_escalation,
)


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

    pending = src.pending_escalation
    assert pending is not None
    assert pending.target_level == "polygon"
    assert pending.sam2_variant == "sam2.1-hiera-base_plus"
    staged_label = Path(pending.staged_path) / "labels" / "a.txt"
    assert staged_label.exists() and len(staged_label.read_text().splitlines()) == 2
    assert (Path(pending.staged_path) / "classes.txt").read_text() == "ant\n"


def test_rerun_without_overwrite_skips_existing_pending(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    first_staged_path = src.pending_escalation.staged_path

    result2 = run_escalation(req, _FakeExec(), overwrite=False)

    assert result2.staged == []
    assert len(result2.skipped) == 1 and result2.skipped[0][0] == "orig"
    assert src.pending_escalation.staged_path == first_staged_path  # untouched


def test_rerun_with_overwrite_restages(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    first_staged_path = src.pending_escalation.staged_path

    result2 = run_escalation(req, _FakeExec(), overwrite=True)

    assert result2.staged == ["orig"]
    assert result2.skipped == []
    # Same content-hashed staging dir reused, not accumulated.
    assert src.pending_escalation.staged_path == first_staged_path
    assert Path(first_staged_path).is_dir()


def test_accept_pending_escalation_promotes_labels_and_resets_reviewed(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    staged_label_text = (
        Path(src.pending_escalation.staged_path) / "labels" / "a.txt"
    ).read_text()
    staged_path = src.pending_escalation.staged_path

    accept_pending_escalation(src)

    assert src.pending_escalation is None
    assert src.level == "polygon"
    assert src.reviewed is False
    assert src.sam2_variant == "sam2.1-hiera-base_plus"
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
    staged_label = (
        Path(src.pending_escalation.staged_path) / "labels" / "train" / "a.txt"
    )
    assert staged_label.exists()
    assert len(staged_label.read_text().splitlines()) == 1


def test_accept_refuses_when_staged_labels_missing_files(tmp_path):
    """If staging skipped a label (e.g. an unreadable image during escalation),
    accept must refuse rather than deleting that image's original label with
    nothing to replace it."""
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
    staged_b = Path(src.pending_escalation.staged_path) / "labels" / "b.txt"
    staged_b.unlink()

    original_a = (Path(src.path) / "labels" / "a.txt").read_text()
    original_b = (Path(src.path) / "labels" / "b.txt").read_text()

    with pytest.raises(RuntimeError):
        accept_pending_escalation(src)

    # Nothing was touched -- refusal happens before any deletion.
    assert src.pending_escalation is not None
    assert (Path(src.path) / "labels" / "a.txt").read_text() == original_a
    assert (Path(src.path) / "labels" / "b.txt").read_text() == original_b


def test_reject_pending_escalation_discards_staging_leaves_source_untouched(tmp_path):
    src = _make_source(tmp_path)
    original_label_text = (Path(src.path) / "labels" / "a.txt").read_text()
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    staged_path = src.pending_escalation.staged_path

    reject_pending_escalation(src)

    assert src.pending_escalation is None
    assert src.level == "obb"
    assert src.reviewed is True
    assert (Path(src.path) / "labels" / "a.txt").read_text() == original_label_text
    assert not Path(staged_path).exists()


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
        (Path(src.pending_escalation.staged_path) / "labels" / "a.txt")
        .read_text()
        .splitlines()
    )
    assert [line.split()[0] for line in staged_lines] == ["0", "1"]

    accept_pending_escalation(src)

    promoted_lines = (Path(src.path) / "labels" / "a.txt").read_text().splitlines()
    assert [line.split()[0] for line in promoted_lines] == ["0", "1"]


def test_read_boxes_from_label_parses_class_id(tmp_path):
    from hydra_suite.detectkit.jobs.sam2_prompts import read_boxes_from_label

    label = tmp_path / "a.txt"
    label.write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"  # obb, class 0
        "2 0.5 0.5 0.2 0.2\n"  # aabb, class 2
    )

    boxes = read_boxes_from_label(label, 100, 100)

    assert [b.class_id for b in boxes] == [0, 2]


def test_accept_without_pending_raises():
    src = OBBSource(name="orig", level="obb")
    with pytest.raises(ValueError):
        accept_pending_escalation(src)


def test_reject_without_pending_raises():
    src = OBBSource(name="orig", level="obb")
    with pytest.raises(ValueError):
        reject_pending_escalation(src)


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
