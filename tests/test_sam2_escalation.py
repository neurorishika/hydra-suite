import types
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.detectkit.gui.models import OBBSource
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


def test_escalation_writes_reviewed_false_derived_source(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    result = run_escalation(req, _FakeExec())

    assert result.derived == ["orig_seg"]
    assert result.primed == 1 and result.fell_back == 1
    new = [s for s in project.sources if s.name == "orig_seg"][0]
    assert new.level == "polygon" and new.reviewed is False
    assert new.derived_from == "orig" and new.sam2_variant == "sam2.1-hiera-base_plus"
    label = Path(new.path) / "labels" / "a.txt"
    assert label.exists() and len(label.read_text().splitlines()) == 2
    assert (Path(new.path) / "images" / "a.jpg").exists()  # image copied


def test_rerun_without_overwrite_skips_and_preserves_existing(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    result = run_escalation(req, _FakeExec())
    assert result.derived == ["orig_seg"]

    derived_sources = [s for s in project.sources if s.name == "orig_seg"]
    assert len(derived_sources) == 1
    derived = derived_sources[0]
    # Simulate the user having reviewed the derived source.
    derived.reviewed = True
    label_path = Path(derived.path) / "labels" / "a.txt"
    original_label_text = label_path.read_text()

    req2 = EscalationRequest(
        project=project,
        source_names=["orig"],
        variant="sam2.1-hiera-base_plus",
        overwrite=False,
    )
    result2 = run_escalation(req2, _FakeExec())

    assert result2.derived == []
    assert len(result2.skipped) == 1
    assert result2.skipped[0][0] == "orig"

    seg_sources = [s for s in project.sources if s.name == "orig_seg"]
    assert len(seg_sources) == 1  # no duplicate entry
    assert seg_sources[0] is derived
    assert seg_sources[0].reviewed is True  # not clobbered
    assert label_path.read_text() == original_label_text  # not clobbered


def test_rerun_with_overwrite_replaces_in_place(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    derived = [s for s in project.sources if s.name == "orig_seg"][0]
    derived.reviewed = True

    req2 = EscalationRequest(
        project=project,
        source_names=["orig"],
        variant="sam2.1-hiera-base_plus",
        overwrite=True,
    )
    result2 = run_escalation(req2, _FakeExec(), overwrite=True)

    assert result2.derived == ["orig_seg"]
    assert result2.skipped == []
    seg_sources = [s for s in project.sources if s.name == "orig_seg"]
    assert len(seg_sources) == 1  # still exactly one entry, no duplicate
    assert seg_sources[0].reviewed is False  # replaced in place, fresh state


def test_worker_runs_with_injected_executor(tmp_path):
    from hydra_suite.detectkit.jobs.sam2_escalation import (
        EscalationRequest,
        Sam2EscalationWorker,
    )

    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project,
        source_names=["orig"],
        variant="sam2.1-hiera-base_plus",
    )
    worker = Sam2EscalationWorker(req, executor=_FakeExec())
    captured = {}
    worker.result_ready.connect(lambda r: captured.update(derived=r.derived))
    worker.execute()  # call directly (no thread) — BaseWorker pattern
    assert captured["derived"] == ["orig_seg"]
