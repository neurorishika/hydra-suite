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
