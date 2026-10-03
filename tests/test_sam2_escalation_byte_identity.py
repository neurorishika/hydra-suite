"""run_escalation with tiling off is byte-identical to pre-SAHI main.

EXPECTED_LOG and EXPECTED_LABEL were recorded against the unmodified
run_escalation (the commit before owner-tile segmentation was wired in) and
are asserted verbatim: the executor call sequence and the staged label bytes
must not move when tile_fraction is None.
"""

import types
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.sam2_escalation import EscalationRequest, run_escalation


class RecordingExec:
    def __init__(self):
        self.log = []
        self._shape = None

    def set_image(self, img):
        self._shape = img.shape[:2]
        self.log.append(("set_image", tuple(img.shape), int(img.sum()) % 1000003))

    def segment(self, box, pos, neg):
        self.log.append(
            (
                "segment",
                tuple(round(float(v), 3) for v in box),
                tuple(tuple(round(float(c), 3) for c in p) for p in pos),
                tuple(tuple(round(float(c), 3) for c in p) for p in neg),
            )
        )
        h, w = self._shape
        m = np.zeros((h, w), bool)
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        m[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2] = True  # inset box mask
        return m, 0.8


def _source(tmp_path):
    root = tmp_path / "src"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    rng = np.random.default_rng(0)
    cv2.imwrite(
        str(root / "images" / "f1.png"),
        rng.integers(0, 255, (300, 400, 3), dtype=np.uint8),
    )
    (root / "labels" / "f1.txt").write_text(
        "0 0.10 0.10 0.20 0.10 0.20 0.25 0.10 0.25\n"
        "1 0.15 0.20 0.30 0.20 0.30 0.35 0.15 0.35\n"
        "0 0.5 0.5 0.2 0.2\n"
    )
    (root / "classes.txt").write_text("ant\nqueen\n")
    return OBBSource(path=str(root), name="src", level="obb")


def _run(tmp_path):
    src = _source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    ex = RecordingExec()
    run_escalation(EscalationRequest(project, ["src"], "v"), ex)
    staged = (Path(src.staged_review.staged_path) / "labels" / "f1.txt").read_text()
    return ex.log, staged


def test_untiled_escalation_matches_recorded_main(tmp_path):
    log, staged = _run(tmp_path)
    assert log == EXPECTED_LOG
    assert staged == EXPECTED_LABEL


EXPECTED_LOG = [
    ("set_image", (300, 400, 3), 637540),
    ("segment", (40.0, 30.0, 80.0, 75.0), ((60.0, 52.5),), ((90.0, 82.5),)),
    ("segment", (60.0, 60.0, 120.0, 105.0), ((90.0, 82.5),), ((60.0, 52.5),)),
    ("segment", (160.0, 120.0, 240.0, 180.0), ((200.0, 150.0),), ()),
]
EXPECTED_LABEL = "0 0.105000 0.106667 0.105000 0.240000 0.192500 0.240000 0.192500 0.106667 0.192500 0.106667\n1 0.155000 0.206667 0.155000 0.340000 0.292500 0.340000 0.292500 0.206667 0.292500 0.206667\n0 0.405000 0.406667 0.405000 0.590000 0.592500 0.590000 0.592500 0.406667 0.592500 0.406667\n"
