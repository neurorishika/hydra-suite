"""SAM2 geometry calibration against polygon ground truth."""

import cv2
import numpy as np

from hydra_suite.core.inference.sam2 import calibration as gc


class PerfectExec:
    """Returns the GT disc (radius 10) around the positive point, in tile coords."""

    def __init__(self):
        self.encodes = 0

    def set_image(self, img):
        self.shape = img.shape[:2]
        self.encodes += 1

    def segment(self, box, pos, neg):
        m = np.zeros(self.shape, np.uint8)
        cx, cy = pos[0]
        cv2.circle(m, (int(round(cx)), int(round(cy))), 10, 1, -1)
        return m.astype(bool), 0.9


class EmptyExec(PerfectExec):
    def segment(self, box, pos, neg):
        return np.zeros(self.shape, bool), 0.1


def _disc(cx, cy, r=10.0):
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    return np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], 1).astype(np.float32)


def _frame(tmp_path, centers, size=(200, 200), name="f.png"):
    p = tmp_path / name
    cv2.imwrite(str(p), np.zeros((*size, 3), np.uint8))
    return p, [_disc(cx, cy) for cx, cy in centers]


def test_perfect_masks_score_high_iou_for_every_fraction(tmp_path):
    frame = _frame(tmp_path, [(50, 50), (150, 150)])
    pts = gc.calibrate_geometry(
        PerfectExec(), [frame], reference_body_px=20.0, tile_fractions=(0.2, None)
    )
    assert {p.tile_fraction for p in pts} == {0.2, None}
    for p in pts:
        assert p.median_iou > 0.85 and p.fallback_rate == 0.0 and p.n_instances == 2
    full = next(p for p in pts if p.tile_fraction is None)
    tiled = next(p for p in pts if p.tile_fraction == 0.2)
    assert full.owned_tiles_per_frame == 1 and full.tile_px is None
    assert tiled.tile_px == 100 and tiled.owned_tiles_per_frame >= 1


def test_empty_masks_are_fallbacks_with_zero_iou(tmp_path):
    frame = _frame(tmp_path, [(50, 50)])
    (pt,) = gc.calibrate_geometry(
        EmptyExec(), [frame], reference_body_px=20.0, tile_fractions=(None,)
    )
    assert pt.fallback_rate == 1.0 and pt.median_iou == 0.0


def test_fraction_unresolvable_on_some_frames_is_dropped(tmp_path):
    small = _frame(tmp_path, [(20, 20)], size=(60, 60), name="s.png")
    big = _frame(tmp_path, [(50, 50)], size=(200, 200), name="b.png")
    pts = gc.calibrate_geometry(
        PerfectExec(), [small, big], reference_body_px=20.0, tile_fractions=(0.2, None)
    )
    # 100 px tiles cover the 60 px frame, so 0.2 does not resolve there
    assert [p.tile_fraction for p in pts] == [None]


def test_recommend_prefers_fewest_tiles_clearing_floor():
    def mk(frac, tiles, iou, fb=0.0, n=50):
        return gc.GeometryCalibrationPoint(frac, None, tiles, 1.0, iou, iou, fb, 0.0, n)

    best, why = gc.recommend_geometry(
        [mk(None, 1, 0.3), mk(0.1, 4, 0.8), mk(0.05, 9, 0.85)],
        iou_floor=0.5,
        fallback_ceil=0.1,
    )
    assert best.tile_fraction == 0.1 and why == ""


def test_recommend_refuses_with_reason():
    def mk(iou, fb, n):
        return gc.GeometryCalibrationPoint(None, None, 1, 1.0, iou, iou, fb, 0.0, n)

    kw = dict(iou_floor=0.5, fallback_ceil=0.1)
    assert gc.recommend_geometry([], **kw)[0] is None
    assert "IoU" in gc.recommend_geometry([mk(0.1, 0.0, 50)], **kw)[1]
    assert "fell back" in gc.recommend_geometry([mk(0.9, 0.9, 50)], **kw)[1]
    assert "instances" in gc.recommend_geometry([mk(0.9, 0.0, 3)], **kw)[1]


def test_cancel_returns_no_partial_points(tmp_path):
    frame = _frame(tmp_path, [(50, 50)])
    assert (
        gc.calibrate_geometry(
            PerfectExec(),
            [frame],
            reference_body_px=20.0,
            tile_fractions=(None,),
            should_stop=lambda: True,
        )
        == []
    )


def test_prompts_mimic_a_box_source():
    polys = [_disc(50, 50), _disc(55, 50), _disc(150, 150)]
    prompts, obbs = gc.prompts_from_ground_truth(polys)
    assert len(prompts) == 3 and all(o.shape == (4, 2) for o in obbs)
    # overlapping neighbours become negatives; the far one does not
    assert len(prompts[0].negative_points) == 1
    assert prompts[2].negative_points == []
    x1, y1, x2, y2 = prompts[2].box_xyxy
    # the OBB may be rotated, so its extent CONTAINS the disc, tightly
    assert 135 <= x1 <= 140.5 and 159.5 <= x2 <= 165
    assert 135 <= y1 <= 140.5 and 159.5 <= y2 <= 165


def test_core_module_is_qt_free():
    import subprocess
    import sys

    code = (
        "import sys, hydra_suite.core.inference.sam2.calibration;"
        "print(any(m.startswith('PySide6') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip().endswith("False"), out.stderr
