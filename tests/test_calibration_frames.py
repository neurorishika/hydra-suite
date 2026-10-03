"""Neutral calibration-frame helpers and the polygon ground-truth criterion."""

import cv2
import numpy as np

from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs import calibration_frames as cf

POLY = "0 0.1 0.1 0.3 0.1 0.35 0.2 0.3 0.3 0.1 0.3\n"  # 5 points
OBB = "0 0.5 0.5 0.7 0.5 0.7 0.7 0.5 0.7\n"
AABB = "0 0.5 0.5 0.1 0.1\n"


def make_source(tmp_path, name, frames, level="polygon"):
    root = tmp_path / name
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for stem, text in frames.items():
        cv2.imwrite(
            str(root / "images" / f"{stem}.jpg"), np.zeros((100, 100, 3), np.uint8)
        )
        (root / "labels" / f"{stem}.txt").write_text(text)
    return OBBSource(path=str(root), name=name, level=level)


def test_is_polygon_line():
    assert cf.is_polygon_line(6) and cf.is_polygon_line(10)
    assert not cf.is_polygon_line(4) and not cf.is_polygon_line(8)
    assert not cf.is_polygon_line(7)


def test_has_polygon_frames_needs_an_all_polygon_frame(tmp_path):
    assert cf.has_polygon_frames(make_source(tmp_path, "p", {"a": POLY}))
    assert not cf.has_polygon_frames(make_source(tmp_path, "b", {"a": OBB + AABB}))
    assert not cf.has_polygon_frames(make_source(tmp_path, "m", {"a": POLY + OBB}))


def test_has_polygon_frames_never_decodes(tmp_path, monkeypatch):
    src = make_source(tmp_path, "p", {"a": POLY})

    def _boom(*_a, **_k):
        raise AssertionError("decoded an image")

    monkeypatch.setattr(cf.cv2, "imread", _boom)
    assert cf.has_polygon_frames(src)


def test_polygon_only_skips_box_and_mixed_frames(tmp_path):
    src = make_source(tmp_path, "s", {"a": POLY, "b": OBB, "c": POLY + OBB})
    frames = cf.labelled_frames_for(src, polygon_only=True)
    assert [p.stem for p, _ in frames] == ["a"]
    assert len(frames[0][1]) == 1 and frames[0][1][0].points.shape == (5, 2)
    assert len(cf.labelled_frames_for(src)) == 3  # default unchanged


def test_stratified_polygon_only(tmp_path):
    a = make_source(tmp_path, "a", {"x": POLY, "y": OBB})
    b = make_source(tmp_path, "b", {"z": OBB})
    out = cf.stratified_calibration_frames([a, b], budget=4, polygon_only=True)
    assert [p.stem for p, _ in out] == ["x"]


def test_scale_mismatch():
    assert cf.scale_mismatch(50.0, 100.0)
    assert not cf.scale_mismatch(50.0, 70.0)
    assert not cf.scale_mismatch(0.0, 70.0)


def test_semantic_module_reexports_same_objects():
    from hydra_suite.detectkit.jobs import semantic_escalation as se

    for name in (
        "labelled_frames_for",
        "stratified_calibration_frames",
        "measure_median_body_px",
        "median_body_px_for",
        "has_labelled_frames",
        "_label_path_for",
        "CALIBRATION_SAMPLE_FRAMES",
    ):
        assert getattr(se, name) is getattr(cf, name)
