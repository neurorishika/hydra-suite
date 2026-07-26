import numpy as np

from hydra_suite.training.contracts import TrainingRole
from hydra_suite.training.dataset_builders import _parse_geometry_label_lines


def test_new_roles_exist():
    assert TrainingRole.DETECT_DIRECT.value == "detect_direct"
    assert TrainingRole.SEGMENT_DIRECT.value == "segment_direct"
    assert TrainingRole.SEQ_CROP_SEGMENT.value == "seq_crop_segment"


def test_parse_polygon_line(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.5 0.3 0.7 0.1 0.5\n", encoding="utf-8"
    )  # 5 pts
    parsed = _parse_geometry_label_lines(p)
    assert parsed[0][0] == 0
    assert parsed[0][1].shape == (5, 2)


def test_parse_detect_line_expands_to_quad(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")  # cx cy w h
    cls, pts = _parse_geometry_label_lines(p)[0]
    assert cls == 2 and pts.shape == (4, 2)
    assert np.allclose(pts[0], [0.4, 0.3])  # x1,y1 = cx-w/2, cy-h/2
