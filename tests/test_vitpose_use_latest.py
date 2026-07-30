from hydra_suite.posekit.gui.main_window import MainWindow


def test_latest_vitpose_path_accepts_existing_pt(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"x")
    assert MainWindow._latest_vitpose_candidate(str(p)) == str(p)


def test_latest_vitpose_path_rejects_missing(tmp_path):
    assert MainWindow._latest_vitpose_candidate(str(tmp_path / "nope.pt")) == ""


def test_latest_vitpose_path_rejects_non_pt(tmp_path):
    p = tmp_path / "model.onnx"
    p.write_bytes(b"x")
    assert MainWindow._latest_vitpose_candidate(str(p)) == ""
