from hydra_suite.posekit.gui.dialogs.training import resolve_finished_weights


def test_vitpose_payload_resolves_best_at_run_dir_root(tmp_path):
    best = tmp_path / "best.pt"
    best.write_bytes(b"x")
    info = {"run_dir": str(tmp_path), "best": str(best)}
    assert resolve_finished_weights(info) == str(best)


def test_vitpose_payload_falls_back_to_run_dir_best(tmp_path):
    best = tmp_path / "best.pt"
    best.write_bytes(b"x")
    info = {"run_dir": str(tmp_path)}  # no explicit "best"/"weights"
    assert resolve_finished_weights(info) == str(best)


def test_yolo_payload_still_resolves_weights_subdir(tmp_path):
    wdir = tmp_path / "weights"
    wdir.mkdir()
    best = wdir / "best.pt"
    best.write_bytes(b"x")
    info = {"weights": str(best), "run_dir": str(tmp_path)}
    assert resolve_finished_weights(info) == str(best)


def test_missing_weights_returns_empty(tmp_path):
    assert resolve_finished_weights({"run_dir": str(tmp_path)}) == ""
