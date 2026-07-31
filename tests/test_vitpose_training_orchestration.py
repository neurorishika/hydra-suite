import json
import sys

from hydra_suite.posekit.core.vitpose_training import (
    build_training_command,
    parse_progress_line,
    prepare_run,
)


def test_prepare_run_writes_valid_config(tmp_path):
    ckpt = tmp_path / "w.pth"
    ckpt.write_bytes(b"x")
    run_dir = tmp_path / "run"
    params = dict(
        init_checkpoint=str(ckpt),
        variant="B",
        num_keypoints=5,
        dataset_dir=str(tmp_path / "ds"),
        device="cpu",
        epochs=10,
        batch_size=8,
    )
    rj = prepare_run(params, run_dir, cache_dir=tmp_path / "cache")
    d = json.loads(rj.read_text())
    assert d["variant"] == "B" and d["num_keypoints"] == 5
    assert d["output_dir"] == str(run_dir)


def test_build_command_uses_current_interpreter(tmp_path):
    rj = tmp_path / "run.json"
    cmd = build_training_command(rj)
    assert cmd[0] == sys.executable
    assert cmd[1:4] == [
        "-m",
        "hydra_suite.core.identity.pose.vitpose.training",
        "--config",
    ] or cmd[1:3] == ["-m", "hydra_suite.core.identity.pose.vitpose.training"]
    assert str(rj) in cmd


def test_parse_progress():
    line = "EPOCH 4 train_loss=0.00123 val_loss=0.00456 pck@0.05=0.8000 pck@0.1=0.9500"
    r = parse_progress_line(line)
    assert r == {
        "epoch": 4,
        "train_loss": 0.00123,
        "val_loss": 0.00456,
        "pck@0.05": 0.8,
        "pck@0.1": 0.95,
    }
    assert parse_progress_line("random log line") is None
