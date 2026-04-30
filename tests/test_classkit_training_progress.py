from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")


def test_stream_ultralytics_output_parses_multiple_epoch_formats() -> None:
    from hydra_suite.training.runner import _stream_ultralytics_output

    class Proc:
        stdout = iter(
            [
                "Epoch 2/5\n",
                "Epoch 3 of 5\n",
                " 4/5 1.23G loss=0.42\n",
            ]
        )

    seen: list[tuple[int, int]] = []
    logs: list[str] = []

    result = _stream_ultralytics_output(
        Proc(),
        logs.append,
        lambda current, total: seen.append((current, total)),
        should_cancel=None,
        command=["yolo"],
    )

    assert result is None
    assert seen == [(2, 5), (3, 5), (4, 5)]
    assert logs[:3] == ["Epoch 2/5", "Epoch 3 of 5", " 4/5 1.23G loss=0.42"]


def test_classkit_training_worker_uses_negative_pct_for_log_messages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hydra_suite.classkit.jobs.task_workers import ClassKitTrainingWorker

    def fake_run_training(spec, run_dir, log_cb, progress_cb, should_cancel):
        log_cb("starting")
        progress_cb(1, 4)
        log_cb("still running")
        return {"success": True}

    monkeypatch.setattr("hydra_suite.training.runner.run_training", fake_run_training)

    worker = ClassKitTrainingWorker(
        role="flat",
        specs=[object()],
        run_dir=str(tmp_path),
        multi_head=False,
    )
    emitted: list[tuple[int, str]] = []
    worker.signals.progress.connect(lambda pct, msg: emitted.append((pct, msg)))

    worker.run()

    assert emitted == [(-1, "starting"), (25, ""), (-1, "still running"), (100, "")]


def test_classkit_training_worker_multihead_progress_advances_early(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hydra_suite.classkit.jobs.task_workers import ClassKitTrainingWorker

    def fake_run_training(spec, run_dir, log_cb, progress_cb, should_cancel):
        progress_cb(1, 10)
        return {"success": True}

    monkeypatch.setattr("hydra_suite.training.runner.run_training", fake_run_training)

    worker = ClassKitTrainingWorker(
        role="multi",
        specs=[object(), object()],
        run_dir=str(tmp_path),
        multi_head=True,
    )
    emitted: list[tuple[int, str]] = []
    worker.signals.progress.connect(lambda pct, msg: emitted.append((pct, msg)))

    worker.run()

    assert emitted[0][0] == 5
    assert emitted[1][0] == 55
    assert emitted[-1] == (100, "")
