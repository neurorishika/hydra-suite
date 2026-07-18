from pathlib import Path

import hydra_suite.posekit.gui.dialogs.training as T


class _FakeProc:
    def __init__(self, lines):
        self.stdout = iter(lines)
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def wait(self):
        self._alive = False
        return 0

    def terminate(self):
        self._alive = False


def test_worker_streams_progress(monkeypatch, tmp_path):
    lines = [
        "EPOCH 0 train_loss=0.5 val_loss=0.4 pck@0.05=0.1 pck@0.1=0.2\n",
        "EPOCH 1 train_loss=0.3 val_loss=0.2 pck@0.05=0.5 pck@0.1=0.7\n",
        "DONE best_pck=0.5 best_epoch=1\n",
    ]
    # stub the dataset build + run prep so the worker does no real IO
    monkeypatch.setattr(
        T,
        "build_coco_keypoints_dataset",
        lambda **kw: {
            "dataset_dir": tmp_path / "ds",
            "coco_path": tmp_path / "ds/annotations.json",
            "labeled_count": 8,
            "manifest": tmp_path / "m.json",
        },
    )
    monkeypatch.setattr(
        T, "prepare_run", lambda params, run_dir, cache_dir: Path(run_dir) / "run.json"
    )
    monkeypatch.setattr(T, "build_training_command", lambda rj: ["true"])
    monkeypatch.setattr(T.subprocess, "Popen", lambda *a, **k: _FakeProc(lines))

    w = T.ViTPoseTrainingWorker(
        image_paths=[],
        labels_dir=tmp_path,
        run_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        class_names=["a"],
        keypoint_names=["k0"],
        skeleton_edges=[],
        variant="B",
        init_checkpoint="vitpose-b-coco",
        num_keypoints=1,
        epochs=2,
        batch=4,
        device="cpu",
    )
    progresses = []
    w.progress.connect(lambda cur, tot: progresses.append((cur, tot)))
    w.run()
    assert (1, 2) in progresses  # epoch 1 of 2 reported


def test_worker_cancel_terminates(monkeypatch, tmp_path):
    fp = _FakeProc([])
    w = T.ViTPoseTrainingWorker(
        image_paths=[],
        labels_dir=tmp_path,
        run_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        class_names=["a"],
        keypoint_names=["k0"],
        skeleton_edges=[],
        variant="B",
        init_checkpoint="x",
        num_keypoints=1,
        epochs=1,
        batch=1,
        device="cpu",
    )
    w._proc = fp
    w.cancel()
    assert fp.poll() == 0  # terminated
