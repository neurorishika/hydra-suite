"""input_size threads from the dialog's worker into RunConfig.

Qt-free by design: this constructs the worker object directly and inspects the
params dict, rather than building the dialog. The repo has known modal-dialog
hangs that stop the full suite completing, so GUI construction stays out of tests.
"""

from __future__ import annotations

import pytest

from hydra_suite.core.identity.pose.vitpose.training.config import validate_run_config

pytest.importorskip("PySide6")


def _worker(**over):
    from hydra_suite.posekit.gui.dialogs.training import ViTPoseTrainingWorker

    kwargs = dict(
        image_paths=[],
        labels_dir="labels",
        run_dir="run",
        cache_dir="cache",
        class_names=["a"],
        keypoint_names=["k0", "k1"],
        skeleton_edges=[],
        variant="B",
        init_checkpoint="ckpt.pth",
        num_keypoints=2,
        epochs=1,
        batch=1,
        device="cpu",
    )
    kwargs.update(over)
    return ViTPoseTrainingWorker(**kwargs)


def test_worker_accepts_and_stores_input_size():
    w = _worker(input_size=[256, 256])
    assert w.input_size == [256, 256]


def test_input_size_defaults_to_none_so_existing_callers_are_unaffected():
    assert _worker().input_size is None


def test_params_dict_shape_is_accepted_by_run_config():
    # Mirrors the dict ViTPoseTrainingWorker.run() builds, with input_size added.
    params = dict(
        init_checkpoint="ckpt.pth",
        variant="B",
        num_keypoints=2,
        dataset_dir="ds",
        output_dir="out",
        device="cpu",
        epochs=1,
        batch_size=1,
        input_size=[256, 256],
    )
    cfg = validate_run_config(params)
    assert cfg.input_size == [256, 256]


def test_run_config_rejects_a_bad_input_size_from_the_dialog():
    params = dict(
        init_checkpoint="ckpt.pth",
        variant="B",
        num_keypoints=2,
        dataset_dir="ds",
        output_dir="out",
        device="cpu",
        epochs=1,
        batch_size=1,
        input_size=[250, 192],  # not a multiple of 32
    )
    with pytest.raises(ValueError, match="input_size"):
        validate_run_config(params)
