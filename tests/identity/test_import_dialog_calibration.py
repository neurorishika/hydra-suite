"""Calibration summary surfaced by the TrackerKit CNN-import dialog.

``describe_cnn_identity_candidate`` is the Qt-free seam used to build the
preview dict for ``CNNIdentityImportDialog``. This test only calls that
module-level function — no QApplication / widget exec — per the identity
overhaul Phase 2 conventions for keeping GUI-adjacent tests fast and
non-hanging.
"""

import torch

from hydra_suite.trackerkit.gui.dialogs.cnn_identity_import_dialog import (
    describe_cnn_identity_candidate,
)
from hydra_suite.training.torchvision_model import save_torchvision_checkpoint


def _save_flat_torchvision(path, *, calibration_temperature=None):
    model = torch.nn.Linear(4, 3)
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=["a", "b", "c"],
        factor_names=["flat"],
        input_size=(64, 64),
        best_val_acc=0.9,
        history=[],
        trainable_layers=0,
        backbone_lr_scale=1.0,
        monochrome=False,
        path=str(path),
        calibration_temperature=calibration_temperature,
    )


def test_describe_candidate_reports_calibrated(tmp_path):
    p = tmp_path / "calibrated.pth"
    _save_flat_torchvision(p, calibration_temperature=[1.7])

    summary = describe_cnn_identity_candidate(str(p))

    assert summary["calibration_status"] == "calibrated"
    assert summary["calibration_temperature"] == (1.7,)


def test_describe_candidate_reports_uncalibrated(tmp_path):
    p = tmp_path / "uncalibrated.pth"
    _save_flat_torchvision(p, calibration_temperature=None)

    summary = describe_cnn_identity_candidate(str(p))

    assert summary["calibration_status"] == "uncalibrated"
    assert summary["calibration_temperature"] is None
