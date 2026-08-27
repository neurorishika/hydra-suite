"""Regression tests for a Task 3 fix-round finding: `fit_policy` must survive
into the TrackerKit `.multihead.json` manifest at every call site that writes
one via `write_classifier_multihead_manifest`, not just
`model_publish.py::_copy_classifier_artifact_to_repository` (already fixed).

Covers `training/service.py::_publish_training_artifacts`'s multi-factor
bundle-publish path -- the most common real path a freshly-trained multihead
classifier bundle takes to disk. `classkit/gui/main_window.py` and
`classkit/gui/dialogs/model_history.py` are Qt GUI call sites and are fixed
identically but not covered here (no GUI harness in this file).
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra_suite.training.model_publish as mp
import hydra_suite.training.service as svc
from hydra_suite.training import runner as R
from hydra_suite.training.contracts import (
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)


def _make_tiny_checkpoint(path: Path, class_names: list[str]) -> None:
    import torch.nn as nn

    model = nn.Linear(4, len(class_names))
    R._save_tiny_checkpoint(
        model=model,
        save_path=str(path),
        class_names=class_names,
        input_size=(32, 32),
        monochrome=False,
        hidden_layers=1,
        hidden_dim=8,
        dropout=0.0,
        best_val_acc=0.5,
        history=[],
    )


def test_publish_training_artifacts_multihead_manifest_carries_fit_policy(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mp, "get_models_root", lambda: tmp_path)

    p1 = tmp_path / "f1.pth"
    p2 = tmp_path / "f2.pth"
    _make_tiny_checkpoint(p1, ["a", "b"])
    _make_tiny_checkpoint(p2, ["c", "d"])
    # _save_tiny_checkpoint always stamps fit_policy="letterbox" (Task 3);
    # confirm the fixture actually carries it before trusting the assertion
    # below.
    import torch

    assert torch.load(p1, weights_only=False)["fit_policy"] == "letterbox"
    assert torch.load(p2, weights_only=False)["fit_policy"] == "letterbox"

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_MULTIHEAD_TINY,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path / "derived"),
        base_model="",
        hyperparams=TrainingHyperParams(imgsz=32),
    )

    key, manifest_path = svc._publish_training_artifacts(
        spec=spec,
        artifact_paths=[str(p1), str(p2)],
        publish_metadata={"factor_names": ["f1", "f2"]},
        run_id="r1",
        dataset_fingerprint_value="fp",
    )

    assert key and manifest_path
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["fit_policy"] == "letterbox"
