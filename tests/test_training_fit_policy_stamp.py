"""Tests for fit_policy stamping: training-time writers + the stamping script.

Root cause (see docs/superpowers/specs/2026-08-27-identity-subsystem-repair-design.md):
classifiers trained before commit 3a2163ac used an anisotropic Resize((sz,sz))
squash; training now uses CanonicalFitTransform (letterbox) unconditionally, so
every artifact training publishes must carry fit_policy="letterbox". Existing
artifacts need scripts/stamp_fit_policy.py to be stamped explicitly by a human.
"""

import json
import subprocess
import sys

import torch

from hydra_suite.training import model_publish as MP
from hydra_suite.training import runner as R
from hydra_suite.training.canonical_transform import FIT_POLICY_TRAINED
from hydra_suite.training.torchvision_model import save_torchvision_checkpoint


def test_fit_policy_trained_constant_is_letterbox():
    assert FIT_POLICY_TRAINED == "letterbox"


def test_checkpoint_dict_carries_letterbox_fit_policy():
    d = R.build_checkpoint_dict(
        arch="tinyclassifier",
        factor_names=["flat"],
        class_names_per_factor=[["a", "b"]],
        input_size=(32, 32),
        monochrome=False,
        state_dict={},
        best_val_acc=0.5,
        history={},
    )
    assert d["fit_policy"] == "letterbox"


def test_checkpoint_dict_passes_extra_keys_through():
    d = R.build_checkpoint_dict(
        arch="resnet18",
        factor_names=["flat"],
        class_names_per_factor=[["a", "b"]],
        input_size=(64, 64),
        monochrome=False,
        state_dict={},
        best_val_acc=None,
        history={},
        trainable_layers=2,
        backbone_lr_scale=0.1,
    )
    assert d["trainable_layers"] == 2
    assert d["backbone_lr_scale"] == 0.1
    assert d["fit_policy"] == "letterbox"


def test_save_tiny_checkpoint_stamps_fit_policy(tmp_path):
    import torch.nn as nn

    model = nn.Linear(4, 2)
    save_path = tmp_path / "tiny.pth"
    R._save_tiny_checkpoint(
        model=model,
        save_path=str(save_path),
        class_names=["a", "b"],
        input_size=(32, 32),
        monochrome=False,
        hidden_layers=1,
        hidden_dim=8,
        dropout=0.0,
        best_val_acc=0.5,
        history=[],
    )
    ckpt = torch.load(save_path, weights_only=False)
    assert ckpt["fit_policy"] == "letterbox"


def test_save_torchvision_checkpoint_stamps_fit_policy(tmp_path):
    import torch.nn as nn

    model = nn.Linear(4, 2)
    save_path = tmp_path / "tv.pth"
    save_torchvision_checkpoint(
        model=model,
        backbone="resnet18",
        class_names=["a", "b"],
        factor_names=["flat"],
        input_size=(64, 64),
        best_val_acc=0.5,
        history={},
        trainable_layers=1,
        backbone_lr_scale=0.1,
        monochrome=False,
        path=save_path,
    )
    ckpt = torch.load(save_path, weights_only=False)
    assert ckpt["fit_policy"] == "letterbox"


def test_emit_yolo_multihead_manifest_stamps_fit_policy(tmp_path):
    pt = tmp_path / "flat.pt"
    pt.write_bytes(b"stub")
    manifest_path = R.emit_yolo_multihead_manifest(
        manifest_path=str(tmp_path / "bundle.multihead.json"),
        factors=[("flat", pt, ["a", "b"])],
        input_size=(64, 64),
        monochrome=False,
    )
    data = json.loads(manifest_path.read_text())
    assert data["fit_policy"] == "letterbox"


def test_write_classifier_multihead_manifest_stamps_fit_policy(tmp_path):
    pt = tmp_path / "flat.pt"
    pt.write_bytes(b"stub")
    manifest_path = MP.write_classifier_multihead_manifest(
        tmp_path / "bundle.multihead.json",
        factor_entries=[{"factor": "flat", "path": str(pt), "class_names": ["a", "b"]}],
        input_size=(64, 64),
        monochrome=False,
        fit_policy="letterbox",
    )
    data = json.loads(manifest_path.read_text())
    assert data["fit_policy"] == "letterbox"


def test_write_classifier_multihead_manifest_omits_fit_policy_when_none(tmp_path):
    pt = tmp_path / "flat.pt"
    pt.write_bytes(b"stub")
    manifest_path = MP.write_classifier_multihead_manifest(
        tmp_path / "bundle2.multihead.json",
        factor_entries=[{"factor": "flat", "path": str(pt), "class_names": ["a", "b"]}],
        input_size=(64, 64),
        monochrome=False,
    )
    data = json.loads(manifest_path.read_text())
    assert "fit_policy" not in data


def test_stamp_script_torch_and_manifest(tmp_path):
    p = tmp_path / "m.pth"
    torch.save({"schema_version": 2, "arch": "tinyclassifier"}, p)
    man = tmp_path / "b.multihead.json"
    man.write_text(json.dumps({"schema_version": 2}))
    for target in (p, man):
        subprocess.run(
            [
                sys.executable,
                "scripts/stamp_fit_policy.py",
                str(target),
                "--policy",
                "squash",
            ],
            check=True,
        )
    assert torch.load(p, weights_only=False)["fit_policy"] == "squash"
    assert json.loads(man.read_text())["fit_policy"] == "squash"


def test_stamp_script_recurses_into_multihead_factor_models(tmp_path):
    sub1 = tmp_path / "factor1.pth"
    torch.save({"schema_version": 2, "arch": "tinyclassifier"}, sub1)
    man = tmp_path / "bundle.multihead.json"
    man.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "classifier_multihead_bundle",
                "factor_models": [
                    {"factor": "flat", "path": "factor1.pth", "class_names": ["a"]}
                ],
            }
        )
    )
    subprocess.run(
        [sys.executable, "scripts/stamp_fit_policy.py", str(man), "--policy", "squash"],
        check=True,
    )
    assert json.loads(man.read_text())["fit_policy"] == "squash"
    assert torch.load(sub1, weights_only=False)["fit_policy"] == "squash"


def test_stamp_script_rejects_non_dict_checkpoint(tmp_path):
    p = tmp_path / "bad.pth"
    torch.save([1, 2, 3], p)
    result = subprocess.run(
        [sys.executable, "scripts/stamp_fit_policy.py", str(p), "--policy", "squash"],
    )
    assert result.returncode == 2
