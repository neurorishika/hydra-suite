import json
import logging

import pytest
import torch

from hydra_suite.core.individual.classification import backend as B


def _tiny_ckpt(tmp_path, **extra):
    ckpt = {
        "schema_version": 2,
        "arch": "tinyclassifier",
        "factor_names": ["flat"],
        "class_names_per_factor": [["a", "b"]],
        "class_names": ["a", "b"],
        "input_size": [32, 32],
        "num_classes": 2,
        "monochrome": False,
        "model_state_dict": {},
        **extra,
    }
    p = tmp_path / "m.pth"
    torch.save(ckpt, p)
    return str(p)


def test_torch_ckpt_without_fit_policy_defaults_to_squash_with_warning(
    tmp_path, caplog
):
    path = _tiny_ckpt(tmp_path)
    with caplog.at_level(logging.WARNING):
        meta = B._select_loader(path).parse_metadata(path)
    assert meta.fit_policy == "squash"
    assert "fit_policy" in caplog.text and "squash" in caplog.text


def test_torch_ckpt_with_fit_policy_letterbox(tmp_path, caplog):
    path = _tiny_ckpt(tmp_path, fit_policy="letterbox")
    with caplog.at_level(logging.WARNING):
        meta = B._select_loader(path).parse_metadata(path)
    assert meta.fit_policy == "letterbox"
    assert "fit_policy" not in caplog.text


def test_invalid_fit_policy_raises(tmp_path):
    path = _tiny_ckpt(tmp_path, fit_policy="stretchy")
    with pytest.raises(B.ClassifierFormatError):
        B._select_loader(path).parse_metadata(path)


def test_multihead_manifest_fit_policy(tmp_path):
    _tiny_ckpt(tmp_path, fit_policy="letterbox")
    man = tmp_path / "bundle.multihead.json"
    man.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "classifier_multihead_bundle",
                "factor_names": ["flat", "flat_1"],
                "factor_models": [
                    {"factor": "flat", "path": "m.pth", "class_names": ["a", "b"]},
                    {"factor": "flat_1", "path": "m.pth", "class_names": ["a", "b"]},
                ],
                "input_size": [32, 32],
                "monochrome": False,
            }
        )
    )
    meta = B._select_loader(str(man)).parse_metadata(str(man))
    assert meta.fit_policy == "squash"  # absent on manifest → legacy squash
    man.write_text(
        json.dumps(
            {
                **json.loads(man.read_text()),
                "fit_policy": "letterbox",
            }
        )
    )
    assert B._select_loader(str(man)).parse_metadata(str(man)).fit_policy == "letterbox"


def test_unstamped_yolo_multihead_bundle_resolves_to_native(tmp_path):
    """Finding I2: YOLO multihead bundles must always resolve to 'native'
    (spec Sec 3.1), regardless of a missing/stamped fit_policy -- ultralytics
    applies its own preprocessing and Layer 2 must not touch it.
    """
    _tiny_ckpt(tmp_path, fit_policy="letterbox")
    man = tmp_path / "bundle.multihead.json"
    man.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "yolo_multihead_bundle",
                "factor_names": ["flat", "flat_1"],
                "factor_models": [
                    {"factor": "flat", "path": "m.pth", "class_names": ["a", "b"]},
                    {"factor": "flat_1", "path": "m.pth", "class_names": ["a", "b"]},
                ],
                "input_size": [32, 32],
                "monochrome": False,
                # no fit_policy key: unstamped
            }
        )
    )
    meta = B._select_loader(str(man)).parse_metadata(str(man))
    assert meta.fit_policy == "native"


def test_yolo_multihead_bundle_ignores_stamped_fit_policy(tmp_path):
    """Even an explicitly stamped (e.g. legacy 'letterbox') fit_policy on a
    yolo_multihead_bundle manifest must be overridden to 'native'.
    """
    _tiny_ckpt(tmp_path, fit_policy="letterbox")
    man = tmp_path / "bundle.multihead.json"
    man.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "yolo_multihead_bundle",
                "factor_names": ["flat", "flat_1"],
                "factor_models": [
                    {"factor": "flat", "path": "m.pth", "class_names": ["a", "b"]},
                    {"factor": "flat_1", "path": "m.pth", "class_names": ["a", "b"]},
                ],
                "input_size": [32, 32],
                "monochrome": False,
                "fit_policy": "letterbox",
            }
        )
    )
    meta = B._select_loader(str(man)).parse_metadata(str(man))
    assert meta.fit_policy == "native"
