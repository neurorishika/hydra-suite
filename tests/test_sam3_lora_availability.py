"""The probe must explain WHY it is unusable, and never import or download."""

import sys

from hydra_suite.training.sam3_lora import availability as av


def test_missing_package_names_itself_and_the_install(monkeypatch):
    monkeypatch.setattr(
        av, "_find_spec", lambda n: None if n == "torchmetrics" else object()
    )
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: True)
    got = av.probe_sam3_training_availability()
    assert not got.usable
    assert "torchmetrics" in got.reason


def test_missing_checkpoint_is_reported_not_downloaded(monkeypatch):
    monkeypatch.setattr(av, "_find_spec", lambda n: object())
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: False)
    got = av.probe_sam3_training_availability()
    assert not got.usable
    assert "checkpoint" in got.reason.lower()


def test_all_present_is_usable(monkeypatch):
    monkeypatch.setattr(av, "_find_spec", lambda n: object())
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: True)
    assert av.probe_sam3_training_availability().usable


def test_probe_does_not_import_sam3(monkeypatch):
    monkeypatch.setattr(av, "_find_spec", lambda n: object())
    monkeypatch.setattr(av, "_checkpoint_present", lambda cache_dir=None: True)
    sys.modules.pop("sam3", None)
    av.probe_sam3_training_availability()
    assert "sam3" not in sys.modules
