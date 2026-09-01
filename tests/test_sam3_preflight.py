"""Preflight refuses before any weights load, and never imports ultralytics."""

import sys

import pytest

from hydra_suite.training.contracts import (
    Sam3LoraParams,
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.sam3_lora import preflight as pf


def _spec(**kw):
    # ack defaults False and preflight refuses without it, so the "healthy"
    # baseline must set it or every test below passes for the wrong reason.
    p = Sam3LoraParams(
        prompt=kw.pop("prompt", "ant"), label_quality_acknowledged=kw.pop("ack", True)
    )
    base = dict(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir="/tmp/d",
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=p,
    )
    base.update(kw)
    return TrainingRunSpec(**base)


@pytest.fixture(autouse=True)
def _healthy(monkeypatch):
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: 48.0)
    monkeypatch.setattr(pf, "_instance_count", lambda d: 100)
    monkeypatch.setattr(pf, "_free_disk_gb", lambda p: 100.0)


def test_empty_prompt_is_refused(monkeypatch):
    assert any("prompt" in r.lower() for r in pf.preflight(_spec(prompt="")))


def test_non_cuda_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: None)
    assert any("cuda" in r.lower() for r in pf.preflight(_spec()))


def test_vram_band_between_refuse_and_warn_is_refused(monkeypatch):
    # 29 GB was the measured requirement; a 24-29 GB card must NOT pass.
    monkeypatch.setattr(pf, "_cuda_free_gb", lambda: 26.0)
    assert any("gb" in r.lower() for r in pf.preflight(_spec()))


def test_too_few_instances_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "_instance_count", lambda d: 5)
    assert any("instance" in r.lower() for r in pf.preflight(_spec()))


def test_resume_from_is_refused(monkeypatch):
    assert any(
        "resume" in r.lower() for r in pf.preflight(_spec(resume_from="/tmp/last.pt"))
    )


def test_unacknowledged_labels_are_refused():
    assert any("acknowledge" in r.lower() for r in pf.preflight(_spec(ack=False)))


def test_low_disk_is_refused(monkeypatch):
    monkeypatch.setattr(pf, "_free_disk_gb", lambda p: 2.0)
    assert any("disk" in r.lower() for r in pf.preflight(_spec()))


def test_healthy_spec_passes_and_imports_nothing_heavy():
    sys.modules.pop("ultralytics", None)
    assert pf.preflight(_spec()) == []
    assert "ultralytics" not in sys.modules
