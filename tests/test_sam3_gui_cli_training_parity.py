"""GUI/CLI parity guard for SAM3 LoRA training configuration.

Both entry points build the same `Sam3LoraParams` dataclass
(`hydra_suite.training.contracts.Sam3LoraParams`) and route it through the
same `DetectTrainingPlan` -> `build_role_entries` -> `TrainingRunSpec`
pipeline (`hydra_suite.detectkit.jobs.training`). Nothing currently asserts
that, so the two paths could silently drift -- exactly the failure mode the
shared `build_engine_params` work (see `test_gui_cli_param_equivalence.py`)
found three latent divergences of.

This test:

1. Drives the REAL GUI construction path (`TrainingDialog._start_training`,
   following the harness in `test_sam3_dialog_wiring.py`) and the REAL
   CLI/JSON construction path (`load_training_plan` -> `plan.role_entries`)
   from the *same* canonical input values, and asserts every
   `Sam3LoraParams` field -- reflectively, via `dataclasses.fields`, so a
   newly added field is covered automatically -- agrees between the two.
2. Separately documents the two known, intended divergences (env_name
   *default* resolution, and PublishPolicy.auto_import *default*) as
   explicit assertions of the expected difference, not as excluded fields.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, fields

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

QApplication.instance() or QApplication([])

from hydra_suite.detectkit.config.training import load_training_plan  # noqa: E402
from hydra_suite.detectkit.gui.dialogs import training_dialog as td  # noqa: E402
from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource  # noqa: E402
from hydra_suite.training.contracts import (  # noqa: E402
    PublishPolicy,
    Sam3LoraParams,
    TrainingRole,
)
from hydra_suite.training.sam3_lora.env import (  # noqa: E402
    DEFAULT_SAM3_ENV,
    resolve_sam3_env,
)

# Every field of Sam3LoraParams, set to a non-default value, so a round trip
# through either path is a meaningful test (not accidentally passing because
# nothing changed). If a new field appears on Sam3LoraParams and this dict
# isn't updated, the `test_reference_covers_every_sam3lora_field` guard below
# fails loudly -- that is the whole point of the reflective comparison.
_REFERENCE_KWARGS = dict(
    prompt="worker ant",
    negative_prompts=["debris", "reflection"],
    rank=8,
    alpha=24,
    dropout=0.2,
    lr=1e-4,
    epochs=7,
    batch=2,
    grad_accum=4,
    mixed_precision="bf16",
    num_negatives=2,
    host_reserve_gb=11.0,
    host_reserve_fraction=0.22,
    cuda_safety_fraction=0.78,
    host_limit_headroom_fraction=1.4,
    watchdog_poll_seconds=2.5,
    adapt_vision_encoder=False,
    adapt_text_encoder=True,
    adapt_geometry_encoder=False,
    adapt_detr_encoder=False,
    adapt_detr_decoder=False,
    adapt_mask_decoder=False,
    adapt_scoring_head=True,
    geometry_mode="custom",
    object_tile_fraction=0.11,
    object_tile_fractions=(0.05, 0.12),
    full_frame_mix=True,
    slice_width=896,
    slice_height=768,
    tile_overlap=0.3,
    keep_empty_tiles=False,
    label_quality_acknowledged=True,
    env_name="hydra-sam3-custom-env",
)


def test_reference_covers_every_sam3lora_field():
    """Fails loudly if a field is added to Sam3LoraParams and not covered here."""

    assert {f.name for f in fields(Sam3LoraParams)} == set(_REFERENCE_KWARGS)


def _make_class_signals():
    class _Signal:
        def connect(self, *_a, **_k):
            pass

    return _Signal


def _drive_gui_sam3_spec(tmp_path, monkeypatch, params: Sam3LoraParams):
    """Build a TrainingRunSpec through the real `_start_training` GUI path."""

    _Signal = _make_class_signals()

    monkeypatch.setattr(
        td.QMessageBox,
        "warning",
        lambda _parent, title, message: pytest.fail(f"{title}: {message}"),
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    src_dir = tmp_path / "ds1"
    src_dir.mkdir(parents=True)
    proj.sources = [OBBSource(path=str(src_dir), name="ds1")]

    dlg = td.TrainingDialog(proj)
    dlg.chk_role_obb_direct.setChecked(False)
    dlg.chk_role_segment_direct.setChecked(False)
    dlg.chk_role_seq_detect.setChecked(False)
    dlg.chk_role_seq_crop_obb.setChecked(False)
    dlg.chk_role_seq_crop_segment.setChecked(False)
    dlg.chk_role_detect_direct.setChecked(False)
    dlg.chk_semantic_sam3.setChecked(True)
    dlg.sam3_panel.set_params(params)

    monkeypatch.setattr(dlg, "_get_orchestrator", lambda: object())
    monkeypatch.setattr(dlg, "_write_to_project", lambda: None)

    def _finish_preparation(_orchestrator, request):
        dlg.role_dataset_dirs = {
            TrainingRole.SEMANTIC_SAM3.value: str(tmp_path / "derived")
        }
        dlg._set_training_running(True)
        dlg._start_training_worker(list(request.roles))

    monkeypatch.setattr(dlg, "_launch_dataset_preparation", _finish_preparation)

    captured = {}

    class _FakeWorker:
        def __init__(self, orchestrator, role_entries):
            captured["role_entries"] = role_entries
            self.log_signal = _Signal()
            self.role_started = _Signal()
            self.role_finished = _Signal()
            self.progress_signal = _Signal()
            self.done_signal = _Signal()
            self.finished = _Signal()

        def isRunning(self):
            return False

        def start(self):
            pass

    monkeypatch.setattr(td, "_TrainingWorker", _FakeWorker)

    dlg._start_training()

    entries = captured.get("role_entries")
    assert entries, "no role_entries reached the worker -- run path is unreachable"
    sam3_entries = [e for e in entries if e["role"] is TrainingRole.SEMANTIC_SAM3]
    assert len(sam3_entries) == 1
    return sam3_entries[0]["spec"], dlg


def _cli_plan_payload(tmp_path, sam3_values: dict, publish_values: dict | None = None):
    payload = {
        "version": 1,
        "workspace": "./workspace",
        "sources": [{"path": "./source", "name": "day-1", "level": "polygon"}],
        "class_names": ["ant"],
        "dataset": {
            "split": {"train": 0.8, "val": 0.2, "test": 0.0},
        },
        "training": {
            "device": "0",
            "seed": 7,
            "epochs": 12,
            "batch": 4,
        },
        "roles": [{"role": "semantic_sam3", "imgsz": 1008}],
        "sam3": sam3_values,
    }
    if publish_values is not None:
        payload["publish"] = publish_values
    return payload


def _cli_sam3_spec(tmp_path, sam3_values: dict, publish_values: dict | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = _cli_plan_payload(tmp_path, sam3_values, publish_values)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    plan = load_training_plan(path)
    entries = plan.role_entries(
        {TrainingRole.SEMANTIC_SAM3.value: str(tmp_path / "derived")}
    )
    sam3_entries = [e for e in entries if e["role"] is TrainingRole.SEMANTIC_SAM3]
    assert len(sam3_entries) == 1
    return sam3_entries[0]["spec"], plan


def test_gui_and_cli_sam3lora_params_agree_field_by_field(tmp_path, monkeypatch):
    """Same explicit inputs on both paths -> byte-identical Sam3LoraParams.

    Both `env_name` and `publish.auto_import` are given EXPLICITLY here so
    this test isolates "does the plumbing preserve every field" from the two
    documented *default*-divergences covered by the tests below.
    """

    reference = Sam3LoraParams(**_REFERENCE_KWARGS)

    gui_spec, _dlg = _drive_gui_sam3_spec(tmp_path / "gui", monkeypatch, reference)

    cli_spec, _plan = _cli_sam3_spec(
        tmp_path / "cli",
        sam3_values=dict(_REFERENCE_KWARGS),
        publish_values={"auto_import": True, "auto_select": False},
    )

    gui_params = gui_spec.sam3_params
    cli_params = cli_spec.sam3_params
    assert gui_params is not None
    assert cli_params is not None

    mismatches = {
        f.name: (getattr(gui_params, f.name), getattr(cli_params, f.name))
        for f in fields(Sam3LoraParams)
        if getattr(gui_params, f.name) != getattr(cli_params, f.name)
    }
    assert not mismatches, f"Sam3LoraParams field(s) diverged: {mismatches}"

    # Publish policy was given explicitly identically on both sides too.
    assert asdict(gui_spec.publish_policy) == asdict(cli_spec.publish_policy)


def test_env_name_default_resolution_matches_and_honors_override(tmp_path, monkeypatch):
    """Fix (a): the GUI must honor HYDRA_SAM3_ENV like the CLI does.

    The raw `env_name` field is allowed to differ by *default* (the GUI
    widget always carries a concrete resolved string; the CLI/JSON default
    is the empty "resolve later" sentinel) -- that is documented here as the
    intended difference, not silently skipped. What must be identical is
    what the shared `resolve_sam3_env` resolves each side's value to.
    """

    monkeypatch.delenv("HYDRA_SAM3_ENV", raising=False)

    # Neither side names the env explicitly.
    default_params = Sam3LoraParams(**{**_REFERENCE_KWARGS, "env_name": ""})
    gui_spec, _dlg = _drive_gui_sam3_spec(
        tmp_path / "gui-default", monkeypatch, default_params
    )
    cli_spec, _plan = _cli_sam3_spec(
        tmp_path / "cli-default",
        sam3_values={k: v for k, v in _REFERENCE_KWARGS.items() if k != "env_name"},
    )

    # Documented intentional difference in the RAW field:
    assert gui_spec.sam3_params.env_name == DEFAULT_SAM3_ENV
    assert cli_spec.sam3_params.env_name == ""
    # But both resolve to the same actual sidecar env with no override set.
    assert resolve_sam3_env(gui_spec.sam3_params.env_name) == resolve_sam3_env(
        cli_spec.sam3_params.env_name
    )
    assert resolve_sam3_env(gui_spec.sam3_params.env_name) == DEFAULT_SAM3_ENV

    # Now set an override and prove BOTH paths honor it -- this is the actual
    # bug fix (a): before it, the GUI's widget always carried the literal
    # DEFAULT_SAM3_ENV string, so it never even asked the environment.
    monkeypatch.setenv("HYDRA_SAM3_ENV", "hydra-sam3-override")
    gui_spec2, _dlg2 = _drive_gui_sam3_spec(
        tmp_path / "gui-override", monkeypatch, default_params
    )
    cli_spec2, _plan2 = _cli_sam3_spec(
        tmp_path / "cli-override",
        sam3_values={k: v for k, v in _REFERENCE_KWARGS.items() if k != "env_name"},
    )
    assert resolve_sam3_env(gui_spec2.sam3_params.env_name) == "hydra-sam3-override"
    assert resolve_sam3_env(cli_spec2.sam3_params.env_name) == "hydra-sam3-override"


def test_auto_import_default_is_a_documented_intentional_divergence(
    tmp_path, monkeypatch
):
    """Fix (b): the choice is now visible/inspectable, not a hidden literal.

    The GUI keeps its historical default (auto-import ON) via the new
    `chk_auto_import` checkbox; the CLI/JSON default stays OFF
    (`detectkit/config/training.py`). This test pins both defaults
    explicitly as the intended, reviewed difference -- an excluded field
    would hide a regression in either default; asserting the exact values
    documents it instead.
    """

    reference = Sam3LoraParams(**_REFERENCE_KWARGS)

    gui_spec, dlg = _drive_gui_sam3_spec(tmp_path / "gui", monkeypatch, reference)
    assert dlg.chk_auto_import.isChecked() is True
    assert gui_spec.publish_policy.auto_import is True

    cli_spec, _plan = _cli_sam3_spec(
        tmp_path / "cli", sam3_values=dict(_REFERENCE_KWARGS), publish_values=None
    )
    assert cli_spec.publish_policy.auto_import is False

    # auto_select is NOT part of this deliberate divergence -- it must still
    # agree, so a future change that also flips it is caught here.
    assert gui_spec.publish_policy.auto_select == cli_spec.publish_policy.auto_select
    assert gui_spec.publish_policy.auto_select is False


def test_publish_policy_has_no_undocumented_fields():
    """Fails loudly if PublishPolicy grows a field this suite doesn't know about."""

    assert {f.name for f in fields(PublishPolicy)} == {"auto_import", "auto_select"}
