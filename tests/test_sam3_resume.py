"""Resuming an interrupted SAM3 LoRA run from an epoch checkpoint."""

from __future__ import annotations

import json
import math

import pytest

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora import cli
from hydra_suite.training.sam3_lora.resume import (
    ResumeError,
    parse_resume_checkpoint,
    read_val_history,
    replay_validation_history,
    resume_spec_mismatches,
    scheduler_fast_forward,
)

# The 2026-10-08 diptera run, killed by a host reboot during epoch 7.
DIPTERA_VAL_LOSSES = [1.80181, 1.73376, 1.91058, 1.56054, 1.47392, 1.73507]


def _make_run(tmp_path, epochs=(1, 2, 3), losses=None, name="old"):
    run_dir = tmp_path / name
    ckpts = run_dir / "checkpoints"
    ckpts.mkdir(parents=True)
    (run_dir / "spec.json").write_text(
        json.dumps(
            {
                "role": "semantic_sam3",
                "seed": 42,
                "derived_dataset_dir": str(tmp_path / "dataset"),
                "sam3_params": {"prompt": "ant", "batch": 8},
            }
        ),
        encoding="utf-8",
    )
    for epoch in epochs:
        path = ckpts / f"epoch_{epoch:03d}.pt"
        path.write_bytes(b"x")
        path.with_name(path.name + ".complete.json").write_text("{}")
    if losses is not None:
        with (run_dir / "val_series.jsonl").open("w") as handle:
            for epoch, loss in enumerate(losses, start=1):
                handle.write(json.dumps({"epoch": epoch, "val_loss_mean": loss}) + "\n")
    return run_dir


# --- checkpoint validation -------------------------------------------------


def test_parse_resume_checkpoint_locates_run_and_epoch(tmp_path):
    run_dir = _make_run(tmp_path)
    point = parse_resume_checkpoint(run_dir / "checkpoints" / "epoch_003.pt")
    assert point.epoch == 3
    assert point.run_dir == run_dir.resolve()


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d / "checkpoints" / "last.pt", "epoch_NNN.pt"),
        (lambda d: d / "checkpoints" / "epoch_009.pt", "not found"),
    ],
)
def test_parse_resume_checkpoint_rejects_bad_paths(tmp_path, mutate, match):
    run_dir = _make_run(tmp_path)
    target = mutate(run_dir)
    if target.name == "last.pt":
        target.write_bytes(b"x")
    with pytest.raises(ResumeError, match=match):
        parse_resume_checkpoint(target)


def test_parse_resume_checkpoint_refuses_torn_checkpoint(tmp_path):
    run_dir = _make_run(tmp_path)
    (run_dir / "checkpoints" / "epoch_002.pt.complete.json").unlink()
    with pytest.raises(ResumeError, match="completion marker"):
        parse_resume_checkpoint(run_dir / "checkpoints" / "epoch_002.pt")


def test_parse_resume_checkpoint_requires_run_spec(tmp_path):
    run_dir = _make_run(tmp_path)
    (run_dir / "spec.json").unlink()
    with pytest.raises(ResumeError, match="spec.json"):
        parse_resume_checkpoint(run_dir / "checkpoints" / "epoch_002.pt")


# --- history replay --------------------------------------------------------


def test_read_val_history_keeps_last_row_per_epoch_and_stops_at_resume(tmp_path):
    run_dir = _make_run(tmp_path, losses=[3.0, 2.0, 1.0])
    with (run_dir / "val_series.jsonl").open("a") as handle:
        handle.write(json.dumps({"epoch": 2, "val_loss_mean": 2.5}) + "\n")
    history = read_val_history(run_dir, through_epoch=2)
    assert [(r["epoch"], r["val_loss_mean"]) for r in history] == [
        (1, 3.0),
        (2, 2.5),
    ]


def test_replay_reproduces_the_diptera_run_state(tmp_path):
    run_dir = _make_run(tmp_path, losses=DIPTERA_VAL_LOSSES)
    early_stop = cli.EarlyStopTracker(patience=5, min_delta=0.01)
    selector = cli.CheckpointSelector("best_val_loss")

    should_stop, last = replay_validation_history(
        read_val_history(run_dir, 6), early_stop, selector
    )

    assert should_stop is False
    assert last["epoch"] == 6
    assert early_stop.best_epoch == 5
    assert early_stop.epochs_without_improvement == 1
    assert selector.selected_epoch == 5
    assert len(selector.candidates) == 6


def test_replay_reports_an_already_earned_stop(tmp_path):
    run_dir = _make_run(tmp_path, losses=[1.0, 2.0, 2.0])
    early_stop = cli.EarlyStopTracker(patience=2, min_delta=0.0)
    selector = cli.CheckpointSelector("best_val_loss")
    should_stop, _ = replay_validation_history(
        read_val_history(run_dir, 3), early_stop, selector
    )
    assert should_stop is True
    assert early_stop.stopped_at_epoch == 3


# --- spec drift --------------------------------------------------------------


def test_resume_spec_mismatches_ignores_shape_only_differences():
    from dataclasses import asdict

    params = asdict(Sam3LoraParams(prompt="ant", object_tile_fractions=(0.05, 0.1)))
    saved = {"seed": 41, "sam3_params": json.loads(json.dumps(params))}
    resumed = dict(params, env_name="other-env")
    assert resume_spec_mismatches(saved, resumed, 41) == []


def test_resume_spec_mismatches_names_every_drifted_setting():
    from dataclasses import asdict

    params = asdict(Sam3LoraParams(prompt="ant"))
    saved = {"seed": 41, "sam3_params": json.loads(json.dumps(params))}
    resumed = dict(params, lr=1e-3)
    problems = resume_spec_mismatches(saved, resumed, 7)
    assert any(p.startswith("seed:") for p in problems)
    assert any(p.startswith("sam3.lr:") for p in problems)


def test_scheduler_fast_forward_lands_on_the_closed_form_lr():
    torch = pytest.importorskip("torch")
    weight = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([weight], lr=5e-5)
    warmup, total = 50, 85880
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cli._cosine_with_warmup(warmup, total)
    )
    scheduler_fast_forward(scheduler, 25764)
    expected = 5e-5 * cli._cosine_with_warmup(warmup, total)(25764)
    assert math.isclose(optimizer.param_groups[0]["lr"], expected, rel_tol=1e-9)


# --- end to end through run_training ----------------------------------------


def _resume_harness(tmp_path, monkeypatch, *, losses_by_epoch, epochs=5):
    """Drive `run_training` with a one-parameter model and no batches.

    Each epoch takes zero optimizer steps; what is under test is where the
    loop starts, what state it starts from, and what it exports.
    """
    torch = pytest.importorskip("torch")

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

    model = _Model()
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec_path = tmp_path / "child_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "derived_dataset_dir": str(tmp_path / "dataset"),
                "sam3_params": {
                    "prompt": "ant",
                    "label_quality_acknowledged": True,
                    "epochs": epochs,
                    "batch": 2,
                    "patience": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    spec = cli._load_spec(spec_path)

    monkeypatch.setattr(cli, "_build_dataloader", lambda *_a, **_k: [object()])
    monkeypatch.setattr(cli, "_runtime_admission_refusal", lambda *_a: None)
    monkeypatch.setattr(cli, "scale_group_summary", lambda _d: {})
    monkeypatch.setattr(cli, "query_count", lambda _d: 4)
    monkeypatch.setattr(
        cli,
        "_build_model_and_loss",
        lambda _p: ("cpu", model, None, None, [model.w]),
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_a: None)
    monkeypatch.setattr(cli, "adapter_state_dict", lambda m: {"w": m.w.detach()})
    monkeypatch.setattr(cli, "_validate_adapter_state", lambda *_a: None)
    logs = []
    monkeypatch.setattr(cli, "emit_log", logs.append)

    seeds = []
    model.weights_seen = []

    def _collate(_descriptors, _batch, *, seed, **_kwargs):
        seeds.append(seed)
        model.weights_seen.append(model.w.item())
        return iter(())

    monkeypatch.setattr(cli, "collate_epoch_batches", _collate)

    def _validate(model_, _spec, _params, *_a, **_k):
        run_dir_path, epoch = _a[-2], _a[-1]
        record = {"epoch": epoch, "val_loss_mean": losses_by_epoch[epoch], "ap": 0.5}
        cli.append_val_record(run_dir_path, record)
        return record

    monkeypatch.setattr(cli, "_record_epoch_validation", _validate)
    evaluated = []
    monkeypatch.setattr(
        cli, "_evaluate_and_write", lambda *a, **k: evaluated.append((a, k))
    )
    return spec, model, seeds, logs, evaluated


def _seed_interrupted_run(tmp_path, torch, values, losses):
    run_dir = _make_run(tmp_path, epochs=(), losses=losses)
    for epoch, value in enumerate(values, start=1):
        path = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        torch.save({"w": torch.tensor([value])}, path)
        path.with_name(path.name + ".complete.json").write_text("{}")
    return run_dir


def test_warm_resume_continues_the_interrupted_run(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    losses = {1: 3.0, 2: 1.0, 3: 2.0, 4: 2.5, 5: 2.6}
    spec, model, seeds, logs, evaluated = _resume_harness(
        tmp_path, monkeypatch, losses_by_epoch=losses
    )
    old = _seed_interrupted_run(tmp_path, torch, [10.0, 20.0, 30.0], [3.0, 1.0, 2.0])
    spec.resume_from = str(old / "checkpoints" / "epoch_003.pt")
    new = tmp_path / "new"
    new.mkdir()

    assert cli.run_training(spec, new) is True

    # Epochs 4 and 5 (indices 3, 4) ran, with the uninterrupted run's seeds.
    assert seeds == [spec.seed + 3, spec.seed + 4]
    # The restored epoch-3 weights were what training continued from.
    assert model.weights_seen == [30.0, 30.0]
    # Epoch 2 stays the raw argmin across the resume and is what is exported.
    exported = torch.load(new / "adapters.pt", weights_only=True)
    assert exported["w"].item() == 20.0
    selection = json.loads((new / "checkpoint_selection.json").read_text())
    assert selection["selected_epoch"] == 2
    assert [c["epoch"] for c in selection["candidates"]] == [1, 2, 3, 4, 5]
    series = [
        json.loads(l) for l in (new / "val_series.jsonl").read_text().splitlines()
    ]
    assert [r["epoch"] for r in series] == [1, 2, 3, 4, 5]
    record = json.loads((new / "resume.json").read_text())[0]
    assert record["mode"] == "warm"
    assert record["resume_epoch"] == 3
    assert record["global_step"] == 3 * record["steps_per_epoch"]
    assert any("RESUMED (WARM) from epoch 3" in line for line in logs)
    # Epoch 4 wrote trainer state for an exact resume next time.
    state = torch.load(new / "checkpoints" / "trainer_state.pt", weights_only=True)
    assert state["epoch"] == 4


def test_exact_resume_uses_trainer_state(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    losses = {1: 3.0, 2: 1.0, 3: 2.0, 4: 2.5, 5: 2.6}
    spec, _model, _seeds, _logs, _ = _resume_harness(
        tmp_path, monkeypatch, losses_by_epoch=losses
    )
    old = _seed_interrupted_run(tmp_path, torch, [10.0, 20.0, 30.0], [3.0, 1.0, 2.0])
    spec.resume_from = str(old / "checkpoints" / "epoch_003.pt")
    first = tmp_path / "first"
    first.mkdir()
    assert cli.run_training(spec, first) is True

    # The launcher writes spec.json into every run dir; the child does not.
    (first / "spec.json").write_text((old / "spec.json").read_text())
    spec2, _m2, seeds2, logs2, _ = _resume_harness(
        tmp_path / "again", monkeypatch, losses_by_epoch=losses
    )
    spec2.resume_from = str(first / "checkpoints" / "epoch_004.pt")
    second = tmp_path / "second"
    second.mkdir()
    assert cli.run_training(spec2, second) is True

    assert seeds2 == [spec2.seed + 4]
    assert json.loads((second / "resume.json").read_text())[0]["mode"] == "exact"
    assert any("RESUMED (EXACT) from epoch 4" in line for line in logs2)


def test_resume_refuses_when_the_selected_checkpoint_was_pruned(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    spec, *_ = _resume_harness(tmp_path, monkeypatch, losses_by_epoch={4: 1.0, 5: 1.0})
    old = _seed_interrupted_run(tmp_path, torch, [10.0, 20.0, 30.0], [3.0, 1.0, 2.0])
    (old / "checkpoints" / "epoch_002.pt").unlink()
    spec.resume_from = str(old / "checkpoints" / "epoch_003.pt")
    new = tmp_path / "new"
    new.mkdir()
    with pytest.raises(RuntimeError, match="epoch_002.pt is missing|selects epoch 2"):
        cli.run_training(spec, new)


def test_resume_refuses_a_finished_run(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    spec, *_ = _resume_harness(tmp_path, monkeypatch, losses_by_epoch={}, epochs=3)
    old = _seed_interrupted_run(tmp_path, torch, [10.0, 20.0, 30.0], [3.0, 1.0, 2.0])
    spec.resume_from = str(old / "checkpoints" / "epoch_003.pt")
    new = tmp_path / "new"
    new.mkdir()
    with pytest.raises(RuntimeError, match="nothing left to train"):
        cli.run_training(spec, new)
