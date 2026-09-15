"""Early stopping for SAM3 LoRA training: the rule, the record, the retention.

Every test here FAILS on the base commit (97c0e68b) -- `patience`/`min_delta`
do not exist on `Sam3LoraParams`, `EarlyStopTracker` does not exist, and
`plan_checkpoint_retention` takes no `protected` argument -- except
`test_defaults_are_behaviour_preserving`, which is a CHARACTERIZATION guard:
it asserts the disabled default path, which is by construction what the base
commit already did. It is here to keep it that way.
"""

from __future__ import annotations

import json

import pytest

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora import cli


def test_checkpoint_selector_uses_raw_argmin_not_early_stop_min_delta():
    selector = cli.CheckpointSelector("best_val_loss")
    tracker = cli.EarlyStopTracker(5, 0.01)
    values = [1.53148, 1.40514, 1.40398, 1.42350, 1.41971, 1.45694, 1.43106]
    stopped = False
    for epoch, value in enumerate(values, start=1):
        selector.observe(epoch, value)
        stopped = tracker.observe(epoch, value)

    assert stopped is True
    assert tracker.best_epoch == 2
    assert selector.selected_epoch == 3
    assert selector.selected_value == pytest.approx(1.40398)


def test_checkpoint_selector_uses_earlier_tie_and_skips_non_finite():
    selector = cli.CheckpointSelector("best_val_loss")
    for epoch, loss in enumerate((1.0, float("nan"), 0.8, 0.8, float("inf")), 1):
        selector.observe(epoch, loss)

    assert selector.selected_epoch == 3
    assert selector.selected_value == pytest.approx(0.8)
    assert selector.candidates == [
        {"epoch": 1, "val_loss_mean": 1.0},
        {"epoch": 3, "val_loss_mean": 0.8},
        {"epoch": 4, "val_loss_mean": 0.8},
    ]


def test_checkpoint_selector_can_select_the_terminal_epoch():
    selector = cli.CheckpointSelector("best_val_loss")
    selector.observe(1, 1.2)
    selector.observe(2, 1.1)
    selector.observe(3, 1.0)  # terminal evaluation on a full-length run

    record = selector.selection_record(
        final_epoch=3, stopped_early=False, fallback_reason=None
    )
    assert record["selected_epoch"] == 3
    assert record["selected_val_loss_mean"] == pytest.approx(1.0)


def test_cpu_adapter_snapshot_is_isolated_from_later_model_updates(monkeypatch):
    torch = pytest.importorskip("torch")
    state = {"block.lora_A": torch.tensor([1.0])}
    monkeypatch.setattr(cli, "adapter_state_dict", lambda _model: state)
    snapshot = cli._cpu_adapter_clone(object())
    state["block.lora_A"].fill_(9.0)

    assert snapshot["block.lora_A"].device.type == "cpu"
    assert snapshot["block.lora_A"].item() == 1.0


def test_checkpoint_selector_record_falls_back_to_last_without_validation():
    selector = cli.CheckpointSelector("best_val_loss")
    record = selector.selection_record(
        final_epoch=4, stopped_early=False, fallback_reason="no_validation_split"
    )

    assert record["selected_epoch"] == 4
    assert record["selection_fallback_reason"] == "no_validation_split"
    assert record["candidates"] == []


def test_checkpoint_selector_distinguishes_nonfinite_validation_from_absence():
    selector = cli.CheckpointSelector("best_val_loss")
    selector.observe(1, float("nan"))
    selector.observe(2, float("inf"))

    assert selector.evaluated_epochs == 2
    assert selector.candidates == []
    record = selector.selection_record(
        final_epoch=2,
        stopped_early=False,
        fallback_reason="no_finite_validation_loss",
    )
    assert record["selection_fallback_reason"] == "no_finite_validation_loss"


# -- Terminal export plan (the four run_training exit paths) -----------------

_AP = {"ap": 0.5}


def _observed(mode, series, *, ap_epochs=()):
    selector = cli.CheckpointSelector(mode)
    for epoch, loss in enumerate(series, start=1):
        record = {"epoch": epoch, "val_loss_mean": loss}
        if epoch in ap_epochs:
            record.update(_AP)
        selector.observe(epoch, loss, record=record)
    return selector


def test_export_plan_full_length_run_restores_an_earlier_best():
    # Full-length run: the terminal epoch is recorded WITH AP; the best is not.
    selector = _observed("best_val_loss", [1.2, 1.0, 1.1], ap_epochs={3})
    plan = cli.plan_terminal_export(
        selector, final_epoch=3, stopped_early=False, have_snapshot=True
    )

    assert plan.selection["selected_epoch"] == 2
    assert plan.export_snapshot is True
    assert plan.load_snapshot is True
    # Epoch 2 carries no AP, so val_stats must re-evaluate the loaded weights
    # rather than reuse epoch 3's record.
    assert plan.reusable_record is None
    assert plan.fallback_reason is None


def test_export_plan_early_stop_reuses_the_selected_record_when_it_has_ap():
    selector = _observed(
        "best_val_loss",
        [1.53148, 1.40514, 1.40398, 1.42350, 1.41971, 1.45694, 1.43106],
        ap_epochs={1, 2, 3, 4, 5, 6, 7},
    )
    plan = cli.plan_terminal_export(
        selector, final_epoch=7, stopped_early=True, have_snapshot=True
    )

    assert plan.selection["selected_epoch"] == 3
    assert plan.selection["stopped_early"] is True
    assert plan.load_snapshot is True
    assert plan.reusable_record["epoch"] == 3


def test_export_plan_terminal_best_needs_no_reload():
    selector = _observed("best_val_loss", [1.2, 1.1, 1.0], ap_epochs={3})
    plan = cli.plan_terminal_export(
        selector, final_epoch=3, stopped_early=False, have_snapshot=True
    )

    assert plan.selection["selected_epoch"] == 3
    assert plan.export_snapshot is True
    assert plan.load_snapshot is False
    assert plan.reusable_record["epoch"] == 3


def test_export_plan_without_validation_falls_back_to_last():
    selector = cli.CheckpointSelector("best_val_loss")
    plan = cli.plan_terminal_export(
        selector, final_epoch=4, stopped_early=False, have_snapshot=False
    )

    assert plan.selection["selected_epoch"] == 4
    assert plan.fallback_reason == "no_validation_split"
    assert plan.export_snapshot is False
    assert plan.load_snapshot is False


def test_export_plan_last_mode_exports_final_and_records_no_fallback():
    # "last" never selects on the series, so an absent validation split is
    # not a fallback -- the record must not claim one happened.
    for series in ([], [1.0, 0.5, 0.9]):
        selector = _observed("last", series)
        final = max(len(series), 1)
        plan = cli.plan_terminal_export(
            selector, final_epoch=final, stopped_early=False, have_snapshot=False
        )
        assert plan.selection["rule"] == "last"
        assert plan.selection["selected_epoch"] == final
        assert plan.fallback_reason is None
        assert plan.selection["selection_fallback_reason"] is None
        assert plan.export_snapshot is False
        assert plan.load_snapshot is False


def test_export_plan_refuses_a_selection_without_a_snapshot():
    selector = _observed("best_val_loss", [1.0, 0.9])
    with pytest.raises(RuntimeError, match="no adapter snapshot"):
        cli.plan_terminal_export(
            selector, final_epoch=2, stopped_early=False, have_snapshot=False
        )


def test_selected_adapter_load_restores_exact_weights():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    snapshot = {"0.bias": torch.tensor([3.0, 4.0])}  # adapter-only subset
    cli._load_selected_adapters(model, snapshot)

    assert torch.equal(model[0].bias.detach(), snapshot["0.bias"])


def test_selected_adapter_load_refuses_a_silent_no_op():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    before = model[0].bias.detach().clone()
    # A renamed key: strict=False would load nothing without complaint.
    with pytest.raises(RuntimeError, match="does not match the model"):
        cli._load_selected_adapters(model, {"renamed.0.bias": torch.tensor([3.0, 4.0])})
    assert torch.equal(model[0].bias.detach(), before)


# -- The stopping rule ------------------------------------------------------


def test_defaults_are_behaviour_preserving():
    """CHARACTERIZATION GUARD (not fail-first): default = never stop early."""
    params = Sam3LoraParams(prompt="ant")
    assert params.patience == 0
    assert cli.EarlyStopTracker(params.patience, params.min_delta).enabled is False


def test_disabled_tracker_never_stops_however_flat_the_curve():
    tracker = cli.EarlyStopTracker(0, 0.005)
    for epoch in range(1, 21):
        assert tracker.observe(epoch, 1.0) is False


def test_stops_after_patience_consecutive_non_improving_epochs():
    tracker = cli.EarlyStopTracker(3, 0.005)
    assert tracker.observe(1, 1.00) is False  # first is always the best
    assert tracker.observe(2, 0.90) is False  # improvement -> counter resets
    assert tracker.observe(3, 0.91) is False  # 1
    assert tracker.observe(4, 0.90) is False  # 2 (equal, not better)
    assert tracker.observe(5, 0.92) is True  # 3 -> stop
    assert tracker.stopped_at_epoch == 5
    assert tracker.best_epoch == 2
    assert tracker.best_value == pytest.approx(0.90)


def test_improvement_must_exceed_min_delta_strictly():
    """Exactly `min_delta` is NOT an improvement; a hair more is."""
    tracker = cli.EarlyStopTracker(2, 0.01)
    tracker.observe(1, 1.00)
    assert tracker.observe(2, 0.99) is False  # exactly min_delta -> no credit
    assert tracker.best_epoch == 1
    assert tracker.observe(3, 0.99) is True  # second non-improvement -> stop

    generous = cli.EarlyStopTracker(2, 0.01)
    generous.observe(1, 1.00)
    assert generous.observe(2, 0.989) is False  # beats it by more than min_delta
    assert generous.best_epoch == 2
    assert generous.epochs_without_improvement == 0


def test_best_advances_only_on_a_qualifying_improvement():
    """A sub-min_delta step does not move the baseline; a cumulative gain does.

    Because ``best_value`` is only rewritten by a qualifying improvement, the
    comparison point never creeps along by sub-threshold amounts. The flip
    side -- deliberate -- is that real cumulative progress still counts: two
    0.004 steps against a 0.005 floor total 0.008 and DO reset patience.
    """
    tracker = cli.EarlyStopTracker(3, 0.005)
    assert tracker.observe(1, 1.000) is False
    assert tracker.observe(2, 0.996) is False  # 0.004 < min_delta: no credit
    assert tracker.best_epoch == 1
    assert tracker.epochs_without_improvement == 1
    assert tracker.observe(3, 0.992) is False  # 0.008 vs the UNMOVED baseline
    assert tracker.best_epoch == 3
    assert tracker.epochs_without_improvement == 0


def test_non_finite_loss_counts_against_patience_but_never_becomes_best():
    tracker = cli.EarlyStopTracker(2, 0.0)
    tracker.observe(1, 0.5)
    assert tracker.observe(2, float("nan")) is False
    assert tracker.observe(3, float("inf")) is True
    assert tracker.best_epoch == 1
    assert tracker.best_value == pytest.approx(0.5)


def test_patience_counts_evaluated_epochs_and_the_record_says_so(monkeypatch):
    """With val cadence 2, patience 3 is 3 validations -- stamped, not guessed."""
    monkeypatch.setenv(cli.VAL_CADENCE_ENV, "2")
    tracker = cli.EarlyStopTracker(2, 0.005)
    tracker.observe(2, 1.0)
    tracker.observe(4, 1.0)
    assert tracker.observe(6, 1.0) is True
    summary = tracker.summary(final_epoch=6)
    assert summary["val_cadence"] == 2
    assert summary["evaluated_epochs"] == 3
    assert summary["monitor"] == "val_loss_mean"


# -- The run artifact -------------------------------------------------------


def test_record_says_why_the_run_stopped(tmp_path):
    tracker = cli.EarlyStopTracker(2, 0.005)
    tracker.observe(1, 0.9)
    tracker.observe(2, 0.95)
    assert tracker.observe(3, 0.95) is True
    path = cli.write_early_stop_record(tmp_path, tracker.summary(final_epoch=3))
    assert path.name == "early_stop.json"
    record = json.loads(path.read_text())
    assert record["stopped_early"] is True
    assert record["stopped_at_epoch"] == 3
    assert record["best_epoch"] == 1
    assert record["best_val_loss_mean"] == pytest.approx(0.9)
    assert record["best_checkpoint"] == "epoch_001.pt"
    assert record["patience"] == 2
    assert record["min_delta"] == pytest.approx(0.005)
    assert record["final_epoch"] == 3


def test_record_of_a_run_that_finished_without_stopping(tmp_path):
    tracker = cli.EarlyStopTracker(5, 0.005)
    for epoch, value in enumerate([1.0, 0.8, 0.6], start=1):
        assert tracker.observe(epoch, value) is False
    record = json.loads(
        cli.write_early_stop_record(
            tmp_path, tracker.summary(final_epoch=3)
        ).read_text()
    )
    assert record["stopped_early"] is False
    assert record["stopped_at_epoch"] is None
    assert record["best_epoch"] == 3


# -- Retention: an early stop must not prune the best epoch -----------------


def _checkpoints(directory, epochs, size=1000):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for epoch in epochs:
        path = directory / cli.epoch_checkpoint_name(epoch)
        path.write_bytes(b"x" * size)
        paths.append(path)
    return paths


def test_unprotected_retention_is_unchanged(tmp_path):
    """CHARACTERIZATION GUARD: the default path still thins the middle."""
    paths = _checkpoints(tmp_path, range(1, 8))
    removed = cli.plan_checkpoint_retention(
        paths, free_bytes=0, adapter_bytes=1000, budget_bytes=3000
    )
    kept = sorted({p.name for p in paths} - {p.name for p in removed})
    assert kept == ["epoch_001.pt", "epoch_002.pt", "epoch_007.pt"]


def test_protected_best_checkpoint_survives_a_binding_budget(tmp_path):
    paths = _checkpoints(tmp_path, range(1, 8))
    removed = cli.plan_checkpoint_retention(
        paths,
        free_bytes=0,
        adapter_bytes=1000,
        budget_bytes=3000,
        protected=["epoch_003.pt"],
    )
    kept = sorted({p.name for p in paths} - {p.name for p in removed})
    assert "epoch_003.pt" in kept
    assert kept == ["epoch_001.pt", "epoch_003.pt", "epoch_007.pt"]


def test_protection_wins_even_when_only_two_checkpoints_fit(tmp_path):
    """max_keep=2 (endpoints only) must still not delete the best epoch."""
    paths = _checkpoints(tmp_path, range(1, 6))
    removed = cli.plan_checkpoint_retention(
        paths,
        free_bytes=0,
        adapter_bytes=1000,
        budget_bytes=1000,  # floors to max_keep = 2
        protected=["epoch_002.pt"],
    )
    kept = sorted({p.name for p in paths} - {p.name for p in removed})
    assert kept == ["epoch_001.pt", "epoch_002.pt", "epoch_005.pt"]


def test_enforce_budget_on_disk_keeps_the_protected_file(tmp_path):
    directory = tmp_path / "checkpoints"
    _checkpoints(directory, range(1, 8))
    for path in list(directory.glob("epoch_*.pt")):
        path.with_name(path.name + ".complete.json").write_text("{}")
    cli.enforce_checkpoint_budget(
        directory,
        budget_bytes=3000,
        log=lambda *_a, **_k: None,
        protected=["epoch_003.pt"],
    )
    survivors = sorted(p.name for p in directory.glob("epoch_*.pt"))
    assert survivors == ["epoch_001.pt", "epoch_003.pt", "epoch_007.pt"]
    # A marker never outlives its artifact.
    markers = sorted(p.name for p in directory.glob("*.complete.json"))
    assert markers == [name + ".complete.json" for name in survivors]


def test_tracker_protects_only_the_current_best(tmp_path):
    tracker = cli.EarlyStopTracker(3, 0.005)
    assert tracker.protected_checkpoint_names() == ()
    tracker.observe(1, 1.0)
    assert tracker.protected_checkpoint_names() == ("epoch_001.pt",)
    tracker.observe(2, 0.5)
    assert tracker.protected_checkpoint_names() == ("epoch_002.pt",)
    tracker.observe(3, 0.51)
    assert tracker.protected_checkpoint_names() == ("epoch_002.pt",)


# -- No new per-epoch compute ----------------------------------------------


def test_stopping_rule_touches_no_model_and_no_new_metric():
    """The rule's whole input is one float the run already computed.

    `observe` takes an epoch number and a loss. There is no model, no batch,
    no dataset and no metric name in its signature, so it structurally cannot
    add a forward pass or a second metric.
    """
    import inspect

    signature = inspect.signature(cli.EarlyStopTracker.observe)
    assert list(signature.parameters) == ["self", "epoch_number", "value"]
