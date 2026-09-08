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
