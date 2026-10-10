"""Resuming an interrupted SAM3 LoRA run from an epoch checkpoint.

Stdlib-only on purpose: the launcher (hydra env), the preflight and the
training sidecar (hydra-sam3 env) all import this, and only the sidecar has
torch.

What a resume restores, and what it cannot:

* Adapter weights come from ``checkpoints/epoch_NNN.pt``.
* The LR schedule is a pure function of the step count, so fast-forwarding
  it to ``N * steps_per_epoch`` reproduces it exactly.
* Data order and augmentation are seeded per epoch (``seed + epoch``), so
  epochs N+1.. draw the same batches the uninterrupted run would have.
* Early stopping and checkpoint selection are replayed from the run's own
  ``val_series.jsonl``.
* Optimizer moments and the RNG streams are restored only when a matching
  ``checkpoints/trainer_state.pt`` exists (written from this change on).
  Without it the resume is a WARM restart: AdamW starts from zero moments
  and dropout draws differ, and the run says so.

A resume never re-prepares the dataset. Preparation is not byte-reproducible
(timestamped dataset names, re-derived labels), and a different tile set
would change ``steps_per_epoch``, the LR curve and -- worst -- the validation
split, making the replayed losses incomparable with the new ones. The resumed
run reuses the interrupted run's ``derived_dataset_dir`` instead.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EPOCH_CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d{3,})\.pt$")
TRAINER_STATE_FILENAME = "trainer_state.pt"
RESUME_RECORD_FILENAME = "resume.json"
VAL_SERIES_FILENAME = "val_series.jsonl"
SPEC_FILENAME = "spec.json"

# Fields of `sam3_params` a resume may legitimately differ on. `batch` is the
# auto-batch request (-1) in a plan but the resolved value in the saved spec;
# the resumed spec is pinned to the saved value before comparison anyway.
# `env_name` names the sidecar env, which does not change what trains.
_RESUME_TOLERATED_PARAM_KEYS = frozenset({"env_name"})


class ResumeError(ValueError):
    """A resume request that cannot be honoured faithfully."""


@dataclass(frozen=True)
class ResumePoint:
    """Where an interrupted run stopped, derived from its checkpoint path."""

    checkpoint: Path
    run_dir: Path
    epoch: int

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint.parent


def parse_resume_checkpoint(path: str | Path) -> ResumePoint:
    """Validate an epoch checkpoint path and locate its run directory.

    Only a COMPLETED epoch checkpoint qualifies: the writer promotes the file
    atomically and then writes ``<name>.complete.json``, so a checkpoint
    without that marker may be torn.
    """

    checkpoint = Path(path).expanduser().resolve()
    match = EPOCH_CHECKPOINT_PATTERN.match(checkpoint.name)
    if match is None:
        raise ResumeError(
            f"SAM3 resume needs an epoch checkpoint named epoch_NNN.pt, got "
            f"{checkpoint.name!r}"
        )
    if not checkpoint.is_file():
        raise ResumeError(f"SAM3 resume checkpoint not found: {checkpoint}")
    marker = checkpoint.with_name(checkpoint.name + ".complete.json")
    if not marker.is_file():
        raise ResumeError(
            f"SAM3 resume checkpoint {checkpoint} has no completion marker "
            f"({marker.name}); it may be partially written."
        )
    if checkpoint.parent.name != "checkpoints":
        raise ResumeError(
            f"SAM3 resume checkpoint {checkpoint} is not inside a run's "
            "checkpoints/ directory"
        )
    run_dir = checkpoint.parent.parent
    if not (run_dir / SPEC_FILENAME).is_file():
        raise ResumeError(
            f"SAM3 resume checkpoint {checkpoint} has no {SPEC_FILENAME} in its "
            f"run directory {run_dir}"
        )
    epoch = int(match.group(1))
    if epoch < 1:
        raise ResumeError(f"SAM3 resume checkpoint names epoch {epoch}")
    return ResumePoint(checkpoint=checkpoint, run_dir=run_dir, epoch=epoch)


def load_run_spec_payload(run_dir: str | Path) -> dict[str, Any]:
    """The interrupted run's serialised spec, as the launcher wrote it."""

    path = Path(run_dir) / SPEC_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResumeError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("sam3_params"), dict
    ):
        raise ResumeError(f"{path} is not a SAM3 run spec")
    if payload.get("role") != "semantic_sam3":
        raise ResumeError(f"{path} is a {payload.get('role')!r} run, not SAM3")
    return payload


def _normalise(value: Any) -> Any:
    """JSON round-trip, so tuples and lists compare equal."""

    return json.loads(json.dumps(value))


def resume_spec_mismatches(
    saved: dict[str, Any], resumed_sam3_params: dict[str, Any], resumed_seed: int
) -> list[str]:
    """Settings that differ between the interrupted run and the resumed one.

    Anything that shapes the schedule, the data or the model has to match, or
    the resumed epochs train a different run than the one being continued.
    """

    problems: list[str] = []
    if int(saved.get("seed", 42)) != int(resumed_seed):
        problems.append(f"seed: {saved.get('seed')} -> {resumed_seed}")
    old = _normalise(saved.get("sam3_params") or {})
    new = _normalise(resumed_sam3_params)
    for key in sorted(set(old) | set(new)):
        if key in _RESUME_TOLERATED_PARAM_KEYS:
            continue
        if old.get(key) != new.get(key):
            problems.append(f"sam3.{key}: {old.get(key)!r} -> {new.get(key)!r}")
    return problems


def read_val_history(run_dir: str | Path, through_epoch: int) -> list[dict[str, Any]]:
    """Validation records for epochs ``<= through_epoch``, one per epoch.

    The series is append-only history, so an earlier resume can leave two
    rows for one epoch; the LAST row for an epoch is the one that run acted
    on. Returned in epoch order.
    """

    path = Path(run_dir) / VAL_SERIES_FILENAME
    if not path.is_file():
        return []
    by_epoch: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            epoch = int(record["epoch"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ResumeError(
                f"{path}:{line_number} is not a validation record: {exc}"
            ) from exc
        if epoch <= int(through_epoch):
            by_epoch[epoch] = record
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def replay_validation_history(
    history: list[dict[str, Any]], early_stop: Any, selector: Any
) -> tuple[bool, dict[str, Any] | None]:
    """Feed recorded epochs through the stopping and selection rules.

    Returns ``(should_stop, last_record)``. ``should_stop`` is True when the
    interrupted run had already earned an early stop at its last recorded
    epoch (it died between deciding and exiting).
    """

    should_stop = False
    last: dict[str, Any] | None = None
    for record in history:
        value = record.get("val_loss_mean")
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        selector.observe(int(record["epoch"]), value, record=record if finite else None)
        should_stop = early_stop.observe(int(record["epoch"]), value)
        last = record
    return should_stop, last


def scheduler_fast_forward(scheduler: Any, steps: int) -> None:
    """Advance an LR scheduler ``steps`` times without an optimizer step.

    Stepping (rather than constructing with ``last_epoch=``) is the one form
    that behaves identically across torch versions. Torch warns once that
    ``scheduler.step()`` ran before ``optimizer.step()``; that is exactly
    what is intended here, so the warning is silenced.
    """

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(int(steps)):
            scheduler.step()
