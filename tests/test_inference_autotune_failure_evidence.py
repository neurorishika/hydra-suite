"""Regression cover for the three CUDA-gate defects around FAILED calibrations.

All three were found running `--inference-autotune automatic` on real RTX 4090
hosts (see the task-12 report in
``.superpowers/sdd/2026-09-07-inference-autotuner-review-remediation``):

* **F5** -- a search that *raises* has still spent the whole budget, but the
  coordinator's ``except`` arm returned without persisting anything, so the
  next run re-paid the same 76 minutes. Measured: an identity clip ran 4608s
  against a 4500s budget, the deadline fired mid-trial, teardown could not reap
  the child tree, ``WorkloadStillOwnedError`` escaped, and no record survived.
* **C2** -- a crashed sidecar surfaced only as the bare string
  ``ordinary-failure`` with the child's traceback discarded, making a broken
  cuDNN install indistinguishable from a bad candidate.
* **F1** -- ``detection_batch_size=2`` crashes sequential-OBB tracking (a
  PRE-EXISTING cache-writer bug, not this branch's). What must hold here is
  that such a candidate is classified as a failure and can never be promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hydra_suite.core.inference.autotune.coordinator import (
    AutotuneCoordinator,
    AutotuneRequest,
)
from hydra_suite.core.inference.autotune.equivalence import CalibrationOutputs
from hydra_suite.core.inference.autotune.measure import MeasurementProtocol
from hydra_suite.core.inference.autotune.models import ProfileState
from hydra_suite.core.inference.autotune.search import (
    CoordinateSearch,
    TrialObservation,
)
from hydra_suite.core.inference.autotune.sidecar import (
    MAX_FAILURE_DETAIL_CHARS,
    redact_diagnostic,
    summarize_child_failure,
)
from hydra_suite.core.inference.autotune.store import InferenceTuningProfileStore

from .autotune_helpers import _key, _planner, _settings

# --------------------------------------------------------------------------
# F5: a raising search must persist a loud stop, not vanish
# --------------------------------------------------------------------------


class _ExplodingExecutor:
    """Stands in for the measured failure: teardown raises after the budget."""

    def run(self, settings, **_kwargs):  # pragma: no cover - never reached
        raise AssertionError("the search should have raised before any trial")


def _raising_search(*_args, **_kwargs):
    raise RuntimeError("child process tree survived SIGKILL; sidecar remains owned")


def test_a_raising_calibration_persists_an_incomplete_record(tmp_path, monkeypatch):
    """F5: the expensive failure must be recorded so it is not re-paid."""
    key = _key()
    store = InferenceTuningProfileStore(tmp_path)
    assert store.load(key) is None

    monkeypatch.setattr(CoordinateSearch, "run", _raising_search)
    request = AutotuneRequest(key, _settings(det=1), _planner(), mode="calibrate")
    coordinator = AutotuneCoordinator(store, trial_executor=_ExplodingExecutor())

    result = coordinator.resolve(request)

    # The run itself must still be safe: settings unchanged.
    assert result.overlay.status == "fallback"
    assert result.overlay.effective == result.overlay.requested
    assert "RuntimeError" in result.overlay.reason

    stored = store.load(key)
    assert stored is not None, (
        "a search that raised after spending the budget left NOTHING behind, so "
        "every later run re-pays the same failure"
    )
    assert stored.state is ProfileState.INCOMPLETE
    assert "RuntimeError" in stored.selection_reason
    # An INCOMPLETE record must never carry a tuned vector.
    assert stored.selected == stored.baseline


def test_a_raising_calibration_under_contention_records_nothing(tmp_path, monkeypatch):
    """Contention is a property of the host, not of this key: a stop measured
    while another workload was running teaches nothing and must not be cached."""
    key = _key()
    store = InferenceTuningProfileStore(tmp_path)
    monkeypatch.setattr(CoordinateSearch, "run", _raising_search)
    request = AutotuneRequest(
        key,
        _settings(det=1),
        _planner(),
        mode="calibrate",
        contention_detected=True,
    )

    AutotuneCoordinator(store, trial_executor=_ExplodingExecutor()).resolve(request)
    assert store.load(key) is None


# --------------------------------------------------------------------------
# C2: a failed trial must carry bounded, de-identified diagnostics
# --------------------------------------------------------------------------


@dataclass
class _FakeExit:
    kind: str
    message: str


@dataclass
class _FakeSupervised:
    classified_exit: _FakeExit
    returncode: int
    output_tail: tuple[str, ...]
    dropped_output_lines: int = 0


def test_child_failure_summary_keeps_the_output_tail():
    supervised = _FakeSupervised(
        _FakeExit("ordinary-failure", "Worker exited with code 1"),
        1,
        (
            '  File "/home/rutalab/gate-head/src/x.py", line 252, in _run_window',
            "RuntimeError: calibration tracking pass did not finish successfully",
        ),
        dropped_output_lines=37,
    )

    detail = summarize_child_failure(supervised)

    assert "Worker exited with code 1" in detail
    assert "returncode=1" in detail
    assert "calibration tracking pass did not finish successfully" in detail
    assert "37 earlier output lines dropped" in detail


def test_diagnostics_never_store_an_absolute_home_path():
    """Profiles are shared artifacts; they must not carry a user's home path."""
    raw = (
        'File "/home/rutalab/gate-head/src/a.py" and '
        '"/Users/someone/Projects/b.py" both failed'
    )
    redacted = redact_diagnostic(raw)
    assert "/home/rutalab" not in redacted
    assert "/Users/someone" not in redacted
    assert "~/gate-head/src/a.py" in redacted
    assert str(Path.home()) not in redacted


def test_diagnostics_are_bounded():
    redacted = redact_diagnostic("x" * (MAX_FAILURE_DETAIL_CHARS * 3))
    assert len(redacted) <= MAX_FAILURE_DETAIL_CHARS
    # The TAIL is what matters -- a traceback's cause is on its last line.
    assert redacted.endswith("x")
    assert redacted.startswith("...")


# --------------------------------------------------------------------------
# F1: a crashing candidate is a failure and can never be promoted
# --------------------------------------------------------------------------


def _healthy_outputs() -> CalibrationOutputs:
    data = pd.DataFrame(
        {
            "FrameID": [0, 1],
            "DetectionID": [0, 10_000],
            "X": [1.0, 1.0],
            "Y": [2.0, 2.0],
            "Theta": [0.0, 0.0],
            "UniqueIdentityKey": ["A", "A"],
        }
    )
    return CalibrationOutputs(data.copy(), data.copy())


class _CrashOnDetectionBatchTwo:
    """Reproduces the sequential-OBB cache-writer crash in miniature.

    The real failure is ``ValueError: cache frame indices must be unique and
    increasing`` raised by ``core/inference/cache/store.py`` whenever
    ``detection_batch_size > 1`` in sequential-OBB mode. The sidecar child
    creates its own ``inference-cache`` directory and hands it to
    ``TrackingEngineCore``, so a trial genuinely exercises that writer and
    genuinely dies. What must hold is that the crash is *screened*, never
    promoted into a production profile.
    """

    def __init__(self):
        self.seen: list[int] = []

    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        self.seen.append(settings.detection_batch_size)
        if settings.detection_batch_size > 1:
            return TrialObservation(
                settings=settings,
                throughput=0.0,
                stage_seconds=0.0,
                outputs=None,
                failure_class="ordinary-failure",
                failure_detail=(
                    "Worker exited with code 1; child output tail: "
                    "ValueError: cache frame indices must be unique and increasing"
                ),
            )
        # The baseline is healthy and deterministic.
        return TrialObservation(
            settings=settings,
            throughput=10.0,
            stage_seconds=1.0,
            outputs=_healthy_outputs(),
            measured_frames=16,
        )


def test_a_crashing_candidate_is_screened_and_never_selected():
    executor = _CrashOnDetectionBatchTwo()
    baseline = _settings(det=1)
    search = CoordinateSearch(
        _planner(),
        executor,
        protocol=MeasurementProtocol(budget_seconds=60.0),
    )

    result = search.run(baseline)

    # Whatever else happens, a crashing vector must never be the winner.
    assert result.selected.detection_batch_size == 1
    assert result.selected == baseline
    assert executor.seen, "the search never ran a trial"


def test_a_crashing_candidates_rejection_names_the_cause():
    """C2 + F1 together: the recorded rejection must explain the crash."""
    rejected: list[tuple[str, str]] = []
    search = CoordinateSearch(
        _planner(),
        _CrashOnDetectionBatchTwo(),
        protocol=MeasurementProtocol(budget_seconds=60.0),
    )
    search._measure(
        (_settings(det=2),),
        phase="stage",
        field_name="detection_batch_size",
        reference=None,
        determinism_floor=None,
        deadline=search.monotonic() + 60.0,
        should_cancel=lambda: False,
        rejected=rejected,
    )

    assert rejected, "a crashing candidate produced no rejection record at all"
    _label, reason = rejected[0]
    assert "measurement_incomplete" in reason
    assert "ordinary-failure" in reason
    assert "cache frame indices must be unique and increasing" in reason, (
        "the rejection must name the CAUSE; before this fix it said only "
        "'ordinary-failure', which is why the CUDA gate could not tell a broken "
        "cuDNN install from a bad candidate"
    )
