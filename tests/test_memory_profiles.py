from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hydra_suite.runtime.memory_profiles import (
    PROFILE_SCHEMA_VERSION,
    AdaptiveAttemptResult,
    AttemptTelemetry,
    MemoryMeasurement,
    MemoryProfileStore,
    PressureField,
    PressureSettings,
    ProbePlan,
    ProfileIdentity,
    recommend_batch_size,
    resource_telemetry,
    run_with_bounded_oom_retries,
)
from hydra_suite.runtime.process_supervisor import ExitKind
from hydra_suite.runtime.resource_budget import AcceleratorKind


def _identity(**changes):
    values = {
        "operation": "detect",
        "model_identity": "sha256:abc",
        "backend": "torch",
        "device_identity": "GPU-1",
        "precision": "fp16",
        "task": "obb",
    }
    values.update(changes)
    return ProfileIdentity(**values)


def _settings(**changes):
    values = {"input_width": 640, "input_height": 640, "batch_size": 1}
    values.update(changes)
    return PressureSettings(**values)


def _measurement(kind=AcceleratorKind.CUDA, **changes):
    values = {
        "identity": _identity(),
        "settings": _settings(),
        "accelerator_kind": kind,
        "host_peak_bytes": 2_000,
        "accelerator_allocated_peak_bytes": 3_000,
        "accelerator_reserved_peak_bytes": 4_000,
    }
    values.update(changes)
    return MemoryMeasurement(**values)


def test_profile_store_round_trip_and_invalidates_schema_and_estimator(tmp_path):
    path = tmp_path / "profiles.json"
    store = MemoryProfileStore(path)
    record = _measurement()
    store.save([record])
    assert store.load() == (record,)

    raw = json.loads(path.read_text())
    raw["schema_version"] = PROFILE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(raw))
    assert store.load() == ()

    store.save([replace(record, estimator_version="obsolete")])
    assert store.load() == ()


def test_profile_identity_separates_device_model_precision_and_adapter():
    base = _identity()
    assert (
        len(
            {
                base,
                _identity(device_identity="GPU-2"),
                _identity(model_identity="sha256:def"),
                _identity(precision="bf16"),
                _identity(adapter_rank=16, adapter_scope="encoder"),
            }
        )
        == 5
    )


def test_probe_must_begin_at_minimum_batch_and_fit_hard_budgets():
    ProbePlan(_identity(), _settings(batch_size=1), 10_000, 5_000)
    with pytest.raises(ValueError, match="batch size one"):
        ProbePlan(_identity(), _settings(batch_size=2), 10_000, 5_000)


def test_recommendation_is_monotonic_for_memory_and_input_size():
    measured = _measurement()
    small_memory = recommend_batch_size(
        measured,
        available_host_bytes=100_000,
        available_accelerator_bytes=20_000,
        input_width=640,
        input_height=640,
        maximum=64,
    )
    large_memory = recommend_batch_size(
        measured,
        available_host_bytes=100_000,
        available_accelerator_bytes=40_000,
        input_width=640,
        input_height=640,
        maximum=64,
    )
    large_input = recommend_batch_size(
        measured,
        available_host_bytes=100_000,
        available_accelerator_bytes=40_000,
        input_width=1280,
        input_height=1280,
        maximum=64,
    )
    assert large_memory >= small_memory
    assert large_input <= large_memory


def test_mps_uses_one_unified_pool_without_double_counting():
    measured = _measurement(
        AcceleratorKind.MPS,
        accelerator_allocated_peak_bytes=7_000,
        accelerator_reserved_peak_bytes=8_000,
        host_peak_bytes=10_000,
    )
    assert (
        recommend_batch_size(
            measured,
            available_host_bytes=100_000,
            available_accelerator_bytes=None,
            input_width=640,
            input_height=640,
            maximum=64,
            safety_fraction=1.0,
        )
        == 10
    )


def test_retry_reduces_specific_pressure_on_fresh_attempt_and_records_history():
    seen = []

    def launch(settings, attempt):
        seen.append((id(settings), settings, attempt))
        kind = ExitKind.ACCELERATOR_OOM if attempt == 0 else ExitKind.SUCCESS
        return AdaptiveAttemptResult(
            attempt > 0,
            kind,
            AttemptTelemetry(attempt, kind, settings, hard_host_bytes=20_000),
        )

    result = run_with_bounded_oom_retries(
        _settings(batch_size=8, tile_chunk=16),
        launch,
        pressure_order=(PressureField.TILE_CHUNK,),
    )
    assert result.result.success
    assert [item[1].tile_chunk for item in seen] == [16, 8]
    assert seen[0][0] != seen[1][0]
    assert result.adjustments == (
        {"attempt": 1, "field": "tile_chunk", "from": 16, "to": 8},
    )


@pytest.mark.parametrize(
    "kind", [ExitKind.HOST_HARD_LIMIT, ExitKind.ORDINARY_FAILURE, ExitKind.CANCELED]
)
def test_retry_does_not_mask_nonrecoverable_failures(kind):
    calls = []

    def launch(settings, attempt):
        calls.append(attempt)
        return AdaptiveAttemptResult(
            False, kind, AttemptTelemetry(attempt, kind, settings, 20_000)
        )

    result = run_with_bounded_oom_retries(
        _settings(batch_size=8),
        launch,
        pressure_order=(PressureField.BATCH_SIZE,),
    )
    assert calls == [0]
    assert result.result.exit_kind is kind


def test_retry_count_is_finite_when_every_fresh_child_ooms():
    calls = []

    def launch(settings, attempt):
        calls.append(attempt)
        return AdaptiveAttemptResult(
            False,
            ExitKind.HOST_SOFT_LIMIT,
            AttemptTelemetry(
                attempt, ExitKind.HOST_SOFT_LIMIT, settings, hard_host_bytes=20_000
            ),
        )

    result = run_with_bounded_oom_retries(
        _settings(batch_size=16),
        launch,
        pressure_order=(PressureField.BATCH_SIZE,),
    )
    assert calls == [0, 1, 2]
    assert len(result.adjustments) == 2
    assert result.result.success is False


def test_structured_telemetry_reports_admission_limits_peaks_and_adjustments():
    budget = SimpleNamespace(
        estimator_version="v1",
        host_peak_bytes=100,
        accelerator_peak_bytes=200,
        reserved_host_bytes=300,
        usable_host_bytes=400,
        usable_accelerator_bytes=500,
        dominant_phase="inference",
        limits=SimpleNamespace(batch_size=2, workers=1, prefetch_batches=3),
    )
    supervised = SimpleNamespace(
        peak_tree_rss_bytes=90,
        minimum_system_available_bytes=310,
        peak_accelerator_bytes=180,
        classified_exit=SimpleNamespace(kind=ExitKind.SUCCESS),
    )
    telemetry = resource_telemetry(
        budget,
        hard_host_bytes=120,
        soft_host_bytes=110,
        result=supervised,
        effective_parameters={"tile_chunk": 4},
        queue_high_water_bytes=70,
        cache_chunk_size=8,
        retry_history=({"field": "tile_chunk", "from": 8, "to": 4},),
    )
    assert telemetry["admission"]["host_peak_bytes"] == 100
    assert telemetry["applied_limits"]["hard_host_bytes"] == 120
    assert telemetry["effective_parameters"]["tile_chunk"] == 4
    assert telemetry["observed"]["peak_tree_rss_bytes"] == 90
    assert telemetry["observed"]["minimum_system_available_bytes"] == 310
    assert telemetry["observed"]["queue_high_water_bytes"] == 70
    assert telemetry["retry_history"][0]["to"] == 4
