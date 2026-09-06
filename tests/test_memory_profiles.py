from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hydra_suite.runtime.memory_profiles import (
    MEASURED_SAFETY_FRACTION,
    PROFILE_SCHEMA_VERSION,
    AdaptiveAttemptResult,
    AttemptTelemetry,
    MemoryMeasurement,
    MemoryProfileStore,
    PressureField,
    PressureSettings,
    ProbePlan,
    ProfileIdentity,
    fit_batch_curve,
    merge_records,
    profile_store_path,
    recommend_batch_size,
    records_for,
    resource_telemetry,
    run_with_bounded_oom_retries,
    select_batch,
)
from hydra_suite.runtime.process_supervisor import ExitKind
from hydra_suite.runtime.resource_budget import AcceleratorKind

GiB = 1024**3


def _records(batch_to_peak, **identity_changes):
    identity = _identity(**identity_changes)
    return tuple(
        _measurement(
            identity=identity,
            settings=_settings(batch_size=batch_size),
            accelerator_allocated_peak_bytes=0,
            accelerator_reserved_peak_bytes=peak,
        )
        for batch_size, peak in batch_to_peak.items()
    )


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


def test_load_degrades_to_empty_on_malformed_json(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text("{not valid json")
    assert MemoryProfileStore(path).load() == ()


def test_load_degrades_to_empty_on_missing_expected_keys(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"schema_version": PROFILE_SCHEMA_VERSION}))
    assert MemoryProfileStore(path).load() == ()


def test_load_degrades_to_empty_on_oversized_file(tmp_path, monkeypatch):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps({"schema_version": PROFILE_SCHEMA_VERSION, "records": []})
    )
    monkeypatch.setattr("hydra_suite.runtime.memory_profiles.MAX_PROFILE_BYTES", 4)
    assert MemoryProfileStore(path).load() == ()


def test_load_degrades_to_empty_on_record_missing_identity_or_settings(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "records": [{"settings": {"input_width": 640, "input_height": 640}}],
            }
        )
    )
    assert MemoryProfileStore(path).load() == ()


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


def test_fit_batch_curve_recovers_a_known_line():
    base, slope = fit_batch_curve(_records({1: 8_000, 2: 12_000, 4: 20_000}))
    assert abs(base - 4_000) < 200 and abs(slope - 4_000) < 200


def test_single_record_never_yields_a_zero_slope():
    _base, slope = fit_batch_curve(_records({1: 8_000}))
    assert slope > 0, "a zero slope would make every batch size look free"


def test_selection_never_predicts_below_an_observed_peak():
    # A superlinear jump at 2 must not be smoothed away by the linear fit.
    assert (
        select_batch(
            _records({1: 8 * GiB, 2: 20 * GiB}), usable_bytes=24 * GiB, maximum=8
        )
        == 1
    )


def test_selection_never_exceeds_the_largest_observed_batch():
    assert (
        select_batch(
            _records({1: 1 * GiB, 2: 2 * GiB}), usable_bytes=512 * GiB, maximum=64
        )
        == 2
    )


def test_selection_returns_zero_when_nothing_fits():
    assert select_batch((), usable_bytes=24 * GiB, maximum=8) == 0
    assert select_batch(_records({1: 40 * GiB}), usable_bytes=24 * GiB, maximum=8) == 0


def test_safety_fraction_is_applied_exactly_once():
    # usable_bytes is RAW free memory; 20 GiB at 0.8 admits a 16 GiB peak.
    assert select_batch(_records({1: 16 * GiB}), usable_bytes=20 * GiB, maximum=1) == 1
    assert select_batch(_records({1: 16 * GiB}), usable_bytes=19 * GiB, maximum=1) == 0


def test_default_safety_fraction_matches_measured_constant():
    assert MEASURED_SAFETY_FRACTION == 0.8


def test_merge_replaces_a_record_for_the_same_batch_size():
    merged = merge_records(_records({1: 1_000}), _records({1: 2_000}))
    assert len(merged) == 1 and merged[0].accelerator_reserved_peak_bytes == 2_000


def test_store_round_trips_one_record_per_batch(tmp_path):
    store = MemoryProfileStore(tmp_path / "sam3.json")
    store.save(_records({1: 1_000, 2: 2_000}))
    assert {r.settings.batch_size for r in store.load()} == {1, 2}


def test_records_for_filters_by_exact_identity():
    matching = _records({1: 1_000, 2: 2_000})
    other = _records({1: 500}, operation="train")
    combined = matching + other
    found = records_for(combined, matching[0].identity)
    assert set(found) == set(matching)


def test_profile_store_path_uses_paths_module(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hydra_suite.runtime.memory_profiles.get_data_dir", lambda: tmp_path
    )
    path = profile_store_path("sam3")
    assert path == tmp_path / "memory_profiles" / "sam3.json"
