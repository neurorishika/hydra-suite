"""The ordering the audit demanded: probe -> final preflight -> limits.

Every test here runs the REAL `train_sam3_lora` parent flow with the sidecar,
the preflight decisions, and the profile store faked out, so the ordering
assertions are about production control flow rather than a mock of it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from hydra_suite.runtime.memory_profiles import (
    MemoryMeasurement,
    MemoryProfileStore,
    PressureSettings,
    ProfileIdentity,
)
from hydra_suite.runtime.process_supervisor import (
    ClassifiedExit,
    ExitKind,
    SupervisedResult,
)
from hydra_suite.runtime.resource_budget import AcceleratorKind
from hydra_suite.training.contracts import (
    Sam3LoraParams,
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.sam3_lora import autobatch as ab
from hydra_suite.training.sam3_lora import cli as sam3_cli
from hydra_suite.training.sam3_lora import train as tr

GiB = 1024**3
_SAM3_PARAM_FIELDS = {f.name for f in dataclasses.fields(Sam3LoraParams)}

# Reserved peaks the fake probe child reports, per candidate batch size.
# 1 and 2 survive; 4 OOMs, which stops the ladder. The two survivors fit a
# line of base 2 GiB + 6 GiB/item, so a 24 GiB free card (0.8 * 24 = 19.2 GiB
# of budget) admits batch 2 and nothing larger was measured.
_PROBE_PEAKS = {1: 8 * GiB, 2: 14 * GiB}

_IDENTITY = ProfileIdentity(
    operation=ab.OPERATION,
    model_identity="model",
    # Must carry the live allocator sub-hash: `run_probe` refuses a record
    # whose child allocator does not match the key it would be filed under,
    # and this harness's fake child reports the live hash below.
    backend=f"env|{ab.sidecar_alloc_conf_hash()}",
    device_identity="Test CUDA|48",
    precision="bf16",
    task="task",
)


def _spec(tmp_path, **overrides):
    params = Sam3LoraParams(prompt="ant", label_quality_acknowledged=True, **overrides)
    return TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=params,
    )


class _Decision:
    """Containment deliberately VARIES with batch.

    With one flat set of limits, an implementation that computed a single
    decision and reused its WorkLimits for every launch would pass -- and
    "candidates 2-8 must not run under batch-1 containment" is the whole
    point of the per-candidate admission.
    """

    def __init__(self, *, admitted=True, refusals=(), batch=1):
        self.batch = int(batch)
        self.admitted = admitted
        self.refusals = tuple(refusals)
        self.warnings = ()
        self.cuda_device = SimpleNamespace(
            uuid="GPU-physical-0", total_bytes=48 * GiB, free_bytes=24 * GiB
        )
        self.dataset = SimpleNamespace(marker="metadata-only")
        self.budget = SimpleNamespace(
            host_peak_bytes=10 * GiB,
            reserved_host_bytes=8 * GiB,
            usable_host_bytes=40 * GiB,
        )
        self.containment_soft_host_bytes = (10 + self.batch) * GiB
        self.containment_hard_host_bytes = (12 + self.batch) * GiB

    def to_dict(self):
        return {"admitted": self.admitted, "refusals": list(self.refusals)}


class _Output:
    def __init__(self):
        self.eof = True

    def drain(self, _timeout=0):
        return [], True, None


def _supervised(kind=ExitKind.SUCCESS, returncode=0):
    return SupervisedResult(
        returncode=returncode,
        classified_exit=ClassifiedExit(kind, f"classified {kind.value}"),
        output_tail=(),
        dropped_output_lines=0,
        watchdog=None,
        cgroup=None,
        output_error=None,
        peak_accelerator_bytes=2 * GiB,
        accelerator_observation_error=None,
        peak_tree_rss_bytes=3 * GiB,
        minimum_system_available_bytes=40 * GiB,
    )


class _Harness:
    def __init__(self):
        self.order: list[str] = []
        self.probe_preflight_batches: list[int] = []
        self.final_preflight_batches: list[int] = []
        self.probe_launch_limits: list[tuple[int, int]] = []
        self.probe_launch_batches: list[int] = []
        self.training_launches: list[tuple[str, ...]] = []
        self.training_launch_limits: list[tuple[int, int]] = []
        self.logs: list[str] = []


def _probe_batch_of(command) -> int | None:
    command = list(command)
    if "--probe" not in command:
        return None
    return int(command[command.index("--probe-batch") + 1])


def _install(
    monkeypatch,
    tmp_path,
    *,
    free_bytes=24 * GiB,
    store_records=(),
    degraded_reasons=(),
    probe_peaks=None,
    cancel_after=None,
    probe_refusals=(),
    sidecar_factory=None,
    stub_analytic=True,
):
    harness = _Harness()
    probe_peaks = _PROBE_PEAKS if probe_peaks is None else probe_peaks
    run_dir = (tmp_path / "run").resolve()
    store_path = tmp_path / "profiles.json"
    if store_records:
        MemoryProfileStore(store_path).save(store_records)

    # DELIBERATE HARNESS AFFORDANCE. These tests prove ORDERING and flow
    # (probe -> final preflight -> limits -> launch, refusals, cancellation,
    # marker behaviour); their fake peaks are stage scenery, not claims about
    # real memory. Stubbing the analytic device estimate keeps every ordering
    # assertion literal instead of coupling it to the requirement formula.
    # The REAL behaviour -- selection must not exceed what the analytic
    # estimate permits -- is covered un-stubbed by
    # `test_selection_never_exceeds_what_the_analytic_estimate_permits`.
    if stub_analytic:
        monkeypatch.setattr(
            tr.preflight_module, "analytic_device_peak_bytes", lambda *a, **k: 0
        )

    monkeypatch.setattr(tr, "get_models_root", lambda: tmp_path / "models")
    monkeypatch.setattr(tr, "_store_path", lambda: store_path)

    def assess(_spec, **_kwargs):
        harness.order.append("final_preflight")
        batch = int(_spec.sam3_params.batch)
        harness.final_preflight_batches.append(batch)
        return _Decision(batch=batch)

    def assess_probe(_spec, *, batch, **_kwargs):
        harness.order.append("probe")
        harness.probe_preflight_batches.append(int(batch))
        if probe_refusals:
            return _Decision(admitted=False, refusals=probe_refusals, batch=batch)
        return _Decision(batch=int(batch))

    monkeypatch.setattr(tr.preflight_module, "assess_preflight", assess)
    monkeypatch.setattr(tr.preflight_module, "assess_probe_preflight", assess_probe)
    monkeypatch.setattr(
        tr.preflight_module,
        "_probe_cuda_device",
        lambda _device: SimpleNamespace(
            uuid="GPU-physical-0",
            name="Test CUDA",
            total_bytes=48 * GiB,
            free_bytes=free_bytes,
        ),
    )
    monkeypatch.setattr(
        tr.autobatch,
        "sam3_workload_fingerprint",
        lambda *a, **k: ab.Sam3FingerprintResult(
            identity=_IDENTITY, degraded_reasons=tuple(degraded_reasons)
        ),
    )
    monkeypatch.setattr(
        tr.autobatch,
        "sam3_dataset_density_profile",
        lambda *a, **k: ab.Sam3DatasetDensityProfile(1, 1, 0, ()),
    )

    real_build = tr.build_limited_launch

    def build(command, limits, **kwargs):
        batch = _probe_batch_of(command)
        if batch is None:
            harness.order.append("build_limited_launch")
            harness.training_launches.append(tuple(command))
            harness.training_launch_limits.append(
                (limits.soft_host_bytes, limits.hard_host_bytes)
            )
        else:
            harness.order.append("probe_launch")
            harness.probe_launch_batches.append(batch)
            harness.probe_launch_limits.append(
                (limits.soft_host_bytes, limits.hard_host_bytes)
            )
        return real_build(command, limits, **kwargs)

    monkeypatch.setattr(tr, "build_limited_launch", build)

    class FakeSidecar:
        def __init__(self, plan, *, prelaunch_check=None, **_kwargs):
            self.plan = plan
            self.output = _Output()
            self.batch = _probe_batch_of(plan.launch.command)
            self.returncode = 0
            self.forced_kind = None
            self.canceled = False
            if prelaunch_check is not None:
                prelaunch_check()
            if self.batch is None:
                artifact = run_dir / "adapters.pt"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"adapter")
                (run_dir / "adapters.pt.complete.json").write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "size_bytes": 7,
                            "sha256": hashlib.sha256(b"adapter").hexdigest(),
                        }
                    ),
                    encoding="utf-8",
                )
                return
            records_dir = run_dir / "probe_records"
            records_dir.mkdir(parents=True, exist_ok=True)
            peak = probe_peaks.get(self.batch)
            payload = (
                {"outcome": "oom", "batch_size": self.batch}
                if peak is None
                else {
                    "outcome": "ok",
                    "batch_size": self.batch,
                    "accelerator_reserved_peak_bytes": peak,
                    "accelerator_allocated_peak_bytes": peak // 2,
                    "host_peak_bytes": GiB,
                    "observed_at_unix_ns": 5,
                    "alloc_conf_hash": ab.sidecar_alloc_conf_hash(),
                }
            )
            (records_dir / f"batch_{self.batch}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            self.returncode = 0 if peak is not None else 3

        @property
        def process(self):
            return SimpleNamespace(poll=lambda: self.returncode)

        def wait(self, *, post_exit_check=None, **_kwargs):
            kind = self.forced_kind or (
                ExitKind.SUCCESS if self.returncode == 0 else ExitKind.ACCELERATOR_OOM
            )
            result = _supervised(kind, returncode=self.returncode)
            if post_exit_check is not None:
                post_exit_check(result)
            return result

        def cancel(self, _grace):
            self.canceled = True

    def dispatch(plan, **kwargs):
        if sidecar_factory is not None:
            return sidecar_factory(FakeSidecar, plan, **kwargs)
        return FakeSidecar(plan, **kwargs)

    monkeypatch.setattr(tr, "SupervisedSidecar", dispatch)

    def should_cancel():
        if cancel_after is None:
            return False
        return len(harness.probe_preflight_batches) >= cancel_after

    harness.run_dir = run_dir
    harness.store_path = store_path
    harness.should_cancel = should_cancel
    return harness


def _run(harness, spec):
    return tr.train_sam3_lora(
        spec,
        str(harness.run_dir),
        log_cb=harness.logs.append,
        should_cancel=harness.should_cancel,
    )


# --- the ordering tests ----------------------------------------------------


def test_resolved_batch_is_written_before_limits_are_built(tmp_path, monkeypatch):
    harness = _install(monkeypatch, tmp_path)

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert result["success"], result.get("error_message")
    assert (
        harness.order.index("probe")
        < harness.order.index("final_preflight")
        < harness.order.index("build_limited_launch")
    )
    spec_on_disk = json.loads((harness.run_dir / "spec.json").read_text())
    assert spec_on_disk["sam3_params"]["batch"] == 2
    assert (
        set(spec_on_disk["sam3_params"]) == _SAM3_PARAM_FIELDS
    ), "extra keys in sam3_params would raise in the child"
    assert spec_on_disk["batch_resolution"]["requested"] == -1
    assert spec_on_disk["batch_resolution"]["resolved"] == 2
    assert spec_on_disk["batch_resolution"]["provenance"] == "measured"
    assert spec_on_disk["batch_resolution"]["fingerprint"]
    artifact = json.loads((harness.run_dir / "batch_resolution.json").read_text())
    assert artifact["resolved"] == 2
    assert artifact["free_bytes"] == 24 * GiB


def test_the_child_can_load_the_rewritten_spec(tmp_path, monkeypatch):
    harness = _install(monkeypatch, tmp_path)
    _run(harness, _spec(tmp_path, batch=-1))

    loaded = sam3_cli._load_spec(harness.run_dir / "spec.json")

    assert loaded.sam3_params.batch == 2


def test_final_preflight_sees_the_resolved_batch_not_minus_one(tmp_path, monkeypatch):
    harness = _install(monkeypatch, tmp_path)
    _run(harness, _spec(tmp_path, batch=-1))

    assert harness.final_preflight_batches == [2, 2]


def test_each_candidate_is_admitted_and_contained_at_its_own_batch(
    tmp_path, monkeypatch
):
    harness = _install(monkeypatch, tmp_path)
    _run(harness, _spec(tmp_path, batch=-1))

    assert harness.probe_preflight_batches == [1, 2, 4]
    assert harness.probe_launch_batches == [1, 2, 4]
    # Each candidate is CONTAINED at its own batch, not at batch 1's limits.
    assert harness.probe_launch_limits == [
        ((10 + batch) * GiB, (12 + batch) * GiB) for batch in (1, 2, 4)
    ]
    assert len(harness.training_launches) == 1
    assert "--probe" not in harness.training_launches[0]
    # ...and training runs under the FINAL decision's limits (batch 2), not
    # under any probe candidate's.
    assert harness.training_launch_limits == [(12 * GiB, 14 * GiB)]


def test_a_successful_probe_is_cached_even_when_selection_refuses(
    tmp_path, monkeypatch
):
    # Measurements are facts about this hardware; free memory is transient.
    harness = _install(monkeypatch, tmp_path, free_bytes=1 * GiB)

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert not result["success"]
    assert result["failure_kind"] == ExitKind.HOST_ADMISSION_REFUSAL.value
    stored = MemoryProfileStore(harness.store_path).load()
    assert [r.settings.batch_size for r in stored] == [1, 2]
    assert not harness.training_launches
    assert not (harness.run_dir / "batch_resolution.json").exists()


def test_zero_survivors_refuse_and_cache_nothing(tmp_path, monkeypatch):
    harness = _install(monkeypatch, tmp_path, probe_peaks={})

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert not result["success"]
    assert not MemoryProfileStore(harness.store_path).load()
    assert not harness.training_launches


def test_cancellation_merges_nothing_and_launches_nothing(tmp_path, monkeypatch):
    harness = _install(monkeypatch, tmp_path, cancel_after=1)

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert result["canceled"]
    assert not MemoryProfileStore(harness.store_path).load()
    assert not harness.training_launches
    assert not (harness.run_dir / "batch_resolution.json").exists()


def test_a_cached_profile_skips_the_probe_entirely(tmp_path, monkeypatch):
    cached = tuple(
        MemoryMeasurement(
            identity=_IDENTITY,
            settings=PressureSettings(
                input_width=640, input_height=640, batch_size=batch
            ),
            accelerator_kind=AcceleratorKind.CUDA,
            host_peak_bytes=GiB,
            accelerator_allocated_peak_bytes=peak // 2,
            accelerator_reserved_peak_bytes=peak,
            observed_at_unix_ns=1,
        )
        for batch, peak in _PROBE_PEAKS.items()
    )
    harness = _install(monkeypatch, tmp_path, store_records=cached)

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert result["success"]
    assert harness.probe_preflight_batches == []
    spec_on_disk = json.loads((harness.run_dir / "spec.json").read_text())
    assert spec_on_disk["sam3_params"]["batch"] == 2
    assert spec_on_disk["batch_resolution"]["provenance"] == "cached"


def test_force_probe_overrides_a_cached_profile(tmp_path, monkeypatch):
    cached = (
        MemoryMeasurement(
            identity=_IDENTITY,
            settings=PressureSettings(input_width=640, input_height=640, batch_size=1),
            accelerator_kind=AcceleratorKind.CUDA,
            host_peak_bytes=GiB,
            accelerator_reserved_peak_bytes=8 * GiB,
            observed_at_unix_ns=1,
        ),
    )
    harness = _install(monkeypatch, tmp_path, store_records=cached)
    monkeypatch.setenv("HYDRA_SAM3_FORCE_PROBE", "1")

    _run(harness, _spec(tmp_path, batch=-1))

    assert harness.probe_preflight_batches == [1, 2, 4]


def test_an_explicit_batch_never_probes_and_is_stamped_explicit(tmp_path, monkeypatch):
    harness = _install(monkeypatch, tmp_path)

    result = _run(harness, _spec(tmp_path, batch=3))

    assert result["success"]
    assert harness.probe_preflight_batches == []
    spec_on_disk = json.loads((harness.run_dir / "spec.json").read_text())
    assert spec_on_disk["sam3_params"]["batch"] == 3
    assert spec_on_disk["batch_resolution"] == {
        **spec_on_disk["batch_resolution"],
        "requested": 3,
        "resolved": 3,
        "provenance": "explicit",
    }


def test_a_degraded_fingerprint_is_logged_as_loudly_as_the_banner(
    tmp_path, monkeypatch
):
    harness = _install(
        monkeypatch, tmp_path, degraded_reasons=("sidecar_env_package_hash_degraded",)
    )

    _run(harness, _spec(tmp_path, batch=-1))

    assert any("auto batch: 2" in line for line in harness.logs)
    assert any(
        "sidecar_env_package_hash_degraded" in line and "DEGRADED" in line
        for line in harness.logs
    )


def test_an_admission_refusal_at_batch_one_reports_its_own_reason(
    tmp_path, monkeypatch
):
    """A missing credential is not "your workload does not fit the GPU". The
    generic no-fit message would send the user hunting for VRAM they have."""

    harness = _install(
        monkeypatch,
        tmp_path,
        probe_refusals=("No Hugging Face credential found.",),
    )

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert not result["success"]
    assert result["failure_kind"] == ExitKind.HOST_ADMISSION_REFUSAL.value
    assert "No Hugging Face credential found." in result["error_message"]
    assert harness.probe_preflight_batches == [1]
    assert not harness.training_launches
    assert not MemoryProfileStore(harness.store_path).load()


def test_a_busy_lease_during_the_probe_is_a_structured_refusal(tmp_path, monkeypatch):
    """`ResourceBusyError` is a RuntimeError, not an OSError, so without the
    probe's own constructor guard it escapes as an unhandled traceback."""

    from hydra_suite.runtime.resource_lease import ResourceBusyError

    def busy(_fake, *_args, **_kwargs):
        raise ResourceBusyError("cuda:GPU-physical-0", None)

    harness = _install(monkeypatch, tmp_path, sidecar_factory=busy)

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert not result["success"]
    assert result["failure_kind"] == ExitKind.HOST_ADMISSION_REFUSAL.value
    assert "probe sidecar launch refused" in result["error_message"]
    assert not MemoryProfileStore(harness.store_path).load()


def test_a_busy_lease_at_a_later_candidate_keeps_the_earlier_measurement(
    tmp_path, monkeypatch
):
    """A measurement is a fact about this hardware. Another job grabbing the
    GPU between candidates is transient, and must not throw away the batch-1
    fact and force the next attempt to re-probe from scratch."""

    from hydra_suite.runtime.resource_lease import ResourceBusyError

    def factory(fake, plan, **kwargs):
        if _probe_batch_of(plan.launch.command) == 2:
            raise ResourceBusyError("cuda:GPU-physical-0", None)
        return fake(plan, **kwargs)

    harness = _install(monkeypatch, tmp_path, sidecar_factory=factory)

    result = _run(harness, _spec(tmp_path, batch=-1))

    stored = MemoryProfileStore(harness.store_path).load()
    assert [r.settings.batch_size for r in stored] == [
        1
    ], "the batch-1 measurement must survive a transient failure at batch 2"
    assert result["success"], result.get("error_message")
    spec_on_disk = json.loads((harness.run_dir / "spec.json").read_text())
    assert spec_on_disk["sam3_params"]["batch"] == 1
    assert spec_on_disk["batch_resolution"]["ladder_terminated_by"] == "refused"


def test_a_host_limit_stop_does_not_become_a_permanent_batch_ceiling(
    tmp_path, monkeypatch
):
    """A cgroup kill caused by an UNRELATED job says nothing about what this
    workload needs. Without this, one bad afternoon caps every later run."""

    def factory(fake, plan, **kwargs):
        sidecar = fake(plan, **kwargs)
        if _probe_batch_of(plan.launch.command) == 2:
            sidecar.returncode = 137
            sidecar.forced_kind = ExitKind.HOST_HARD_LIMIT
        return sidecar

    harness = _install(monkeypatch, tmp_path, sidecar_factory=factory)

    first = _run(harness, _spec(tmp_path, batch=-1))
    assert first["success"]
    resolution = json.loads((harness.run_dir / "batch_resolution.json").read_text())
    assert resolution["ladder_terminated_by"] == "host_limit"
    assert [
        r.settings.batch_size for r in MemoryProfileStore(harness.store_path).load()
    ] == [1]

    # A second run must RE-PROBE rather than silently inheriting the ceiling.
    harness.probe_preflight_batches.clear()
    second = _run(harness, _spec(tmp_path, batch=-1))

    assert second["success"]
    assert harness.probe_preflight_batches == [
        1,
        2,
    ], "a transient host limit must not authorise skipping the untried rungs"


def test_the_safety_fraction_is_applied_exactly_once(tmp_path, monkeypatch):
    """Peaks and free bytes chosen so single vs double application of the 0.8
    fraction give DIFFERENT answers: batch 2 needs 14 GiB; 0.8 * 20 = 16 GiB
    admits it, 0.8 * 0.8 * 20 = 12.8 GiB does not."""

    harness = _install(monkeypatch, tmp_path, free_bytes=20 * GiB)

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert result["success"]
    spec_on_disk = json.loads((harness.run_dir / "spec.json").read_text())
    assert (
        spec_on_disk["sam3_params"]["batch"] == 2
    ), "batch 1 here would mean the safety fraction was applied twice"


def test_every_probe_candidate_announces_itself_before_it_launches(
    tmp_path, monkeypatch
):
    """The probe path used to be entirely silent.

    `run_probe` logs nothing, the child writes nothing until its last step
    lands, and `_run_probe_candidate` passes a no-op progress callback, so
    neither the log nor the GUI showed anything while the ladder ran. Each
    candidate must at least announce its batch and its step count before it
    launches; the child adds a heartbeat every ten steps on top of this.
    """

    harness = _install(monkeypatch, tmp_path)

    _run(harness, _spec(tmp_path, batch=-1))

    announced = [line for line in harness.logs if "probing batch" in line]
    assert [line for line in announced if "probing batch 1" in line]
    assert [line for line in announced if "probing batch 2" in line]
    assert all(str(ab.PROBE_STEPS) in line for line in announced), announced
    # Announced BEFORE the launch, not after the measurement came back.
    assert harness.order.index("probe_launch") > 0


def test_a_between_rung_batch_is_charged_the_observed_peak_of_the_rung_above(
    tmp_path, monkeypatch
):
    """`select_batch` scans contiguously, so a batch nobody measured (3, with
    rungs 1, 2 and 4) is reachable -- and its fitted envelope is a guess.

    `device_requirement_bytes` charges such a batch the OBSERVED peak at the
    next rung above, a guaranteed-safe upper bound by monotonicity. So this
    ladder (4/6/10 GiB, fitting exactly base 2 + 2 GiB/item) must NOT resolve
    to 3 on an 11 GiB card: the fit says 8 GiB, which the 0.8 x 11 = 8.8 GiB
    budget would admit, but the real peak at 4 is 10 GiB. Selection steps down
    to 2, whose 6 GiB is an observation.

    This is also the one case that makes `_resolve_measured_batch`'s downward
    scan do real work rather than re-check what `select_batch` already
    cleared.
    """

    harness = _install(
        monkeypatch,
        tmp_path,
        probe_peaks={1: 4 * GiB, 2: 6 * GiB, 4: 10 * GiB},
        free_bytes=11 * GiB,
    )

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert result["success"]
    resolution = json.loads((harness.run_dir / "batch_resolution.json").read_text())
    assert resolution["resolved"] == 2, (
        "batch 3 here would mean an unvalidated fit was allowed to decide "
        "against a 10 GiB observation one rung up"
    )
    assert resolution["requirement_basis"] == "measured"
    assert resolution["requirement_bytes"] == 6 * GiB
    assert not resolution["requirement_measured_extrapolated"]


def test_an_unreadable_ladder_marker_re_probes_rather_than_trusting_the_cache(
    tmp_path, monkeypatch
):
    """Failing open here would re-arm the permanent-ceiling bug the marker
    exists to prevent -- for every cached workload at once, silently. A
    marker that cannot be trusted must mean "re-probe", not "trust"."""

    cached = tuple(
        MemoryMeasurement(
            identity=_IDENTITY,
            settings=PressureSettings(
                input_width=640, input_height=640, batch_size=batch
            ),
            accelerator_kind=AcceleratorKind.CUDA,
            host_peak_bytes=GiB,
            accelerator_reserved_peak_bytes=peak,
            observed_at_unix_ns=1,
        )
        for batch, peak in _PROBE_PEAKS.items()
    )
    harness = _install(monkeypatch, tmp_path, store_records=cached)
    (tmp_path / "profiles.json.incomplete.json").write_text("{truncated", "utf-8")

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert result["success"]
    assert harness.probe_preflight_batches == [
        1,
        2,
        4,
    ], "a corrupt marker must degrade to re-probing, not to the cache"


def test_the_ladder_marker_is_cleared_by_a_later_authoritative_ladder(
    tmp_path, monkeypatch
):
    """Set and read were proven; the CLEAR was not. A marker that is never
    cleared would re-probe forever."""

    host_limited = [True]

    def factory(fake, plan, **kwargs):
        sidecar = fake(plan, **kwargs)
        if host_limited[0] and _probe_batch_of(plan.launch.command) == 2:
            sidecar.returncode = 137
            sidecar.forced_kind = ExitKind.HOST_HARD_LIMIT
        return sidecar

    harness = _install(monkeypatch, tmp_path, sidecar_factory=factory)
    marker = tmp_path / "profiles.json.incomplete.json"

    _run(harness, _spec(tmp_path, batch=-1))
    assert json.loads(marker.read_text()), "a host-limit stop must be marked"

    # The box quietens down; the ladder now ends for an authoritative reason.
    host_limited[0] = False
    harness.probe_preflight_batches.clear()
    result = _run(harness, _spec(tmp_path, batch=-1))

    assert result["success"]
    assert harness.probe_preflight_batches == [1, 2, 4]
    assert json.loads(marker.read_text()) == {}, "the mark must be cleared"

    # ...and a third run may now legitimately take the cache.
    harness.probe_preflight_batches.clear()
    _run(harness, _spec(tmp_path, batch=-1))
    assert harness.probe_preflight_batches == []


def test_the_cached_path_does_not_claim_the_ladder_was_complete(tmp_path, monkeypatch):
    """A cached run walked no ladder, so it has no standing to report how the
    original one ended."""

    cached = tuple(
        MemoryMeasurement(
            identity=_IDENTITY,
            settings=PressureSettings(
                input_width=640, input_height=640, batch_size=batch
            ),
            accelerator_kind=AcceleratorKind.CUDA,
            host_peak_bytes=GiB,
            accelerator_reserved_peak_bytes=peak,
            observed_at_unix_ns=1,
        )
        for batch, peak in _PROBE_PEAKS.items()
    )
    harness = _install(monkeypatch, tmp_path, store_records=cached)

    _run(harness, _spec(tmp_path, batch=-1))

    resolution = json.loads((harness.run_dir / "batch_resolution.json").read_text())
    assert resolution["provenance"] == "cached"
    assert resolution["ladder_terminated_by"] == "cached"


def test_no_accessor_makes_an_unreadable_marker_look_complete():
    """The `get`-only sentinel of round 2 left `in`, truthiness, and
    iteration falling through to the empty dict beneath -- silently restoring
    the fail-open behaviour for the next person who reached for a different
    accessor. Every accessor must answer for the whole key space, or refuse."""

    sentinel = tr._AllLaddersIncomplete()

    assert sentinel.get("any-fingerprint") is not None
    assert sentinel.get("any-fingerprint", None) is not None
    assert sentinel["any-fingerprint"]
    assert "any-fingerprint" in sentinel
    assert bool(sentinel) is True
    assert not (sentinel.get("any-fingerprint") is None)

    # The enumerating accessors cannot be answered honestly -- the sentinel
    # stands for an unbounded key space -- and must refuse rather than return
    # a value that reads as "nothing is incomplete".
    for accessor in (
        list,
        len,
        lambda m: m.keys(),
        lambda m: m.items(),
        lambda m: m.values(),
    ):
        with pytest.raises(TypeError, match="unreadable ladder marker"):
            accessor(sentinel)


def test_a_corrupt_marker_self_heals_on_an_authoritative_ladder(tmp_path, monkeypatch):
    """The clear path must rebuild the file from empty, not try to pop a key
    out of something it could not read -- otherwise a corrupt marker
    re-probes forever."""

    harness = _install(monkeypatch, tmp_path)
    marker = tmp_path / "profiles.json.incomplete.json"
    marker.write_text("{truncated", "utf-8")

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert result["success"]
    assert json.loads(marker.read_text()) == {}, "the marker must self-heal"

    harness.probe_preflight_batches.clear()
    _run(harness, _spec(tmp_path, batch=-1))
    assert harness.probe_preflight_batches == [], "and then allow the cache"


def test_selection_never_exceeds_what_the_analytic_estimate_permits(
    tmp_path, monkeypatch
):
    """The item-2 guard, re-pointed at the EXTRAPOLATION cap, still un-stubbed.

    Its original premise -- "a measurement may only raise the analytic
    estimate" -- no longer holds: with `expandable_segments:True` enforced for
    the sidecar, the allocator config in the fingerprint, and `PROBE_STEPS`
    at the -2.82% plateau, a measurement now DECIDES at or below an observed
    rung. What survives, and what this guard now pins, is
    `device_requirement_bytes`'s remaining cap: PAST the largest observed
    rung the measured envelope is a linear fit -- a guess -- so
    `max(analytic, measured)` still applies there.

    Why this asserts through the shared function rather than on a resolved
    batch: `memory_profiles.select_batch` iterates only up to the largest
    observed rung, and an explicit positive batch bypasses
    `_resolve_measured_batch` entirely, so no batch beyond the rungs is
    REACHABLE through the auto flow. The beyond-rung case is reached by
    preflight admission of an explicit batch, through this same one function.
    Everything here is real: the records come out of the store the real
    ladder wrote, and the analytic estimate is not stubbed.

    The numbers: the ladder measures batch 1 at 6 GiB and OOMs at 2, so the
    only rung is 1. A one-point fit extrapolates to 12 GiB at batch 2, which
    the 0.8 x 16 GiB = 12.8 GiB budget would happily admit -- but the
    (empty-dataset, rank-16) analytic estimate at batch 2 is ~14.0 GiB (base
    ~12 GiB + 1 extra item x the measured 2 GiB/item
    `_EXTRA_BATCH_DEVICE_BYTES`). The cap must win.
    """

    harness = _install(
        monkeypatch,
        tmp_path,
        stub_analytic=False,
        free_bytes=16 * GiB,
        probe_peaks={1: 6 * GiB},
    )

    result = _run(harness, _spec(tmp_path, batch=-1))
    assert result["success"]

    # The auto flow itself cannot exceed the observed rung.
    spec_on_disk = json.loads((harness.run_dir / "spec.json").read_text())
    assert spec_on_disk["sam3_params"]["batch"] == 1
    resolution = json.loads((harness.run_dir / "batch_resolution.json").read_text())
    assert resolution["requirement_provenance"] == "measured"
    assert not resolution["requirement_measured_extrapolated"]

    # Now the beyond-rung question, on the records the ladder really wrote.
    records = MemoryProfileStore(tmp_path / "profiles.json").load()
    assert [record.settings.batch_size for record in records] == [1]
    analytic_at_2 = tr.preflight_module.analytic_device_peak_bytes(
        _spec(tmp_path, batch=2).sam3_params,
        tr.preflight_module.dataset_profile(str(tmp_path / "dataset")),
        batch_size=2,
    )
    fitted_at_2 = 12 * GiB
    assert fitted_at_2 < analytic_at_2, "the fit must be the LOWER number here"

    requirement = tr.preflight_module.device_requirement_bytes(
        analytic_at_2, records, 2
    )
    assert requirement.bytes == analytic_at_2, (
        "records observed only at batch 1 must not authorise a measured-only "
        "verdict at batch 2; the analytic cap must still apply"
    )
    assert requirement.provenance == "max_extrapolated"
    assert requirement.measured_extrapolated
    assert not requirement.decided_by_measurement
    assert requirement.measured_bytes == fitted_at_2


def test_multi_gpu_device_string_is_announced_at_launch(tmp_path, monkeypatch):
    """SAM3 pins ONE device by UUID; narrowing "0,1" must not be silent.

    The durable `resource_preflight.json` warning is not enough on its own --
    the user watching the run must see WHICH GPU it actually took.
    """

    harness = _install(monkeypatch, tmp_path)
    spec = _spec(tmp_path, batch=-1)
    spec.device = "0,1"

    _run(harness, spec)

    announced = [line for line in harness.logs if "names several GPUs" in line]
    assert len(announced) == 1
    assert "cuda:0" in announced[0] and "GPU-physical-0" in announced[0]


def test_single_device_strings_are_not_announced_as_multi_gpu(tmp_path, monkeypatch):
    harness = _install(monkeypatch, tmp_path)
    spec = _spec(tmp_path, batch=-1)
    spec.device = "0"

    _run(harness, spec)

    assert not any("names several GPUs" in line for line in harness.logs)
