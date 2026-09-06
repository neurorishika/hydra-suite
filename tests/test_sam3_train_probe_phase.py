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
    backend="env",
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
    def __init__(self, *, admitted=True, refusals=()):
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
        self.containment_soft_host_bytes = 11 * GiB
        self.containment_hard_host_bytes = 12 * GiB

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
):
    harness = _Harness()
    probe_peaks = _PROBE_PEAKS if probe_peaks is None else probe_peaks
    run_dir = (tmp_path / "run").resolve()
    store_path = tmp_path / "profiles.json"
    if store_records:
        MemoryProfileStore(store_path).save(store_records)

    monkeypatch.setattr(tr, "get_models_root", lambda: tmp_path / "models")
    monkeypatch.setattr(tr, "_store_path", lambda: store_path)

    def assess(_spec, **_kwargs):
        harness.order.append("final_preflight")
        harness.final_preflight_batches.append(int(_spec.sam3_params.batch))
        return _Decision()

    def assess_probe(_spec, *, batch, **_kwargs):
        harness.order.append("probe")
        harness.probe_preflight_batches.append(int(batch))
        if probe_refusals:
            return _Decision(admitted=False, refusals=probe_refusals)
        return _Decision()

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
            kind = (
                ExitKind.SUCCESS if self.returncode == 0 else ExitKind.ACCELERATOR_OOM
            )
            result = _supervised(kind, returncode=self.returncode)
            if post_exit_check is not None:
                post_exit_check(result)
            return result

        def cancel(self, _grace):
            self.canceled = True

    monkeypatch.setattr(tr, "SupervisedSidecar", sidecar_factory or FakeSidecar)

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
    assert len(harness.training_launches) == 1
    assert "--probe" not in harness.training_launches[0]


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

    def busy(*_args, **_kwargs):
        raise ResourceBusyError("cuda:GPU-physical-0", None)

    harness = _install(monkeypatch, tmp_path, sidecar_factory=busy)

    result = _run(harness, _spec(tmp_path, batch=-1))

    assert not result["success"]
    assert result["failure_kind"] == ExitKind.HOST_ADMISSION_REFUSAL.value
    assert "probe sidecar launch refused" in result["error_message"]
    assert not MemoryProfileStore(harness.store_path).load()
