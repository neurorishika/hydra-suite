from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hydra_suite.runtime.memory_profiles import (
    MemoryMeasurement,
    MemoryProfileStore,
    PressureSettings,
    records_for,
)
from hydra_suite.runtime.resource_budget import AcceleratorKind
from hydra_suite.training.contracts import (
    Sam3LoraParams,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.sam3_lora import autobatch as ab

GiB = 1024**3


def _make_env(
    tmp_path: Path,
    env_name: str,
    conda_packages: tuple[str, ...] = ("sam3-1.0-0", "torch-2.4-cu121"),
    pip_packages: tuple[str, ...] = (),
) -> Path:
    """Build a fake sidecar env prefix with both a conda-meta dir and a
    lib/python3*/site-packages dir, mirroring the real mehek `sam3-lora` env
    where sam3/torch/CUDA are pip-installed, not conda-installed.
    """

    prefix = tmp_path / "envs" / env_name
    conda_meta = prefix / "conda-meta"
    conda_meta.mkdir(parents=True)
    for pkg in conda_packages:
        (conda_meta / f"{pkg}.json").write_text("{}")
    site_packages = prefix / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    for pkg in pip_packages:
        (site_packages / f"{pkg}.dist-info").mkdir()
    return prefix


# Backward-compatible alias for the old fixture name.
_make_conda_meta = _make_env


def _patch_conda(monkeypatch, prefixes: dict[str, Path]):
    def fake_env_list_json():
        return {"envs": [str(p) for p in prefixes.values()]}

    monkeypatch.setattr(ab, "_conda_env_list_json", fake_env_list_json)
    ab._PACKAGE_HASH_CACHE.clear()


def _checkpoint(
    tmp_path: Path, name: str = "checkpoint.pt", content: bytes = b"a"
) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _spec(tmp_path: Path, **overrides) -> TrainingRunSpec:
    checkpoint = overrides.pop("checkpoint", None) or _checkpoint(tmp_path)
    params_overrides = overrides.pop("params_overrides", {})
    params_kwargs = dict(
        prompt="ant",
        rank=16,
        alpha=32,
        mixed_precision="bf16",
        num_negatives=3,
        negative_prompts=["background"],
        env_name=overrides.pop("env_name", "hydra-sam3"),
    )
    params_kwargs.update(params_overrides)
    params = Sam3LoraParams(**params_kwargs)
    hyperparams = TrainingHyperParams(imgsz=overrides.pop("imgsz", 1008))
    return TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path / "derived"),
        base_model=str(checkpoint),
        hyperparams=hyperparams,
        sam3_params=params,
    )


def _dev(name: str = "NVIDIA A6000", total_bytes: int = 48 * GiB, uuid: str = "GPU-1"):
    return SimpleNamespace(name=name, total_bytes=total_bytes, uuid=uuid)


def _dataset(max_instances=10, p95=6, num_negatives=3, pool=("background",)):
    return ab.Sam3DatasetDensityProfile(
        max_instances_per_tile=max_instances,
        p95_instances_per_tile=p95,
        num_negatives=num_negatives,
        negative_prompt_pool=tuple(pool),
    )


_FIXED_CHECKPOINT_NAME = "shared_checkpoint.pt"


def _fingerprint(tmp_path, monkeypatch, prefix, **kwargs):
    _patch_conda(monkeypatch, {"hydra-sam3": prefix})
    checkpoint = kwargs.get("checkpoint")
    if checkpoint is None:
        checkpoint = tmp_path / _FIXED_CHECKPOINT_NAME
        if not checkpoint.exists():
            checkpoint.write_bytes(b"a")
    spec = _spec(
        tmp_path,
        checkpoint=checkpoint,
        **{
            k: v
            for k, v in kwargs.items()
            if k in ("imgsz", "env_name", "params_overrides")
        },
    )
    dataset = _dataset(
        max_instances=kwargs.get("max_instances", 10),
        p95=kwargs.get("p95", 6),
        num_negatives=kwargs.get("num_negatives", 3),
        pool=kwargs.get("pool", ("background",)),
    )
    dev = kwargs.get("dev") or _dev()
    return ab.sam3_workload_fingerprint(spec, cuda_device=dev, dataset=dataset).identity


@pytest.fixture
def prefix(tmp_path):
    # Conda-meta carries unrelated conda-installed packages; sam3/torch/CUDA
    # live only as pip dist-info, matching the real mehek `sam3-lora` env.
    return _make_env(
        tmp_path,
        "hydra-sam3",
        conda_packages=("python-3.11.0-h1234", "pip-24.0-py311"),
        pip_packages=("sam3-0.1.0", "torch-2.11.0+cu128"),
    )


def test_same_workload_hits_the_cache(tmp_path, monkeypatch, prefix):
    checkpoint = _checkpoint(tmp_path)
    a = _fingerprint(tmp_path, monkeypatch, prefix, checkpoint=checkpoint)
    b = _fingerprint(tmp_path, monkeypatch, prefix, checkpoint=checkpoint)
    assert a == b


def test_change_sidecar_env_packages_misses_cache(tmp_path, monkeypatch, prefix):
    """A new conda-meta entry alone changes the key."""
    base = _fingerprint(tmp_path, monkeypatch, prefix)
    (prefix / "conda-meta" / "some-new-package-1.0-0.json").write_text("{}")
    mutated = _fingerprint(tmp_path, monkeypatch, prefix)
    assert base != mutated


def test_change_sidecar_env_dist_info_misses_cache(tmp_path, monkeypatch, prefix):
    """A new pip *.dist-info entry alone changes the key.

    This is the exact scenario the controller found on mehek: sam3/torch/CUDA
    are pip-installed (dist-info only, no conda-meta entry), so an upgrade
    there must still miss the cache even though conda-meta is untouched.
    """
    base = _fingerprint(tmp_path, monkeypatch, prefix)
    site_packages = prefix / "lib" / "python3.11" / "site-packages"
    (site_packages / "torch-2.12.0+cu128.dist-info").mkdir()
    mutated = _fingerprint(tmp_path, monkeypatch, prefix)
    assert base != mutated


def test_change_checkpoint_bytes_misses_cache(tmp_path, monkeypatch, prefix):
    checkpoint = _checkpoint(tmp_path)
    base = _fingerprint(tmp_path, monkeypatch, prefix, checkpoint=checkpoint)
    checkpoint.write_bytes(b"aaaa")
    mutated = _fingerprint(tmp_path, monkeypatch, prefix, checkpoint=checkpoint)
    assert base != mutated


def test_change_sidecar_env_name_misses_cache(tmp_path, monkeypatch, prefix):
    other_prefix = _make_env(tmp_path, "other-env")
    _patch_conda(monkeypatch, {"hydra-sam3": prefix, "other-env": other_prefix})
    spec_a = _spec(tmp_path, env_name="hydra-sam3")
    spec_b = _spec(tmp_path, env_name="other-env", checkpoint=Path(spec_a.base_model))
    dataset = _dataset()
    dev = _dev()
    a = ab.sam3_workload_fingerprint(spec_a, cuda_device=dev, dataset=dataset).identity
    b = ab.sam3_workload_fingerprint(spec_b, cuda_device=dev, dataset=dataset).identity
    assert a != b


def test_change_physical_gpu_misses_cache(tmp_path, monkeypatch, prefix):
    base = _fingerprint(tmp_path, monkeypatch, prefix, dev=_dev(name="NVIDIA A6000"))
    mutated = _fingerprint(tmp_path, monkeypatch, prefix, dev=_dev(name="NVIDIA A100"))
    assert base != mutated


def test_change_precision_misses_cache(tmp_path, monkeypatch, prefix):
    base = _fingerprint(tmp_path, monkeypatch, prefix)
    mutated = _fingerprint(
        tmp_path, monkeypatch, prefix, params_overrides={"mixed_precision": "fp32"}
    )
    assert base != mutated


def test_change_tile_px_misses_cache(tmp_path, monkeypatch, prefix):
    base = _fingerprint(tmp_path, monkeypatch, prefix, imgsz=1008)
    mutated = _fingerprint(tmp_path, monkeypatch, prefix, imgsz=640)
    assert base != mutated


def test_change_lora_rank_misses_cache(tmp_path, monkeypatch, prefix):
    base = _fingerprint(tmp_path, monkeypatch, prefix, params_overrides={"rank": 16})
    mutated = _fingerprint(tmp_path, monkeypatch, prefix, params_overrides={"rank": 32})
    assert base != mutated


def test_change_max_instances_per_tile_misses_cache(tmp_path, monkeypatch, prefix):
    base = _fingerprint(tmp_path, monkeypatch, prefix, max_instances=10)
    mutated = _fingerprint(tmp_path, monkeypatch, prefix, max_instances=40)
    assert base != mutated


def test_change_negative_prompt_pool_misses_cache(tmp_path, monkeypatch, prefix):
    base = _fingerprint(tmp_path, monkeypatch, prefix, pool=("background",))
    mutated = _fingerprint(tmp_path, monkeypatch, prefix, pool=("other",))
    assert base != mutated


def test_change_num_negatives_misses_cache(tmp_path, monkeypatch, prefix):
    base = _fingerprint(tmp_path, monkeypatch, prefix, num_negatives=3)
    mutated = _fingerprint(tmp_path, monkeypatch, prefix, num_negatives=5)
    assert base != mutated


def test_a_denser_dataset_does_not_reuse_a_sparse_measurement(
    tmp_path, monkeypatch, prefix
):
    sparse = _fingerprint(tmp_path, monkeypatch, prefix, max_instances=3)
    dense = _fingerprint(tmp_path, monkeypatch, prefix, max_instances=40)
    store = MemoryProfileStore(tmp_path / "sam3.json")
    measurement = MemoryMeasurement(
        identity=sparse,
        settings=PressureSettings(input_width=1008, input_height=1008, batch_size=1),
        accelerator_kind=AcceleratorKind.CUDA,
        host_peak_bytes=1,
        accelerator_reserved_peak_bytes=8 * GiB,
    )
    store.save((measurement,))
    assert records_for(store.load(), dense) == ()


def test_device_identity_excludes_uuid():
    identity = ab.device_identity_for("NVIDIA A6000", 48 * GiB)
    assert "GPU-1" not in identity
    assert identity == f"NVIDIA A6000|{48 * GiB}"


def test_package_hash_degrades_when_prefix_unresolvable(monkeypatch):
    monkeypatch.setattr(
        ab,
        "_conda_run_version_query",
        lambda env_name: "sam3==1.0 torch==2.4 cu121",
    )
    digest, degraded = ab.sam3_env_package_hash(None, "hydra-sam3")
    assert degraded is True
    assert isinstance(digest, str) and digest


def test_checkpoint_identity_degrades_for_a_bare_repo_id():
    digest_a, degraded_a = ab._checkpoint_identity("facebook/sam3")
    digest_b, degraded_b = ab._checkpoint_identity("facebook/sam3-v2")
    assert degraded_a is True and degraded_b is True
    # Different repo ids must still differ (it's not a constant sentinel)...
    assert digest_a != digest_b
    # ...but two different real revisions published under the SAME repo id
    # are indistinguishable from a bare string alone -- that's exactly why
    # this path is degraded rather than treated as a healthy key.


def test_fingerprint_reports_degraded_reasons_for_unresolvable_env_and_checkpoint(
    tmp_path, monkeypatch
):
    _patch_conda(monkeypatch, {})  # no envs registered -> prefix unresolvable
    monkeypatch.setattr(
        ab, "_conda_run_version_query", lambda env_name: "unresolved-probe"
    )
    spec = _spec(tmp_path, checkpoint=Path("facebook/sam3"))
    result = ab.sam3_workload_fingerprint(spec, cuda_device=_dev(), dataset=_dataset())
    assert "sidecar_env_package_hash_degraded" in result.degraded_reasons
    assert "checkpoint_not_a_stat_able_local_file" in result.degraded_reasons


def test_degraded_key_never_collides_with_a_healthy_one_for_the_same_workload(
    tmp_path, monkeypatch, prefix
):
    healthy = _fingerprint(tmp_path, monkeypatch, prefix)

    _patch_conda(monkeypatch, {})  # force env resolution to fail
    monkeypatch.setattr(
        ab, "_conda_run_version_query", lambda env_name: "unresolved-probe"
    )
    spec = _spec(tmp_path, checkpoint=tmp_path / _FIXED_CHECKPOINT_NAME)
    degraded = ab.sam3_workload_fingerprint(
        spec, cuda_device=_dev(), dataset=_dataset()
    ).identity
    assert healthy != degraded


# ---------------------------------------------------------------------------
# Probe half (Task 3): the candidate ladder, validation, and selection.
# ---------------------------------------------------------------------------


def _identity(precision: str = "bf16"):
    from hydra_suite.runtime.memory_profiles import ProfileIdentity

    return ProfileIdentity(
        operation=ab.OPERATION,
        model_identity="model",
        backend="env",
        device_identity="dev|1",
        precision=precision,
        task="task",
    )


def _record(identity, batch, reserved, *, allocated=None, host=1):
    return MemoryMeasurement(
        identity=identity,
        settings=PressureSettings(
            input_width=1008, input_height=1008, batch_size=batch
        ),
        accelerator_kind=AcceleratorKind.CUDA,
        host_peak_bytes=host,
        accelerator_allocated_peak_bytes=(reserved if allocated is None else allocated),
        accelerator_reserved_peak_bytes=reserved,
        observed_at_unix_ns=1,
    )


def test_probe_candidates_are_a_bounded_power_of_two_ladder(tmp_path):
    assert ab.probe_candidates(_spec(tmp_path)) == (1, 2, 4, 8)


def test_probe_stops_at_the_first_oom_and_never_exceeds_it(tmp_path):
    import torch

    seen = []

    def step(batch):
        seen.append(batch)
        if batch >= 4:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return {"accelerator_reserved_peak_bytes": batch * GiB}

    records = ab.run_probe(_spec(tmp_path), tmp_path, step_fn=step)

    assert seen == [1, 2, 4], "must stop probing after the first OOM"
    assert max(r.settings.batch_size for r in records) == 2


def test_a_probe_that_ooms_at_batch_one_fails_closed(tmp_path):
    import torch

    def step(_batch):
        raise torch.cuda.OutOfMemoryError("out of memory")

    with pytest.raises(ab.ProbeFailedError):
        ab.run_probe(_spec(tmp_path), tmp_path, step_fn=step)
    assert not list(tmp_path.glob("*.json")), "a doomed config must not be cached"


def test_a_host_refusal_stops_the_ladder_rather_than_skipping_it(tmp_path):
    seen = []

    def step(batch):
        seen.append(batch)
        if batch >= 2:
            raise ab.ProbeCandidateRefused("host demand exceeds the budget")
        return {"accelerator_reserved_peak_bytes": GiB}

    records = ab.run_probe(_spec(tmp_path), tmp_path, step_fn=step)

    assert seen == [1, 2]
    assert [r.settings.batch_size for r in records] == [1]


def test_run_probe_checks_cancellation_before_each_candidate(tmp_path):
    seen = []

    def step(batch):
        seen.append(batch)
        return {"accelerator_reserved_peak_bytes": GiB}

    with pytest.raises(ab.ProbeCanceled):
        ab.run_probe(
            _spec(tmp_path),
            tmp_path,
            step_fn=step,
            should_cancel=lambda: len(seen) >= 1,
        )
    assert seen == [1]


def test_explicit_batch_is_honoured_without_probing(tmp_path):
    batch, provenance = ab.resolve_batch(
        _spec(tmp_path, params_overrides={"batch": 4}),
        records=(),
        usable_bytes=24 * GiB,
        maximum=8,
    )
    assert (batch, provenance) == (4, "explicit")


def test_resolve_batch_refuses_rather_than_flooring_at_one(tmp_path):
    identity = _identity()
    records = (_record(identity, 1, 20 * GiB),)
    batch, provenance = ab.resolve_batch(
        _spec(tmp_path, params_overrides={"batch": -1}),
        records=records,
        usable_bytes=2 * GiB,
        maximum=8,
    )
    assert (batch, provenance) == (0, "measured")


def test_validation_discards_foreign_and_non_monotone_records(tmp_path):
    identity = _identity()
    foreign = _identity(precision="fp32")
    records = (
        _record(identity, 1, 10 * GiB),
        _record(foreign, 2, 12 * GiB),
        _record(identity, 4, 5 * GiB),  # peak fell as batch grew: impossible
        _record(identity, 2, 14 * GiB),
    )
    kept = ab.validate_probe_records(records, identity)
    assert [
        (r.settings.batch_size, r.accelerator_reserved_peak_bytes) for r in kept
    ] == [
        (1, 10 * GiB),
        (2, 14 * GiB),
    ]


def test_validation_discards_a_zero_peak_record(tmp_path):
    identity = _identity()
    assert ab.validate_probe_records((_record(identity, 1, 0),), identity) == ()
