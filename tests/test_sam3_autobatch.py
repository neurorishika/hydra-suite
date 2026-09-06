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


def test_a_child_allocator_that_contradicts_the_cache_key_is_not_stored(tmp_path):
    """The fingerprint is computed by the PARENT from `sam3_env_environ()`;
    the child now reports what it actually ran under. A child that saw the
    default allocator must not have its measurement filed under an
    `expandable_segments` key -- that record would be reused by runs it does
    not describe (42% apart on the same box)."""

    identity = ab.sam3_workload_fingerprint(
        _spec(tmp_path), cuda_device=_dev(), dataset=_dataset()
    ).identity
    assert identity.backend.endswith(f"|{ab.sidecar_alloc_conf_hash()}")

    def step(batch):
        return {
            "accelerator_reserved_peak_bytes": batch * GiB,
            "alloc_conf_hash": ab.sidecar_alloc_conf_hash({}),
        }

    with pytest.raises(ab.ProbeAllocatorMismatch):
        ab.run_probe(_spec(tmp_path), tmp_path, step_fn=step, identity=identity)
    assert not list(tmp_path.glob("*.json")), "a mis-keyed record must not cache"


def test_the_unfingerprinted_exemption_cannot_reach_a_decision(tmp_path):
    """`run_probe` without an identity is exempt from the allocator check.

    That exemption is only safe because such records can never gate anything:
    `_unfingerprinted_identity` is a placeholder no real fingerprint can
    equal, and `validate_probe_records` discards every record under it. Pinned
    here so the exemption cannot quietly become a hole.
    """

    def silent(batch):
        return {"accelerator_reserved_peak_bytes": batch * GiB}

    ladder = ab.run_probe(_spec(tmp_path), tmp_path, step_fn=silent)
    real = ab.sam3_workload_fingerprint(
        _spec(tmp_path), cuda_device=_dev(), dataset=_dataset()
    ).identity

    assert ladder, "the ladder itself still runs"
    assert ab.validate_probe_records(tuple(ladder), real) == ()


def test_only_a_self_reporting_child_allocator_is_accepted(tmp_path):
    """Silence is refused exactly like a contradiction.

    A record with no self-report is one we cannot honestly key, and it is
    allowed to be the sole gate on GPU admission -- the gap it could hide is
    the same 42% a contradiction hides. Refusing it costs nothing: there is no
    production profile store yet, so it retro-invalidates an empty set, and
    the only production producer (`cli.py::run_probe_measurement`) always
    reports.
    """

    identity = ab.sam3_workload_fingerprint(
        _spec(tmp_path), cuda_device=_dev(), dataset=_dataset()
    ).identity

    def matching(batch):
        return {
            "accelerator_reserved_peak_bytes": batch * GiB,
            "alloc_conf_hash": ab.sidecar_alloc_conf_hash(),
        }

    def silent(batch):
        return {"accelerator_reserved_peak_bytes": batch * GiB}

    ladder = ab.run_probe(
        _spec(tmp_path), tmp_path, step_fn=matching, identity=identity
    )
    assert [record.settings.batch_size for record in ladder] == [1, 2, 4, 8]

    with pytest.raises(ab.ProbeAllocatorMismatch):
        ab.run_probe(_spec(tmp_path), tmp_path, step_fn=silent, identity=identity)
    assert not list(tmp_path.glob("*.json")), "an unkeyable record must not cache"


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


def test_change_allocator_config_misses_cache(tmp_path, monkeypatch, prefix):
    """Task 7: only the sidecar's PYTORCH_CUDA_ALLOC_CONF changes -> key changes.

    A measurement taken under `expandable_segments:True` understates a
    default-allocator run by ~31% (measured on mehek); the two must never
    share a cache key.

    `sam3_env_environ()` now hardcodes `expandable_segments:True` into the
    sidecar's composed environment, so mutating the parent's `os.environ`
    no longer moves what the sidecar actually sees. Vary the sidecar
    composition directly (the seam `sidecar_alloc_conf_hash` actually reads)
    instead of `os.environ`.
    """
    monkeypatch.setattr(ab, "sam3_env_environ", lambda: {})
    base = _fingerprint(tmp_path, monkeypatch, prefix)
    monkeypatch.setattr(
        ab,
        "sam3_env_environ",
        lambda: {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    mutated = _fingerprint(tmp_path, monkeypatch, prefix)
    assert base != mutated


def test_allocator_config_enforced_hashes_expandable_not_unset(
    tmp_path, monkeypatch, prefix
):
    """Task 8: with enforcement in place, a run TODAY hashes the
    `expandable_segments:True` state, never the unset state.

    `sam3_env_environ()` unconditionally sets `PYTORCH_CUDA_ALLOC_CONF`, so
    the parent's `os.environ` cannot make the sidecar see "unset" anymore.
    This is a meaningful invariant, not an artifact: it is what makes any
    two measured records comparable -- every measurement taken through the
    real sidecar today shares one allocator identity.
    """
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    today = _fingerprint(tmp_path, monkeypatch, prefix)

    monkeypatch.setattr(ab, "sam3_env_environ", lambda: {})
    unset = _fingerprint(tmp_path, monkeypatch, prefix)

    assert today != unset


def test_allocator_config_pair_reordering_does_not_change_key(
    tmp_path, monkeypatch, prefix
):
    """Order of comma-separated key:value pairs is not semantically meaningful."""
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.8",
    )
    a = _fingerprint(tmp_path, monkeypatch, prefix)
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF",
        "garbage_collection_threshold:0.8,expandable_segments:True",
    )
    b = _fingerprint(tmp_path, monkeypatch, prefix)
    assert a == b


def test_allocator_config_unset_default_and_expandable_all_differ(
    tmp_path, monkeypatch, prefix
):
    """unset, explicitly-default, and expandable_segments:True are three
    distinct states and must hash three different ways.

    `sam3_env_environ()` now hardcodes the sidecar's allocator config, so
    the parent's `os.environ` no longer selects the sidecar's state; the
    sidecar composition itself (`ab.sam3_env_environ`) must be varied.
    """
    monkeypatch.setattr(ab, "sam3_env_environ", lambda: {})
    unset = _fingerprint(tmp_path, monkeypatch, prefix)

    monkeypatch.setattr(
        ab, "sam3_env_environ", lambda: {"PYTORCH_CUDA_ALLOC_CONF": "backend:native"}
    )
    explicit_default = _fingerprint(tmp_path, monkeypatch, prefix)

    monkeypatch.setattr(
        ab,
        "sam3_env_environ",
        lambda: {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    expandable = _fingerprint(tmp_path, monkeypatch, prefix)

    assert len({unset, explicit_default, expandable}) == 3


def test_allocator_config_case_insensitive(tmp_path, monkeypatch, prefix):
    """Case is normalised away, so True/true/TRUE all hash identically."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    a = _fingerprint(tmp_path, monkeypatch, prefix)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "EXPANDABLE_SEGMENTS:true")
    b = _fingerprint(tmp_path, monkeypatch, prefix)
    assert a == b


def test_change_probe_steps_misses_cache(tmp_path, monkeypatch, prefix):
    """The probe PROTOCOL is part of the workload.

    Two steps measure ~35% less than sixty on the same run (7.34 vs 9.93 GiB),
    so raising `PROBE_STEPS` without moving the key would silently reuse every
    stored record as a stale, lower number forever.
    """

    base = _fingerprint(tmp_path, monkeypatch, prefix)
    monkeypatch.setattr(ab, "PROBE_STEPS", ab.PROBE_STEPS + 1)
    mutated = _fingerprint(tmp_path, monkeypatch, prefix)
    assert base != mutated


def test_change_custom_slice_size_misses_cache(tmp_path, monkeypatch, prefix):
    base = _fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        params_overrides={
            "geometry_mode": "custom",
            "slice_width": 512,
            "slice_height": 512,
        },
    )
    mutated = _fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        params_overrides={
            "geometry_mode": "custom",
            "slice_width": 1024,
            "slice_height": 512,
        },
    )
    assert base != mutated


def test_slice_size_is_ignored_outside_custom_geometry(tmp_path, monkeypatch, prefix):
    """`tile_size_for_mode` ignores the slice dims unless the mode is custom.

    Hashing them unconditionally would force a re-probe over a stale number
    that cannot change the tile size, and so cannot change the memory.
    """

    base = _fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        params_overrides={"geometry_mode": "auto_object", "slice_width": 512},
    )
    mutated = _fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        params_overrides={"geometry_mode": "auto_object", "slice_width": 4096},
    )
    assert base == mutated


# ---------------------------------------------------------------------------
# Task 6 -- the resolved tile-px SET is part of the workload.
#
# The probe is the authority on what fits in VRAM. A fingerprint that cannot
# tell a single-scale workload from a multi-scale one lets a run reuse a probe
# measured on different geometry -- and every short probe here has
# UNDER-reported the full-run peak (7.72 -> 12.99 GiB), so a wrongly reused
# record is an OOM, not an inefficiency.
# ---------------------------------------------------------------------------


def _multiscale_fingerprint(
    tmp_path,
    monkeypatch,
    prefix,
    *,
    fractions=(),
    full_frame_mix=False,
    tile_px_set=(),
    checkpoint=None,
):
    _patch_conda(monkeypatch, {"hydra-sam3": prefix})
    if checkpoint is None:
        checkpoint = tmp_path / _FIXED_CHECKPOINT_NAME
        if not checkpoint.exists():
            checkpoint.write_bytes(b"a")
    spec = _spec(
        tmp_path,
        checkpoint=checkpoint,
        params_overrides={
            "object_tile_fractions": tuple(fractions),
            "full_frame_mix": bool(full_frame_mix),
        },
    )
    dataset = ab.Sam3DatasetDensityProfile(
        max_instances_per_tile=10,
        p95_instances_per_tile=6,
        num_negatives=3,
        negative_prompt_pool=("background",),
        tile_px_set=tuple((int(w), int(h)) for w, h in tile_px_set),
    )
    return ab.sam3_workload_fingerprint(
        spec, cuda_device=_dev(), dataset=dataset
    ).identity


def test_multiscale_spec_misses_the_single_scale_probe(tmp_path, monkeypatch, prefix):
    """The OOM this task exists to prevent: one scale vs four, same scalar."""
    single = _multiscale_fingerprint(
        tmp_path, monkeypatch, prefix, tile_px_set=((727, 727),)
    )
    multi = _multiscale_fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        fractions=(0.03, 0.055, 0.11, 0.22),
        tile_px_set=((907, 907), (727, 727), (363, 363), (181, 181)),
    )
    assert single != multi


def test_two_different_scale_sets_miss_each_other(tmp_path, monkeypatch, prefix):
    a = _multiscale_fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        fractions=(0.055, 0.11),
        tile_px_set=((727, 727), (363, 363)),
    )
    b = _multiscale_fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        fractions=(0.055, 0.22),
        tile_px_set=((727, 727), (181, 181)),
    )
    assert a != b


def test_full_frame_mix_alone_misses_the_cache(tmp_path, monkeypatch, prefix):
    """Full frames are the largest images in the set; they drive the peak."""
    off = _multiscale_fingerprint(
        tmp_path, monkeypatch, prefix, fractions=(0.055,), tile_px_set=((727, 727),)
    )
    on = _multiscale_fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        fractions=(0.055,),
        full_frame_mix=True,
        tile_px_set=((727, 727),),
    )
    assert off != on


def test_identical_multiscale_specs_share_one_key(tmp_path, monkeypatch, prefix):
    """Probe reuse must still work -- otherwise the cache is dead weight.

    (Passes before this change too; it is the guard against over-keying.)
    """
    kwargs = dict(
        fractions=(0.03, 0.055),
        full_frame_mix=True,
        tile_px_set=((907, 907), (727, 727)),
    )
    a = _multiscale_fingerprint(tmp_path, monkeypatch, prefix, **kwargs)
    b = _multiscale_fingerprint(tmp_path, monkeypatch, prefix, **kwargs)
    assert a == b


def test_scale_set_order_does_not_change_the_key(tmp_path, monkeypatch, prefix):
    """The set is canonicalised, not f-string-interpolated by accident."""
    a = _multiscale_fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        fractions=(0.03, 0.055),
        tile_px_set=((907, 907), (727, 727)),
    )
    b = _multiscale_fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        fractions=(0.055, 0.03),
        tile_px_set=((727, 727), (907, 907)),
    )
    assert a == b


def test_unreadable_scale_set_is_degraded_and_still_distinct(
    tmp_path, monkeypatch, prefix
):
    """A multi-scale run whose built set could not be read must NOT collide.

    Two different requested sets with no readable manifest are two different
    workloads; falling back to a shared marker would recreate the exact bug.
    """
    _patch_conda(monkeypatch, {"hydra-sam3": prefix})
    checkpoint = _checkpoint(tmp_path)
    dataset = _dataset()
    results = []
    for fractions in ((0.03, 0.055), (0.11, 0.22)):
        spec = _spec(
            tmp_path,
            checkpoint=checkpoint,
            params_overrides={"object_tile_fractions": fractions},
        )
        results.append(
            ab.sam3_workload_fingerprint(spec, cuda_device=_dev(), dataset=dataset)
        )
    assert results[0].identity != results[1].identity
    for result in results:
        assert "multiscale_tile_px_set_unreadable" in result.degraded_reasons


def test_default_params_still_key_on_a_reused_multiscale_manifest(
    tmp_path, monkeypatch, prefix
):
    """Finding 2: gating the suffix on PARAMS alone, while the tile set is
    read from the MANIFEST, lets a default-params spec run against a reused
    dataset directory whose manifest already carries a multi-scale
    `tile_px_set` skip the suffix entirely -- reusing a single-scale VRAM
    probe for a multi-scale workload, precisely the OOM path this task
    exists to close."""
    identity = _multiscale_fingerprint(
        tmp_path,
        monkeypatch,
        prefix,
        # Default params: no requested fan-out, no full-frame mix.
        fractions=(),
        full_frame_mix=False,
        # But the manifest on disk already carries a multi-scale set.
        tile_px_set=((907, 907), (727, 727), (363, 363), (181, 181)),
    )
    assert "tile_px_set=" in identity.task


def test_single_scale_key_is_unchanged_by_the_scale_set_work(
    tmp_path, monkeypatch, prefix
):
    """No gratuitous invalidation: today's default run keeps today's key.

    (Characterization -- passes before the change; it is what makes the
    conditional inclusion above safe for existing stored probes.)
    """
    identity = _multiscale_fingerprint(
        tmp_path, monkeypatch, prefix, tile_px_set=((727, 727),)
    )
    assert "tile_px_set=" not in identity.task
    assert identity.task.startswith("imgsz=1008|overlap=0.25|object_tile_fraction=")
