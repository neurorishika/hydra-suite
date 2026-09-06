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


def _make_conda_meta(tmp_path: Path, env_name: str, packages: list[str]) -> Path:
    prefix = tmp_path / "envs" / env_name
    conda_meta = prefix / "conda-meta"
    conda_meta.mkdir(parents=True)
    for pkg in packages:
        (conda_meta / f"{pkg}.json").write_text("{}")
    return prefix


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
    return ab.sam3_workload_fingerprint(spec, cuda_device=dev, dataset=dataset)


@pytest.fixture
def prefix(tmp_path):
    return _make_conda_meta(tmp_path, "hydra-sam3", ["sam3-1.0-0", "torch-2.4-cu121"])


def test_same_workload_hits_the_cache(tmp_path, monkeypatch, prefix):
    checkpoint = _checkpoint(tmp_path)
    a = _fingerprint(tmp_path, monkeypatch, prefix, checkpoint=checkpoint)
    b = _fingerprint(tmp_path, monkeypatch, prefix, checkpoint=checkpoint)
    assert a == b


def test_change_sidecar_env_packages_misses_cache(tmp_path, monkeypatch, prefix):
    base = _fingerprint(tmp_path, monkeypatch, prefix)
    (prefix / "conda-meta" / "sam3-1.1-0.json").write_text("{}")
    mutated = _fingerprint(tmp_path, monkeypatch, prefix)
    assert base != mutated


def test_change_checkpoint_bytes_misses_cache(tmp_path, monkeypatch, prefix):
    checkpoint = _checkpoint(tmp_path)
    base = _fingerprint(tmp_path, monkeypatch, prefix, checkpoint=checkpoint)
    checkpoint.write_bytes(b"aaaa")
    mutated = _fingerprint(tmp_path, monkeypatch, prefix, checkpoint=checkpoint)
    assert base != mutated


def test_change_sidecar_env_name_misses_cache(tmp_path, monkeypatch, prefix):
    other_prefix = _make_conda_meta(
        tmp_path, "other-env", ["sam3-1.0-0", "torch-2.4-cu121"]
    )
    _patch_conda(monkeypatch, {"hydra-sam3": prefix, "other-env": other_prefix})
    spec_a = _spec(tmp_path, env_name="hydra-sam3")
    spec_b = _spec(tmp_path, env_name="other-env", checkpoint=Path(spec_a.base_model))
    dataset = _dataset()
    dev = _dev()
    a = ab.sam3_workload_fingerprint(spec_a, cuda_device=dev, dataset=dataset)
    b = ab.sam3_workload_fingerprint(spec_b, cuda_device=dev, dataset=dataset)
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
