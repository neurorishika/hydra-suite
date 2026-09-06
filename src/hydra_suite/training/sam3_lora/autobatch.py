"""Measured auto batch sizing for SAM3 LoRA training.

This module owns the *cache key* half of measured auto batch sizing: given a
training spec, the physical CUDA device, and a dataset density summary, it
builds the `ProfileIdentity` that decides whether a stored memory measurement
may be reused, or must be re-measured.

The probe half (launching candidate batch sizes in the sidecar and recording
peaks) is a separate task and lives elsewhere; this module never imports
``sam3`` or ``torch`` and never launches a subprocess with a full model in it.

Every field folded into the identity is chosen because it changes the memory
a training step consumes. Get the key too coarse and a measurement taken on a
sparse dataset sizes a dense one into an OOM; too fine and nothing ever hits.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hydra_suite.runtime.memory_profiles import ProfileIdentity
from hydra_suite.training.contracts import Sam3LoraParams, TrainingRunSpec
from hydra_suite.training.sam3_lora.env import resolve_sam3_env

OPERATION = "sam3_lora_train"

# In-process cache of the sidecar-env package hash, keyed on
# (prefix, conda-meta directory mtime_ns). A fresh `conda install`/`pip
# install` inside the env changes conda-meta's mtime, so this never needs
# invalidating by hand.
_PACKAGE_HASH_CACHE: dict[tuple[str, int], tuple[str, bool]] = {}


@dataclass(frozen=True, slots=True)
class Sam3DatasetDensityProfile:
    """Dataset facts that drive per-tile device memory during training.

    Distinct from `Sam3DatasetProfile` (preflight.py's admission estimator
    facts): this carries only what the fingerprint needs to tell a sparse
    dataset from a dense one, plus the negative-prompt composition, which is
    per-tile device state (extra prompt queries run through the model on
    every tile).
    """

    max_instances_per_tile: int
    p95_instances_per_tile: int
    num_negatives: int
    negative_prompt_pool: tuple[str, ...] = field(default_factory=tuple)


def device_identity_for(name: str, total_vram_bytes: int) -> str:
    """The one shared convention for `ProfileIdentity.device_identity`.

    Deliberately name + total VRAM only -- never the GPU UUID. A UUID would
    make a profile non-portable between two physically identical cards,
    which defeats caching across a fleet of otherwise-identical machines.
    Every producer of a `ProfileIdentity` (this task and later ones) must
    call this helper rather than hand-rolling the format.
    """

    return f"{name}|{int(total_vram_bytes)}"


def sam3_env_prefix(env_name: str) -> Optional[Path]:
    """Resolve a conda env name to its install prefix.

    Parent-side only: shells out to `conda env list --json`, never imports
    or launches the sidecar. Returns `None` if conda is unavailable or the
    env name is not registered, so callers can fall back gracefully instead
    of raising out of a fingerprint computation.
    """

    try:
        raw = _conda_env_list_json()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    envs = raw.get("envs") if isinstance(raw, dict) else None
    if not isinstance(envs, list):
        return None
    for entry in envs:
        candidate = Path(str(entry))
        if candidate.name == env_name:
            return candidate
    return None


def _conda_env_list_json() -> dict:
    """Thin, mockable wrapper around `conda env list --json`."""

    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return json.loads(result.stdout)


def _conda_run_version_query(env_name: str) -> str:
    """Fallback, degraded package-identity query when the prefix can't be
    resolved directly: one `conda run` version probe, not a directory scan.
    """

    result = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            "-c",
            "import sam3, torch; print(sam3.__version__, torch.__version__, "
            "torch.version.cuda)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return result.stdout.strip()


def sam3_env_package_hash(prefix: Optional[Path], env_name: str) -> tuple[str, bool]:
    """Return `(hash, degraded)` identifying the sidecar env's packages.

    The primary path is a pure filesystem listing: the sorted filenames of
    `<prefix>/conda-meta/*.json` (each is a `name-version-build` string), so
    the hash changes whenever any package in the env -- sam3, torch, the CUDA
    runtime -- is upgraded. It costs a `listdir`, not a subprocess, and is
    cached in-process on `(prefix, conda-meta mtime_ns)`.

    Only when the prefix cannot be resolved does this fall back to one
    `conda run` version query, and it reports `degraded=True` so callers
    know the key is weaker than the filesystem-listing form (e.g. it will not
    notice a CUDA runtime bump that doesn't change `sam3`/`torch` versions).
    """

    if prefix is not None:
        conda_meta = prefix / "conda-meta"
        try:
            mtime_ns = conda_meta.stat().st_mtime_ns
        except OSError:
            mtime_ns = None
        if mtime_ns is not None:
            cache_key = (str(prefix), mtime_ns)
            cached = _PACKAGE_HASH_CACHE.get(cache_key)
            if cached is not None:
                return cached
            try:
                names = sorted(p.name for p in conda_meta.glob("*.json"))
            except OSError:
                names = None
            if names:
                digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
                value = (digest, False)
                _PACKAGE_HASH_CACHE[cache_key] = value
                return value

    # Degraded fallback: prefix unresolved or conda-meta unreadable/empty.
    try:
        probe = _conda_run_version_query(env_name)
    except (OSError, subprocess.SubprocessError):
        probe = f"unresolved:{env_name}"
    digest = hashlib.sha256(probe.encode("utf-8")).hexdigest()
    return digest, True


def _checkpoint_identity(base_model: str) -> str:
    """`(size, mtime_ns)` of the resolved checkpoint file, hashed.

    Falls back to hashing the raw `base_model` string (e.g. a bare HF repo
    id that has no local file yet) so the fingerprint always produces a
    stable string rather than raising.
    """

    path = Path(base_model)
    try:
        stat = path.stat()
        payload = f"file:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        payload = f"ref:{base_model}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _adapter_scope(params: Sam3LoraParams) -> str:
    """Which submodules receive LoRA adapters, plus alpha.

    `ProfileIdentity` has no separate adapter-alpha field, so alpha is
    folded into this one free-text scope string alongside the sorted set of
    adapted submodules.
    """

    flags = {
        "vision_encoder": params.adapt_vision_encoder,
        "text_encoder": params.adapt_text_encoder,
        "geometry_encoder": params.adapt_geometry_encoder,
        "detr_encoder": params.adapt_detr_encoder,
        "detr_decoder": params.adapt_detr_decoder,
        "mask_decoder": params.adapt_mask_decoder,
        "scoring_head": params.adapt_scoring_head,
    }
    scope = "+".join(sorted(name for name, enabled in flags.items() if enabled))
    return f"{scope or 'none'}|alpha={int(params.alpha)}"


def _dataset_density_hash(dataset: Sam3DatasetDensityProfile) -> str:
    """Stable sub-hash of the dataset density profile.

    Folded into `task` so a sparse-vs-dense dataset mismatch misses the
    cache: the max/p95 active instances per tile drive per-tile attention
    cost, and the negative-prompt count/composition drive per-tile extra
    device queries.
    """

    payload = json.dumps(
        {
            "max_instances_per_tile": int(dataset.max_instances_per_tile),
            "p95_instances_per_tile": int(dataset.p95_instances_per_tile),
            "num_negatives": int(dataset.num_negatives),
            "negative_prompt_pool": sorted(dataset.negative_prompt_pool),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sam3_workload_fingerprint(
    spec: TrainingRunSpec,
    *,
    cuda_device,
    dataset: Sam3DatasetDensityProfile,
) -> ProfileIdentity:
    """Build the `ProfileIdentity` cache key for one SAM3 LoRA training spec.

    `cuda_device` is the *observed* physical device (an object exposing
    `.name` and `.total_bytes`, e.g. `CudaDeviceObservation` from
    `preflight.py`) -- never the requested `"cuda"` string, and never the
    device UUID (see `device_identity_for`).
    """

    params = spec.sam3_params
    if params is None:
        raise ValueError("spec.sam3_params is required to fingerprint a SAM3 run")

    env_name = resolve_sam3_env(params.env_name or None)
    prefix = sam3_env_prefix(env_name)
    package_hash, _degraded = sam3_env_package_hash(prefix, env_name)

    model_identity = f"{_checkpoint_identity(spec.base_model)}|{package_hash}"
    backend = f"{env_name}|{package_hash}"
    device_identity = device_identity_for(cuda_device.name, cuda_device.total_bytes)

    imgsz = int(spec.hyperparams.imgsz)
    task_payload = (
        f"imgsz={imgsz}|overlap={params.tile_overlap}|"
        f"object_tile_fraction={params.object_tile_fraction}|"
        f"density={_dataset_density_hash(dataset)}"
    )

    return ProfileIdentity(
        operation=OPERATION,
        model_identity=model_identity,
        backend=backend,
        device_identity=device_identity,
        precision=params.mixed_precision,
        task=task_payload,
        tiling_mode=params.geometry_mode,
        adapter_scope=_adapter_scope(params),
        adapter_rank=int(params.rank),
    )
