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
from hydra_suite.training.sam3_lora.preflight import CudaDeviceObservation

OPERATION = "sam3_lora_train"

# Mixed into hashed material whenever a fallback path is taken, so a
# degraded key can never collide with a healthy one for the same nominal
# workload (see `Sam3FingerprintResult.degraded_reasons`).
_DEGRADED_MARKER = "degraded"

# In-process cache of the sidecar-env package hash, keyed on
# (prefix, conda-meta mtime_ns, site-packages mtime_ns). A fresh `conda
# install` or `pip install` inside the env changes one of those two
# directories' mtimes, so this never needs invalidating by hand.
_PACKAGE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True, slots=True)
class Sam3FingerprintResult:
    """The cache key plus any reasons it is weaker than a full measurement.

    `degraded_reasons` is non-empty when some part of the identity fell back
    to a coarser signal (the sidecar env prefix couldn't be resolved, or the
    checkpoint didn't resolve to a stat-able local file). The reasons are
    also mixed into the hashed material (see `_DEGRADED_MARKER`), so a
    degraded key can never collide with a healthy one for the same nominal
    workload -- callers that additionally want to log the reason(s) loudly
    at launch can inspect this field without widening `ProfileIdentity`.
    """

    identity: ProfileIdentity
    degraded_reasons: tuple[str, ...] = ()


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


def _site_packages_dir(prefix: Path) -> Optional[Path]:
    """Find `<prefix>/lib/python3*/site-packages`, without hardcoding a version.

    There is ordinarily exactly one `python3*` directory under `lib/`; if
    there are zero or several (an unusual/broken env), this returns `None`
    rather than guessing, and the caller degrades.
    """

    lib_dir = prefix / "lib"
    try:
        candidates = sorted(lib_dir.glob("python3*/site-packages"))
    except OSError:
        return None
    return candidates[0] if len(candidates) == 1 else None


def sam3_env_package_hash(prefix: Optional[Path], env_name: str) -> tuple[str, bool]:
    """Return `(hash, degraded)` identifying the sidecar env's packages.

    Verified against a real sidecar env on the CUDA box: `sam3`, `torch`, and
    the whole CUDA stack are **pip-installed** there (visible only as
    `*.dist-info` under `lib/python3*/site-packages/`), not `conda-meta`
    entries -- `conda-meta` alone would silently fail to notice a `sam3` or
    `torch` upgrade, exactly the collision this key exists to prevent. So the
    primary path hashes the union of two pure filesystem listings:

    - the sorted filenames of `<prefix>/conda-meta/*.json`
      (`name-version-build` strings, conda-installed packages), and
    - the sorted filenames of `<prefix>/lib/python3*/site-packages/*.dist-info`
      (pip-installed packages, e.g. `sam3`, `torch`, `nvidia_cudnn_cu12`).

    Each entry is tagged `conda:<name>` or `pip:<name>` before hashing, so a
    package moving from one installer to the other also changes the key.
    Both are `listdir`-class operations (milliseconds), no subprocess. The
    result is cached in-process on `(prefix, conda-meta mtime_ns,
    site-packages mtime_ns)`, so a fresh install under either directory
    invalidates the cache automatically.

    Only when the prefix cannot be resolved, or neither directory yields any
    entries, does this fall back to one `conda run` version query, and it
    reports `degraded=True` so callers know the key is weaker than the
    filesystem-listing form.
    """

    if prefix is not None:
        conda_meta = prefix / "conda-meta"
        site_packages = _site_packages_dir(prefix)

        try:
            conda_mtime = conda_meta.stat().st_mtime_ns
        except OSError:
            conda_mtime = -1
        try:
            site_mtime = site_packages.stat().st_mtime_ns if site_packages else -1
        except OSError:
            site_mtime = -1

        if conda_mtime != -1 or site_mtime != -1:
            cache_key = (str(prefix), conda_mtime, site_mtime)
            cached = _PACKAGE_HASH_CACHE.get(cache_key)
            if cached is not None:
                return cached, False

            entries: list[str] = []
            try:
                entries.extend(f"conda:{p.name}" for p in conda_meta.glob("*.json"))
            except OSError:
                pass
            if site_packages is not None:
                try:
                    entries.extend(
                        f"pip:{p.name}" for p in site_packages.glob("*.dist-info")
                    )
                except OSError:
                    pass

            if entries:
                digest = hashlib.sha256(
                    "\n".join(sorted(entries)).encode("utf-8")
                ).hexdigest()
                _PACKAGE_HASH_CACHE[cache_key] = digest
                return digest, False

    # Degraded fallback: prefix unresolved, or neither listing yielded
    # anything (broken/empty env layout).
    try:
        probe = _conda_run_version_query(env_name)
    except (OSError, subprocess.SubprocessError):
        probe = f"unresolved:{env_name}"
    digest = hashlib.sha256(f"{_DEGRADED_MARKER}:{probe}".encode("utf-8")).hexdigest()
    return digest, True


def _checkpoint_identity(base_model: str) -> tuple[str, bool]:
    """`(size, mtime_ns)` of the resolved checkpoint file, hashed.

    Returns `(hash, degraded)`. When `base_model` does not resolve to a
    stat-able local file (e.g. a bare HF repo id with no local checkpoint
    yet), this is DEGRADED rather than hashing the bare `base_model` string:
    a plain string hash would be a deterministic function of the repo id
    alone, so two different upstream revisions published under the same
    repo id would silently collide -- exactly what the brief forbids as an
    identity source. The degraded marker is mixed in so that path can never
    produce a key indistinguishable from a real, stat-backed one.
    """

    path = Path(base_model)
    try:
        stat = path.stat()
    except OSError:
        payload = f"{_DEGRADED_MARKER}:ref:{base_model}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), True
    payload = f"file:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), False


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
    cuda_device: CudaDeviceObservation,
    dataset: Sam3DatasetDensityProfile,
) -> Sam3FingerprintResult:
    """Build the `ProfileIdentity` cache key for one SAM3 LoRA training spec.

    `cuda_device` is the *observed* physical device (`CudaDeviceObservation`
    from `preflight.py`, exposing `.name` and `.total_bytes`) -- never the
    requested `"cuda"` string, and never the device UUID (see
    `device_identity_for`).

    Returns a `Sam3FingerprintResult` rather than a bare `ProfileIdentity`:
    when any sub-identity falls back to a degraded signal (the sidecar env
    prefix couldn't be resolved, or the checkpoint isn't a stat-able local
    file), that reason is both mixed into the hashed material (so a degraded
    key can never collide with a healthy one) and surfaced on the result so
    a caller can log it loudly at launch, without widening `ProfileIdentity`
    itself (a shared, already-closed schema).
    """

    params = spec.sam3_params
    if params is None:
        raise ValueError("spec.sam3_params is required to fingerprint a SAM3 run")

    env_name = resolve_sam3_env(params.env_name or None)
    prefix = sam3_env_prefix(env_name)
    package_hash, env_degraded = sam3_env_package_hash(prefix, env_name)
    checkpoint_hash, checkpoint_degraded = _checkpoint_identity(spec.base_model)

    degraded_reasons: list[str] = []
    if env_degraded:
        degraded_reasons.append("sidecar_env_package_hash_degraded")
    if checkpoint_degraded:
        degraded_reasons.append("checkpoint_not_a_stat_able_local_file")

    model_identity = f"{checkpoint_hash}|{package_hash}"
    backend = f"{env_name}|{package_hash}"
    device_identity = device_identity_for(cuda_device.name, cuda_device.total_bytes)

    imgsz = int(spec.hyperparams.imgsz)
    task_payload = (
        f"imgsz={imgsz}|overlap={params.tile_overlap}|"
        f"object_tile_fraction={params.object_tile_fraction}|"
        f"density={_dataset_density_hash(dataset)}"
    )

    identity = ProfileIdentity(
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
    return Sam3FingerprintResult(
        identity=identity, degraded_reasons=tuple(degraded_reasons)
    )
