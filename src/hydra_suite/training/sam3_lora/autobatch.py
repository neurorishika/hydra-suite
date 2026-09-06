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
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from hydra_suite.runtime.memory_profiles import (
    MEASURED_SAFETY_FRACTION,
    MemoryMeasurement,
    PressureSettings,
    ProfileIdentity,
    records_for,
    select_batch,
)
from hydra_suite.runtime.resource_budget import AcceleratorKind
from hydra_suite.training.contracts import Sam3LoraParams, TrainingRunSpec
from hydra_suite.training.sam3_lora.env import resolve_sam3_env, sam3_env_environ
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


_ALLOC_CONF_ENV_VAR = "PYTORCH_CUDA_ALLOC_CONF"

# Measured on mehek (see task-7 brief): the caching allocator's default
# growth policy shows a reserved-memory staircase that opens ~35% over 60
# steps while allocated (live) bytes grow ~2% -- pure fragmentation, not
# real usage. Under `expandable_segments:True` the staircase does not
# appear at all out to 120 steps. A measurement taken under one allocator
# config understates -- never overstates -- what the other needs, so the
# two configs must never share a cache key.
_UNSET_ALLOC_CONF_MARKER = "unset"


def _normalize_pytorch_alloc_conf(raw: Optional[str]) -> str:
    """Canonical string form of `PYTORCH_CUDA_ALLOC_CONF` for hashing.

    The setting is a comma-separated `key:value` list whose ORDER carries no
    semantics -- `a:1,b:2` and `b:2,a:1` configure the allocator identically
    -- so the pairs are sorted before joining, or two equivalent configs
    would silently miss each other's cache forever.

    Keys and values are lowercased: PyTorch's own parser is case-sensitive
    only by accident of how it happens to compare strings, and a user typing
    `True` vs `true` vs `TRUE` for the same boolean must not fragment the
    cache into three copies of the same measurement.

    `None` (the variable is unset in the sidecar environment) normalizes to
    a distinct sentinel, `"unset"`, that can never collide with any set
    value -- including one that normalizes to the empty string (e.g. the
    variable set to `""`) or to a config that happens to reproduce PyTorch's
    compiled-in defaults. Three sidecar states -- unset, explicitly set to a
    default-equivalent value, and `expandable_segments:True` -- must hash
    three different ways: unset and "explicitly default" are related but
    distinguishable (a future PyTorch could change what "default" means),
    and both differ from the workload this task exists to isolate.
    """

    if raw is None:
        return _UNSET_ALLOC_CONF_MARKER

    pairs: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, _, value = part.partition(":")
            pairs.append(f"{key.strip().lower()}:{value.strip().lower()}")
        else:
            pairs.append(part.lower())
    pairs.sort()
    return "set:" + ",".join(pairs)


def sidecar_alloc_conf_hash(environment: Optional[Mapping[str, str]] = None) -> str:
    """Sub-hash of the sidecar child's normalised allocator configuration.

    Reads from the SIDECAR environment -- the composed mapping the probe and
    training children actually run under (`os.environ` overridden by
    `sam3_env_environ()`, exactly as `train.py`'s `_child_environment`
    builds it) -- not the parent process's own `os.environ` in isolation.
    `sam3_env_environ()` now hardcodes `PYTORCH_CUDA_ALLOC_CONF=
    expandable_segments:True`, so this always resolves to that value
    regardless of the parent's own `os.environ`; composing the same way
    `_child_environment` does means any future change there is picked up
    automatically rather than silently diverging from what the child sees.

    `environment` is accepted for tests; production callers pass nothing and
    get the live composed environment.
    """

    if environment is None:
        environment = {**os.environ, **sam3_env_environ()}
    raw = environment.get(_ALLOC_CONF_ENV_VAR)
    normalized = _normalize_pytorch_alloc_conf(raw)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


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


def sam3_dataset_density_profile(
    spec: TrainingRunSpec,
) -> Sam3DatasetDensityProfile:
    """Read the built COCO metadata into the density facts the key needs.

    Metadata only -- it reuses `preflight`'s existing bounded COCO reader and
    negative-prompt resolution rather than adding a second parser, so the
    density the fingerprint records is the density admission already saw.
    """

    from .preflight import _resolved_negative_prompts, dataset_profile

    params = spec.sam3_params
    if params is None:
        raise ValueError("spec.sam3_params is required to profile a SAM3 dataset")
    profile = dataset_profile(spec.derived_dataset_dir)
    prompts = tuple(
        prompt
        for prompt in _resolved_negative_prompts(spec.derived_dataset_dir, params)
        if isinstance(prompt, str)
    )
    return Sam3DatasetDensityProfile(
        max_instances_per_tile=int(profile.max_active_instances_per_tile),
        p95_instances_per_tile=int(profile.p95_active_instances_per_tile),
        num_negatives=int(params.num_negatives),
        negative_prompt_pool=prompts,
    )


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

    `backend` also carries the sidecar's normalised `PYTORCH_CUDA_ALLOC_CONF`
    (see `sidecar_alloc_conf_hash`): the caching allocator's growth policy is
    a memory-driving property of the run, not just of the software stack,
    and a measurement taken under one config understates -- never
    overstates -- what another config needs.
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
    alloc_conf_hash = sidecar_alloc_conf_hash()
    backend = f"{env_name}|{package_hash}|{alloc_conf_hash}"
    device_identity = device_identity_for(cuda_device.name, cuda_device.total_bytes)

    imgsz = int(spec.hyperparams.imgsz)
    # `slice_width`/`slice_height` change the tile size -- and therefore the
    # memory -- but ONLY in custom geometry mode (`tile_size_for_mode`
    # ignores them otherwise). Hashing them unconditionally would make two
    # identical auto-mode runs re-probe over a stale, unused number.
    slice_payload = (
        f"{int(params.slice_width)}x{int(params.slice_height)}"
        if params.geometry_mode == "custom"
        else "mode_derived"
    )
    # The probe PROTOCOL is part of the workload: the branch's own evidence is
    # that step count moves the measured peak by ~35% (2 steps 7.34 GiB vs 60
    # steps 9.93 GiB). Without this, raising `PROBE_STEPS` would silently
    # reuse every stored record -- each a stale, lower number -- forever.
    task_payload = (
        f"imgsz={imgsz}|overlap={params.tile_overlap}|"
        f"object_tile_fraction={params.object_tile_fraction}|"
        f"slice={slice_payload}|"
        f"probe_steps={PROBE_STEPS}|"
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


# ---------------------------------------------------------------------------
# Probe half: the candidate ladder, record validation, and batch selection.
#
# Still parent-side and still torch-free. The actual measurement happens
# inside one contained sidecar child per candidate (`cli.py --probe
# --probe-batch N`); this module only drives the ladder and interprets what
# the children reported.
# ---------------------------------------------------------------------------

# The ladder. Powers of two because each rung must be worth a full model
# reload: a linear ladder would pay that cost for a batch size the curve fit
# can already extrapolate. 8 is the ceiling because SAM3 LoRA's per-item cost
# is measured in GiB, so a card that fits 16 is not a card this gate is for.
MAX_AUTO_BATCH = 8
PROBE_CANDIDATES: tuple[int, ...] = (1, 2, 4, 8)

# Steps THROUGH `optimizer.step()` per candidate. At least two are needed on
# principle: the first materialises Adam's lazy exp_avg/exp_avg_sq state, so a
# one-step probe understates the peak by the whole optimizer state.
#
# 30 is chosen from measurement, not taste. Truncation error of a K-step probe
# against a completed 2430-step run of the same configuration, under the
# `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` that `sam3_env_environ()`
# now hardcodes for every probe and training child:
#
#       2 steps   -6.42%
#      30 steps   -2.82%
#      60 steps   -2.82%
#     120 steps   -2.53%
#
# 30 reaches the plateau; 60 buys nothing over it while doubling probe cost
# (one full model load per candidate, four candidates). Under the DEFAULT
# allocator the same comparison is -41%, which is why the flag is enforced
# rather than left to the caller's shell -- and why the allocator config is
# part of the fingerprint (see `sidecar_alloc_conf_hash`).
#
# `PROBE_STEPS` is hashed into the workload fingerprint's `task` payload, so
# changing it invalidates every stored record. That is INTENDED: records taken
# with a different probe protocol are measurements of a different thing.
PROBE_STEPS = 30

# Set to "1" to re-measure even when the store already has records for this
# exact fingerprint (e.g. after a driver upgrade the key does not capture).
FORCE_PROBE_ENV_VAR = "HYDRA_SAM3_FORCE_PROBE"

PROFILE_SCOPE = "sam3_lora"
PROBE_RECORDS_DIRNAME = "probe_records"


# Why the candidate ladder stopped. Only `host_limit` is TRANSIENT: an
# unrelated job triggering a cgroup kill says nothing about what this
# workload needs, so a ladder that ended that way is INCOMPLETE and must not
# authorise skipping the untried rungs on a later run. `oom` and `refused`
# are properties of this workload on this hardware and are authoritative;
# `complete` means every candidate was tried.
LADDER_COMPLETE = "complete"
LADDER_OOM = "oom"
LADDER_REFUSED = "refused"
LADDER_HOST_LIMIT = "host_limit"
TRANSIENT_LADDER_TERMINATIONS = frozenset({LADDER_HOST_LIMIT})


class ProbeLadder(tuple):
    """The surviving records, plus WHY the ladder stopped.

    A `tuple` subclass rather than a wrapper object so every caller can keep
    treating the result as the sequence of records it is; `terminated_by`
    rides along for the one caller that needs to know whether the ladder was
    cut short by something transient.
    """

    terminated_by: str

    def __new__(
        cls, records: Sequence[MemoryMeasurement], terminated_by: str
    ) -> "ProbeLadder":
        ladder = super().__new__(cls, records)
        ladder.terminated_by = terminated_by
        return ladder


class ProbeHostLimitError(RuntimeError):
    """The child hit a HOST memory limit -- transient, not a device verdict."""


class ProbeFailedError(RuntimeError):
    """No candidate survived -- not even batch 1. Fail closed, cache nothing."""


class ProbeOutOfMemoryError(RuntimeError):
    """This candidate did not fit. A measurement, not a failure; stops the ladder."""


class ProbeCandidateRefused(RuntimeError):
    """Admission refused this candidate. Host demand is monotone: stop the ladder."""


class ProbeCanceled(RuntimeError):
    """The user cancelled mid-ladder. Merge nothing, write nothing, launch nothing."""


def _is_out_of_memory(exc: BaseException) -> bool:
    """True for our own OOM signal and for torch's, without importing torch.

    `torch.cuda.OutOfMemoryError` subclasses `RuntimeError`, so catching
    `RuntimeError` here would swallow every unrelated failure and record a
    broken configuration as "does not fit". Match the type name instead.
    """

    return isinstance(exc, ProbeOutOfMemoryError) or type(exc).__name__ in (
        "OutOfMemoryError",
        "CudaOutOfMemoryError",
    )


def probe_candidates(spec: TrainingRunSpec) -> tuple[int, ...]:
    """The batch sizes to try, smallest first."""

    del spec  # Reserved: a future dataset-size cap would consult the spec.
    return tuple(
        candidate for candidate in PROBE_CANDIDATES if candidate <= MAX_AUTO_BATCH
    )


def _unfingerprinted_identity(spec: TrainingRunSpec) -> ProfileIdentity:
    """A deliberately non-matching identity for records taken without a key.

    `run_probe` accepts `identity=None` so it can be exercised without conda,
    a GPU, or a dataset. Records built this way carry an identity that no
    real fingerprint can equal, so `validate_probe_records` discards them and
    they can never reach the store by accident.
    """

    params = spec.sam3_params
    return ProfileIdentity(
        operation=OPERATION,
        model_identity="unfingerprinted",
        backend="unfingerprinted",
        device_identity="unfingerprinted",
        precision=getattr(params, "mixed_precision", "bf16") if params else "bf16",
        task="unfingerprinted",
    )


class ProbeAllocatorMismatch(ProbeFailedError):
    """A probe child ran under an allocator config the key does not describe."""


def _reject_allocator_mismatch(
    identity: ProfileIdentity,
    batch_size: int,
    peaks: Mapping[str, Any],
) -> None:
    """Refuse a record whose child allocator disagrees with its cache key.

    The fingerprint is built by the PARENT from `sam3_env_environ()`, so it
    describes what the parent BELIEVES the child ran under. The child now
    reports what it actually saw. In the product path (`train.py` composes
    `_child_environment` from the same function) these always agree; for any
    child launched another way they can diverge, and the divergence is worth
    42% of reserved VRAM (9.82 GiB without `expandable_segments:True` vs
    6.89 GiB with it, same box, same 30 steps).

    The error direction of a mismatch happens to be safe today -- a
    default-allocator record OVER-states -- but now that a measurement
    DECIDES rather than only raising, that safety must not rest on luck.
    Fail closed: a record we cannot honestly key is not stored at all.

    A child that reports no hash at all is accepted, so an older child or a
    test double is not retro-invalidated; only a hash that CONTRADICTS the
    key is refused.
    """

    reported = peaks.get("alloc_conf_hash")
    if not reported:
        return
    if not identity.backend.endswith(f"|{reported}"):
        raise ProbeAllocatorMismatch(
            f"The SAM3 probe child at batch {batch_size} ran under allocator "
            f"config {reported}, which is not the one its cache key describes "
            f"({identity.backend}). Nothing was cached: a measurement filed "
            "under the wrong allocator key would be reused by runs it does "
            "not describe."
        )


def _measurement(
    spec: TrainingRunSpec,
    identity: ProfileIdentity,
    batch_size: int,
    peaks: Optional[Mapping[str, int]],
) -> MemoryMeasurement:
    peaks = peaks or {}
    imgsz = max(1, int(spec.hyperparams.imgsz))
    reserved = max(0, int(peaks.get("accelerator_reserved_peak_bytes", 0)))
    allocated = max(0, int(peaks.get("accelerator_allocated_peak_bytes", 0)))
    return MemoryMeasurement(
        identity=identity,
        settings=PressureSettings(
            input_width=imgsz, input_height=imgsz, batch_size=batch_size
        ),
        accelerator_kind=AcceleratorKind.CUDA,
        host_peak_bytes=max(0, int(peaks.get("host_peak_bytes", 0))),
        # Clamped rather than trusted: `MemoryMeasurement` refuses a record
        # whose allocated peak exceeds its reserved peak, and a child that
        # reported such a pair is confused, not authoritative.
        accelerator_allocated_peak_bytes=min(allocated, reserved),
        accelerator_reserved_peak_bytes=reserved,
        observed_at_unix_ns=int(peaks.get("observed_at_unix_ns", 0) or time.time_ns()),
    )


def run_probe(
    spec: TrainingRunSpec,
    run_dir: str | Path,
    *,
    step_fn: Callable[[int], Optional[Mapping[str, int]]],
    identity: Optional[ProfileIdentity] = None,
    candidates: Optional[Sequence[int]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> ProbeLadder:
    """Walk the candidate ladder, returning one record per surviving candidate.

    `step_fn(batch_size)` performs ONE candidate's measurement and returns its
    peaks (or `None` when the caller has no peaks to report, e.g. a test
    fake). In production it launches one fresh contained sidecar per
    candidate: candidates 2-8 must never run under batch-1 containment, and a
    model reload costs seconds while wrong containment costs a wedged box.

    The ladder stops -- it does not skip -- on the first candidate that
    either runs out of memory or is refused admission. Both are monotone in
    batch size, so nothing larger could have survived either. The reason is
    reported on `ProbeLadder.terminated_by`, because a HOST-limit stop is a
    transient event about the box rather than a verdict about this workload,
    and must not silently become a permanent batch ceiling.

    Raises `ProbeFailedError` when nothing survives, including at batch 1:
    that configuration cannot run on this hardware and must not be cached.
    An admission refusal at batch 1 is reported through the same error, but
    carrying the refusal's own reason rather than a no-fit message.
    """

    del run_dir  # The child owns record I/O; the ladder owns control flow.
    should_cancel = should_cancel or (lambda: False)
    resolved_identity = identity or _unfingerprinted_identity(spec)
    records: list[MemoryMeasurement] = []
    terminated_by = LADDER_COMPLETE
    for candidate in candidates or probe_candidates(spec):
        if should_cancel():
            raise ProbeCanceled(f"cancelled before probing batch {candidate}")
        try:
            peaks = step_fn(candidate)
        except (ProbeCandidateRefused, ProbeHostLimitError) as exc:
            if not records:
                # A refusal at the FIRST candidate is an admission or
                # environment problem -- a missing credential, unacknowledged
                # labels, a busy lease -- not "this workload does not fit the
                # GPU". Reporting the generic no-fit message would send the
                # user hunting for VRAM they already have. Carry the real
                # reason verbatim.
                raise ProbeFailedError(str(exc)) from exc
            # Survivors are FACTS about this hardware and workload and are
            # kept regardless of why the ladder stopped. Only zero survivors
            # may cache nothing.
            terminated_by = (
                LADDER_HOST_LIMIT
                if isinstance(exc, ProbeHostLimitError)
                else LADDER_REFUSED
            )
            break
        except BaseException as exc:  # noqa: BLE001 - re-raised unless it is an OOM
            if not _is_out_of_memory(exc):
                raise
            terminated_by = LADDER_OOM
            break
        _reject_allocator_mismatch(resolved_identity, candidate, peaks or {})
        records.append(_measurement(spec, resolved_identity, candidate, peaks))
    if not records:
        raise ProbeFailedError(
            "No SAM3 batch size survived the memory probe, including batch 1. "
            "This workload does not fit on this device; nothing was cached."
        )
    return ProbeLadder(records, terminated_by)


def validate_probe_records(
    records: Sequence[MemoryMeasurement],
    identity: ProfileIdentity,
) -> tuple[MemoryMeasurement, ...]:
    """Keep only records that are safe to store and select against.

    A record is discarded (never stored) when it does not belong to
    `identity`, when its reserved peak is not positive (a measurement that
    measured nothing), or when it breaks monotonicity -- a larger batch that
    reports a SMALLER peak than a smaller batch is not a cheaper batch, it is
    a broken measurement, and storing it would let `select_batch` admit a
    batch size that never actually fit.
    """

    ordered = sorted(
        records_for(records, identity), key=lambda record: record.settings.batch_size
    )
    kept: list[MemoryMeasurement] = []
    highest_peak = 0
    for record in ordered:
        reserved = record.accelerator_reserved_peak_bytes
        if reserved <= 0:
            continue
        if record.accelerator_allocated_peak_bytes > reserved:
            continue
        if reserved < highest_peak:
            continue
        highest_peak = reserved
        kept.append(record)
    return tuple(kept)


def resolve_batch(
    spec: TrainingRunSpec,
    records: Sequence[MemoryMeasurement],
    *,
    usable_bytes: int,
    maximum: int = MAX_AUTO_BATCH,
) -> tuple[int, str]:
    """Return `(batch, provenance)` for one run.

    A positive `sam3_params.batch` is the user's explicit choice and is
    honoured without consulting any measurement. Otherwise the measured
    records decide.

    `usable_bytes` is **raw free device bytes**. The safety fraction is
    applied exactly once, inside `select_batch`; pre-multiplying here would
    discount the budget twice.

    A `0` result is a REFUSAL, never a floor of 1: the records say batch 1
    does not fit in what is free right now, and launching a run that is
    measured not to fit wastes minutes of setup to reach a certain OOM.
    """

    params = spec.sam3_params
    requested = int(getattr(params, "batch", 1)) if params is not None else 1
    if requested > 0:
        return requested, "explicit"
    return (
        select_batch(
            records,
            usable_bytes=int(usable_bytes),
            maximum=max(1, int(maximum)),
            safety_fraction=MEASURED_SAFETY_FRACTION,
        ),
        "measured",
    )
