"""Safe, bounded throughput tuning for sliced-detector prediction batches.

The tuner deliberately changes only *how many* already-planned tiles are sent
to ``predict`` at once.  It never changes tile order, geometry, model kwargs,
or result processing, so a selected batch has the same detector outputs as an
explicitly selected batch.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable

from .slicing import MAX_TILE_CHUNK


@dataclass(frozen=True)
class TileBatchAutotuneKey:
    """Everything that can materially affect tile-predict throughput/capacity."""

    artifact: str
    backend: str
    device: str
    imgsz: int
    tile_wh: tuple[int, int]
    full_frame: bool
    task: str
    per_job_bytes: int
    byte_budget: int
    admitted_max: int


_CACHE: dict[TileBatchAutotuneKey, int] = {}
_CACHE_LOCK = Lock()
_ARTIFACT_HASHES: dict[tuple[str, int, int], str] = {}


def clear_tile_batch_autotune_cache() -> None:
    """Clear process-local results (principally useful to deterministic tests)."""
    with _CACHE_LOCK:
        _CACHE.clear()
        _ARTIFACT_HASHES.clear()


def stable_artifact_identity(path: str | None, model: object) -> str:
    """Return a stable artifact identity without relying on object identity.

    A content digest avoids collisions between changed checkpoints at the same
    pathname.  Missing/unreadable paths safely fall back to a conservative
    type/name identity, which is intentionally not persisted across processes.
    """
    candidate = (
        path or getattr(model, "model_path", None) or getattr(model, "path", None)
    )
    if candidate:
        try:
            artifact = Path(candidate).expanduser().resolve()
            stat = artifact.stat()
            cache_key = (str(artifact), stat.st_size, stat.st_mtime_ns)
            with _CACHE_LOCK:
                digest = _ARTIFACT_HASHES.get(cache_key)
            if digest is None:
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                with _CACHE_LOCK:
                    _ARTIFACT_HASHES[cache_key] = digest
            return f"sha256:{digest}"
        except (OSError, ValueError):
            pass
    # ``name`` is a common executor-provided artifact identifier.  Do not use
    # repr(model): it commonly includes a process-specific address.
    name = getattr(model, "name", None) or getattr(model, "artifact_id", None)
    return f"model:{type(model).__module__}.{type(model).__qualname__}:{name or ''}"


def stable_device_identity(device: str) -> str:
    """Include the physical CUDA GPU when it can be queried safely."""
    device = str(device)
    if device.startswith("cuda"):
        try:
            import torch

            return f"{device}:{torch.cuda.get_device_name(device)}"
        except Exception:
            pass
    return device


def tile_batch_candidates(admitted_max: int) -> tuple[int, ...]:
    """Small geometric candidate set, always including the admitted maximum."""
    maximum = max(1, min(int(admitted_max), MAX_TILE_CHUNK))
    candidates = {1, maximum}
    size = 2
    while size < maximum:
        candidates.add(size)
        size *= 2
    return tuple(sorted(candidates))


def select_tile_batch_size(
    key: TileBatchAutotuneKey,
    *,
    benchmark: Callable[[int], float],
    candidates: tuple[int, ...] | None = None,
) -> int:
    """Benchmark candidates after a warmup and cache the fastest safe batch.

    A non-finite/non-positive duration, or any model/timer failure, simply
    excludes that candidate.  If nothing can be measured, batch one is the
    conservative fallback.  The caller executes its normal prediction path
    after this probe; probe results are never consumed as detector output.
    """
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    choices = tuple(
        c
        for c in (candidates or tile_batch_candidates(key.admitted_max))
        if 1 <= c <= key.admitted_max
    )
    if not choices:
        return 1
    try:
        benchmark(choices[0])  # one warmup, excluded from measurement
    except Exception:
        return 1
    best_size, best_rate = 1, -1.0
    for size in choices:
        try:
            elapsed = float(benchmark(size))
        except Exception:
            continue
        if not (elapsed > 0.0 and elapsed < float("inf")):
            continue
        rate = size / elapsed
        if rate > best_rate:
            best_size, best_rate = size, rate
    if best_rate < 0:
        return 1
    with _CACHE_LOCK:
        # Another worker may have tuned the same model while this one warmed up.
        return _CACHE.setdefault(key, best_size)


def synchronized_timer(device: str) -> Callable[[], float]:
    """Use CUDA synchronization when available; otherwise monotonic wall time."""
    if str(device).startswith("cuda"):
        try:
            import torch

            def now() -> float:
                torch.cuda.synchronize(device)
                return time.perf_counter()

            return now
        except Exception:
            pass
    return time.perf_counter
