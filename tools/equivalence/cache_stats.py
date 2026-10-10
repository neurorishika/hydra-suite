"""Per-frame stored-detection and per-animal-row statistics for one cache dir.

Usage: python tools/equivalence/cache_stats.py <video_dir>/.inference_cache_<stem>

Members are resolved through ``cache_set.json``. Each member ``*.npz`` is a
chunked-store manifest whose rows live in immutable chunks, so rows are read
through ``ChunkedArrayStore`` (key check disabled: this is a read-only
inspector). Per-frame counts include processed frames with zero rows, so the
mean and percentiles are over every frame the store covers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from hydra_suite.core.inference.cache.base import CacheKey
from hydra_suite.core.inference.cache.chunked import ChunkedArrayStore
from hydra_suite.core.inference.cache.set_manifest import load_cache_set
from hydra_suite.core.inference.limits import MAX_DETECTIONS_PER_FRAME

_ANY_KEY = CacheKey(schema_version=0, model_id="", config_hash="")


def _kind(member: str) -> str:
    stem = member[: -len(".npz")]
    return "cnn" if stem.startswith("cnn_") else stem


def _per_frame_counts(path: Path, kind: str) -> np.ndarray | None:
    store = ChunkedArrayStore(path, _ANY_KEY, kind, require_key=False)
    if not store.is_valid():
        return None
    if store.is_legacy:
        arrays = store.load_legacy()
        if arrays is None or "frame_indices" not in arrays:
            return None
        _, counts = np.unique(arrays["frame_indices"], return_counts=True)
        return counts
    written = np.asarray(sorted(store.written_frames()), dtype=np.int64)
    counts = np.zeros(len(written), dtype=np.int64)
    for arrays in store.iter_chunk_arrays():
        rows = np.asarray(arrays["frame_indices"], dtype=np.int64)
        counts += np.bincount(
            np.searchsorted(written, rows), minlength=len(written)
        ).astype(np.int64)
    return counts


def main(cache_dir: str) -> int:
    d = Path(cache_dir)
    manifest = load_cache_set(d)
    if manifest is None:
        print(f"{d}: no valid cache_set.json", file=sys.stderr)
        return 1
    print(f"{d}  generation={manifest.generation_id}")
    for member in sorted(manifest.members):
        counts = _per_frame_counts(d / manifest.members[member], _kind(member))
        if counts is None:
            print(f"{member}: unreadable or invalid store")
            continue
        if len(counts) == 0:
            print(f"{member}: frames=0")
            continue
        label = "detections" if member == "detection.npz" else "rows"
        print(
            f"{member}: frames={len(counts)} {label}/frame mean={counts.mean():.1f} "
            f"p50={np.median(counts):.0f} p99={np.percentile(counts, 99):.0f} "
            f"max={counts.max()} at_limit={(counts >= MAX_DETECTIONS_PER_FRAME).sum()}"
        )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
