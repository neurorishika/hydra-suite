"""Identity evidence sidecar cache.

Identity Phase 0: accumulates ``IdentityEvidence`` objects during a tracking
run and persists them as a compressed NumPy archive next to the detection and
CNN caches.  The evidence sidecar is written equivalently from both the live
streaming path and the replay fallback path so the offline decoder sees the
same artifact regardless of which execution path was used.

Naming convention::

    <base>_identity_evidence_<signature>.npz

On-disk layout
--------------
Metadata keys (stored once):

    evidence_schema_version  int64 scalar  currently 1
    catalog_labels           U255 (C,)     label for each catalog index

Per-frame keys (``f{N}`` prefix, e.g. ``f0``, ``f1500``):

    f{N}_det_ids         int64 (E,)        detection slot IDs
    f{N}_source_types    U32   (E,)        EvidenceSource enum value strings
    f{N}_sources         U32   (E,)        human-readable source_name strings
    f{N}_log_probs       float64 (E, C)    log-posterior rows
    f{N}_catalog_size    int64 scalar      catalog size C for validation
    f{N}_cal_sig         U255  (E,)        calibration signatures
    f{N}_rt_sig          U64   (E,)        runtime signatures
    f{N}_obs_mask        bool  (E, C)      observed masks (all-True when absent)

where E = number of evidence items in the frame and C = catalog_size.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from hydra_suite.core.identity.evidence import EvidenceSource, IdentityEvidence

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


class IdentityEvidenceCache:
    """Write-once-per-run evidence sidecar for one tracking run.

    Usage (write mode)::

        cache = IdentityEvidenceCache(
            path, catalog_labels=catalog.labels, mode="w"
        )
        cache.save_frame(frame_idx, [ev_det0, ev_det1, ...])
        ...
        cache.flush()

    Usage (read mode)::

        cache = IdentityEvidenceCache(path, mode="r")
        evidences = cache.load_frame(frame_idx)  # list[IdentityEvidence]

    Parameters
    ----------
    cache_path:
        Path to the ``.npz`` sidecar file.
    catalog_labels:
        Required in write mode; the full label tuple from
        ``IdentityCatalog.labels``.
    mode:
        ``'w'`` for write, ``'r'`` for read.
    """

    def __init__(
        self,
        cache_path: str | Path,
        catalog_labels: Optional[tuple[str, ...]] = None,
        mode: str = "w",
    ) -> None:
        self._path = Path(cache_path)
        self._mode = mode
        self._catalog_labels: Optional[tuple[str, ...]] = catalog_labels
        self._data: dict = {}
        self._loaded: bool = False

        if mode not in ("r", "w"):
            raise ValueError(f"mode must be 'r' or 'w', got {mode!r}")
        if mode == "w" and catalog_labels is None:
            raise ValueError("catalog_labels is required in write mode")
        if mode == "r":
            self._load()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def save_frame(
        self, frame_idx: int, evidences: list[IdentityEvidence]
    ) -> None:
        """Accumulate evidence for *frame_idx* in memory.

        Multiple calls with the same *frame_idx* will overwrite the previous
        entry for that frame.
        """
        if self._mode != "w":
            raise RuntimeError("IdentityEvidenceCache is not open for writing")
        if not evidences:
            return

        catalog_size: int = evidences[0].catalog_size
        n = len(evidences)

        det_ids = np.array([e.detection_id for e in evidences], dtype=np.int64)
        source_types = np.array([str(e.source) for e in evidences], dtype="U32")
        sources = np.array([e.source_name for e in evidences], dtype="U32")
        log_probs = np.stack(
            [e.log_probs for e in evidences], axis=0
        ).astype(np.float64)
        cal_sigs = np.array(
            [e.calibration_signature for e in evidences], dtype="U255"
        )
        rt_sigs = np.array(
            [e.runtime_signature for e in evidences], dtype="U64"
        )

        obs_rows = []
        for e in evidences:
            if e.observed_mask is not None:
                obs_rows.append(e.observed_mask.astype(bool))
            else:
                obs_rows.append(np.ones(catalog_size, dtype=bool))
        obs_mask = np.stack(obs_rows, axis=0)

        key = f"f{frame_idx}"
        self._data[f"{key}_det_ids"] = det_ids
        self._data[f"{key}_source_types"] = source_types
        self._data[f"{key}_sources"] = sources
        self._data[f"{key}_log_probs"] = log_probs
        self._data[f"{key}_catalog_size"] = np.int64(catalog_size)
        self._data[f"{key}_cal_sig"] = cal_sigs
        self._data[f"{key}_rt_sig"] = rt_sigs
        self._data[f"{key}_obs_mask"] = obs_mask

    def flush(self) -> None:
        """Write all accumulated evidence to disk as a compressed .npz file.

        The parent directory is created if it does not exist.

        Raises
        ------
        RuntimeError
            If the cache is open in read mode.
        """
        if self._mode != "w":
            raise RuntimeError("IdentityEvidenceCache is not open for writing")

        self._path.parent.mkdir(parents=True, exist_ok=True)
        meta: dict = {
            "evidence_schema_version": np.int64(_SCHEMA_VERSION),
        }
        if self._catalog_labels is not None:
            meta["catalog_labels"] = np.array(self._catalog_labels, dtype="U255")

        np.savez_compressed(str(self._path), **meta, **self._data)
        log.debug("Wrote identity evidence cache: %s (%d keys)", self._path, len(self._data))

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        if not self._path.exists():
            raise FileNotFoundError(
                f"Identity evidence cache not found: {self._path}"
            )
        raw = np.load(str(self._path), allow_pickle=False)
        try:
            self._data = {k: raw[k] for k in raw.files}
        finally:
            raw.close()
        if "catalog_labels" in self._data:
            self._catalog_labels = tuple(
                str(s) for s in self._data["catalog_labels"]
            )
        self._loaded = True

    def load_frame(self, frame_idx: int) -> list[IdentityEvidence]:
        """Return all evidence items stored for *frame_idx*.

        Returns an empty list if the frame is not present in the cache.
        """
        key = f"f{frame_idx}"
        det_key = f"{key}_det_ids"
        if det_key not in self._data:
            return []

        det_ids = self._data[f"{key}_det_ids"]
        source_types = self._data[f"{key}_source_types"]
        sources = self._data[f"{key}_sources"]
        log_probs = self._data[f"{key}_log_probs"]
        cat_size = int(self._data[f"{key}_catalog_size"])
        cal_sigs = self._data[f"{key}_cal_sig"]
        rt_sigs = self._data[f"{key}_rt_sig"]
        obs_mask_arr = self._data.get(f"{key}_obs_mask")

        results: list[IdentityEvidence] = []
        for i in range(len(det_ids)):
            src_val = str(source_types[i])
            try:
                src = EvidenceSource(src_val)
            except ValueError:
                src = EvidenceSource.MISSING

            om: Optional[np.ndarray] = None
            if obs_mask_arr is not None:
                om = obs_mask_arr[i].astype(bool)

            results.append(
                IdentityEvidence(
                    frame_idx=frame_idx,
                    detection_id=int(det_ids[i]),
                    source=src,
                    source_name=str(sources[i]),
                    log_probs=log_probs[i].astype(np.float64),
                    catalog_size=cat_size,
                    calibration_signature=str(cal_sigs[i]),
                    runtime_signature=str(rt_sigs[i]),
                    observed_mask=om,
                )
            )

        return results

    def get_cached_frames(self) -> list[int]:
        """Return a sorted list of all frame indices present in the cache."""
        frame_indices: set[int] = set()
        for k in self._data:
            # Pattern: f{N}_det_ids
            if k.endswith("_det_ids") and k.startswith("f"):
                body = k[1 : k.index("_")]
                try:
                    frame_indices.add(int(body))
                except ValueError:
                    pass
        return sorted(frame_indices)

    @property
    def catalog_labels(self) -> Optional[tuple[str, ...]]:
        """Catalog label tuple, if available (always set in write mode or after
        load from a cache that was written with catalog metadata)."""
        return self._catalog_labels

    def __len__(self) -> int:
        return len(self.get_cached_frames())

    def close(self) -> None:
        """Match cache-style interfaces used elsewhere in the tracking stack."""
        return None
