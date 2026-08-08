"""Identity evidence sidecar cache.

Identity Phase 0: accumulates ``IdentityEvidence`` objects during a tracking
run and persists them as a compressed NumPy archive next to the detection and
CNN caches. The evidence sidecar is written equivalently from both the live
streaming path and the replay fallback path so identity post-processing sees
the same artifact regardless of which execution path was used.

Naming convention::

    <base>_identity_evidence_<signature>.npz

On-disk layout
--------------
Metadata keys (stored once):

    evidence_schema_version         int64 scalar  currently 2
    catalog_labels                  U255 (C,)     global catalog label per index
    src_catalog_labels__{source}    U255 (Cs,)    one entry per distinct
                                                   source_name ever saved --
                                                   that source's OWN catalog
                                                   basis (may differ in size
                                                   from the global catalog;
                                                   see ``catalog_labels_by_source``)

Per-frame keys are grouped by ``source_name`` (schema v2+) so sources with
different catalog sizes (e.g. two CNN phases with different label counts, or
a CNN phase vs. AprilTag) can coexist within one frame without forcing a
uniform-width stack:

    f{N}_source_order          U255 (S,)          source_name append order

and, for each ``source_name`` appearing in frame N (``Es`` = that source's
evidence count that frame, ``Cs`` = that source's OWN catalog size):

    f{N}__{source}_det_ids       int64 (Es,)
    f{N}__{source}_source_types  U32   (Es,)       EvidenceSource enum strings
    f{N}__{source}_log_probs     float64 (Es, Cs)  log-posterior rows, in
                                                    THIS source's own basis
    f{N}__{source}_catalog_size  int64 scalar      Cs, for validation
    f{N}__{source}_cal_sig       U255  (Es,)        calibration signatures
    f{N}__{source}_rt_sig        U64   (Es,)        runtime signatures
    f{N}__{source}_obs_mask      bool  (Es, Cs)      observed masks
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from hydra_suite.core.individual.identity.evidence import (
    EvidenceSource,
    IdentityEvidence,
)

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 2
_SRC_CATALOG_PREFIX = "src_catalog_labels__"


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
        Required in write mode; the full (global) label tuple from
        ``IdentityCatalog.labels``. Also used as the fallback basis for any
        source with no entry in ``catalog_labels_by_source``.
    mode:
        ``'w'`` for write, ``'r'`` for read.
    catalog_labels_by_source:
        Optional per-``source_name`` catalog basis, e.g.
        ``IdentityEvidenceStage.catalog_labels_by_source``. A CNN phase's
        entry is that phase's OWN phase-local catalog, which may be smaller
        than (and differently ordered from) the global ``catalog_labels`` --
        required so the tracking worker can remap phase-basis evidence to
        the global catalog exactly as the old per-source
        ``IdentityEvidenceEmitter`` sidecars did. ``None``/omitted means
        every source is assumed to already be on the global basis (single
        global catalog for the whole cache -- pre-final-fix-wave schema
        v1 behavior).
    """

    def __init__(
        self,
        cache_path: str | Path,
        catalog_labels: Optional[tuple[str, ...]] = None,
        mode: str = "w",
        catalog_labels_by_source: Optional[dict[str, tuple[str, ...]]] = None,
    ) -> None:
        self._path = Path(cache_path)
        self._mode = mode
        self._catalog_labels: Optional[tuple[str, ...]] = catalog_labels
        self._catalog_labels_by_source: dict[str, tuple[str, ...]] = dict(
            catalog_labels_by_source or {}
        )
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

    def save_frame(self, frame_idx: int, evidences: list[IdentityEvidence]) -> None:
        """Accumulate evidence for *frame_idx* in memory.

        Multiple calls with the same *frame_idx* will overwrite the previous
        entry for that frame.

        Evidence is grouped by ``source_name`` and stored per-group (schema
        v2+): sources may have different catalog sizes (e.g. a phase-local
        CNN catalog vs. the global AprilTag catalog), which would make a
        single uniform-width stack across all of a frame's evidence
        impossible.
        """
        if self._mode != "w":
            raise RuntimeError("IdentityEvidenceCache is not open for writing")
        if not evidences:
            return

        groups: dict[str, list[IdentityEvidence]] = {}
        order: list[str] = []
        for e in evidences:
            if e.source_name not in groups:
                groups[e.source_name] = []
                order.append(e.source_name)
            groups[e.source_name].append(e)

        key = f"f{frame_idx}"
        self._data[f"{key}_source_order"] = np.array(order, dtype="U255")

        for source_name, evs in groups.items():
            catalog_size: int = evs[0].catalog_size
            gkey = f"{key}__{source_name}"

            det_ids = np.array([e.detection_id for e in evs], dtype=np.int64)
            source_types = np.array([str(e.source) for e in evs], dtype="U32")
            log_probs = np.stack([e.log_probs for e in evs], axis=0).astype(np.float64)
            cal_sigs = np.array([e.calibration_signature for e in evs], dtype="U255")
            rt_sigs = np.array([e.runtime_signature for e in evs], dtype="U64")

            obs_rows = []
            for e in evs:
                if e.observed_mask is not None:
                    obs_rows.append(e.observed_mask.astype(bool))
                else:
                    obs_rows.append(np.ones(catalog_size, dtype=bool))
            obs_mask = np.stack(obs_rows, axis=0)

            self._data[f"{gkey}_det_ids"] = det_ids
            self._data[f"{gkey}_source_types"] = source_types
            self._data[f"{gkey}_log_probs"] = log_probs
            self._data[f"{gkey}_catalog_size"] = np.int64(catalog_size)
            self._data[f"{gkey}_cal_sig"] = cal_sigs
            self._data[f"{gkey}_rt_sig"] = rt_sigs
            self._data[f"{gkey}_obs_mask"] = obs_mask

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
        for source_name, labels in self._catalog_labels_by_source.items():
            meta[f"{_SRC_CATALOG_PREFIX}{source_name}"] = np.array(labels, dtype="U255")

        np.savez_compressed(str(self._path), **meta, **self._data)
        log.debug(
            "Wrote identity evidence cache: %s (%d keys)", self._path, len(self._data)
        )

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        if not self._path.exists():
            raise FileNotFoundError(f"Identity evidence cache not found: {self._path}")
        raw = np.load(str(self._path), allow_pickle=False)
        try:
            self._data = {k: raw[k] for k in raw.files}
        finally:
            raw.close()
        if "catalog_labels" in self._data:
            self._catalog_labels = tuple(str(s) for s in self._data["catalog_labels"])
        self._catalog_labels_by_source = {}
        for k in list(self._data.keys()):
            if k.startswith(_SRC_CATALOG_PREFIX):
                source_name = k[len(_SRC_CATALOG_PREFIX) :]
                self._catalog_labels_by_source[source_name] = tuple(
                    str(s) for s in self._data[k]
                )
        self._loaded = True

    def load_frame(self, frame_idx: int) -> list[IdentityEvidence]:
        """Return all evidence items stored for *frame_idx*.

        Returns an empty list if the frame is not present in the cache.
        """
        key = f"f{frame_idx}"
        order_key = f"{key}_source_order"
        if order_key not in self._data:
            return []

        results: list[IdentityEvidence] = []
        for source_name in (str(s) for s in self._data[order_key]):
            gkey = f"{key}__{source_name}"
            det_key = f"{gkey}_det_ids"
            if det_key not in self._data:
                continue

            det_ids = self._data[det_key]
            source_types = self._data[f"{gkey}_source_types"]
            log_probs = self._data[f"{gkey}_log_probs"]
            cat_size = int(self._data[f"{gkey}_catalog_size"])
            cal_sigs = self._data[f"{gkey}_cal_sig"]
            rt_sigs = self._data[f"{gkey}_rt_sig"]
            obs_mask_arr = self._data.get(f"{gkey}_obs_mask")

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
                        source_name=source_name,
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

    def catalog_labels_for_source(self, source_name: str) -> Optional[tuple[str, ...]]:
        """The catalog basis *source_name*'s evidence rows were built against.

        Returns that source's own phase-local basis when the sidecar has one
        recorded (schema v2+, via ``catalog_labels_by_source`` at
        construction). Falls back to the global ``catalog_labels`` when the
        source has no recorded basis (e.g. a v1 sidecar, predating this
        per-source tracking, or a source that is genuinely global-basis like
        AprilTag) -- callers (the tracking worker's
        ``_remap_source_log_probs_to_catalog``) treat that fallback as
        "already on the global basis", matching v1 behavior exactly.
        """
        return self._catalog_labels_by_source.get(source_name, self._catalog_labels)

    def __len__(self) -> int:
        return len(self.get_cached_frames())

    def close(self) -> None:
        """Match cache-style interfaces used elsewhere in the tracking stack."""
        return None
