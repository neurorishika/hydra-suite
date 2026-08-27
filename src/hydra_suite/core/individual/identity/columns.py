"""Canonical, provenance-explicit identity column vocabulary.

Single source of truth for every CSV/DataFrame column name the identity
system reads or writes. Column names use PascalCase with no underscores,
prefixed by the provenance family they belong to:

- ``IdentityRealtime*``: written per-frame by the tracking worker's
  online (Kalman-time) identity assignment.
- ``IdentityEvidence*``: shared evidence fields consumed by both the
  realtime and offline paths.
- ``IdentityFinal*``: written by offline resolution/smoothing — the
  final, authoritative identity call for a track.

This module is pure: no imports beyond the standard library, and no
imports from any app layer (trackerkit/classkit/refinekit/detectkit/
filterkit/integrations). Core must never import upward.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# --- IdentityRealtime* -------------------------------------------------
REALTIME_ID = "IdentityRealtimeID"
REALTIME_LABEL = "IdentityRealtimeLabel"
REALTIME_CONFIDENCE = "IdentityRealtimeConfidence"
REALTIME_MARGIN = "IdentityRealtimeMargin"
REALTIME_ENTROPY = "IdentityRealtimeEntropy"
REALTIME_COMMITTED = "IdentityRealtimeCommitted"
REALTIME_SLOTLOCK = "IdentityRealtimeSlotLock"

# --- IdentityEvidence* --------------------------------------------------
EVIDENCE_TOPLABEL = "IdentityEvidenceTopLabel"
EVIDENCE_CONFIDENCE = "IdentityEvidenceConfidence"
EVIDENCE_SOURCES = "IdentityEvidenceSources"
EVIDENCE_CONFLICT_FLAG = "IdentityEvidenceConflictFlag"

# --- IdentityFinal* ------------------------------------------------------
FINAL_LABEL = "IdentityFinalLabel"
FINAL_ID = "IdentityFinalID"
FINAL_CONFIDENCE = "IdentityFinalConfidence"
FINAL_SOURCE = "IdentityFinalSource"
FINAL_FRAGMENT_SCORE = "IdentityFinalFragmentScore"
FINAL_SMOOTHED_LABEL = "IdentityFinalSmoothedLabel"
FINAL_SMOOTHED_CONFIDENCE = "IdentityFinalSmoothedConfidence"
FINAL_CONFLICT_RESOLVED = "IdentityFinalConflictResolved"

# --- Cross-family ---------------------------------------------------------
UNIQUE_IDENTITY_KEY = "UniqueIdentityKey"


class IdentityFinalSource:
    """Vocabulary for ``IdentityFinalSource`` column values."""

    REALTIME = "realtime"
    OFFLINE = "offline"
    TAG = "tag"
    NON_IDENTIFYING = "nonidentifying"
    """A declared non-identifying composite (e.g. an untagged animal).

    The label is descriptive only -- ``IdentityFinalID`` stays at the unknown
    slot (0), so nothing downstream can mistake it for a resolved identity.
    """
    NONE = "none"
    """Explicit "no identity was resolved for this row" token.

    Never write ``""``; readers normalise ``""``/NaN to this value
    (``identity_postprocess.normalize_final_source_series``).
    """


def normalize_final_source_series(source: "pd.Series") -> "pd.Series":
    """Map NaN / blank ``IdentityFinalSource`` cells (legacy CSVs, columns
    created before the solver ran) to the explicit ``IdentityFinalSource.NONE``
    token; strip whitespace from real tokens."""

    token = source.astype(object).where(source.notna(), "").astype(str).str.strip()
    return token.where(token != "", IdentityFinalSource.NONE)


def identity_realtime_columns() -> list:
    """Ordered realtime column block appended to each raw tracking row.

    Mirrors the worker's positional row writer:
    (catalog_index, label, confidence, margin, entropy, committed,
    evidence_sources, conflict_flag, slot_lock_label).
    """
    return [
        REALTIME_ID,
        REALTIME_LABEL,
        REALTIME_CONFIDENCE,
        REALTIME_MARGIN,
        REALTIME_ENTROPY,
        REALTIME_COMMITTED,
        EVIDENCE_SOURCES,
        EVIDENCE_CONFLICT_FLAG,
        REALTIME_SLOTLOCK,
    ]
