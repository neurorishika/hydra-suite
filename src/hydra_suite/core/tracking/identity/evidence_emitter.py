"""Identity evidence sidecar path derivation.

Identity Phase 7 / Task 4: the streaming-time ``IdentityEvidenceEmitter``
(which converted CNN ``ClassPrediction`` outputs to ``IdentityEvidence``
objects and accumulated them into an ``IdentityEvidenceCache`` sidecar as the
tracking worker streamed frames) has been retired. Evidence is now written
entirely at inference time by ``IdentityEvidenceStage``
(``core/inference/stages/identity_evidence.py``), driven by
``core/inference/runner.py``. Parity between the two paths is preserved as a
committed golden fixture (see ``tests/data/identity_evidence_goldens/``) so
the retirement carries no behavior change.

This module now only holds ``build_evidence_cache_path``, still used by
``core/inference/runner.py`` to derive the sidecar path from the detection
cache base path.
"""

from __future__ import annotations

from pathlib import Path


def build_evidence_cache_path(
    base_cache_path: str,
    source_name: str,
    signature: str,
) -> Path:
    """Derive the evidence sidecar path from the detection cache base path.

    Convention::

        <base>_identity_evidence_<source>_<signature>.npz
    """
    p = Path(base_cache_path)
    stem = p.stem.replace("_detections", "").replace("_detection", "")
    sidecar_name = f"{stem}_identity_evidence_{source_name}_{signature}.npz"
    return p.parent / sidecar_name
