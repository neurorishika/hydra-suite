"""Candidate-pool construction backed by FilterKit dedup primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hydra_suite.filterkit.core import FilterKitCore

from .frame_source import FrameRef, FrameSource

DedupMethod = Literal["phash", "ahash", "dhash", "histogram", "none"]


@dataclass
class CandidatePoolConfig:
    """Configuration for `build_candidate_pool`."""

    dedup_method: DedupMethod = "phash"
    dedup_threshold: int = 8  # Hamming for hashes; bins for histogram
    max_candidates: int | None = None


def build_candidate_pool(
    source: FrameSource,
    cfg: CandidatePoolConfig,
) -> list[FrameRef]:
    """Return a deduplicated, optionally capped list of candidate FrameRefs.

    Iterates `source`, computes the configured perceptual signature for each
    frame, and keeps only frames whose signature is sufficiently distinct from
    all previously-kept frames.
    """
    fk = FilterKitCore()
    kept: list[FrameRef] = []
    kept_signatures: list = []

    for ref in source:
        if cfg.max_candidates is not None and len(kept) >= cfg.max_candidates:
            break

        if cfg.dedup_method == "none":
            kept.append(ref)
            continue

        img = source.read(ref)
        if img is None:
            continue
        sig = fk.compute_signature(img, method=cfg.dedup_method)

        is_dup = any(
            fk.is_duplicate(sig, prev, cfg.dedup_threshold, cfg.dedup_method)
            for prev in kept_signatures
        )
        if not is_dup:
            kept.append(ref)
            kept_signatures.append(sig)

    return kept
