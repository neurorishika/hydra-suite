"""Candidate-pool construction backed by FilterKit dedup primitives.

Layer note
----------
`hydra_suite.filterkit.core.FilterKitCore` lives under the `filterkit/` app
package, so the `from hydra_suite.filterkit.core import FilterKitCore` below
inverts the strict App -> Data dependency rule documented in CLAUDE.md.

`FilterKitCore` itself is a pure-utility class (perceptual hashing + BK-tree
indexing, no Qt/GUI dependencies). The clean fix is to relocate it to
`hydra_suite/utils/perceptual_dedup.py` and have both FilterKit and this module
import from there. That refactor is intentionally out of scope for the AL
detection-dataset feature; treat this import as a documented carve-out until
the Simplification Sprint lands the relocation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from hydra_suite.filterkit.core import (  # noqa: I900 (layer carve-out, see module docstring)
    FilterKitCore,
)

from .frame_source import FrameRef, FrameSource

DedupMethod = Literal["phash", "ahash", "dhash", "histogram", "none"]


@dataclass
class CandidatePoolConfig:
    """Configuration for `build_candidate_pool`."""

    dedup_method: DedupMethod = "phash"
    dedup_threshold: int = 8  # Hamming for hashes; bins for histogram

    # Hard ceiling on the returned pool, and a memory/OOM guard rather than a
    # scoring knob. Downstream, DetectKit's AL round holds every candidate
    # frame in memory at once and hands the whole list to ONE
    # `InferenceRunner.detect_batch_raw` call, which forwards it to `run_obb`
    # with no windowing of its own (unlike the tracking batch pass, which
    # windows by `detection_batch_size`). An unbounded pool on a long,
    # high-motion video is therefore thousands of full-resolution frames
    # resident simultaneously AND one enormous model batch. 128 keeps both
    # bounded (~800 MB of decoded 1080p) while still giving the batched pass
    # plenty to work with; raise it deliberately, with the frame size and the
    # detection batch in mind. `None` restores the old unbounded behaviour for
    # callers that know their source is small.
    max_candidates: int | None = 128

    # Windowed dedup: compare each frame's signature against at most the last
    # `dedup_window` *kept* signatures instead of the full history. `None`
    # (default) means unbounded -- identical to the pre-windowing global scan,
    # so leaving this unset keeps existing behavior exactly for any
    # typical-sized fixture/video.
    dedup_window: int | None = None

    # Frame-difference motion prefilter: frames whose grayscale mean-absolute
    # difference from a rolling reference frame is below `motion_threshold`
    # skip full dedup/signature scoring entirely (cheap early-out before the
    # perceptual hash is even computed). Default 0.0 is permissive -- a real
    # per-pixel mean-abs-diff is virtually never < 0.0, so by default the
    # prefilter is a no-op and every frame still reaches full scoring, same as
    # current behavior.
    motion_threshold: float = 0.0

    # Periodic-sampling floor: even when consecutive frames sit below
    # `motion_threshold` (e.g. a long static stretch), let one through anyway
    # once `periodic_sample_every` frames have been skipped since the last
    # frame that was let through, so long static stretches still get
    # occasionally sampled. Irrelevant while `motion_threshold` is at its
    # permissive default (the skip branch is never taken), so any default
    # value here is safe; kept modest for when motion_threshold is tightened.
    periodic_sample_every: int = 30


def build_candidate_pool(
    source: FrameSource,
    cfg: CandidatePoolConfig,
) -> list[FrameRef]:
    """Return a deduplicated, optionally capped list of candidate FrameRefs.

    Iterates `source`, computes the configured perceptual signature for each
    frame, and keeps only frames whose signature is sufficiently distinct from
    the most recently kept frames (within `cfg.dedup_window`, or all
    previously-kept frames when unset).

    Before scoring, an inline frame-difference prefilter skips frames whose
    visual change from a rolling reference frame is negligible
    (`cfg.motion_threshold`), with a periodic-sampling floor
    (`cfg.periodic_sample_every`) so long static stretches still contribute an
    occasional sample. This prefilter and the dedup step both operate on
    frames read sequentially via `source`, so they benefit from
    `VideoFrameSource`'s single-capture sequential-read reuse.

    `cfg.max_candidates` stops the scan as soon as the cap is reached, so on a
    source longer than the cap the pool covers the source's BEGINNING only. To
    spread a capped pool across a long video, give the source a stride
    (`VideoFrameSource(path, stride=n)`) rather than raising the cap -- the cap
    exists to bound the caller's memory and model-batch size (see
    `CandidatePoolConfig.max_candidates`).
    """
    fk = FilterKitCore()
    kept: list[FrameRef] = []
    kept_signatures: deque = deque(maxlen=cfg.dedup_window)

    reference_gray: np.ndarray | None = None
    frames_since_allowed = 0

    for ref in source:
        if cfg.max_candidates is not None and len(kept) >= cfg.max_candidates:
            break

        if cfg.dedup_method == "none":
            kept.append(ref)
            continue

        img = source.read(ref)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        if reference_gray is not None:
            delta = float(
                np.abs(gray.astype(np.int16) - reference_gray.astype(np.int16)).mean()
            )
            periodic_override = frames_since_allowed >= cfg.periodic_sample_every
            if delta < cfg.motion_threshold and not periodic_override:
                # Negligible visual change: skip full dedup/signature scoring
                # for this frame entirely -- it never enters the candidate
                # pool consideration. Reference frame is intentionally left
                # untouched here (refreshed only on a letting-through) so slow
                # drift is measured cumulatively, not frame-to-frame.
                frames_since_allowed += 1
                continue

        # Frame is let through -- either real motion, the periodic-sampling
        # floor, or the very first frame. Refresh the rolling reference on
        # every letting-through so slow lighting drift doesn't lock the
        # threshold against a stale reference.
        reference_gray = gray
        frames_since_allowed = 0

        sig = fk.compute_signature(img, method=cfg.dedup_method)

        is_dup = any(
            fk.is_duplicate(sig, prev, cfg.dedup_threshold, cfg.dedup_method)
            for prev in kept_signatures
        )
        if not is_dup:
            kept.append(ref)
            kept_signatures.append(sig)

    return kept
