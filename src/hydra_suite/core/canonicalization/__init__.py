"""Canonicalization utilities for MAT.

Submodules:
    geometry — Layer 1: project-fixed rigid canvas geometry (``CanonicalGeometry``).
    fit      — Layer 2: letterbox fit of any image into a model's input tensor.
    crop     — real-time OBB-based canonical crop extraction for tracking.
"""

from hydra_suite.core.canonicalization.crop import (  # noqa: F401
    CanonicalCropResult,
    apply_headtail_rotation,
    compute_alignment_affine,
    compute_crop_dimensions,
    compute_native_crop_dimensions,
    compute_native_scale_affine,
    extract_and_classify_batch,
    extract_canonical_crop,
    gpu_canonical_crop,
    gpu_canonical_crop_batch,
    invert_keypoints,
)
