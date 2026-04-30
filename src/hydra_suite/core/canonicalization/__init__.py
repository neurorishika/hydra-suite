"""Canonicalization utilities for MAT.

Submodules:
    dataset — metadata-driven canonicalization for individual-dataset exports.
    crop    — real-time OBB-based canonical crop extraction for tracking.
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
    invert_keypoints,
)
from hydra_suite.core.canonicalization.dataset import (  # noqa: F401
    MatMetadataCanonicalizer,
    get_canon_transform,
)
