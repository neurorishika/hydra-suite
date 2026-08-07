"""AprilTag auto-labeler for ClassKit.

Runs MAT's AprilTagDetector across multiple preprocessing profiles on each
cropped image. Confidence is the fraction of profiles agreeing on the same
tag ID. Only labels above the confidence threshold are committed.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from ....core.individual.classification.apriltag import AprilTagConfig, AprilTagDetector

# ---------------------------------------------------------------------------
# Preprocessing profiles
# All must return 3-channel BGR uint8 — required because AprilTagDetector's
# _detect_composite always runs cv2.cvtColor(composite, COLOR_BGR2GRAY).
# ---------------------------------------------------------------------------


def _profile_raw(image: np.ndarray) -> np.ndarray:
    """Pass image through unchanged."""
    return image


def _profile_clahe(image: np.ndarray) -> np.ndarray:
    """CLAHE on grayscale, returned as BGR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _profile_gamma_boost(image: np.ndarray) -> np.ndarray:
    """Gamma γ=0.5: brightens dark images."""
    lut = np.array(
        [int(((i / 255.0) ** 0.5) * 255) for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, lut)


def _profile_gamma_darken(image: np.ndarray) -> np.ndarray:
    """Gamma γ=2.0: darkens overexposed images."""
    lut = np.array(
        [int(((i / 255.0) ** 2.0) * 255) for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, lut)


def _make_unsharp_strong(
    config: AprilTagConfig,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return an unsharp-mask profile using 2× the user-configured amount."""
    ks = config.unsharp_kernel_size  # Tuple[int, int] e.g. (5, 5)
    sigma = config.unsharp_sigma
    amount = config.unsharp_amount * 2.0

    def _profile(image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, ks, sigma)
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    return _profile


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LabelResult:
    """Outcome of auto-labeling one image."""

    label: Optional[str]
    """'tag_N', 'no_tag', or None (uncertain — leave unlabeled)."""

    confidence: float
    """count_of_majority_tag / N_total_profiles."""

    n_profiles_run: int
    """Always equal to len(PREPROCESSING_PROFILES) = 5."""

    detected_tag_id: Optional[int]
    """Raw integer majority tag ID, or None."""

    all_no_tag: bool = False
    """True when every profile returned no observations (or all ambiguous)."""


# ---------------------------------------------------------------------------
# Labeler
# ---------------------------------------------------------------------------

_N_PROFILES = 5  # raw, clahe, gamma_boost, gamma_darken, unsharp_strong


class AprilTagAutoLabeler:
    """Detect AprilTags in a single crop using multiple preprocessing profiles.

    The internal ``AprilTagDetector`` is constructed with its own preprocessing
    disabled (``unsharp_amount=0.0``, ``contrast_factor=1.0``) so that each
    profile fully controls what the detector sees.
    """

    def __init__(self, config: AprilTagConfig, confidence_threshold: float = 0.6):
        self.config = config
        self.threshold = confidence_threshold
        # Disable detector's built-in preprocessing so profiles are in full control
        internal_config = dataclasses.replace(
            config, unsharp_amount=0.0, contrast_factor=1.0
        )
        self._detector = AprilTagDetector(internal_config)
        self._profiles: List[Callable[[np.ndarray], np.ndarray]] = [
            _profile_raw,
            _profile_clahe,
            _profile_gamma_boost,
            _profile_gamma_darken,
            _make_unsharp_strong(config),
        ]

    def _run_profile(
        self, image: np.ndarray, profile_fn: Callable[[np.ndarray], np.ndarray]
    ) -> Optional[int]:
        """Apply one profile and run detection. Returns tag_id or None."""
        preprocessed = profile_fn(image)
        observations = self._detector.detect_in_crops(
            crops=[preprocessed], offsets_xy=[(0, 0)]
        )
        if not observations:
            return None  # NO_TAG
        if len(observations) == 1:
            return observations[0].tag_id
        # Multiple observations: pick lowest-hamming; if tied → AMBIGUOUS → None
        min_hamming = min(o.hamming for o in observations)
        winners = [o for o in observations if o.hamming == min_hamming]
        if len(winners) == 1:
            return winners[0].tag_id
        return None  # AMBIGUOUS

    def label_image(self, image: np.ndarray) -> LabelResult:
        """Label one BGR crop image."""
        n = len(self._profiles)
        per_profile: List[Optional[int]] = [
            self._run_profile(image, pf) for pf in self._profiles
        ]
        detected = [tag_id for tag_id in per_profile if tag_id is not None]

        if not detected:
            return LabelResult(
                label="no_tag",
                confidence=1.0,
                n_profiles_run=n,
                detected_tag_id=None,
                all_no_tag=True,
            )

        counts = Counter(detected)
        majority_id, majority_count = counts.most_common(1)[0]
        confidence = majority_count / n

        if confidence >= self.threshold:
            return LabelResult(
                label=f"tag_{majority_id}",
                confidence=confidence,
                n_profiles_run=n,
                detected_tag_id=majority_id,
            )
        return LabelResult(
            label=None,
            confidence=confidence,
            n_profiles_run=n,
            detected_tag_id=majority_id,
        )

    def close(self) -> None:
        """Release detector resources."""
        self._detector.close()


# ---------------------------------------------------------------------------
# Batch entrypoint
# ---------------------------------------------------------------------------


def autolabel_images(
    image_paths: List[Path],
    config: AprilTagConfig,
    threshold: float,
) -> List[LabelResult]:
    """Auto-label a list of image paths.

    Returns one ``LabelResult`` per path, in the same order. Returns an empty
    list when ``image_paths`` is empty. Images that cannot be read (bad path or
    corrupt file) produce ``LabelResult(label=None, confidence=0.0, ...)``.
    """
    if not image_paths:
        return []

    labeler = AprilTagAutoLabeler(config, threshold)
    results: List[LabelResult] = []
    try:
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                results.append(
                    LabelResult(
                        label=None,
                        confidence=0.0,
                        n_profiles_run=_N_PROFILES,
                        detected_tag_id=None,
                    )
                )
                continue
            results.append(labeler.label_image(image))
    finally:
        labeler.close()
    return results
