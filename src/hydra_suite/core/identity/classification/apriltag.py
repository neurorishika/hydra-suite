"""
AprilTag detector wrapper with composite-strip decoding.

Implements the fast decode strategy from the standalone ``temp.py`` script:
crop each animal detection from the frame, tile all crops into a single wide
composite image, run the AprilTag detector *once* on the composite, then
re-project each tag detection back to absolute frame coordinates.

The ``apriltag`` package is imported lazily so that the rest of MAT keeps
working in environments where it is not installed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)
APRILTAG_FORK_REPO = "https://github.com/Social-Evolution-and-Behavior/apriltag"
APRILTAG_FORK_COMMIT = "c43a9b6e6b7dcfe0e7647a78eff6655a1d743c2c"
APRILTAG_REQUIRED_FAMILY = "tag36ARTag"

# ---------------------------------------------------------------------------
# Lazy import of the apriltag library
# ---------------------------------------------------------------------------
_apriltag_module: Any = None


def _has_required_apriltag_family(module: Any) -> bool:
    """Return whether the installed apriltag build exposes the lab-only family."""
    try:
        module.apriltag("__probe__")
    except Exception as exc:
        families = set(re.findall(r"^\s+(\S+)$", str(exc), re.MULTILINE))
        if families:
            return APRILTAG_REQUIRED_FAMILY in families
    return False


def _get_apriltag():
    """Import ``apriltag`` once and cache the module reference."""
    global _apriltag_module
    if _apriltag_module is None:
        try:
            import apriltag as _at  # type: ignore[import-untyped]

            if not _has_required_apriltag_family(_at):
                raise ImportError(
                    "HYDRA requires the lab apriltag fork with tag36ARTag support. "
                    f"Install {APRILTAG_FORK_REPO} at commit {APRILTAG_FORK_COMMIT}, "
                    "or run `make install-apriltag-fork` in the active environment."
                )
            _apriltag_module = _at
        except ImportError:
            raise ImportError(
                "The lab apriltag fork is required for AprilTag detection but is "
                "not installed in this environment. Install "
                f"{APRILTAG_FORK_REPO} at commit {APRILTAG_FORK_COMMIT}, or run "
                "`make install-apriltag-fork` in the active environment."
            )
    return _apriltag_module


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class AprilTagConfig:
    """All tunables for the AprilTag detection step."""

    family: str = "tag36h11"
    threads: int = 4
    max_hamming: int = 1
    decimate: float = 1.0
    blur: float = 0.8
    refine_edges: bool = True
    decode_sharpening: float = 0.25

    # Pre-processing (unsharp mask + contrast)
    unsharp_kernel_size: Tuple[int, int] = (5, 5)
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 1.5
    contrast_factor: float = 1.5

    # Filtering
    max_tag_id: Optional[int] = None  # ignore IDs above this (None = no limit)
    # Crop padding as a fraction of OBB size — reuses INDIVIDUAL_CROP_PADDING
    padding_fraction: float = 0.1

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "AprilTagConfig":
        """Build from the MAT tracking-parameters dictionary."""
        return cls(
            family=str(params.get("APRILTAG_FAMILY", "tag36h11")),
            threads=int(params.get("APRILTAG_THREADS", 4)),
            max_hamming=int(params.get("APRILTAG_MAX_HAMMING", 1)),
            decimate=float(params.get("APRILTAG_DECIMATE", 1.0)),
            blur=float(params.get("APRILTAG_BLUR", 0.8)),
            refine_edges=bool(params.get("APRILTAG_REFINE_EDGES", True)),
            decode_sharpening=float(params.get("APRILTAG_DECODE_SHARPENING", 0.25)),
            unsharp_kernel_size=tuple(params.get("APRILTAG_UNSHARP_KERNEL", (5, 5))),
            unsharp_sigma=float(params.get("APRILTAG_UNSHARP_SIGMA", 1.0)),
            unsharp_amount=float(params.get("APRILTAG_UNSHARP_AMOUNT", 1.5)),
            contrast_factor=float(params.get("APRILTAG_CONTRAST_FACTOR", 1.5)),
            max_tag_id=(
                int(params["APRILTAG_MAX_TAG_ID"])
                if params.get("APRILTAG_MAX_TAG_ID") is not None
                else None
            ),
            padding_fraction=float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),
        )


# ---------------------------------------------------------------------------
# Single-tag observation returned by the detector
# ---------------------------------------------------------------------------


@dataclass
class TagObservation:
    """One AprilTag detection in absolute frame coordinates."""

    tag_id: int
    center_xy: Tuple[float, float]
    corners: np.ndarray  # (4, 2) absolute frame coords
    det_index: int  # index into the detection list for this frame
    hamming: int = 0


# ---------------------------------------------------------------------------
# Image preprocessing helpers (mirrors temp.py logic)
# ---------------------------------------------------------------------------


def _unsharp_mask(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.5,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def _contrast_enhance(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Simple linear contrast enhancement around the mean."""
    mean = np.mean(image)
    enhanced = np.clip(mean + factor * (image.astype(np.float32) - mean), 0, 255)
    return enhanced.astype(np.uint8)


# ---------------------------------------------------------------------------
# Core detector class
# ---------------------------------------------------------------------------


class AprilTagDetector:
    """Detect AprilTags inside OBB/bbox crops using composite-strip decoding.

    Usage::

        detector = AprilTagDetector(AprilTagConfig.from_params(params))
        observations = detector.detect_in_crops(frame, crops, offsets)
        detector.close()   # optional, releases C resources
    """

    def __init__(self, config: AprilTagConfig):
        self.config = config
        self._detector = None  # lazily created

    def _ensure_detector(self) -> Any:
        """Create the underlying C detector on first use."""
        if self._detector is not None:
            return self._detector

        at = _get_apriltag()
        self._detector = at.apriltag(
            family=self.config.family,
            threads=self.config.threads,
            maxhamming=self.config.max_hamming,
            decimate=self.config.decimate,
            blur=self.config.blur,
            refine_edges=1 if self.config.refine_edges else 0,
            decode_sharpening=self.config.decode_sharpening,
        )
        return self._detector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_in_crops(
        self,
        crops: Sequence[np.ndarray],
        offsets_xy: Sequence[Tuple[int, int]],
        det_indices: Optional[Sequence[int]] = None,
    ) -> List[TagObservation]:
        """Detect tags in pre-extracted crops.

        Parameters
        ----------
        crops:
            List of BGR crops already cut from the frame.
        offsets_xy:
            ``(x_offset, y_offset)`` of each crop's top-left corner in the
            original frame.
        det_indices:
            Explicit detection-index per crop.
        """
        if not crops:
            return []
        idx = list(det_indices) if det_indices is not None else list(range(len(crops)))
        return self._detect_composite(list(crops), list(offsets_xy), idx)

    def close(self) -> None:
        """Release native detector resources."""
        self._detector = None

    # ------------------------------------------------------------------
    # Composite-strip strategy (mirrors temp.py)
    # ------------------------------------------------------------------

    def _detect_composite(
        self,
        crops: List[np.ndarray],
        offsets_xy: List[Tuple[int, int]],
        index_map: List[int],
    ) -> List[TagObservation]:
        detector = self._ensure_detector()
        cfg = self.config

        # 1. Build horizontal composite strip
        strip_height = max(c.shape[0] for c in crops)
        strip_width = sum(c.shape[1] for c in crops)
        composite = np.zeros((strip_height, strip_width, 3), dtype=np.uint8)

        canvas_x_offsets: List[int] = []
        cursor = 0
        for c in crops:
            ch, cw = c.shape[:2]
            composite[0:ch, cursor : cursor + cw] = c
            canvas_x_offsets.append(cursor)
            cursor += cw

        # 2. Pre-process: grayscale → unsharp mask → contrast enhance
        gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
        sharp = _unsharp_mask(
            gray,
            kernel_size=cfg.unsharp_kernel_size,
            sigma=cfg.unsharp_sigma,
            amount=cfg.unsharp_amount,
        )
        enhanced = _contrast_enhance(sharp, factor=cfg.contrast_factor)

        # 3. Single detect call on the whole strip
        raw_tags = detector.detect(enhanced)

        # 4. Re-project each tag to original frame coordinates
        cum_widths = np.cumsum([c.shape[1] for c in crops])
        observations: List[TagObservation] = []

        for tag in raw_tags:
            tag_id = int(tag["id"])
            if cfg.max_tag_id is not None and tag_id > cfg.max_tag_id:
                continue

            cx, cy = tag["center"]
            hamming = int(tag.get("hamming", 0))

            # Determine which crop this tag centre falls in
            crop_idx = int(np.searchsorted(cum_widths, cx, side="right"))
            if crop_idx >= len(crops):
                continue  # safety: tag centre outside all crops

            crop_x0 = canvas_x_offsets[crop_idx]
            x_off, y_off = offsets_xy[crop_idx]

            rel_x = cx - crop_x0
            rel_y = cy  # strip is top-aligned

            abs_x = float(x_off + rel_x)
            abs_y = float(y_off + rel_y)

            # Re-project the 4 corners
            # The key name varies between apriltag bindings:
            #   dt-apriltag uses "corners",  apriltag-python uses "lb-rb-rt-lt"
            raw_corners = tag.get("lb-rb-rt-lt")
            if raw_corners is None:
                raw_corners = tag.get("corners")
            if raw_corners is None:
                continue

            abs_corners = np.asarray(raw_corners, dtype=np.float32).copy()
            if abs_corners.ndim != 2 or abs_corners.shape[0] != 4:
                continue
            abs_corners[:, 0] += x_off - crop_x0
            abs_corners[:, 1] += y_off

            observations.append(
                TagObservation(
                    tag_id=tag_id,
                    center_xy=(abs_x, abs_y),
                    corners=abs_corners,
                    det_index=index_map[crop_idx],
                    hamming=hamming,
                )
            )

        return observations
