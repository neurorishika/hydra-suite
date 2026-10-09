"""Canonical SAHI tiling contract shared by every kit.

Spec: docs/superpowers/specs/2026-10-09-sahi-unification-design.md (§3).
TrackerKit's vocabulary is canonical. This module is pure (numpy only) so
core, training, data and every kit can import it; ``slice_geometry`` stays
the grid module and is not modified.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, NamedTuple

import numpy as np

from .slice_geometry import DEFAULT_MIN_AREA_RATIO, resolve_scales

logger = logging.getLogger(__name__)

GEOMETRY_MODES = ("auto_model", "auto_object", "custom")
FRAGMENT_POLICIES = ("drop", "crowd", "mask")
# ``nmm`` is NOT here: it runs the same code path as ``greedy_nmm`` today
# (stages/merge.py), so ``canonicalize`` reads it as ``greedy_nmm``.
MERGE_POLICIES = ("nms", "greedy_nmm")
MERGE_METRICS = ("iou", "ios", "polygon_iou")
OVERLAP_MAX = 0.9  # same ceiling SliceConfig/_slice_config_from_params clamp to
# Derived overlap = max(fraction) + margin. 0.05 reproduces TrackerKit's 0.2
# default at its 0.15 default fraction. Overlap px >= body px is what puts
# every animal whole inside at least one tile (overlap*tile >= frac*tile).
OVERLAP_MARGIN = 0.05
DEFAULT_OVERLAP = 0.2
FRACTION_MIN = 0.01  # tile_size_for_mode's clamp
FRACTION_MAX = 0.9

Backend = Literal["yolo_train", "yolo_infer", "sam3", "sam2"]

_SPEC_FIELDS = (
    "enabled",
    "geometry_mode",
    "object_tile_fractions",
    "reference_body_px",
    "slice_width",
    "slice_height",
    "overlap",
    "min_area_ratio",
    "fragment_policy",
    "merge_policy",
    "merge_metric",
    "merge_threshold",
)


class Sourced(NamedTuple):
    """A resolved value plus where it came from (shown as a UI badge)."""

    value: Any
    source: str  # user | override | profile | stamped | dataset | derived | default


@dataclass(frozen=True)
class BackendDefaults:
    object_tile_fractions: tuple[float, ...]
    geometry_mode: str
    fragment_policy: str
    merge_policy: str
    merge_metric: str
    merge_threshold: float
    min_area_ratio: float


BACKEND_DEFAULTS: dict[str, BackendDefaults] = {
    # Multi-scale robustness set used by headless SliceTrainingConfig today.
    "yolo_train": BackendDefaults(
        (0.05, 0.10, 0.15, 0.20),
        "auto_object",
        "drop",
        "greedy_nmm",
        "ios",
        0.5,
        DEFAULT_MIN_AREA_RATIO,
    ),
    # SliceConfig defaults (TrackerKit is canonical; must not move).
    "yolo_infer": BackendDefaults(
        (0.15,),
        "auto_model",
        "drop",
        "greedy_nmm",
        "ios",
        0.5,
        DEFAULT_MIN_AREA_RATIO,
    ),
    # Sam3LoraParams.object_tile_fraction; SAM3 merge is polygon-IoU NMS after
    # a containment gate (semantic/tiling.py), fragments become is_crowd.
    "sam3": BackendDefaults(
        (0.055,),
        "auto_object",
        "crowd",
        "nms",
        "polygon_iou",
        0.5,
        DEFAULT_MIN_AREA_RATIO,
    ),
    # Stock SAM2: full frame until calibrated (2026-10-03 spec: fractions do
    # not transfer between models). SAM2 owner tiles do not merge.
    "sam2": BackendDefaults(
        (),
        "auto_object",
        "drop",
        "nms",
        "iou",
        0.5,
        DEFAULT_MIN_AREA_RATIO,
    ),
}


def _clamp_fraction(value: float) -> float:
    return max(FRACTION_MIN, min(FRACTION_MAX, float(value)))


def operating_fraction(fractions) -> float | None:
    """The ONE inference scale for a fraction set: np.median, clamped like the planner.

    The same rule TrackerKit applies to median(target_sizes)/imgsz and SAM3's
    dataset_build stamps as prefill_object_tile_fraction.
    """
    values = [float(f) for f in fractions]
    if not values:
        return None
    return _clamp_fraction(float(np.median(np.asarray(values))))


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class TilingSpec:
    """One SAHI tiling configuration in canonical (TrackerKit) vocabulary."""

    enabled: bool = False
    geometry_mode: str = "auto_model"
    object_tile_fractions: tuple[float, ...] = ()
    reference_body_px: float = 0.0
    slice_width: int = 0
    slice_height: int = 0
    overlap: float | None = None  # None = unset -> resolve_overlap derives it
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO
    fragment_policy: str = "drop"
    merge_policy: str = "greedy_nmm"
    merge_metric: str = "ios"
    merge_threshold: float = 0.5

    def __post_init__(self) -> None:
        fractions = tuple(float(f) for f in self.object_tile_fractions)
        object.__setattr__(self, "object_tile_fractions", fractions)
        object.__setattr__(self, "enabled", bool(self.enabled))
        problems: list[str] = []
        if self.geometry_mode not in GEOMETRY_MODES:
            problems.append(f"geometry_mode {self.geometry_mode!r}")
        if any(not (math.isfinite(f) and 0.0 < f <= 1.0) for f in fractions):
            problems.append(f"object_tile_fractions {fractions!r} outside (0, 1]")
        body = _finite(self.reference_body_px)
        if body is None or body < 0:
            problems.append(f"reference_body_px {self.reference_body_px!r}")
        for name in ("slice_width", "slice_height"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= 8192
            ):
                problems.append(f"{name} {value!r} outside [0, 8192]")
        if self.overlap is not None:
            overlap = _finite(self.overlap)
            if overlap is None or not 0.0 <= overlap <= OVERLAP_MAX:
                problems.append(f"overlap {self.overlap!r} outside [0, {OVERLAP_MAX}]")
        area = _finite(self.min_area_ratio)
        if area is None or not 0.0 <= area <= 1.0:
            problems.append(f"min_area_ratio {self.min_area_ratio!r}")
        if self.fragment_policy not in FRAGMENT_POLICIES:
            problems.append(f"fragment_policy {self.fragment_policy!r}")
        if self.merge_policy not in MERGE_POLICIES:
            problems.append(f"merge_policy {self.merge_policy!r}")
        if self.merge_metric not in MERGE_METRICS:
            problems.append(f"merge_metric {self.merge_metric!r}")
        threshold = _finite(self.merge_threshold)
        if threshold is None or not 0.0 <= threshold <= 1.0:
            problems.append(f"merge_threshold {self.merge_threshold!r}")
        if problems:
            raise ValueError("Invalid TilingSpec: " + "; ".join(problems))

    @classmethod
    def defaults(cls, backend: Backend) -> "TilingSpec":
        d = BACKEND_DEFAULTS[backend]
        return cls(
            geometry_mode=d.geometry_mode,
            object_tile_fractions=d.object_tile_fractions,
            min_area_ratio=d.min_area_ratio,
            fragment_policy=d.fragment_policy,
            merge_policy=d.merge_policy,
            merge_metric=d.merge_metric,
            merge_threshold=d.merge_threshold,
        )

    def operating_fraction(self) -> float | None:
        return operating_fraction(self.object_tile_fractions)

    def training_tile_sizes(self, imgsz: int) -> list[tuple[int, int]]:
        """TRAINING fan-out: one tile size per scale. Inference uses ONE scale
        (operating_fraction + resolve_tile_size), never this."""
        scalar = (
            self.operating_fraction()
            or BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions[0]
        )
        return resolve_scales(
            geometry_mode=self.geometry_mode,
            imgsz=int(imgsz),
            reference_body_px=float(self.reference_body_px),
            fractions=self.object_tile_fractions,
            object_tile_fraction=scalar,
            slice_width=self.slice_width,
            slice_height=self.slice_height,
        )

    def to_mapping(self) -> dict[str, Any]:
        data = asdict(self)
        data["object_tile_fractions"] = list(self.object_tile_fractions)
        return data
