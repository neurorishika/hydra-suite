"""Canonical SAHI tiling contract shared by every kit.

Spec: docs/superpowers/specs/2026-10-09-sahi-unification-design.md (§3).
TrackerKit's vocabulary is canonical. This module is pure (numpy only) so
core, training, data and every kit can import it; ``slice_geometry`` stays
the grid module and is not modified.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, NamedTuple

import numpy as np

from .slice_geometry import DEFAULT_MIN_AREA_RATIO, resolve_scales, tile_size_for_mode

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

    @classmethod
    def from_mapping(
        cls,
        mapping: Any,
        *,
        backend: Backend | None = None,
        legacy_px_imgsz: float | None = None,
    ) -> "TilingSpec":
        """Read-lenient constructor: defaults (or a backend's) overlaid with ``canonicalize``."""
        canonical, _ = canonicalize(mapping, legacy_px_imgsz=legacy_px_imgsz)
        return cls.from_canonical(canonical, backend=backend)

    @classmethod
    def from_canonical(
        cls, canonical: dict[str, Any], *, backend: Backend | None = None
    ) -> "TilingSpec":
        """Build from ``canonicalize`` output (avoids canonicalizing twice)."""
        base = cls.defaults(backend) if backend else cls()
        return replace(
            base, **{k: v for k, v in canonical.items() if k in _SPEC_FIELDS}
        )


# Canonical name -> accepted keys in precedence order (canonical first).
# The ONLY place legacy SAHI names are translated (spec §3.5).
SLICE_ALIASES: dict[str, tuple[str, ...]] = {
    "enabled": ("enabled", "slice_enabled"),
    "geometry_mode": ("geometry_mode", "slice_geometry_mode"),
    "reference_body_px": (
        "reference_body_px",
        "trained_body_px",
        "slice_trained_body_px",
        "measured_reference_body_px",
    ),
    "slice_width": ("slice_width",),
    "slice_height": ("slice_height",),
    "overlap": (
        "overlap",
        "slice_overlap",
        "tile_overlap",
        "overlap_width_ratio",
        "overlap_height_ratio",
    ),
    "min_area_ratio": ("min_area_ratio", "min_retained_area_frac"),
    "fragment_policy": ("fragment_policy",),
    "merge_policy": ("merge_policy", "slice_merge_policy"),
    "merge_metric": ("merge_metric", "slice_merge_metric"),
    "merge_threshold": ("merge_threshold", "slice_merge_threshold", "merge_iou"),
}
_FRACTION_SET_KEYS = ("object_tile_fractions", "target_size_fractions")
_FRACTION_SCALAR_KEYS = (
    "object_tile_fraction",
    "slice_object_tile_fraction",
    "prefill_object_tile_fraction",
    "tile_fraction",
)
_LEGACY_PX_KEY = "target_sizes"
# slice_meta._training_values' literal fallback; mirrored for bit-exact reads.
_LEGACY_READER_DEFAULT_FRACTION = 0.15


def _as_list(raw: Any) -> list[Any]:
    if raw is None or isinstance(raw, (str, bytes)):
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _fraction_list(raw: Any) -> list[float]:
    out: list[float] = []
    for item in _as_list(raw):
        value = _finite(item)
        if value is not None and 0.0 < value <= 1.0:
            out.append(value)
    return out


_WARNED: set[tuple[str, str]] = set()

_DROP = object()  # sentinel: _normalize found the value unusable


def _warn_once(name: str, value: Any, message: str, *args: Any) -> None:
    """Legacy-value warnings fire once per (field, value) per process."""
    key = (name, repr(value))
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, *args)


def _clamped(name: str, value: float, lo: float, hi: float) -> float:
    if value < lo or value > hi:
        clamped = max(lo, min(hi, value))
        _warn_once(
            name,
            value,
            "SAHI %s=%r is outside [%s, %s]; using %s",
            name,
            value,
            lo,
            hi,
            clamped,
        )
        return clamped
    return value


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, str):
        return raw.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(raw)


def _positive_int(raw: Any) -> int:
    """An image size usable as a denominator, else 0."""
    value = _finite(raw)
    if value is None or value < 1:
        return 0
    return min(8192, int(value))


def _normalize(name: str, raw: Any) -> Any:
    """Lenient per-field read. Returns _DROP when the value is unusable."""
    if name == "enabled":
        return _parse_bool(raw)
    if name in ("geometry_mode", "fragment_policy", "merge_metric", "merge_policy"):
        value = str(raw)
        if name == "merge_policy" and value == "nmm":
            return "greedy_nmm"
        allowed = {
            "geometry_mode": GEOMETRY_MODES,
            "fragment_policy": FRAGMENT_POLICIES,
            "merge_metric": MERGE_METRICS,
            "merge_policy": MERGE_POLICIES,
        }[name]
        if value not in allowed:
            _warn_once(
                name,
                value,
                "SAHI %s=%r is not one of %s; ignoring it",
                name,
                value,
                allowed,
            )
            return _DROP
        return value
    number = _finite(raw)
    if number is None:
        _warn_once(
            name, raw, "SAHI %s=%r is not a finite number; ignoring it", name, raw
        )
        return _DROP
    if name in ("slice_width", "slice_height"):
        return int(_clamped(name, float(int(number)), 0, 8192))
    if name == "reference_body_px":
        if number < 0:
            _warn_once(
                name,
                number,
                "SAHI reference_body_px=%r is negative; ignoring it",
                number,
            )
            return _DROP
        return number
    if name == "overlap":
        return _clamped(name, number, 0.0, OVERLAP_MAX)
    return _clamped(name, number, 0.0, 1.0)  # min_area_ratio, merge_threshold


def canonicalize(
    mapping: Any, *, legacy_px_imgsz: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate any SAHI mapping (config, project, plan, sidecar) to canonical keys.

    Read-lenient: out-of-range values are clamped or dropped with a warning,
    never raised. Returns ``(canonical, extras)``; ``extras`` keeps every key
    this function did not consume, unchanged.
    """
    src = dict(mapping or {})
    canonical: dict[str, Any] = {}
    consumed: set[str] = set()

    for name, aliases in SLICE_ALIASES.items():
        present = [key for key in aliases if key in src and src[key] is not None]
        consumed.update(key for key in aliases if key in src)
        if not present:
            continue
        chosen = present[0]
        value = _normalize(name, src[chosen])
        if value is _DROP:
            continue
        canonical[name] = value
        if name == "overlap" and {"overlap_width_ratio", "overlap_height_ratio"} <= set(
            present
        ):
            if src["overlap_width_ratio"] != src["overlap_height_ratio"]:
                _warn_once(
                    "overlap_axes",
                    (src["overlap_width_ratio"], src["overlap_height_ratio"]),
                    "SAHI overlap_width_ratio=%r and overlap_height_ratio=%r differ; "
                    "the canonical single overlap uses the width ratio",
                    src["overlap_width_ratio"],
                    src["overlap_height_ratio"],
                )
        if chosen == "merge_iou" and not any(
            key in src for key in SLICE_ALIASES["merge_metric"]
        ):
            canonical["merge_metric"] = "polygon_iou"

    fractions: list[float] = []
    for key in _FRACTION_SET_KEYS:
        consumed.add(key)
        fractions = _fraction_list(src.get(key))
        if fractions:
            break

    consumed.add(_LEGACY_PX_KEY)
    raw_targets = [_finite(t) for t in _as_list(src.get(_LEGACY_PX_KEY))]
    raw_targets = [t for t in raw_targets if t is not None]
    stated_imgsz = _positive_int(src.get("imgsz"))
    if not fractions and any(t > 0 for t in raw_targets):
        denominator = stated_imgsz or _finite(legacy_px_imgsz)
        if not denominator or denominator <= 0:
            raise ValueError(
                "target_sizes are pixels at a model input size; the mapping has no "
                "imgsz, so pass legacy_px_imgsz (640 for legacy YOLO) explicitly"
            )
        fractions = [
            t / denominator for t in raw_targets if 0.0 < t / denominator <= 1.0
        ]
    # operating_fraction mirrors core/inference/slice_meta._training_values
    # bit-for-bit (what TrackerKit serves today), then a stamped prefill.
    if raw_targets and stated_imgsz:
        canonical["operating_fraction"] = _clamp_fraction(
            float(np.median(np.asarray(raw_targets))) / stated_imgsz
        )

    for key in _FRACTION_SCALAR_KEYS:
        consumed.add(key)
    if not fractions:
        for key in _FRACTION_SCALAR_KEYS:
            if key not in src:
                continue
            if key == "tile_fraction":
                value = _finite(src[key])
                if value is None or value <= 0:
                    canonical.setdefault("enabled", False)
                    break
            got = _fraction_list(src[key])
            if got:
                fractions = got[:1]
                break
            if src[key] is not None:
                _warn_once(
                    key,
                    src[key],
                    "SAHI %s=%r is outside (0, 1]; ignoring it",
                    key,
                    src[key],
                )
    if fractions:
        canonical["object_tile_fractions"] = tuple(fractions)

    prefill = _finite(src.get("prefill_object_tile_fraction"))
    if "operating_fraction" not in canonical and prefill is not None and prefill > 0:
        canonical["operating_fraction"] = _clamp_fraction(prefill)
    if "operating_fraction" not in canonical and (
        "object_tile_fraction" in src or raw_targets
    ):
        bare = _finite(src.get("object_tile_fraction"))
        canonical["operating_fraction"] = (
            _LEGACY_READER_DEFAULT_FRACTION if bare is None else _clamp_fraction(bare)
        )

    extras = {key: value for key, value in src.items() if key not in consumed}
    return canonical, extras


def _positive(value: Any) -> float | None:
    parsed = _finite(value)
    return parsed if parsed is not None and parsed > 0 else None


def resolve_reference_body_px(
    *,
    override: Any = None,
    dataset_median: Any = None,
    stamped: Any = None,
    tracker_reference: Any = None,
) -> Sourced:
    """Body px that sizes tiles: override -> dataset -> stamped -> TrackerKit's own.

    ``tracker_reference`` is REFERENCE_BODY_SIZE x RESIZE_FACTOR, read only;
    this function never writes it.
    """
    for value, source in (
        (override, "override"),
        (dataset_median, "dataset"),
        (stamped, "stamped"),
        (tracker_reference, "user"),
    ):
        parsed = _positive(value)
        if parsed is not None:
            return Sourced(parsed, source)
    return Sourced(0.0, "default")


def resolve_operating_fraction(
    *,
    backend: Backend,
    profile: Any = None,
    stamped_operating: Any = None,
    stamped_fractions=(),
) -> Sourced:
    """The ONE inference scale: profile -> stamped -> backend default."""
    parsed = _positive(profile)
    if parsed is not None:
        return Sourced(_clamp_fraction(parsed), "profile")
    parsed = _positive(stamped_operating)
    if parsed is not None:
        return Sourced(_clamp_fraction(parsed), "stamped")
    stamped = operating_fraction(_fraction_list(stamped_fractions))
    if stamped is not None:
        return Sourced(stamped, "stamped")
    return Sourced(
        operating_fraction(BACKEND_DEFAULTS[backend].object_tile_fractions), "default"
    )


def resolve_object_tile_fractions(
    *, backend: Backend, user: Any = None, profile: Any = None, stamped: Any = ()
) -> Sourced:
    """The fraction SET (training): user -> profile -> stamped -> backend default."""
    for value, source in ((user, "user"), (profile, "profile"), (stamped, "stamped")):
        usable = _fraction_list(value)
        if usable:
            return Sourced(tuple(usable), source)
    return Sourced(BACKEND_DEFAULTS[backend].object_tile_fractions, "default")


def resolve_overlap(
    *, override: Any = None, saved: Any = None, fractions=()
) -> Sourced:
    """Overlap: override -> saved -> derived from the largest scale -> default.

    A SAVED overlap is never re-derived, so existing configs (TrackerKit's
    persisted 0.2) keep their exact value.
    """
    for value, source in ((override, "override"), (saved, "user")):
        parsed = _finite(value)
        if parsed is not None and 0.0 <= parsed <= OVERLAP_MAX:
            return Sourced(parsed, source)
    usable = _fraction_list(fractions)
    if usable:
        return Sourced(
            min(OVERLAP_MAX, round(max(usable) + OVERLAP_MARGIN, 6)), "derived"
        )
    return Sourced(DEFAULT_OVERLAP, "default")


def resolve_tile_size(
    spec: TilingSpec, *, imgsz: int, fraction: float | None
) -> Sourced:
    """Tile (w, h) for one scale; editable only in custom mode with explicit sizes."""
    size = tile_size_for_mode(
        geometry_mode=spec.geometry_mode,
        imgsz=int(imgsz),
        reference_body_px=float(spec.reference_body_px),
        object_tile_fraction=(
            float(fraction)
            if fraction is not None
            else BACKEND_DEFAULTS["yolo_infer"].object_tile_fractions[0]
        ),
        slice_width=spec.slice_width,
        slice_height=spec.slice_height,
    )
    explicit = spec.geometry_mode == "custom" and (
        spec.slice_width > 0 or spec.slice_height > 0
    )
    return Sourced(size, "user" if explicit else "derived")
