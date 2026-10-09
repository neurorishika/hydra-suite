"""Canonical SAHI tiling contract shared by every kit.

Spec: docs/superpowers/specs/2026-10-09-sahi-unification-design.md (§3).
TrackerKit's vocabulary is canonical. This module is pure (numpy only) so
core, training, data and every kit can import it; ``slice_geometry`` stays
the grid module and is not modified.
"""

from __future__ import annotations

import logging
import math
import numbers
import operator
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, NamedTuple

import numpy as np

from .slice_geometry import (
    DEFAULT_MIN_AREA_RATIO,
    LEGACY_TARGET_SIZE_IMGSZ,
    resolve_scales,
)

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
# every animal whole inside at least one tile. That holds for the REALIZED
# (rounded) tile while body/fraction lies inside the planner's [64, 4096] px
# tile clamp; outside it the clamped tile no longer scales with the body.
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
    # not transfer between models). SAM2 owner tiles do not merge, so the
    # fragment/merge entries below are inert placeholders, not behaviour.
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
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, np.ndarray) and value.ndim:
        return None  # an array is not one number (and float() of it is deprecated)
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):  # float(10**400) overflows
        return None
    return parsed if math.isfinite(parsed) else None


def _positive(value: Any) -> float | None:
    parsed = _finite(value)
    return parsed if parsed is not None and parsed > 0 else None


def _strict_real(value: Any) -> float | None:
    """A real number (int/float/numpy, never bool or str) as a finite float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return None
    parsed = _finite(value)
    return parsed


def _strict_index(value: Any) -> int | None:
    """An integral value (int or numpy integer; never bool, float or str)."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


def _strict_fractions(raw: Any) -> tuple[float, ...] | None:
    """An iterable of real numbers as a float tuple; None when it is not one."""
    if isinstance(raw, (str, bytes)):
        return None
    try:
        items = list(raw)
    except TypeError:
        return None
    out: list[float] = []
    for item in items:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real):
            return None
        out.append(float(item))
    return tuple(out)


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
        # Write-strict: wrong TYPES are a ValueError (never a TypeError) and
        # numpy scalars are coerced to Python so to_mapping() is JSON-safe.
        problems: list[str] = []
        enabled = self.enabled
        if isinstance(enabled, (bool, np.bool_)):
            object.__setattr__(self, "enabled", bool(enabled))
        else:
            problems.append(f"enabled {enabled!r} is not a bool")
        fractions = _strict_fractions(self.object_tile_fractions)
        if fractions is None:
            problems.append(
                f"object_tile_fractions {self.object_tile_fractions!r} is not "
                "an iterable of numbers"
            )
            fractions = ()
        object.__setattr__(self, "object_tile_fractions", fractions)
        if any(not (math.isfinite(f) and 0.0 < f <= 1.0) for f in fractions):
            problems.append(f"object_tile_fractions {fractions!r} outside (0, 1]")
        if not isinstance(self.geometry_mode, str) or (
            self.geometry_mode not in GEOMETRY_MODES
        ):
            problems.append(f"geometry_mode {self.geometry_mode!r}")
        for name, lo, hi in (
            ("reference_body_px", 0.0, math.inf),
            ("overlap", 0.0, OVERLAP_MAX),
            ("min_area_ratio", 0.0, 1.0),
            ("merge_threshold", 0.0, 1.0),
        ):
            raw = getattr(self, name)
            if name == "overlap" and raw is None:
                continue  # unset -> resolve_overlap derives it
            value = _strict_real(raw)
            if value is None or not lo <= value <= hi:
                problems.append(f"{name} {raw!r} outside [{lo}, {hi}]")
            else:
                object.__setattr__(self, name, value)
        for name in ("slice_width", "slice_height"):
            raw = getattr(self, name)
            value = _strict_index(raw)
            if value is None or not 0 <= value <= 8192:
                problems.append(f"{name} {raw!r} outside [0, 8192]")
            else:
                object.__setattr__(self, name, value)
        for name, allowed in (
            ("fragment_policy", FRAGMENT_POLICIES),
            ("merge_policy", MERGE_POLICIES),
            ("merge_metric", MERGE_METRICS),
        ):
            raw = getattr(self, name)
            if not isinstance(raw, str) or raw not in allowed:
                problems.append(f"{name} {raw!r}")
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
_WARN_CAP = 1024  # bounded: cleared (not frozen) when full, so warnings never stop
_WARN_KEY_CHARS = 200

_DROP = object()  # sentinel: _normalize found the value unusable


def reset_warnings() -> None:
    """Forget which legacy values were already warned about (tests, long sessions)."""
    _WARNED.clear()


def _warn_once(name: str, value: Any, message: str, *args: Any) -> None:
    """Legacy-value warnings fire once per (field, value) per process."""
    key = (name, repr(value)[:_WARN_KEY_CHARS])
    if key in _WARNED:
        return
    if len(_WARNED) >= _WARN_CAP:
        _WARNED.clear()
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


def _tile_px(raw: Any) -> int | None:
    """``tile_px`` as one square side: a positive scalar or a square ``[w, h]``."""
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            return None
        width, height = _finite(raw[0]), _finite(raw[1])
        if width is None or width != height:
            return None
        raw = width
    value = _finite(raw)
    if value is None or value < 1:
        return None
    return min(8192, int(value))


def canonicalize(
    mapping: Any, *, legacy_px_imgsz: float | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate any SAHI mapping (config, project, plan, sidecar) to canonical keys.

    Read-lenient: out-of-range values are clamped or dropped with a warning,
    never raised. Keys whose value is ``None`` are treated as absent (they
    never produce a canonical value); a non-mapping input (list, str, ...)
    reads as empty, ``({}, {})``. Returns ``(canonical, extras)``; ``extras``
    keeps every key this function did not consume, unchanged.

    Legacy ``target_sizes`` are pixels at some model input size. They are
    divided by the mapping's own ``imgsz`` or by ``legacy_px_imgsz`` (pass
    ``slice_geometry.LEGACY_TARGET_SIZE_IMGSZ`` for legacy YOLO); with
    neither, they are ignored with a warning, never rescaled by a guess.
    """
    if not isinstance(mapping, Mapping):
        return {}, {}
    src = dict(mapping)
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
            if _finite(src["overlap_width_ratio"]) != _finite(
                src["overlap_height_ratio"]
            ):
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

    # Escalation/calibration ``tile_px``: an explicit square tile, i.e. custom
    # geometry, unless the mapping already names its own size or mode.
    consumed.add("tile_px")
    if src.get("tile_px") is not None:
        side = _tile_px(src["tile_px"])
        if side is None:
            if _finite(src["tile_px"]) != 0:  # 0 = "no tile size", silently
                _warn_once(
                    "tile_px",
                    src["tile_px"],
                    "SAHI tile_px=%r is not a positive square size; ignoring it",
                    src["tile_px"],
                )
        elif "slice_width" not in canonical and "slice_height" not in canonical:
            canonical["slice_width"] = canonical["slice_height"] = side
            canonical.setdefault("geometry_mode", "custom")

    consumed.update(_FRACTION_SET_KEYS)
    fractions: list[float] = []
    for key in _FRACTION_SET_KEYS:
        fractions = _fraction_list(src.get(key))
        if fractions:
            break
    from_set = bool(fractions)

    # Not slice_geometry.target_fractions_from: that raises on a bad
    # denominator and keeps fractions > 1; a read here must never raise and
    # emits only (0, 1].
    consumed.add(_LEGACY_PX_KEY)
    raw_targets = [_finite(t) for t in _as_list(src.get(_LEGACY_PX_KEY))]
    raw_targets = [t for t in raw_targets if t is not None]
    stated_imgsz = _positive_int(src.get("imgsz"))
    legacy_targets_unread = False
    if not fractions and any(t > 0 for t in raw_targets):
        denominator = stated_imgsz or _positive(legacy_px_imgsz)
        if denominator:
            fractions = [
                t / denominator for t in raw_targets if 0.0 < t / denominator <= 1.0
            ]
        else:
            legacy_targets_unread = True
            _warn_once(
                _LEGACY_PX_KEY,
                src.get(_LEGACY_PX_KEY),
                "SAHI target_sizes=%r are pixels at an unstated model input size "
                "(no imgsz; legacy YOLO used LEGACY_TARGET_SIZE_IMGSZ=%s); "
                "ignoring them rather than rescaling by a guess",
                src.get(_LEGACY_PX_KEY),
                LEGACY_TARGET_SIZE_IMGSZ,
            )

    consumed.update(_FRACTION_SCALAR_KEYS)
    if "tile_fraction" in src:
        # Escalation requests: None/0 = full frame, a positive value = tiled.
        requested = _finite(src["tile_fraction"])
        canonical.setdefault("enabled", requested is not None and requested > 0)
    if not fractions and not legacy_targets_unread:
        for key in _FRACTION_SCALAR_KEYS:
            if key not in src or src[key] is None:
                continue
            got = _fraction_list(src[key])
            if got:
                fractions = got[:1]
                break
            if not (key == "tile_fraction" and _finite(src[key]) == 0):
                _warn_once(
                    key,
                    src[key],
                    "SAHI %s=%r is outside (0, 1]; ignoring it",
                    key,
                    src[key],
                )
    if fractions:
        canonical["object_tile_fractions"] = tuple(fractions)

    # operating_fraction ladder. (a) mirrors core/inference/slice_meta.
    # _training_values bit-for-bit (what TrackerKit serves today); (b) a stamped
    # prefill; (c) the median of a stamped SET; (d) the bare legacy scalar.
    if raw_targets and stated_imgsz:
        canonical["operating_fraction"] = _clamp_fraction(
            float(np.median(np.asarray(raw_targets))) / stated_imgsz
        )
    prefill = _positive(src.get("prefill_object_tile_fraction"))
    if "operating_fraction" not in canonical and prefill is not None:
        canonical["operating_fraction"] = _clamp_fraction(prefill)
    if "operating_fraction" not in canonical and from_set:
        canonical["operating_fraction"] = operating_fraction(fractions)
    bare_key = next(
        (k for k in ("object_tile_fraction", "slice_object_tile_fraction") if k in src),
        None,
    )
    if "operating_fraction" not in canonical and (bare_key or raw_targets):
        bare = _finite(src.get(bare_key)) if bare_key else None
        canonical["operating_fraction"] = (
            _LEGACY_READER_DEFAULT_FRACTION if bare is None else _clamp_fraction(bare)
        )

    extras = {key: value for key, value in src.items() if key not in consumed}
    return canonical, extras


# Resolution (``resolve_*``) lives in ``tiling_resolve`` to keep this module
# under the size guideline; re-exported so ``from hydra_suite.utils.tiling_spec
# import resolve_overlap`` keeps working. Imported last: both modules bind each
# other's names only after their own definitions, so either import order works.
from .tiling_resolve import (  # noqa: E402, F401  isort: skip
    resolve_object_tile_fractions,
    resolve_operating_fraction,
    resolve_overlap,
    resolve_reference_body_px,
    resolve_tile_size,
)
