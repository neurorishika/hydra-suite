"""Sourced SAHI resolution: which value wins, and where it came from.

Split out of ``tiling_spec`` (the contract) to keep both modules under the
size guideline; every name here is re-exported by ``tiling_spec``. Each
function returns a :class:`~hydra_suite.utils.tiling_spec.Sourced` so a UI
can badge the value's origin.
"""

from __future__ import annotations

from typing import Any

from .slice_geometry import tile_size_for_mode


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


def _fraction(name: str, raw: Any) -> float | None:
    """One fraction under the SAME rule ``canonicalize`` uses.

    Present iff finite and in (0, 1]; 0 / None / unparseable are absent (0 is
    the escalation "full frame" sentinel, spec §3.5), > 1 or < 0 is absent
    with a warning. A present value outside the planner's
    [FRACTION_MIN, FRACTION_MAX] is clamped with a warning.
    """
    value = _finite(raw)
    if value is None or value == 0.0:
        return None
    if not 0.0 < value <= 1.0:
        _warn_once(
            name, value, "SAHI %s=%r is outside (0, 1]; ignoring it", name, value
        )
        return None
    return _clamped(name, value, FRACTION_MIN, FRACTION_MAX)


def _clamped_fraction_list(name: str, raw: Any) -> list[float]:
    """Members kept under :func:`_fraction`'s rule, in order."""
    kept = (_fraction(name, item) for item in _as_list(raw))
    return [value for value in kept if value is not None]


def resolve_operating_fraction(
    *,
    backend: Backend,
    profile: Any = None,
    stamped_operating: Any = None,
    stamped_fractions=(),
) -> Sourced:
    """The ONE inference scale: profile -> stamped -> backend default.

    Values follow :func:`_fraction` (0 means absent, never a 0.01 tile).
    """
    for value, source, name in (
        (profile, "profile", "profile object_tile_fraction"),
        (stamped_operating, "stamped", "stamped object_tile_fraction"),
    ):
        parsed = _fraction(name, value)
        if parsed is not None:
            return Sourced(parsed, source)
    stamped = operating_fraction(
        _clamped_fraction_list("stamped object_tile_fractions", stamped_fractions)
    )
    if stamped is not None:
        return Sourced(stamped, "stamped")
    return Sourced(
        operating_fraction(BACKEND_DEFAULTS[backend].object_tile_fractions), "default"
    )


def resolve_object_tile_fractions(
    *, backend: Backend, user: Any = None, profile: Any = None, stamped: Any = ()
) -> Sourced:
    """The fraction SET (training): user -> profile -> stamped -> backend default.

    Members follow :func:`_fraction` (same keep/drop rule as ``canonicalize``).
    """
    for value, source in ((user, "user"), (profile, "profile"), (stamped, "stamped")):
        usable = _clamped_fraction_list(f"{source} object_tile_fractions", value)
        if usable:
            return Sourced(tuple(usable), source)
    return Sourced(BACKEND_DEFAULTS[backend].object_tile_fractions, "default")


def resolve_overlap(
    *, override: Any = None, saved: Any = None, fractions=()
) -> Sourced:
    """Overlap: override -> saved -> derived from the largest scale -> default.

    A SAVED overlap is never re-derived, so existing configs (TrackerKit's
    persisted 0.2) keep their exact value; a finite out-of-range one is
    clamped into [0, OVERLAP_MAX] with a warning, an unparseable one skipped.
    The derived overlap (largest fraction + OVERLAP_MARGIN) keeps every
    animal whole in some tile while body/fraction lies inside the planner's
    [64, 4096] px tile clamp.
    """
    for value, source in ((override, "override"), (saved, "user")):
        parsed = _finite(value)
        if parsed is not None:
            return Sourced(
                _clamped(f"{source} overlap", parsed, 0.0, OVERLAP_MAX), source
            )
    usable = _clamped_fraction_list("object_tile_fractions", fractions)
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


# Bound last (see the matching import at the bottom of ``tiling_spec``): the
# functions above only use these names at call time, so either module can be
# imported first without a partially-initialised-module ImportError.
# importlib.reload(tiling_spec) must be followed by reload(tiling_resolve),
# or this module keeps the stale classes.
from .tiling_spec import (  # noqa: E402  isort: skip
    BACKEND_DEFAULTS,
    DEFAULT_OVERLAP,
    FRACTION_MAX,
    FRACTION_MIN,
    OVERLAP_MARGIN,
    OVERLAP_MAX,
    Backend,
    Sourced,
    TilingSpec,
    _as_list,
    _clamped,
    _finite,
    _positive,
    _warn_once,
    operating_fraction,
)
