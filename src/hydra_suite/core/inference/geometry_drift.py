"""Train/serve geometry-drift guard and effective-geometry provenance.

WHY THIS EXISTS. A SAM3 LoRA run was built at ``object_tile_fraction =
0.055`` -- the ``training/contracts.py`` default -- while the checkpoint it
existed to be compared against had been served at 0.10: 1766 px tiles versus
971 px. The divergence entered SILENTLY, and the resulting evaluation showed
the model effect flipping sign between the two geometries, so neither
comparison supported a model claim (see
``docs/superpowers/specs/2026-09-06-sam3-training-run-and-evaluation.md``).
Nothing in ``src/`` warned, for two separate reasons:

1. the only drift guard in the tree lived inside a Qt dialog
   (``detectkit/gui/dialogs/semantic_escalation_dialog.py``), so it served one
   of four geometry paths and no headless run at all; and
2. the effective geometry's *source* was never printed anywhere, so a
   contract default and a deliberate choice looked identical in every
   artifact.

This module is the extraction of (1) plus the fix for (2). It is pure and
Qt-free so every path -- both dataset builders, the serving-side
``--sahi-profile`` resolution, and the dialog itself -- can share one guard.

**Warn, never refuse.** A deliberate re-scale is legitimate; refusing would
break valid workflows. The guard's only job is to make a silent divergence
loud. Nothing here raises, and callers are expected to log, not abort.

The verdict is a typed record, never pre-formatted prose: a GUI renders a
modal from its fields and a CLI renders a log line from the same fields,
with neither parsing the other's text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

# Float-noise equality, inherited verbatim from the dialog this guard was
# extracted from (``abs(current - sidecar) > 1e-6``). This is NOT a measured
# tolerance and must never be treated as one: it is the epsilon that keeps a
# round-tripped float from reading as a re-scale. Any caller that genuinely
# needs a *measured* band must pass its own ``tolerance`` and justify it at
# that call site.
FLOAT_NOISE_TOLERANCE = 1e-6

#: The geometry fields worth guarding on a published artifact's sidecar, in
#: report order. ``train_tile_px`` is stamped by
#: ``training/sam3_lora/publish_worker.py`` and, before this module, had no
#: reader anywhere in ``src/`` -- this guard is its first consumer, so a
#: sidecar predating the stamp simply reports NO_STAMPED for it.
GUARDED_FIELDS: tuple[str, ...] = (
    "reference_body_px",
    "object_tile_fraction",
    "train_tile_px",
)


class DriftStatus(Enum):
    """The outcomes of comparing a stamped value to an effective one."""

    #: The artifact makes no claim about this field (absent, unparseable, or
    #: a falsy 0.0). Verbatim dialog behaviour: ``if sidecar_body_px:``.
    NO_STAMPED = "no_stamped"
    #: The artifact claims a value and the caller has none (<= 0). The dialog
    #: fills its widget from the stamp; non-interactive callers only log.
    PREFILL = "prefill"
    #: Both values agree within ``tolerance``.
    MATCH = "match"
    #: Both values are present and disagree. Warn; never refuse.
    MISMATCH = "mismatch"
    #: The artifact stamps a SET of scales and the caller serves ONE of them.
    #: That is neither agreement nor divergence: the served geometry was
    #: trained on, but only a slice of what the artifact learned is in use.
    WITHIN_SET = "within_set"
    #: A value IS present and this reader cannot represent it. Distinct from
    #: NO_STAMPED on purpose: before this status a multi-scale stamp came back
    #: as "no claim", so the guard disarmed itself and every surface reported
    #: nothing to check. An unreadable stamp is a LOUD failure to guard, never
    #: a quiet absence of one.
    UNREADABLE = "unreadable"


class GeometrySource(Enum):
    """Where an effective geometry value came from.

    The 0.055 incident was undetectable precisely because this was never
    recorded: a default and a deliberate choice were indistinguishable in
    every artifact the run produced.
    """

    EXPLICIT = "explicit"
    CALIBRATION_PROFILE = "calibration_profile"
    CORPUS_DERIVED = "corpus_derived"
    CONTRACT_DEFAULT = "contract_default"
    STAMPED_TRAINING = "stamped_training"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        """A short human phrase for logs and dialogs."""
        return {
            GeometrySource.EXPLICIT: "explicit user value",
            GeometrySource.CALIBRATION_PROFILE: "calibration profile",
            GeometrySource.CORPUS_DERIVED: "corpus-derived",
            GeometrySource.CONTRACT_DEFAULT: "contract default",
            GeometrySource.STAMPED_TRAINING: "stamped training geometry",
            GeometrySource.UNKNOWN: "unknown source",
        }[self]


@dataclass(frozen=True)
class GeometryDriftVerdict:
    """One field's stamped-vs-effective comparison.

    Deliberately carries values, not prose, so the DetectKit modal and a
    headless log line can each render their own message from the same record.
    """

    field: str
    status: DriftStatus
    stamped_value: GeometryValue | None
    effective_value: GeometryValue | None
    tolerance: float = FLOAT_NOISE_TOLERANCE
    baseline_label: str | None = None

    @property
    def is_mismatch(self) -> bool:
        return self.status is DriftStatus.MISMATCH

    @property
    def should_prefill(self) -> bool:
        return self.status is DriftStatus.PREFILL


#: A geometry value is a scalar, a ``(w, h)`` pair, or a SET of pairs.
#:
#: What each writer actually writes, verified on real artifacts rather than
#: inferred: ``publish.py`` collapses a single square ``tile_px`` pair to a
#: SCALAR before the sidecar is written, so a published single-scale sidecar
#: carries ``train_tile_px: 971`` -- a number, not a pair. Both shapes are
#: accepted because the dataset-build side compares the manifest's ``[w, h]``
#: pair directly. A multi-scale artifact stamps ``train_tile_px_set``, a list
#: of pairs, which is the third shape here.
GeometryValue = float | tuple[float, float] | tuple[tuple[float, float], ...]


class _Unrepresentable:
    """Sentinel: a value IS present but this reader cannot represent it."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unrepresentable geometry value>"


UNREPRESENTABLE = _Unrepresentable()


def _as_pair_or_none(value: Any) -> tuple[float, float] | None:
    """One ``(w, h)`` pair from a 1- or 2-element numeric sequence."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return None
    try:
        parts = [float(part) for part in value]
    except (TypeError, ValueError):
        return None
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    return None


def _as_geometry_value(value: Any) -> GeometryValue | None | _Unrepresentable:
    """Coerce a stamped or effective geometry value.

    Three outcomes, and the distinction between the last two is the whole
    point: ``None`` means NO CLAIM (absent, or an unparseable scalar -- the
    dialog's verbatim behaviour), a value means a claim this reader can
    compare, and ``UNREPRESENTABLE`` means a claim it CANNOT. Returning
    ``None`` for the third case is what silently disarmed this guard for
    multi-scale artifacts.

    A 1- or 2-element numeric sequence is a tile size and becomes a
    ``(w, h)`` pair compared ELEMENT-WISE, so a non-square stamp is never
    silently collapsed to its width. A non-empty sequence whose every element
    is itself such a pair is a SCALE SET.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not len(value):
            return None
        pair = _as_pair_or_none(value)
        if pair is not None:
            return pair
        members = tuple(_as_pair_or_none(item) for item in value)
        if all(member is not None for member in members):
            return tuple(members)  # type: ignore[return-value]
        # Present, sequence-shaped, and not a tile size or a set of them.
        return UNREPRESENTABLE
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_set(value: GeometryValue) -> bool:
    return isinstance(value, tuple) and bool(value) and isinstance(value[0], tuple)


def _as_set(value: GeometryValue) -> tuple[tuple[float, float], ...]:
    """Promote any representable value to a set of pairs."""
    if _is_set(value):
        return value  # type: ignore[return-value]
    return (_as_pair(value),)  # type: ignore[arg-type]


def stamped_tile_px_set(
    meta: Mapping[str, Any] | None,
) -> tuple[tuple[float, float], ...] | None:
    """BACK-COMPAT READER for a sidecar's trained tile geometry.

    Returns every trained scale as ``(w, h)`` pairs, or ``None`` when the
    artifact makes no claim. Both already-published checkpoints carry a
    SCALAR ``train_tile_px`` (e.g. ``971``); they must keep loading, so the
    scalar and pair shapes are read first-class rather than migrated. A
    multi-scale artifact stamps ``train_tile_px_set``, which wins when
    present.
    """
    if not meta:
        return None
    for key in ("train_tile_px_set", "train_tile_px"):
        if key not in meta:
            continue
        value = _as_geometry_value(meta.get(key))
        if value is None or isinstance(value, _Unrepresentable):
            continue
        if not _is_claimed(value):
            continue
        return _as_set(value)
    return None


def stamped_object_tile_fraction(meta: Mapping[str, Any] | None) -> float | None:
    """The sidecar's tile fraction for PREFILL, scalar or multi-scale.

    A multi-scale artifact omits the bare ``object_tile_fraction`` -- a
    median under a measurement's name is how a scale set silently becomes
    "the training tile size" downstream -- and stamps the median under the
    explicitly named ``prefill_object_tile_fraction`` instead. Consumers that
    need one number for a spin box read it through here so a multi-scale
    model prefills its own median rather than a hardcoded default.
    """
    if not meta:
        return None
    for key in ("object_tile_fraction", "prefill_object_tile_fraction"):
        raw = meta.get(key)
        if isinstance(raw, bool) or raw is None:
            continue
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _as_pair(value: GeometryValue) -> tuple[float, float]:
    """Promote a scalar to a square pair so both sides compare element-wise."""
    return value if isinstance(value, tuple) else (value, value)


def _is_claimed(value: GeometryValue | None) -> bool:
    """Verbatim dialog semantics (``if sidecar_body_px:``): 0 makes no claim."""
    if value is None:
        return False
    if _is_set(value):
        return any(any(bool(part) for part in pair) for pair in value)
    return any(bool(part) for part in _as_pair(value))


def _needs_prefill(value: GeometryValue) -> bool:
    if _is_set(value):
        return any(part <= 0 for pair in value for part in pair)
    return any(part <= 0 for part in _as_pair(value))


def _format_value(value: GeometryValue | None | _Unrepresentable) -> str:
    if value is None:
        return "?"
    if isinstance(value, _Unrepresentable):
        return "<unreadable>"
    if _is_set(value):
        return "{" + ", ".join(f"{w:g}x{h:g}" for w, h in value) + "}"
    if isinstance(value, tuple):
        return f"{value[0]:g}x{value[1]:g}"
    return f"{value:g}"


def compare_geometry_value(
    field: str,
    stamped: Any,
    effective: Any,
    *,
    tolerance: float = FLOAT_NOISE_TOLERANCE,
    baseline_label: str | None = None,
) -> GeometryDriftVerdict:
    """Compare one stamped geometry value against the effective one.

    Semantics are lifted verbatim from the escalation dialog: a falsy or
    unparseable stamp is NO_STAMPED (never a mismatch-with-zero); an
    effective value of <= 0 is a PREFILL opportunity; otherwise the two are
    compared with ``tolerance``.
    """
    stamped_value = _as_geometry_value(stamped)
    effective_value = _as_geometry_value(effective)
    if isinstance(stamped_value, _Unrepresentable) or isinstance(
        effective_value, _Unrepresentable
    ):
        # A present value this reader cannot represent. NEVER NO_STAMPED: the
        # guard failing to read a stamp must look different from an artifact
        # that never made a claim, or an unguarded model ships while every
        # surface says there was nothing to check.
        return GeometryDriftVerdict(
            field,
            DriftStatus.UNREADABLE,
            stamped_value if not isinstance(stamped_value, _Unrepresentable) else None,
            (
                effective_value
                if not isinstance(effective_value, _Unrepresentable)
                else None
            ),
            tolerance,
            baseline_label,
        )
    if not _is_claimed(stamped_value):
        return GeometryDriftVerdict(
            field,
            DriftStatus.NO_STAMPED,
            stamped_value,
            effective_value,
            tolerance,
            baseline_label,
        )
    if effective_value is None:
        # The caller supplied something unreadable; make no claim rather than
        # inventing a drift out of a parse failure.
        return GeometryDriftVerdict(
            field,
            DriftStatus.NO_STAMPED,
            stamped_value,
            None,
            tolerance,
            baseline_label,
        )
    if _needs_prefill(effective_value):
        return GeometryDriftVerdict(
            field,
            DriftStatus.PREFILL,
            stamped_value,
            effective_value,
            tolerance,
            baseline_label,
        )
    if _is_set(stamped_value) or _is_set(effective_value):
        status = _compare_sets(stamped_value, effective_value, tolerance)
    else:
        status = (
            DriftStatus.MATCH
            if all(
                abs(eff - stamp) <= tolerance
                for eff, stamp in zip(
                    _as_pair(effective_value), _as_pair(stamped_value)
                )
            )
            else DriftStatus.MISMATCH
        )
    return GeometryDriftVerdict(
        field, status, stamped_value, effective_value, tolerance, baseline_label
    )


def _pairs_agree(
    left: tuple[float, float], right: tuple[float, float], tolerance: float
) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def _compare_sets(
    stamped: GeometryValue, effective: GeometryValue, tolerance: float
) -> DriftStatus:
    """Set-aware comparison. A subset relation is its OWN verdict.

    Serving one scale out of a stamped set is not a mismatch (that geometry
    WAS trained on) and not a match (the artifact learned more than is being
    used). Collapsing it into either would be a lie in one direction or the
    other, so it gets ``WITHIN_SET``.
    """
    stamped_set = _as_set(stamped)
    effective_set = _as_set(effective)
    if len(stamped_set) == len(effective_set) and all(
        any(_pairs_agree(item, other, tolerance) for other in effective_set)
        for item in stamped_set
    ):
        return DriftStatus.MATCH
    if all(
        any(_pairs_agree(item, other, tolerance) for other in stamped_set)
        for item in effective_set
    ) or all(
        any(_pairs_agree(item, other, tolerance) for other in effective_set)
        for item in stamped_set
    ):
        return DriftStatus.WITHIN_SET
    return DriftStatus.MISMATCH


def sidecar_drift_verdicts(
    stamped_meta: Mapping[str, Any] | None,
    effective: Mapping[str, Any],
    *,
    fields: Sequence[str] = GUARDED_FIELDS,
    baseline_label: str | None = None,
    tolerance: float = FLOAT_NOISE_TOLERANCE,
) -> tuple[GeometryDriftVerdict, ...]:
    """Compare a published artifact's stamped geometry to the effective one.

    Only fields the caller actually supplies an effective value for are
    compared, so a caller guards exactly what it controls. Fields the sidecar
    does not stamp come back NO_STAMPED rather than being dropped, so a
    caller can distinguish "agrees" from "was never recorded".
    """
    if not stamped_meta:
        return ()
    return tuple(
        compare_geometry_value(
            field,
            stamped_meta.get(field),
            effective[field],
            tolerance=tolerance,
            baseline_label=baseline_label,
        )
        for field in fields
        if field in effective
    )


def format_drift_warning(verdict: GeometryDriftVerdict) -> str:
    """A one-line warning for LOG callers.

    Not for the DetectKit modal: that dialog keeps its own user-facing
    wording so its behaviour is unchanged by this extraction.
    """
    where = f" ({verdict.baseline_label})" if verdict.baseline_label else ""
    if verdict.status is DriftStatus.UNREADABLE:
        return (
            f"Geometry drift guard DISABLED for {verdict.field}: the value "
            f"stamped on the reference artifact{where} could not be read as a "
            "tile size or a set of them, so this run is UNGUARDED on that "
            "field. Treat it as unverified, not as agreement."
        )
    if verdict.status is DriftStatus.WITHIN_SET:
        return (
            f"Geometry: {verdict.field} is stamped as a SET "
            f"{_format_value(verdict.stamped_value)} on the reference "
            f"artifact{where}; this run uses "
            f"{_format_value(verdict.effective_value)}, which is part of it. "
            "Not a divergence, but only a slice of the trained geometry is in "
            "use -- a comparison across the full set is not supported by it."
        )
    return (
        f"Geometry drift: {verdict.field} is stamped as "
        f"{_format_value(verdict.stamped_value)} on the reference artifact"
        f"{where}, but this run uses "
        f"{_format_value(verdict.effective_value)}. Train/serve tile scale "
        "can diverge silently when these disagree -- verify this is "
        "intentional (e.g. a deliberate re-scale) before trusting a "
        "comparison between them."
    )


def effective_geometry_log_fields(
    values: Mapping[str, Any],
    sources: Mapping[str, GeometrySource] | None = None,
) -> list[tuple[str, Any, GeometrySource]]:
    """Pair each effective geometry value with its provenance, in insertion order."""
    sources = sources or {}
    return [
        (key, value, sources.get(key, GeometrySource.UNKNOWN))
        for key, value in values.items()
    ]


def log_effective_geometry(
    logger: logging.Logger,
    context: str,
    values: Mapping[str, Any],
    sources: Mapping[str, GeometrySource] | None = None,
) -> None:
    """Log the effective geometry AND its source at INFO.

    Printed unconditionally at every dataset build and every run: the value
    alone is not enough, because the incident's 0.055 was a perfectly
    plausible-looking number.
    """
    rendered = ", ".join(
        f"{key}={value!r} [{source.label}]"
        for key, value, source in effective_geometry_log_fields(values, sources)
    )
    logger.info("%s geometry: %s", context, rendered or "(none)")


def log_drift_verdicts(
    logger: logging.Logger,
    verdicts: Sequence[GeometryDriftVerdict],
) -> None:
    """Warn on every mismatch. Never refuses, never raises."""
    for verdict in verdicts:
        if verdict.is_mismatch or verdict.status is DriftStatus.UNREADABLE:
            logger.warning("%s", format_drift_warning(verdict))
        elif verdict.status is DriftStatus.WITHIN_SET:
            logger.info("%s", format_drift_warning(verdict))
        elif verdict.should_prefill:
            # Non-interactive callers must NOT adopt the stamped value: doing
            # so would change what the run trains. Report only.
            logger.info(
                "Geometry: %s is unset for this run; the reference artifact "
                "was built at %s (not adopted).",
                verdict.field,
                _format_value(verdict.stamped_value),
            )
