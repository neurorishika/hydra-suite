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
    """The four outcomes of comparing a stamped value to an effective one."""

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


#: A geometry value is either a scalar or a ``(w, h)`` pair. Tile sizes are
#: stamped as pairs (``publish_worker.py`` copies the build manifest's
#: ``tile_px = [w, h]``), so a scalar-only reader would silently report
#: NO_STAMPED for every real sidecar -- the reader must parse what the writer
#: writes.
GeometryValue = float | tuple[float, float]


def _as_geometry_value(value: Any) -> GeometryValue | None:
    """Coerce a stamped or effective geometry value; None means "no claim".

    Scalars coerce like the dialog did (unparseable -> no claim). A 1- or
    2-element sequence is a tile size: it becomes a ``(w, h)`` pair, which is
    compared ELEMENT-WISE. A non-square stamp is therefore never silently
    collapsed to its width -- if a caller serves a square tile where the
    artifact was trained on a non-square one, that is a real divergence and
    must read as one.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            parts = [float(part) for part in value]
        except (TypeError, ValueError):
            return None
        if len(parts) == 1:
            return (parts[0], parts[0])
        if len(parts) == 2:
            return (parts[0], parts[1])
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_pair(value: GeometryValue) -> tuple[float, float]:
    """Promote a scalar to a square pair so both sides compare element-wise."""
    return value if isinstance(value, tuple) else (value, value)


def _is_claimed(value: GeometryValue | None) -> bool:
    """Verbatim dialog semantics (``if sidecar_body_px:``): 0 makes no claim."""
    if value is None:
        return False
    return any(bool(part) for part in _as_pair(value))


def _needs_prefill(value: GeometryValue) -> bool:
    return any(part <= 0 for part in _as_pair(value))


def _format_value(value: GeometryValue | None) -> str:
    if value is None:
        return "?"
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
    status = (
        DriftStatus.MATCH
        if all(
            abs(eff - stamp) <= tolerance
            for eff, stamp in zip(_as_pair(effective_value), _as_pair(stamped_value))
        )
        else DriftStatus.MISMATCH
    )
    return GeometryDriftVerdict(
        field, status, stamped_value, effective_value, tolerance, baseline_label
    )


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
        if verdict.is_mismatch:
            logger.warning("%s", format_drift_warning(verdict))
        elif verdict.should_prefill:
            # Non-interactive callers must NOT adopt the stamped value: doing
            # so would change what the run trains. Report only.
            logger.info(
                "Geometry: %s is unset for this run; the reference artifact "
                "was built at %s (not adopted).",
                verdict.field,
                _format_value(verdict.stamped_value),
            )
