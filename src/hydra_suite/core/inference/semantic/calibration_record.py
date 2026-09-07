"""D10 -- where a semantic calibration result is persisted: the MODEL SIDECAR.

A calibration result answers "at what operating point should THIS model be
served on THIS corpus". Until now it lived only in the DetectKit project
JSON, which meant a headless SAM3 serving run could not see it: the model
travelled, the calibration did not. Moving it onto the sidecar is what
unlocks headless serving parity (spec D10).

**The constraint that shapes every line here.**
``semantic/sam3.py:_sidecar_for_checkpoint`` REFUSES to serve on a sidecar it
cannot parse or that is missing ``imgsz``. Two real published checkpoints in
the wild carry a scalar ``train_tile_px: 971``. A shape change without a
back-compat reader would therefore be a hard outage for every published
model, so:

* the writer is strictly ADDITIVE read-modify-write -- it adds one key and
  rewrites nothing, so a sidecar keeps every field serving validates on;
* the reader treats an absent block as "no claim" (never an error), so an
  existing sidecar loads and serves unchanged;
* nothing on disk is ever backfilled or migrated. New persistence applies
  going forward only.

**Shape convention.** This reuses the ``scale_grouped_batching`` precedent
already on the sidecar: ONE named dict key holding scalar/plain-JSON fields,
rather than a new top-level family of keys. Nothing new is invented.

**Scope difference, stated because it is real.** The sidecar block is
PER MODEL; the DetectKit project's ``semantic_calibration`` is PER PROJECT.
Two projects calibrating the same published model are last-write-wins on the
sidecar. The project copy is therefore still written exactly as before -- it
is the per-project record and the legacy read path -- and
``resolve_serving_calibration`` reports which one it handed back rather than
silently merging them.

Qt-free, dependency-free (stdlib only) so a slim SAM3 sidecar env can import
it: see ``tests/test_core_import_is_light.py``.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: The single named block, following the ``scale_grouped_batching`` shape.
SERVING_CALIBRATION_KEY = "serving_calibration"

#: What a calibration record carries onto a sidecar, in write order.
#:
#: ``preview_artifact`` is DELIBERATELY absent. The dialog stores a
#: project-relative preview path; a sidecar is a portable per-model artifact
#: and a project-local path would dangle on any other machine. Previews stay
#: in the project.
SIDECAR_RECORD_FIELDS: tuple[str, ...] = (
    "created_at",
    "variant",
    "prompt",
    "source_names",
    "parameters",
    "reason",
    "recommended_index",
    "points",
)

#: Refuse to grow a sidecar without bound; the same discipline as
#: ``publish.py``'s ``MAX_SIDECAR_BYTES``.
MAX_RECORD_BYTES = 4 * 1024 * 1024


class CalibrationOrigin(Enum):
    """Where a resolved calibration actually came from.

    ``PROJECT_LEGACY`` is reported, never migrated: rewriting a user's
    project file behind their back is exactly the silent mutation this
    module exists to avoid.
    """

    SIDECAR = "sidecar"
    PROJECT_LEGACY = "project_legacy"
    ABSENT = "absent"

    @property
    def label(self) -> str:
        return {
            CalibrationOrigin.SIDECAR: "model sidecar",
            CalibrationOrigin.PROJECT_LEGACY: (
                "DetectKit project JSON (legacy-only: this calibration is not "
                "on the model sidecar, so a headless run cannot see it)"
            ),
            CalibrationOrigin.ABSENT: "no calibration recorded",
        }[self]


def calibration_record_for_sidecar(saved: Mapping[str, Any]) -> dict[str, Any]:
    """Project the dialog's saved calibration onto the sidecar's shape.

    Pure: the input mapping is never mutated. Fields absent from *saved* are
    simply omitted rather than defaulted, so the block never invents a claim.
    """
    return {field: saved[field] for field in SIDECAR_RECORD_FIELDS if field in saved}


def serving_calibration(meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """BACK-COMPAT READER. The sidecar's calibration block, or ``None``.

    ``None`` means the artifact makes NO CLAIM -- which is what every
    already-published sidecar returns, and is never an error. A block that is
    present but not a well-formed record is also ``None`` here rather than an
    exception: this reader must never be the thing that stops a model
    serving.
    """
    if not meta:
        return None
    block = meta.get(SERVING_CALIBRATION_KEY)
    if not isinstance(block, dict) or not block:
        return None
    points = block.get("points")
    if points is not None and not isinstance(points, list):
        return None
    return dict(block)


def resolve_serving_calibration(
    sidecar_meta: Mapping[str, Any] | None,
    project_calibration: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, CalibrationOrigin]:
    """The calibration in force, plus an honest statement of where it lives.

    Sidecar first (portable, visible headlessly), project JSON second. A
    project-only calibration KEEPS WORKING and is handed back verbatim,
    tagged ``PROJECT_LEGACY`` so a caller can say so; it is NOT copied onto
    the sidecar here, because a read must not mutate anything.
    """
    record = serving_calibration(sidecar_meta)
    if record is not None:
        return record, CalibrationOrigin.SIDECAR
    if project_calibration:
        # Handed back VERBATIM -- the same object, not a normalised copy --
        # so nothing about the user's project record is silently reshaped on
        # the way out.
        return project_calibration, CalibrationOrigin.PROJECT_LEGACY  # type: ignore[return-value]
    return None, CalibrationOrigin.ABSENT


def _write_json_atomic(path: Path, payload: Any) -> None:
    """tmp + ``os.replace``, mirroring ``publish.py:_write_json_atomic``."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_serving_calibration(
    sidecar_path: Path | str | None,
    saved: Mapping[str, Any],
) -> bool:
    """ADDITIVE read-modify-write of the calibration block onto a sidecar.

    Returns ``True`` when the block was written. Returns ``False`` -- never
    raises, never creates, never truncates -- when there is no sidecar, it
    cannot be parsed, or the record is implausibly large. Refusing to touch a
    sidecar this function cannot fully understand is the whole safety
    argument: a partial rewrite would be an outage, because SAM3 refuses to
    serve on a malformed sidecar.
    """
    if not sidecar_path:
        return False
    path = Path(sidecar_path)
    if not path.is_file():
        logger.info(
            "Semantic calibration not persisted to a sidecar: %s does not "
            "exist (a stock variant ships none). The project's own copy is "
            "unaffected.",
            path,
        )
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Refusing to write calibration onto %s: the existing sidecar "
            "could not be parsed (%s). It is left exactly as found.",
            path,
            exc,
        )
        return False
    if not isinstance(meta, dict):
        logger.warning(
            "Refusing to write calibration onto %s: the sidecar is not a "
            "JSON object.",
            path,
        )
        return False
    record = calibration_record_for_sidecar(saved)
    try:
        encoded = json.dumps(record)
    except (TypeError, ValueError) as exc:
        logger.warning("Calibration record is not JSON-serialisable (%s).", exc)
        return False
    if len(encoded.encode("utf-8")) > MAX_RECORD_BYTES:
        logger.warning(
            "Calibration record (%d bytes) exceeds the sidecar budget; not " "written.",
            len(encoded),
        )
        return False
    # Additive: every pre-existing key survives byte-for-byte in value.
    meta[SERVING_CALIBRATION_KEY] = record
    _write_json_atomic(path, meta)
    logger.info("Persisted semantic calibration onto the model sidecar %s.", path)
    return True
