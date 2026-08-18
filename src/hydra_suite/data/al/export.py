"""Three-root active-learning dataset layout.

One AL round writes up to three sibling source roots -- one per geometry level
the model can support -- each independently a valid DetectKit source:

    <round_dir>/
      polygon/  images/ labels/ classes.txt source.json   (authoritative)
      obb/      images/ labels/ classes.txt source.json   (derived)
      aabb/     images/ labels/ classes.txt source.json   (derived)

Images in derived roots are hardlinks to the authoritative root's images, so
disk cost stays at roughly 1x regardless of how many levels are written.

The whole round is staged in a sibling temporary directory and moved into place
only on success, so a failure never registers a half-written source.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from hydra_suite.utils.geometry_levels import GeometryLevel

from .escalation import LabelRecord, achievable_levels, derive_down
from .labels import write_label_file

logger = logging.getLogger(__name__)

SOURCE_KIND = "trackerkit_al"
MANIFEST_SCHEMA_VERSION = 2


@dataclass
class ExportedFrame:
    """One frame's exportable content plus its strict-label drop accounting."""

    frame_id: int
    image_name: str
    records: list[LabelRecord]
    is_context: bool = False
    drops: dict[str, int] = field(default_factory=dict)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink `src` to `dst`, falling back to a copy across devices."""
    try:
        dst.hardlink_to(src)
    except (OSError, NotImplementedError) as exc:
        logger.warning("Hardlink failed (%s); copying %s -> %s", exc, src, dst)
        shutil.copy2(src, dst)


def _write_root(
    root: Path,
    frames: Sequence[ExportedFrame],
    images: Mapping[int, np.ndarray],
    shape_cache: dict[int, tuple[int, int]],
    level: GeometryLevel,
    class_names: Sequence[str],
    provenance: dict,
    authoritative_root: Path | None,
    native_level: GeometryLevel,
    autofilled: Sequence[str] = (),
) -> dict:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        img_dst = images_dir / frame.image_name
        if authoritative_root is None:
            # Only the authoritative root touches `images`, and only once per
            # frame -- its (height, width) is cached here so derived roots
            # below never index `images` again (it may be a lazy, single-read
            # mapping backed by a video decode).
            image = images[frame.frame_id]
            cv2.imwrite(str(img_dst), image)
            shape_cache[frame.frame_id] = image.shape[:2]
        else:
            _link_or_copy(authoritative_root / "images" / frame.image_name, img_dst)

        records = derive_down(frame.records, level)
        height, width = shape_cache[frame.frame_id]
        write_label_file(
            labels_dir / (Path(frame.image_name).stem + ".txt"),
            records,
            frame_size=(height, width),
            level=level,
        )

    (root / "classes.txt").write_text("\n".join(class_names) + "\n")

    meta = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "level": level.label,
        "native_level": native_level.label,
        "authoritative": authoritative_root is None,
        "derived_from": None if authoritative_root is None else native_level.label,
        "reviewed": authoritative_root is None,
        "source_kind": SOURCE_KIND,
        "class_names": list(class_names),
        "class_names_autofilled": list(autofilled),
        "provenance": dict(provenance),
    }
    (root / "source.json").write_text(json.dumps(meta, indent=2))
    return meta


def export_al_dataset(
    *,
    round_dir: str | Path,
    frames: Sequence[ExportedFrame],
    images: Mapping[int, np.ndarray],
    native_level: GeometryLevel,
    levels: Sequence[GeometryLevel],
    class_names: Sequence[str],
    provenance: dict,
    extra_totals: Mapping[str, int] | None = None,
) -> dict:
    """Write one AL round as up to three sibling source roots.

    `images` is a Mapping[frame_id, ndarray] consulted exactly once per frame
    -- only by the authoritative root, which is written first. A plain dict
    works, but callers exporting many frames should pass a lazy mapping (one
    that decodes on `__getitem__`) so the whole export never has to be
    resident in memory at once; derived roots hardlink images and read a
    cached (height, width) instead of touching `images` again.

    `extra_totals` is merged into the manifest's `totals` block, so a caller
    can surface its own per-frame failure counters (e.g. `detection_failed`)
    in the same place as the exporter's own drop accounting.

    Returns a manifest dict describing every root written plus round-level
    totals. Raises ValueError if any requested level exceeds `native_level`,
    or if no frame carries any label geometry.
    """
    allowed = achievable_levels(native_level)
    for lvl in levels:
        if lvl not in allowed:
            raise ValueError(
                f"level {lvl.label!r} is not achievable from native level "
                f"{native_level.label!r}: upward derivation is refused"
            )

    # A frame with zero surviving records must NOT be exported. YOLO reads an
    # empty .txt as "this image contains no objects" -- a background sample.
    # For an AL round that is fabricated negative ground truth: the frame was
    # picked *because* detection struggled there, so "I could not compute
    # geometry" would be written to disk as "there is no geometry here". The
    # frame is dropped and counted instead; nothing invented, nothing hidden.
    exportable = [f for f in frames if f.records]
    skipped_no_records = [f for f in frames if not f.records]
    if not exportable:
        raise ValueError(
            f"no frame in this round carries any label geometry "
            f"({len(skipped_no_records)} of {len(frames)} frames had zero "
            "surviving detections). Exporting them would write empty label "
            "files, which YOLO reads as 'these images contain no objects' -- "
            "fabricated negative ground truth. Lower the export confidence "
            "threshold, or check that the detection model matches this video."
        )
    frames = exportable

    # Class ids come from the model; `class_names` comes from the project.
    # A multi-class checkpoint emitting id 3 into a root whose classes.txt has
    # one line produces an unreadable dataset, so reconcile the two here and
    # record the reconciliation rather than letting them silently disagree.
    max_class_id = max(
        (int(rec.class_id) for f in frames for rec in f.records), default=-1
    )
    resolved_names = list(class_names)
    autofilled: list[str] = []
    while len(resolved_names) <= max_class_id:
        placeholder = f"class_{len(resolved_names)}"
        resolved_names.append(placeholder)
        autofilled.append(placeholder)
    if autofilled:
        logger.warning(
            "Detector emitted class ids up to %d but only %d class name(s) were "
            "supplied; padded classes.txt with %s. Rename these in DetectKit.",
            max_class_id,
            len(class_names),
            ", ".join(autofilled),
        )
    class_names = resolved_names

    round_path = Path(round_dir)
    staging = round_path.parent / (round_path.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # Highest level first, so the authoritative root exists before any derived
    # root tries to hardlink its images.
    ordered = sorted(set(levels), reverse=True)
    roots: list[dict] = []
    shape_cache: dict[int, tuple[int, int]] = {}
    try:
        authoritative_root: Path | None = None
        for lvl in ordered:
            root = staging / lvl.label
            meta = _write_root(
                root,
                frames,
                images,
                shape_cache,
                lvl,
                class_names,
                provenance,
                authoritative_root,
                native_level,
                autofilled,
            )
            meta["path"] = str(round_path / lvl.label)
            roots.append(meta)
            if authoritative_root is None:
                authoritative_root = root

        totals = {
            "frames_exported": len(frames),
            "frames_skipped_no_records": len(skipped_no_records),
            "dropped_lost": sum(
                int(f.drops.get("lost", 0)) for f in frames + skipped_no_records
            ),
            "dropped_unmatched": sum(
                int(f.drops.get("unmatched", 0)) for f in frames + skipped_no_records
            ),
            "objects": sum(len(f.records) for f in frames),
        }
        if extra_totals:
            totals.update({str(k): int(v) for k, v in extra_totals.items()})
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "round_dir": str(round_path),
            "native_level": native_level.label,
            "roots": roots,
            "totals": totals,
            "selected_frame_ids": [f.frame_id for f in frames if not f.is_context],
            "context_frame_ids": [f.frame_id for f in frames if f.is_context],
            "skipped_frame_ids_no_records": [f.frame_id for f in skipped_no_records],
            "class_names": list(class_names),
            "class_names_autofilled": autofilled,
            "provenance": dict(provenance),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    staging.rename(round_path)
    return manifest
