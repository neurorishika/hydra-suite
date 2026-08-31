"""Frame-granular review of a StagedReview, applied immediately.

One accept path for every producer. A producer's only job is to fill a
staging directory's ``labels/`` + ``classes.txt`` + ``run.json``; everything
here is producer-agnostic, which is the whole point of the StagedReview
refactor.

Decisions live in the STAGING directory, not the project JSON: a 10k-frame
source would otherwise add 10k entries to every project save, and the
staging directory is already the object whose lifetime matches the
review's.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import cv2

from hydra_suite.data.al.escalation import LabelRecord, derive_down
from hydra_suite.data.al.labels import read_label_file, write_label_file
from hydra_suite.data.al.merge import MergeMode, merge_records
from hydra_suite.detectkit.gui.constants import IMG_EXTS
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.utils.geometry_levels import GeometryLevel

logger = logging.getLogger(__name__)

DEFAULT_MERGE_IOU = 0.5

ACCEPTED_OVERWRITE = "accepted_overwrite"
ACCEPTED_ADD_NEW = "accepted_add_new"
REJECTED = "rejected"

DECISIONS_FILE = "decisions.json"
SNAPSHOT_DIR = "labels_before"
SNAPSHOT_STATE = "state_before.json"


def read_decisions(staged_root: str | Path) -> dict[str, str]:
    """Per-frame outcomes recorded so far. Absent or corrupt reads as empty."""
    path = Path(staged_root) / DECISIONS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def write_decisions(staged_root: str | Path, decisions: dict[str, str]) -> None:
    """Persist per-frame outcomes, overwriting the file."""
    path = Path(staged_root) / DECISIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, indent=2, sort_keys=True), encoding="utf-8")


def staged_frames(staged_root: str | Path) -> list[str]:
    """Every staged frame, as sorted POSIX paths relative to ``labels/``.

    POSIX and relative because they are the review's keys: they index
    decisions.json, which round-trips through JSON on every platform, and
    they mirror the source's images/ tree exactly (the same key
    `_origin_image_for` and `find_staged_label_for_image` already rely on).
    """
    labels = Path(staged_root) / "labels"
    if not labels.is_dir():
        return []
    return sorted(p.relative_to(labels).as_posix() for p in labels.rglob("*.txt"))


def review_key_for_image(source_path: str | Path, image_path: str | Path) -> str | None:
    """The review key for a frame: its path under images/, suffixed .txt.

    THE one definition. The staged label mirrors the image's images-relative
    path, so this string indexes decisions.json, names the staged label, and
    names the source label -- all three. Computing it anywhere else from a
    label path instead would drift the moment
    `find_staged_label_for_image`'s stem or recursive fallback fires.

    Returns None when the image is not under the source's images/ at all,
    which the callers treat as "this frame is not part of the review".
    """
    try:
        rel = Path(image_path).relative_to(Path(source_path) / "images")
    except ValueError:
        return None
    return rel.with_suffix(".txt").as_posix()


def review_progress(staged_root: str | Path) -> tuple[int, int]:
    """(decided, total) for the progress counter."""
    frames = staged_frames(staged_root)
    decided = read_decisions(staged_root)
    return sum(1 for f in frames if f in decided), len(frames)


def ensure_snapshot(source: OBBSource, staged_root: str | Path) -> None:
    """Snapshot the source's pre-review state, once, before the first accept.

    Captures ``labels/`` AND the two other things an accept can change: the
    source's ``level`` (a promoting accept rewrites it) and ``classes.txt``
    (accepting staged classes can extend it). Restoring labels alone would
    leave a reverted source claiming a level its labels no longer have.

    Idempotent by the existence of the snapshot directory: the second and
    later accepts must NOT re-snapshot, or the snapshot would drift forward
    to whatever the last accept produced and revert would be a no-op.
    """
    root = Path(staged_root)
    snapshot = root / SNAPSHOT_DIR
    if snapshot.exists():
        return

    source_root = Path(source.path)
    source_labels = source_root / "labels"
    if source_labels.is_dir():
        shutil.copytree(source_labels, snapshot)
    else:
        snapshot.mkdir(parents=True)

    classes = source_root / "classes.txt"
    (root / SNAPSHOT_STATE).write_text(
        json.dumps(
            {
                "level": source.level,
                "classes_txt": classes.read_text() if classes.is_file() else "",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def revert_review(source: OBBSource, staged_root: str | Path) -> None:
    """Restore the source to its pre-review state and clear every decision.

    Available only while the review is open: completing a review deletes the
    staging directory, and the snapshot with it. The review bar says so.
    """
    root = Path(staged_root)
    snapshot = root / SNAPSHOT_DIR
    if not snapshot.is_dir():
        raise RuntimeError(
            "There is no snapshot to revert to -- no frame of this review has "
            "been accepted yet, so the source is already in its original state."
        )

    source_root = Path(source.path)
    source_labels = source_root / "labels"
    # rmtree BEFORE copytree, not ignore_errors: a half-deleted labels/ makes
    # copytree raise FileExistsError and wedges the source. Raising here
    # leaves it untouched instead.
    if source_labels.exists():
        shutil.rmtree(source_labels)
    shutil.copytree(snapshot, source_labels)

    try:
        state = json.loads((root / SNAPSHOT_STATE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    if state.get("level"):
        source.level = str(state["level"])
    # `in`, not truthiness: a source whose classes.txt was absent or empty
    # before the review must NOT keep the names resolve_staged_class_ids
    # appended. "Restore the class list" has to mean restore, including to
    # nothing.
    if "classes_txt" in state:
        (source_root / "classes.txt").write_text(str(state["classes_txt"]))

    write_decisions(root, {})


def _read_names(path: Path) -> list[str]:
    """Read class names from classes.txt, degrading to [] if absent."""
    try:
        return [
            ln.strip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    except OSError:
        return []


def resolve_staged_class_ids(
    source: OBBSource,
    staged_root: str | Path,
) -> dict[int, int]:
    """Map staged class ids onto the source's, extending it when needed.

    Frame-granular ADD_NEW puts two class-id spaces in one label file for
    the first time: staged ids index the STAGING dir's classes.txt (SAM3's
    are its prompt's, all class 0), source ids index the source's. The old
    wholesale accept dodged this by copying classes.txt over the source's,
    which is not available when only some frames are accepted.

    Matching is BY NAME. A staged name the source does not have is APPENDED
    to the source's classes.txt and takes the new id. Appending -- rather
    than merging and re-sorting -- is what keeps every label already on disk
    valid: no existing id is ever renumbered.

    Idempotent: running it twice appends nothing the second time.
    """
    source_root = Path(source.path)
    classes_path = source_root / "classes.txt"
    source_names = _read_names(classes_path)
    staged_names = _read_names(Path(staged_root) / "classes.txt") or ["object"]

    mapping: dict[int, int] = {}
    appended = False
    for staged_id, name in enumerate(staged_names):
        if name not in source_names:
            source_names.append(name)
            appended = True
        mapping[staged_id] = source_names.index(name)

    if appended:
        classes_path.write_text("\n".join(source_names) + "\n", encoding="utf-8")
    return mapping


def _lift(records: list[LabelRecord], target: GeometryLevel) -> list[LabelRecord]:
    """Re-tag records to `target` WITHOUT moving a point.

    An OBB quad is a valid 4-point polygon; `_polygon_points` encodes it as
    one by repeating the final vertex, so lifting is purely a change of
    declared level. `derive_down` cannot express this -- it refuses upward
    derivation, correctly, because for a genuine level gap upward derivation
    would invent information. Here there is no gap to invent across: the
    points are already there.

    Records at or above `target` are returned unchanged.
    """
    out: list[LabelRecord] = []
    for rec in records:
        if rec.level >= target:
            out.append(rec)
            continue
        out.append(
            LabelRecord(
                class_id=rec.class_id,
                confidence=rec.confidence,
                points=rec.points,
                level=target,
            )
        )
    return out


def _review_of(source: OBBSource):
    review = source.staged_review
    if review is None:
        raise ValueError(f"Source '{source.name}' has no staged review.")
    return review


def _level_of(value: str, fallback: GeometryLevel) -> GeometryLevel:
    """Parse a level string from project JSON, degrading rather than raising.

    Both `OBBSource.level` and `StagedReview.target_level` are unvalidated
    strings loaded from disk, exactly as `resolve_pending_level` in the
    overlay providers already treats them.
    """
    try:
        return GeometryLevel.from_str(value)
    except ValueError:
        logger.warning(
            "Unknown geometry level %r; treating as %s", value, fallback.label
        )
        return fallback


def _image_for(source: OBBSource, rel: str) -> Path | None:
    """The source image a staged label at *rel* came from.

    The staged label's relative path mirrors the image's under `images/` --
    that is the review's key -- so this is a direct sibling lookup. The
    extension loop matches `_origin_image_for` in semantic_escalation,
    including its case handling: on the case-sensitive Linux lab shares this
    is deployed to, trying only the lowercase extension silently orphans
    `a.Jpg`.
    """
    stem = Path(source.path) / "images" / Path(rel).with_suffix("")
    for ext in sorted(IMG_EXTS):
        for candidate in (
            stem.with_name(stem.name + ext),
            stem.with_name(stem.name + ext.upper()),
        ):
            if candidate.is_file():
                return candidate
    return None


def _frame_size(source: OBBSource, rel: str) -> tuple[int, int]:
    """(height, width) of the frame, read from the image on disk."""
    image = _image_for(source, rel)
    if image is None:
        raise RuntimeError(
            f"No image found for staged frame '{rel}' in source '{source.name}'; "
            "the staged label has nothing to apply to."
        )
    frame = cv2.imread(str(image))
    if frame is None:
        raise RuntimeError(f"Could not read image {image} for staged frame '{rel}'.")
    return int(frame.shape[0]), int(frame.shape[1])


def _record_decision(staged_root: Path, rel: str, decision: str) -> None:
    decisions = read_decisions(staged_root)
    decisions[rel] = decision
    write_decisions(staged_root, decisions)


def accept_frame(
    source: OBBSource,
    rel: str,
    *,
    mode: MergeMode,
    iou_threshold: float = DEFAULT_MERGE_IOU,
) -> None:
    """Apply one staged frame to the source, immediately.

    Immediately rather than into a pending set, because the entire point of
    reviewing on the frame is seeing the result land on the ground-truth
    layer as you work. `merge_records`' "a merge can only add" invariant is
    what makes that safe; the Task-5 snapshot is what makes it reversible.

    LEVEL PROMOTION. If the staged level is ABOVE the source's, the source
    is promoted: `source.level` is set and its existing labels are lifted
    (an OBB quad becomes a 4-point polygon, which `_polygon_points` encodes
    with a repeated final vertex precisely so it never reads back as an
    OBB). Promotion is a property of the SOURCE, so the first promoting
    accept sets the level and the rest of the review proceeds at the new
    one. If the staged level is BELOW the source's, the staged records are
    derived down instead and the source's level is untouched.

    VERBATIM EXISTING LINES. Under ADD_NEW without promotion the existing
    file's lines are copied through byte for byte and only the surviving
    staged records are appended. Re-encoding them through
    `write_label_file` would round-trip denormalise -> normalise -> %.6f and
    shift bytes on labels the user never touched. Promotion is the one case
    that necessarily rewrites every line, and the snapshot covers it.
    """
    review = _review_of(source)
    staged_root = Path(review.staged_path)
    staged_label = staged_root / "labels" / rel
    if not staged_label.is_file():
        raise RuntimeError(
            f"Staged label '{rel}' is missing from {staged_root / 'labels'}; "
            "nothing was changed."
        )

    ensure_snapshot(source, staged_root)

    height, width = _frame_size(source, rel)
    source_level = _level_of(source.level, GeometryLevel.OBB)
    staged_level = _level_of(review.target_level, GeometryLevel.POLYGON)
    promoting = staged_level > source_level
    target_level = staged_level if promoting else source_level

    class_map = resolve_staged_class_ids(source, staged_root)
    staged_records = [
        LabelRecord(
            class_id=class_map.get(rec.class_id, rec.class_id),
            confidence=rec.confidence,
            points=rec.points,
            level=rec.level,
        )
        for rec in read_label_file(staged_label, (height, width))
    ]

    source_label = Path(source.path) / "labels" / rel
    source_label.parent.mkdir(parents=True, exist_ok=True)
    existing = read_label_file(source_label, (height, width))

    # Lift BEFORE merging: merge_records refuses upward derivation, and both
    # a promoting accept (existing below target) and a staged result below
    # the source's level (staged below target) would otherwise raise. The
    # lift moves no points; see `_lift`.
    existing = _lift(existing, target_level)
    staged_records = _lift(staged_records, target_level)

    if mode is MergeMode.OVERWRITE:
        write_label_file(
            source_label,
            derive_down(staged_records, target_level),
            (height, width),
            target_level,
        )
    else:
        merged = merge_records(
            existing,
            staged_records,
            mode=MergeMode.ADD_NEW,
            iou_threshold=iou_threshold,
            level=target_level,
        )
        survivors = merged[len(existing) :]
        if promoting or not source_label.is_file():
            # Promotion rewrites every line by necessity; a frame with no
            # prior label has nothing verbatim to preserve.
            write_label_file(
                source_label,
                derive_down(merged, target_level),
                (height, width),
                target_level,
            )
        else:
            # Bytes, not text: read_text()/write_text() apply universal-
            # newline translation, which would silently rewrite a CRLF
            # label file to LF -- exactly the kind of drift this branch
            # exists to avoid on lines the user never touched.
            prior = source_label.read_bytes()
            if prior and not prior.endswith(b"\n"):
                prior += b"\n"
            with source_label.open("wb") as fp:
                fp.write(prior)
            # Append only. write_label_file truncates, so the survivors are
            # formatted into a temp buffer and appended.
            if survivors:
                buffer = staged_root / ".append.tmp"
                try:
                    write_label_file(
                        buffer,
                        derive_down(survivors, target_level),
                        (height, width),
                        target_level,
                    )
                    with source_label.open("ab") as fp:
                        fp.write(buffer.read_bytes())
                finally:
                    buffer.unlink(missing_ok=True)

    if promoting:
        _promote_source(source, target_level, skip=rel, frame_size=(height, width))
        source.level = target_level.label

    _record_decision(
        staged_root,
        rel,
        ACCEPTED_OVERWRITE if mode is MergeMode.OVERWRITE else ACCEPTED_ADD_NEW,
    )


def _promote_source(
    source: OBBSource,
    target_level: GeometryLevel,
    *,
    skip: str,
    frame_size: tuple[int, int],
) -> None:
    """Lift every OTHER label file in the source to `target_level`.

    Only the encoding changes: `_polygon_points` repeats an OBB quad's final
    vertex so it reads back as a polygon, without moving a coordinate. The
    frame that triggered promotion is skipped because `accept_frame` has
    already written it.

    Each file is re-read at ITS OWN frame size, not the triggering frame's:
    a source's images need not all be the same resolution, and normalising
    with the wrong size would silently move every point.
    """
    labels_dir = Path(source.path) / "labels"
    for path in sorted(labels_dir.rglob("*.txt")):
        rel = path.relative_to(labels_dir).as_posix()
        if rel == skip:
            continue
        try:
            size = _frame_size(source, rel)
        except RuntimeError:
            size = frame_size
        records = read_label_file(path, size)
        if not records:
            continue
        lifted = [
            LabelRecord(
                class_id=rec.class_id,
                confidence=rec.confidence,
                points=rec.points,
                level=target_level,
            )
            for rec in records
        ]
        write_label_file(path, lifted, size, target_level)


def reject_frame(source: OBBSource, rel: str) -> None:
    """Record that a staged frame is not wanted. Changes nothing on disk."""
    review = _review_of(source)
    _record_decision(Path(review.staged_path), rel, REJECTED)


def accept_all(
    source: OBBSource,
    *,
    mode: MergeMode,
    iou_threshold: float = DEFAULT_MERGE_IOU,
) -> int:
    """Accept every frame not yet decided. Returns how many were accepted."""
    review = _review_of(source)
    staged_root = Path(review.staged_path)
    decided = read_decisions(staged_root)
    count = 0
    for rel in staged_frames(staged_root):
        if rel in decided:
            continue
        accept_frame(source, rel, mode=mode, iou_threshold=iou_threshold)
        count += 1
    return count


def reject_all(source: OBBSource) -> int:
    """Reject every frame not yet decided. Returns how many were rejected."""
    review = _review_of(source)
    staged_root = Path(review.staged_path)
    decided = read_decisions(staged_root)
    count = 0
    for rel in staged_frames(staged_root):
        if rel in decided:
            continue
        reject_frame(source, rel)
        count += 1
    return count


def is_complete(source: OBBSource) -> bool:
    """True when every staged frame has a decision.

    getattr rather than a bare attribute access, matching the overlay
    providers' style: this is called on whatever the window currently
    considers "the source", which is not guaranteed to be an OBBSource.
    """
    review = getattr(source, "staged_review", None)
    if review is None:
        return False
    decided, total = review_progress(review.staged_path)
    return total > 0 and decided >= total


def finish_review(source: OBBSource, project_dir: str | Path | None = None) -> None:
    """Close the review: remove the staging dir and clear the source's field.

    This DELETES the snapshot along with the staging dir, so revert is only
    available while a review is open. The review bar says so before calling
    this.

    `reviewed` drops to False only if at least one frame was ACCEPTED --
    the same meaning it has everywhere else, "machine-derived and not yet
    human-confirmed". Flipping it unconditionally would exclude a source
    from training because the user rejected every proposal, which is the
    opposite of what rejecting everything means.

    The delete goes through `remove_staged_escalation_dir`, which bounds it
    to the project's artifacts/pending_escalations/ -- `staged_path`
    round-trips through the saved project file, so it is untrusted input
    from disk and every delete here is a recursive rmtree.
    """
    from .sam2_escalation import remove_staged_escalation_dir

    review = _review_of(source)
    decisions = read_decisions(review.staged_path)
    if any(outcome != REJECTED for outcome in decisions.values()):
        source.reviewed = False
    remove_staged_escalation_dir(review.staged_path, project_dir)
    source.staged_review = None
