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

from hydra_suite.detectkit.gui.models import OBBSource

logger = logging.getLogger(__name__)

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
