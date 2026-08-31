"""DetectKit UI utility functions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from hydra_suite.training.class_mapping import build_class_id_map, read_classes_txt

from .constants import IMG_EXTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UI settings persistence
# ---------------------------------------------------------------------------


def get_ui_settings_path() -> Path:
    """Return the path to the DetectKit UI-settings JSON file."""
    try:
        from hydra_suite.paths import _user_data_dir

        return _user_data_dir() / "detectkit" / "ui_settings.json"
    except Exception:
        return Path.home() / ".detectkit" / "ui_settings.json"


def load_ui_settings() -> dict:
    """Load saved UI settings (window size, last dirs, etc.)."""
    sp = get_ui_settings_path()
    if not sp.exists():
        return {}
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        logger.debug("Failed to read UI settings", exc_info=True)
    return {}


def save_ui_settings(settings: dict) -> None:
    """Persist UI settings."""
    sp = get_ui_settings_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(settings, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Image / label discovery
# ---------------------------------------------------------------------------


def list_images_in_source(source_path: str) -> list[Path]:
    """Return sorted list of image files found under *source_path*.

    Checks ``source_path/images/`` first; falls back to the source root.
    """
    root = Path(source_path)
    images_dir = root / "images"
    search_root = images_dir if images_dir.is_dir() else root

    results: list[Path] = []
    for p in search_root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            results.append(p)
    results.sort()
    return results


def ensure_detectkit_source_structure(source_path: str | Path) -> Path:
    """Ensure a DetectKit source has ``images/``, ``labels/``, and ``classes.txt``."""
    root = Path(source_path).expanduser().resolve()
    missing: list[str] = []
    if not (root / "images").is_dir():
        missing.append("images/")
    if not (root / "labels").is_dir():
        missing.append("labels/")
    if not (root / "classes.txt").is_file():
        missing.append("classes.txt")
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"{root} is missing required DetectKit source entries: {missing_text}."
        )
    return root


def source_class_id_map(
    source_path: str | Path,
    project_class_names: list[str],
) -> dict[int, int]:
    """Build a source->project class-id map by class name."""
    source_class_names = read_classes_txt(source_path)
    return build_class_id_map(source_class_names, project_class_names)


def find_label_for_image(
    image_path: Path,
    source_path: str,
) -> Optional[Path]:
    """Locate the OBB label file corresponding to *image_path*.

    Strategies tried in order:
    1. Mirror the ``images/`` sub-path into ``labels/`` (YOLO convention).
    2. Stem match directly inside ``<source>/labels/``.
    3. Recursive search under ``<source>/labels/`` for stem match.
    """
    root = Path(source_path)
    labels_dir = root / "labels"
    stem = image_path.stem

    # Strategy 1: mirror images -> labels
    images_dir = root / "images"
    if images_dir.is_dir():
        try:
            rel = image_path.relative_to(images_dir)
            candidate = labels_dir / rel.with_suffix(".txt")
            if candidate.exists():
                return candidate
        except ValueError:
            pass

    # Strategy 2: direct stem match in labels/
    if labels_dir.is_dir():
        candidate = labels_dir / f"{stem}.txt"
        if candidate.exists():
            return candidate

    # Strategy 3: recursive search
    if labels_dir.is_dir():
        for p in labels_dir.rglob(f"{stem}.txt"):
            return p

    return None


def find_staged_label_for_image(
    image_path: Path,
    source_path: str,
    staged_path: str,
) -> Optional[Path]:
    """Locate a STAGED escalation label for *image_path*.

    A staging directory has ``labels/`` but no ``images/``, so
    ``find_label_for_image`` cannot mirror inside it -- the relative path has
    to be taken from the SOURCE's images tree and applied to the staging
    dir. Falls back to a stem match, then a recursive one, for flat sources.
    """
    staged_root = Path(staged_path)
    labels_dir = staged_root / "labels"
    if not labels_dir.is_dir():
        return None

    images_dir = Path(source_path) / "images"
    if images_dir.is_dir():
        try:
            candidate = labels_dir / image_path.relative_to(images_dir).with_suffix(
                ".txt"
            )
            if candidate.exists():
                return candidate
        except ValueError:
            pass

    candidate = labels_dir / f"{image_path.stem}.txt"
    if candidate.exists():
        return candidate
    for found in labels_dir.rglob(f"{image_path.stem}.txt"):
        return found
    return None


def staged_class_names(staged_path: str) -> list[str]:
    """Class names from a staging dir's classes.txt (the escalation prompt).

    The staged ids index THIS list, not the project's, so the overlay must
    read it here rather than reuse the project's class names.
    """
    path = Path(staged_path) / "classes.txt"
    try:
        names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    except OSError:
        return ["object"]
    return names or ["object"]


def labels_to_clear(
    source_path: str | Path, image_paths: list[Path] | None = None
) -> list[Path]:
    """Return the exact set of label files ``clear_labels_for_source`` will
    truncate for *source_path* (optionally filtered to *image_paths*).

    This is the single source of truth for "what files are in scope" for a
    clear-labels action -- both the actual clearer and any UI code that
    needs to preview/count the same set (confirmation dialogs, partial-
    failure comparisons) must call this rather than re-deriving the rule.

    Unfiltered (image_paths=None): every "*.txt" under source_path/labels/,
    recursively, except a stray classes.txt (which belongs at the source
    root, not under labels/, but is skipped defensively if found there).

    Filtered: only the label files matching the given image paths, resolved
    via (1) mirroring images/<rel> -> labels/<rel>.txt, then (2) a direct
    stem match at the labels/ root -- deliberately NOT an unanchored
    recursive search (see find_label_for_image's Strategy 3, which this
    function does not use): with a split layout, two images in different
    splits can share a stem, and an unanchored search would silently
    resolve to the WRONG image's label file. An image that doesn't resolve
    via (1) or (2) is skipped, not an error -- "no label file for this
    image" is a legitimate, common state. Duplicate resolutions (two images
    sharing one label file) are de-duplicated to a single entry.
    """
    source_root = Path(source_path)
    labels_dir = source_root / "labels"
    if not labels_dir.is_dir():
        return []

    if image_paths is None:
        return [p for p in labels_dir.rglob("*.txt") if p.name != "classes.txt"]

    images_dir = source_root / "images"
    seen: set[Path] = set()
    label_paths: list[Path] = []
    for raw_image_path in image_paths:
        image_path = Path(raw_image_path)
        candidate: Path | None = None
        if images_dir.is_dir():
            try:
                rel = image_path.relative_to(images_dir)
                mirrored = labels_dir / rel.with_suffix(".txt")
                if mirrored.exists():
                    candidate = mirrored
            except ValueError:
                pass
        if candidate is None:
            flat = labels_dir / f"{image_path.stem}.txt"
            if flat.exists():
                candidate = flat
        if candidate is not None and candidate not in seen:
            seen.add(candidate)
            label_paths.append(candidate)

    return label_paths


def clear_labels_for_source(
    source_path: str | Path, image_paths: list[Path] | None = None
) -> int:
    """Truncate label files to empty for a source. Returns the count cleared.

    See ``labels_to_clear`` for the exact rule for which files are in scope.
    Never deletes a file or touches images/classes.txt -- "clear" means
    truncate to empty content, matching the "Clear labels from frame" name
    (the image's own row in the browser persists).
    """
    cleared = 0
    for label_path in labels_to_clear(source_path, image_paths):
        try:
            label_path.write_text("", encoding="utf-8")
            cleared += 1
        except Exception:
            logger.warning("Failed to clear labels at %s", label_path, exc_info=True)

    return cleared


def parse_obb_label(
    label_path: Path,
    img_w: int,
    img_h: int,
    class_id_map: dict[int, int] | None = None,
) -> list[dict]:
    """Parse a DetectKit label file and return pixel-coordinate polygons.

    Supports every line shape DetectKit's own sources and the AL exporter
    (``data/al/labels.py``) can produce, all with normalised [0, 1]
    coordinates:

    - AABB (5 fields): ``class_id cx cy w h`` -> an axis-aligned quad.
    - OBB/quad (9 fields): ``class_id x1 y1 x2 y2 x3
      y3 x4 y4`` -> a quad.
    - Polygon (odd field count >= 7): ``class_id x1 y1 x2 y2 ... xn yn``
      -> the contour's own points (n >= 3).

    Returns a list of dicts with keys ``class_id`` (int) and
    ``polygon_px`` (list of ``(x, y)`` tuples in pixels -- 4 for an
    AABB/OBB line, n for a genuine polygon). Invalid lines (too few
    fields, an even-but-not-4-and-not->=6 coordinate count, or a
    degenerate <3-point shape) are silently skipped.
    """
    results: list[dict] = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except Exception:
        return results

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            if class_id_map is not None:
                mapped_class_id = class_id_map.get(class_id)
                if mapped_class_id is None:
                    continue
                class_id = int(mapped_class_id)

            coord_parts = parts[1:]
            if len(coord_parts) == 4:
                cx, cy, w, h = (float(v) for v in coord_parts)
                x1, y1 = cx - w / 2.0, cy - h / 2.0
                x2, y2 = cx + w / 2.0, cy + h / 2.0
                coords = [x1, y1, x2, y1, x2, y2, x1, y2]
            elif len(coord_parts) >= 6 and len(coord_parts) % 2 == 0:
                coords = [float(v) for v in coord_parts]
            else:
                continue

            polygon_px = [
                (coords[i] * img_w, coords[i + 1] * img_h)
                for i in range(0, len(coords), 2)
            ]
            if len(polygon_px) < 3:
                continue
            results.append(
                {
                    "class_id": class_id,
                    "polygon_px": polygon_px,
                }
            )
        except (ValueError, IndexError):
            continue

    return results
