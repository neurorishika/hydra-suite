"""Which labelled frames calibration may use, and how to sample them.

Shared by SAM3 (semantic) and SAM2 (geometry) escalation calibration and by
direct/YOLO calibration; moved here from ``semantic_escalation`` so the
geometry path does not have to import the semantic job. ``semantic_escalation``
re-exports every name for existing importers.

Calibration scores masks, so it needs real ground truth: ``polygon_only``
restricts sampling to frames whose labels are ALL polygons. The default stays
``False`` so direct/YOLO calibration keeps sampling every labelled frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.detectkit.gui.constants import IMG_EXTS
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.utils.hidden_files import is_hidden_file

# A calibration fitted at one body size only transfers to targets of a
# similar size: tile fraction is relative to the animal.
SCALE_MISMATCH_RATIO = 1.5


def is_polygon_line(n_coords: int) -> bool:
    """A label line with *n_coords* coordinate values is a polygon.

    Even, >= 6, and not 8: AABB lines carry 4 values and OBB/quad lines 8.
    The AL label writer pads 4-point polygons (``data/al/labels.py``)
    precisely so a polygon line never carries 8, which makes this decidable.
    """
    return n_coords >= 6 and n_coords % 2 == 0 and n_coords != 8


def _is_polygon_label_text(text: str) -> bool:
    """True when the label text has a line and EVERY line is a polygon."""
    counts = [len(line.split()) - 1 for line in text.splitlines() if line.strip()]
    return bool(counts) and all(is_polygon_line(n) for n in counts)


def scale_mismatch(
    a_px: float, b_px: float, ratio: float = SCALE_MISMATCH_RATIO
) -> bool:
    """Median body sizes differ by more than *ratio*; unknown sizes never warn."""
    if a_px <= 0 or b_px <= 0:
        return False
    return max(a_px, b_px) / min(a_px, b_px) > ratio


def _label_path_for(images_dir: Path, labels_dir: Path, img_path: Path) -> Path:
    return labels_dir / img_path.relative_to(images_dir).with_suffix(".txt")


def has_labelled_frames(source: OBBSource) -> bool:
    """True if any frame in *source* carries a non-empty label file.

    Deliberately does NOT call ``labelled_frames_for``: answering "are there
    any labels?" by decoding every labelled image cost the GUI thread a full
    image-set decode every time the dialog opened. This is a label-FILE scan
    and touches no pixels.
    """
    root = Path(source.path)
    images_dir, labels_dir = root / "images", root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return False
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS or is_hidden_file(img_path):
            continue
        label_path = _label_path_for(images_dir, labels_dir, img_path)
        try:
            if label_path.exists() and label_path.read_text().strip():
                return True
        except OSError:  # pragma: no cover - unreadable label file
            continue
    return False


def has_polygon_frames(source: OBBSource) -> bool:
    """True if *source* has a frame whose labels are ALL polygons.

    Calibration needs ground truth for masks. A box is not one, and a frame
    mixing boxes and polygons cannot be scored either: its box-labelled
    animals would go unmatched. A label-FILE scan, like
    ``has_labelled_frames``; no image is decoded.
    """
    raw_path = getattr(source, "path", "")
    if not raw_path:
        return False
    root = Path(raw_path)
    images_dir, labels_dir = root / "images", root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return False
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS or is_hidden_file(img_path):
            continue
        label_path = _label_path_for(images_dir, labels_dir, img_path)
        try:
            if label_path.exists() and _is_polygon_label_text(label_path.read_text()):
                return True
        except OSError:  # pragma: no cover - unreadable label file
            continue
    return False


# Enough frames for a stable median without decoding a whole image set on the
# GUI thread while the dialog is opening.
MEDIAN_BODY_SAMPLE_FRAMES = 20
# F4: a PROJECT-WIDE budget, not just a per-source one. The per-source limit
# alone still decoded 20 frames x every source in the project on the GUI
# thread at dialog open -- on 4512^2 frames that is seconds to minutes of a
# frozen window with no feedback. The median only needs a sample, so the
# sample is bounded globally and the truncation is SURFACED (see
# measure_median_body_px), never silently applied.
MEDIAN_BODY_TOTAL_FRAMES = 20
CALIBRATION_SAMPLE_FRAMES = 12


def measure_median_body_px(
    sources,
    *,
    sample_frames: int = MEDIAN_BODY_SAMPLE_FRAMES,
    max_total_frames: int = MEDIAN_BODY_TOTAL_FRAMES,
) -> tuple[float, int, bool]:
    """(median longest side px, frames sampled, whether the budget truncated).

    Link 2 of the ``reference_body_px`` resolution chain (project setting ->
    this -> the user). Returns 0.0 when nothing can be measured. Without it,
    a project with no ``slice_training.reference_body_px`` silently runs with
    tiling OFF -- the measured-worst configuration (F1 0.719 -> 0.075).

    ``max_total_frames`` bounds the decode across ALL sources; the caller is
    expected to report the sample size so the cap is visible rather than a
    silent change of what "median of your labels" means.
    """
    sides: list[float] = []
    used = 0
    truncated = False
    for source in sources:
        if used >= max_total_frames:
            truncated = True
            break
        budget = min(sample_frames, max_total_frames - used)
        # Ask for one MORE than the budget: if it comes back, this source had
        # frames we are declining to read, which is exactly what `truncated`
        # is supposed to tell the caller. Setting the flag only on the next
        # iteration misses the single-source case the cap exists for.
        frames = labelled_frames_for(source, limit=budget + 1)
        if len(frames) > budget:
            truncated = True
            frames = frames[:budget]
        used += len(frames)
        for _path, records in frames:
            for rec in records:
                pts = np.asarray(rec.points, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] < 2:
                    continue
                extent = pts.max(axis=0) - pts.min(axis=0)
                longest = float(max(extent[0], extent[1]))
                if longest > 0:
                    sides.append(longest)
    if not sides:
        return 0.0, used, truncated
    return float(np.median(np.asarray(sides, dtype=np.float64))), used, truncated


def _image_size(img_path: Path) -> tuple[int, int] | None:
    """(width, height) from the file HEADER; decodes only if Pillow is absent."""
    try:
        from PIL import Image

        with Image.open(img_path) as im:
            return int(im.width), int(im.height)
    except ImportError:  # pragma: no cover - Pillow ships with torchvision
        image = cv2.imread(str(img_path))
        return None if image is None else (image.shape[1], image.shape[0])
    except OSError:
        return None


def quick_median_body_px(
    sources, *, max_frames: int = MEDIAN_BODY_TOTAL_FRAMES
) -> float:
    """Median longest label side across *sources*, WITHOUT decoding pixels.

    For UI feedback that reruns on every selection change (the dialogs'
    scale-mismatch warning). ``measure_median_body_px`` decodes each sampled
    frame, which froze the GUI on large frames (F4); this reads only label
    files and image headers. Returns 0.0 when nothing is measurable.
    """
    from hydra_suite.detectkit.gui.utils import parse_obb_label

    sides: list[float] = []
    used = 0
    for source in sources:
        raw_path = getattr(source, "path", "")
        if not raw_path:
            continue
        root = Path(raw_path)
        images_dir, labels_dir = root / "images", root / "labels"
        if not images_dir.is_dir():
            continue
        for img_path in sorted(
            p
            for p in images_dir.rglob("*")
            if p.suffix.lower() in IMG_EXTS and not is_hidden_file(p)
        ):
            if used >= max_frames:
                break
            label_path = _label_path_for(images_dir, labels_dir, img_path)
            if not label_path.exists():
                continue
            size = _image_size(img_path)
            if size is None:
                continue
            parsed = parse_obb_label(label_path, size[0], size[1])
            if not parsed:
                continue
            used += 1
            for item in parsed:
                pts = np.asarray(item["polygon_px"], dtype=np.float32).reshape(-1, 2)
                extent = pts.max(axis=0) - pts.min(axis=0)
                longest = float(max(extent[0], extent[1]))
                if longest > 0:
                    sides.append(longest)
    return float(np.median(sides)) if sides else 0.0


def median_body_px_for(
    sources, *, sample_frames: int = MEDIAN_BODY_SAMPLE_FRAMES
) -> float:
    """The median alone; see ``measure_median_body_px`` for the sample size."""
    return measure_median_body_px(sources, sample_frames=sample_frames)[0]


def labelled_frames_for(
    source: OBBSource, *, limit: int = 0, polygon_only: bool = False
) -> list[tuple[Path, list[LabelRecord]]]:
    """(image path, LabelRecords) for every non-empty labelled frame.

    Uses ``gui/utils.parse_obb_label``, which already handles 5-field AABB,
    9-field quad and odd-count polygon lines. Deliberately NOT
    ``sam2_prompts.read_boxes_from_label``, which accepts only 4- and
    8-value lines and silently drops polygon lines (jobs/sam2_prompts.py:49-60)
    -- calibration must work at ANY geometry level, because choosing an
    operating point needs instance COUNTS, not masks.
    """
    from hydra_suite.detectkit.gui.utils import parse_obb_label

    root = Path(source.path)
    images_dir, labels_dir = root / "images", root / "labels"
    out: list[tuple[Path, list[LabelRecord]]] = []
    for img_path in sorted(
        p
        for p in images_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTS and not is_hidden_file(p)
    ):
        if limit and len(out) >= limit:
            break
        label_path = _label_path_for(images_dir, labels_dir, img_path)
        if not label_path.exists():
            continue
        text = label_path.read_text()
        if not text.strip():
            continue
        if polygon_only and not _is_polygon_label_text(text):
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        parsed = parse_obb_label(label_path, w, h)
        if not parsed:
            continue
        out.append(
            (
                img_path,
                [
                    LabelRecord(
                        class_id=int(d["class_id"]),
                        confidence=1.0,
                        points=np.asarray(d["polygon_px"], dtype=np.float32).reshape(
                            -1, 2
                        ),
                        level=GeometryLevel.POLYGON,
                    )
                    for d in parsed
                ],
            )
        )
    return out


def stratified_calibration_frames(
    sources,
    *,
    budget: int = CALIBRATION_SAMPLE_FRAMES,
    polygon_only: bool = False,
) -> list[tuple[Path, list[LabelRecord]]]:
    """Return a deterministic, globally bounded sample spread across sources."""
    sources = list(sources)
    budget = max(1, int(budget))
    if not sources:
        return []
    per_source = max(1, budget // len(sources))
    sampled = [
        labelled_frames_for(source, limit=per_source, polygon_only=polygon_only)
        for source in sources
    ]
    output: list[tuple[Path, list[LabelRecord]]] = []
    for row in range(per_source):
        for frames in sampled:
            if row < len(frames):
                output.append(frames[row])
                if len(output) >= budget:
                    return output
    return output
