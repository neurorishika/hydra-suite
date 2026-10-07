"""YOLO pose label file I/O and keypoint migration utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hydra_suite.posekit.core.extensions import CrashSafeWriter, LabelVersioning
from hydra_suite.utils.hidden_files import is_hidden_file

from .models import Keypoint, compute_bbox_from_kpts
from .utils import _clamp01, _xyxy_to_cxcywh

logger = logging.getLogger("pose_label")


def load_yolo_pose_label(
    label_path: Path, k: int
) -> Optional[Tuple[int, List[Keypoint], Optional[Tuple[float, float, float, float]]]]:
    """Load a YOLO pose label file and convert entries into Keypoint objects."""
    if not label_path.exists():
        return None
    try:
        txt = label_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    if not txt:
        return None
    line = txt.splitlines()[0].strip()
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        rest = parts[5:]
    except ValueError:
        return None

    kpts: List[Keypoint] = []
    if len(rest) == 2 * k:
        for i in range(k):
            try:
                x = float(rest[2 * i + 0])
                y = float(rest[2 * i + 1])
                kpts.append(Keypoint(x=x, y=y, v=2))
            except ValueError:
                kpts.append(Keypoint(0.0, 0.0, 0))
    elif len(rest) == 3 * k:
        for i in range(k):
            try:
                x = float(rest[3 * i + 0])
                y = float(rest[3 * i + 1])
                v = int(float(rest[3 * i + 2]))
                kpts.append(Keypoint(x=x, y=y, v=v))
            except ValueError:
                kpts.append(Keypoint(0.0, 0.0, 0))
    else:
        # best-effort parse triples
        n = min(len(rest) // 3, k)
        for i in range(k):
            if i < n:
                try:
                    x = float(rest[3 * i + 0])
                    y = float(rest[3 * i + 1])
                    v = int(float(rest[3 * i + 2]))
                    kpts.append(Keypoint(x=x, y=y, v=v))
                except ValueError:
                    kpts.append(Keypoint(0.0, 0.0, 0))
            else:
                kpts.append(Keypoint(0.0, 0.0, 0))

    return cls, kpts, (cx, cy, bw, bh)


def save_yolo_pose_label(
    label_path: Path,
    cls: int,
    img_w: int,
    img_h: int,
    kpts_px: List[Keypoint],
    bbox_xyxy_px: Optional[Tuple[float, float, float, float]],
    pad_frac: float,
    create_backup: bool = True,
) -> None:
    """Write one annotation in YOLO pose format with optional versioned backup."""
    # Create backup of existing label before overwriting
    if create_backup and label_path.exists():
        try:
            versioning = LabelVersioning(label_path.parent)
            versioning.backup_label(label_path)
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    if bbox_xyxy_px is None:
        bbox_xyxy_px = compute_bbox_from_kpts(kpts_px, pad_frac, img_w, img_h)

    if bbox_xyxy_px is None:
        try:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            CrashSafeWriter.write_label(label_path, "")
        except Exception as e:
            logger.error(f"Failed to write empty label: {e}")
        return

    x1, y1, x2, y2 = bbox_xyxy_px
    cx, cy, bw, bh = _xyxy_to_cxcywh(x1, y1, x2, y2)

    cxn = _clamp01(cx / img_w)
    cyn = _clamp01(cy / img_h)
    bwn = _clamp01(bw / img_w)
    bhn = _clamp01(bh / img_h)

    vals = [str(int(cls)), f"{cxn:.6f}", f"{cyn:.6f}", f"{bwn:.6f}", f"{bhn:.6f}"]
    for kp in kpts_px:
        if kp.v == 0:
            vals += ["0.000000", "0.000000", "0"]
        else:
            vals += [
                f"{_clamp01(kp.x / img_w):.6f}",
                f"{_clamp01(kp.y / img_h):.6f}",
                str(int(kp.v)),
            ]

    # Use crash-safe atomic write
    content = " ".join(vals) + "\n"
    CrashSafeWriter.write_label(label_path, content)


def _build_keypoint_mapping(
    old_kpt_names: List[str],
    new_kpt_names: List[str],
    mode: str,
) -> Dict[int, Optional[int]]:
    """Build a mapping from new keypoint index to old keypoint index."""
    old_k = len(old_kpt_names)
    new_k = len(new_kpt_names)
    mapping: Dict[int, Optional[int]] = {}
    if mode == "index":
        for ni in range(new_k):
            mapping[ni] = ni if ni < old_k else None
    else:
        name_to_old: Dict[str, int] = {}
        for i, n in enumerate(old_kpt_names):
            if n not in name_to_old:
                name_to_old[n] = i
        for ni, n in enumerate(new_kpt_names):
            mapping[ni] = name_to_old.get(n, None)
    return mapping


def _parse_old_keypoint_triples(
    rest: List[str], old_k: int
) -> Optional[List[Tuple[str, str, str]]]:
    """Parse keypoint triples (x, y, v) from the tail of a label line.

    Returns None if the data is too short to parse safely.
    """
    if len(rest) >= 3 * old_k:
        return [(rest[3 * i], rest[3 * i + 1], rest[3 * i + 2]) for i in range(old_k)]
    if len(rest) >= 2 * old_k:
        return [(rest[2 * i], rest[2 * i + 1], "2") for i in range(old_k)]
    return None


def migrate_labels_keypoints(
    labels_dir: Path,
    old_kpt_names: List[str],
    new_kpt_names: List[str],
    mode: str = "name",  # "name" or "index"
) -> Tuple[int, int]:
    """
    Update all .txt labels in labels_dir to match new keypoint list length/order.

    mode="name": map by unique keypoint names, fallback to 0 if missing
    mode="index": preserve positions by old index (truncate/extend)
    Returns: (files_modified, files_total)
    """
    txts = sorted(p for p in labels_dir.glob("*.txt") if not is_hidden_file(p))
    if not txts:
        return (0, 0)

    old_k = len(old_kpt_names)
    new_k = len(new_kpt_names)
    mapping = _build_keypoint_mapping(old_kpt_names, new_kpt_names, mode)

    files_modified = 0
    for lp in txts:
        try:
            raw = lp.read_text(encoding="utf-8").strip()
        except Exception:
            continue

        if not raw:
            continue
        line = raw.splitlines()[0].strip()
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = parts[0]
        cx, cy, bw, bh = parts[1:5]
        rest = parts[5:]

        old_trip = _parse_old_keypoint_triples(rest, old_k)
        if old_trip is None:
            continue

        new_trip: List[str] = []
        for ni in range(new_k):
            oi = mapping.get(ni, None)
            if oi is None or oi >= len(old_trip):
                new_trip.extend(["0.000000", "0.000000", "0"])
            else:
                x, y, v = old_trip[oi]
                new_trip.extend([x, y, v])

        out_line = " ".join([cls, cx, cy, bw, bh] + new_trip) + "\n"
        if out_line.strip() != raw.strip():
            try:
                lp.write_text(out_line, encoding="utf-8")
                files_modified += 1
            except Exception:
                pass

    return files_modified, len(txts)
