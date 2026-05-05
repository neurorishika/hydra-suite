#!/usr/bin/env python3
"""
Extensions for PoseKit Labeler:
- Crash-safe recovery with temp files
- Versioned label backups
- Per-frame metadata/tags
- Cluster-stratified dataset splitting
- Embedding/clustering utilities
- YOLO pose I/O
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

logger = logging.getLogger("pose_label.extensions")


# -----------------------------
# Frame Metadata
# -----------------------------
@dataclass
class FrameMetadata:
    """Metadata for a single frame."""

    image_path: str
    tags: Set[str] = field(
        default_factory=set
    )  # e.g., "occluded", "weird_posture", "motion_blur"
    notes: str = ""
    cluster_id: Optional[int] = None

    def to_json(self) -> dict:
        """Serialize frame metadata to a JSON-safe dictionary."""
        return {
            "image_path": self.image_path,
            "tags": list(sorted(self.tags)),
            "notes": self.notes,
            "cluster_id": self.cluster_id,
        }

    @staticmethod
    def from_json(data: dict) -> "FrameMetadata":
        """Build a FrameMetadata object from serialized JSON data."""
        return FrameMetadata(
            image_path=data["image_path"],
            tags=set(data.get("tags", [])),
            notes=data.get("notes", ""),
            cluster_id=data.get("cluster_id"),
        )


class MetadataManager:
    """Manages per-frame metadata."""

    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.metadata: Dict[str, FrameMetadata] = {}
        self.load()

    def load(self: object) -> object:
        """Load metadata from disk."""
        if not self.metadata_path.exists():
            return
        try:
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.metadata = {
                img_path: FrameMetadata.from_json(meta)
                for img_path, meta in data.items()
            }
            logger.info(f"Loaded metadata for {len(self.metadata)} frames")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    def save(self: object) -> object:
        """Save metadata to disk."""
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                img_path: meta.to_json() for img_path, meta in self.metadata.items()
            }
            self.metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.debug(f"Saved metadata for {len(self.metadata)} frames")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def get_metadata(self, image_path: str) -> FrameMetadata:
        """Get metadata for a frame, creating if needed."""
        if image_path not in self.metadata:
            self.metadata[image_path] = FrameMetadata(image_path=image_path)
        return self.metadata[image_path]


# -----------------------------
# Crash-safe recovery
# -----------------------------
class CrashSafeWriter:
    """Writes labels atomically using temp files."""

    @staticmethod
    def write_label(label_path: Path, content: str) -> object:
        """Write label file atomically using temp file + rename."""
        try:
            # Write to temp file in same directory (ensures same filesystem)
            temp_dir = label_path.parent
            temp_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=temp_dir,
                delete=False,
                prefix=f".{label_path.stem}_",
                suffix=".tmp",
            ) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            # Atomic rename
            tmp_path.replace(label_path)
            logger.debug(f"Wrote label atomically: {label_path.name}")

        except Exception as e:
            logger.error(f"Failed to write label {label_path}: {e}")
            raise


# -----------------------------
# Versioned label backups
# -----------------------------
class LabelVersioning:
    """Manages versioned backups of labels."""

    def __init__(self, labels_dir: Path, max_versions: int = 5):
        self.labels_dir = labels_dir
        self.backup_dir = labels_dir.parent / "labels_history"
        self.max_versions = max_versions
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def backup_label(self: object, label_path: Path) -> object:
        """Create a versioned backup of a label file."""
        if not label_path.exists():
            return

        try:
            stem = label_path.stem

            # Find existing backups
            existing = sorted(self.backup_dir.glob(f"{stem}.v*.txt"))

            # Rotate old backups
            if len(existing) >= self.max_versions:
                # Remove oldest
                for old in existing[: len(existing) - self.max_versions + 1]:
                    old.unlink()
                existing = sorted(self.backup_dir.glob(f"{stem}.v*.txt"))

            # Determine next version number
            if existing:
                last_version = int(existing[-1].stem.split(".v")[1])
                next_version = last_version + 1
            else:
                next_version = 1

            # Create backup
            backup_path = self.backup_dir / f"{stem}.v{next_version:03d}.txt"
            shutil.copy2(label_path, backup_path)
            logger.debug(f"Created backup: {backup_path.name}")

        except Exception as e:
            logger.error(f"Failed to backup {label_path}: {e}")


# -----------------------------
# Cluster-stratified splitting
# -----------------------------
def cluster_stratified_split(
    image_paths: List[Path],
    cluster_ids: List[int],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    min_per_cluster: int = 1,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split frames into train/val/test preserving cluster distribution.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Normalize fractions
    total = train_frac + val_frac + test_frac
    train_frac /= total
    val_frac /= total
    test_frac /= total

    # Group by cluster
    clusters: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for cid, indices in clusters.items():
        n = len(indices)

        # Shuffle
        indices = list(indices)
        rng.shuffle(indices)

        if n < 3:
            # Too small to split, put all in train
            train_indices.extend(indices)
            continue

        # Calculate splits
        n_train = max(min_per_cluster, int(n * train_frac))
        n_val = max(min_per_cluster if val_frac > 0 else 0, int(n * val_frac))
        n_test = n - n_train - n_val

        if n_test < min_per_cluster and test_frac > 0:
            # Adjust to maintain minimum
            if n_train > min_per_cluster:
                n_train -= 1
                n_test += 1

        # Split
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val :])

    return train_indices, val_indices, test_indices


# -----------------------------
# Keypoint Dataclass
# -----------------------------
@dataclass
class Keypoint:
    """Represents a single keypoint with pixel coordinates and visibility."""

    x: float = 0.0
    y: float = 0.0
    v: int = 0  # 0=missing, 1=occluded, 2=visible


# -----------------------------
# Embedding/Clustering Utilities
# -----------------------------


def _stable_hash_dict(d: dict) -> str:
    """Create stable hash from dictionary for cache keys."""
    s = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def _choose_device(pref: str = "auto") -> str:
    """Choose compute device for embeddings."""
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _read_image_pil(path: Path):
    """Read image as PIL RGB."""
    from PIL import Image

    return Image.open(path).convert("RGB")


def _maybe_downscale_pil(img, max_side: int):
    """Downscale PIL image if larger than max_side."""
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), resample=2)  # BILINEAR


def _enhance_pil_for_pose(img_pil):
    """Apply CLAHE and unsharp mask enhancement to PIL image."""
    from ..gui.utils import enhance_for_pose

    # enhance_for_pose expects BGR np array
    arr = np.array(img_pil)  # RGB
    bgr = arr[:, :, ::-1].copy()
    bgr2 = enhance_for_pose(bgr)
    rgb2 = bgr2[:, :, ::-1].copy()
    from PIL import Image

    return Image.fromarray(rgb2)


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between L2-normalized embeddings."""
    # assumes A, B are L2-normalized
    return A @ B.T


def _farthest_point_centers(emb: np.ndarray, k: int, seed: int = 0) -> List[int]:
    """Select k centers using farthest-point sampling on embeddings."""
    # emb: (n,d) L2-normalized
    n = emb.shape[0]
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    centers = [first]
    # maintain best similarity to any chosen center (higher is closer)
    best_sim = _cosine_sim_matrix(emb, emb[[first]]).reshape(-1)
    for _ in range(1, min(k, n)):
        # pick point with lowest similarity to nearest center
        idx = int(np.argmin(best_sim))
        centers.append(idx)
        new_sim = _cosine_sim_matrix(emb, emb[[idx]]).reshape(-1)
        best_sim = np.maximum(best_sim, new_sim)
    return centers


def cluster_embeddings_cosine(
    emb: np.ndarray,
    k: int,
    method: str = "hierarchical",
    seed: int = 0,
    hierarchical_limit: int = 2500,
) -> np.ndarray:
    """
    Cluster embeddings using cosine similarity.
    Returns cluster labels in [0..k-1].
    """
    n = emb.shape[0]
    k = max(1, min(int(k), n))

    if method == "hierarchical" and n <= hierarchical_limit:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        # pdist with cosine => 1 - cosine_similarity
        dist = pdist(emb, metric="cosine")
        Z = linkage(dist, method="ward")
        labels = fcluster(Z, t=k, criterion="maxclust")
        return (labels - 1).astype(np.int32)

    # fallback: farthest-point centers + nearest-center assignment
    centers = _farthest_point_centers(emb, k=k, seed=seed)
    C = emb[centers]
    sims = _cosine_sim_matrix(emb, C)  # (n,k)
    return np.argmax(sims, axis=1).astype(np.int32)


def pick_frames_stratified(
    emb: np.ndarray,
    cluster_id: np.ndarray,
    want_n: int,
    eligible_indices: List[int],
    min_per_cluster: int = 1,
    seed: int = 0,
    strategy: str = "centroid_then_diverse",  # or "centroid"
) -> List[int]:
    """
    Select frames stratified by cluster.
    emb: embeddings for eligible_indices (same order)
    cluster_id: cluster labels for eligible_indices
    returns: selected *global* indices (from eligible_indices mapping)
    """
    want_n = max(0, min(int(want_n), len(eligible_indices)))
    if want_n == 0:
        return []

    # cluster -> local positions
    clusters: Dict[int, List[int]] = {}
    for pos, cid in enumerate(cluster_id):
        clusters.setdefault(int(cid), []).append(pos)

    # order clusters by size (largest first)
    cluster_keys = sorted(clusters.keys(), key=lambda c: len(clusters[c]), reverse=True)

    # If more clusters than want_n, keep only top want_n clusters (still diverse-ish)
    if len(cluster_keys) > want_n:
        cluster_keys = cluster_keys[:want_n]

    # initial quotas
    quotas = {c: 0 for c in cluster_keys}
    remaining = want_n

    # give each cluster min_per_cluster (as possible)
    for c in cluster_keys:
        give = min(min_per_cluster, len(clusters[c]), remaining)
        quotas[c] += give
        remaining -= give
        if remaining <= 0:
            break

    if remaining > 0:
        # proportional allocation
        total_sz = sum(len(clusters[c]) for c in cluster_keys)
        for c in cluster_keys:
            frac = len(clusters[c]) / max(total_sz, 1)
            extra = int(round(frac * remaining))
            # don't exceed cluster size
            cap = len(clusters[c]) - quotas[c]
            extra = min(extra, cap, remaining)
            quotas[c] += extra
            remaining -= extra
            if remaining <= 0:
                break

        # leftover
        while remaining > 0:
            for c in cluster_keys:
                if quotas[c] < len(clusters[c]):
                    quotas[c] += 1
                    remaining -= 1
                    if remaining <= 0:
                        break

    selected_local: List[int] = []

    for c in cluster_keys:
        positions = clusters[c]
        q = quotas[c]
        if q <= 0:
            continue

        if strategy == "centroid":
            # pick nearest to cluster centroid
            pos_arr = np.array(positions)
            sub_emb = emb[pos_arr]
            centroid = sub_emb.mean(axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            sims = _cosine_sim_matrix(sub_emb, centroid).reshape(-1)
            top = np.argsort(-sims)[:q]
            selected_local.extend(pos_arr[top].tolist())

        else:  # "centroid_then_diverse"
            pos_arr = np.array(positions)
            sub_emb = emb[pos_arr]
            centroid = sub_emb.mean(axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            if q == 1:
                sims = _cosine_sim_matrix(sub_emb, centroid).reshape(-1)
                best = int(np.argmax(sims))
                selected_local.append(int(pos_arr[best]))
            else:
                # pick centroid first
                sims = _cosine_sim_matrix(sub_emb, centroid).reshape(-1)
                best = int(np.argmax(sims))
                picked_local = [best]
                selected_local.append(int(pos_arr[best]))

                # then diverse
                best_sim_to_any = sims.copy()
                for _ in range(1, q):
                    cand = [i for i in range(len(pos_arr)) if i not in picked_local]
                    if not cand:
                        break
                    cand_sims = best_sim_to_any[cand]
                    # pick the one with lowest similarity to any selected
                    rel_idx = int(np.argmin(cand_sims))
                    chosen = cand[rel_idx]
                    picked_local.append(chosen)
                    selected_local.append(int(pos_arr[chosen]))
                    # update best_sim_to_any
                    new_sim = _cosine_sim_matrix(sub_emb, sub_emb[[chosen]]).reshape(-1)
                    best_sim_to_any = np.maximum(best_sim_to_any, new_sim)

    # map local positions back to global frame indices
    out = [eligible_indices[pos] for pos in selected_local]
    # ensure exact want_n (rare over/under due to edge cases)
    out = out[:want_n]
    return out


# -----------------------------
# YOLO Pose I/O
# -----------------------------


def load_yolo_pose_label(
    label_path: Path, k: int
) -> Optional[Tuple[int, List[Keypoint], Optional[Tuple[float, float, float, float]]]]:
    """Load YOLO pose label from .txt file."""
    if not label_path.exists():
        return None
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return None
    line = txt.splitlines()[0].strip()
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, bw, bh = map(float, parts[1:5])
    rest = parts[5:]

    kpts: List[Keypoint] = []
    if len(rest) == 2 * k:
        # old format: x y
        for i in range(k):
            kpts.append(Keypoint(float(rest[2 * i]), float(rest[2 * i + 1]), 2))
    elif len(rest) == 3 * k:
        # new format: x y v
        for i in range(k):
            kpts.append(
                Keypoint(
                    float(rest[3 * i]), float(rest[3 * i + 1]), int(rest[3 * i + 2])
                )
            )
    else:
        # fallback: fill missing
        for _ in range(k):
            kpts.append(Keypoint(0.0, 0.0, 0))
        # parse what we can
        for i in range(min(k, len(rest) // 3)):
            kpts[i] = Keypoint(
                float(rest[3 * i]), float(rest[3 * i + 1]), int(rest[3 * i + 2])
            )

    return (cls, kpts, (cx, cy, bw, bh))


def _clamp01(x: float) -> float:
    """Clamp value to [0, 1] with NaN protection."""
    if not np.isfinite(x):
        logger.warning(f"Non-finite value clamped to 0: {x}")
        return 0.0
    return max(0.0, min(1.0, x))


def _xyxy_to_cxcywh(x1, y1, x2, y2):
    """Convert bbox from xyxy to cxcywh format."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return (cx, cy, w, h)


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
    """Save YOLO pose label with crash-safe writing and optional backup."""
    MIN_BBOX_DIM = 0.001  # Minimum normalized bbox dimension

    # Create backup of existing label before overwriting
    if create_backup and label_path.exists():
        try:
            LabelVersioning(label_path.parent).backup_label(label_path)
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    if bbox_xyxy_px is None:
        # compute from keypoints
        from ..gui.models import compute_bbox_from_kpts

        bbox_xyxy_px = compute_bbox_from_kpts(kpts_px, pad_frac, img_w, img_h)

    if bbox_xyxy_px is None:
        logger.warning(f"Cannot save label for {label_path}: no valid bbox")
        return

    x1, y1, x2, y2 = bbox_xyxy_px

    # Validate bbox coordinates
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        logger.error(f"Non-finite bbox coordinates for {label_path}: {bbox_xyxy_px}")
        return

    cx, cy, bw, bh = _xyxy_to_cxcywh(x1, y1, x2, y2)

    # Validate bbox dimensions before normalization
    if not all(np.isfinite(v) for v in (cx, cy, bw, bh)):
        logger.error(
            f"Non-finite bbox after conversion for {label_path}: cx={cx}, cy={cy}, w={bw}, h={bh}"
        )
        return

    cxn = _clamp01(cx / max(img_w, 1))
    cyn = _clamp01(cy / max(img_h, 1))
    bwn = _clamp01(bw / max(img_w, 1))
    bhn = _clamp01(bh / max(img_h, 1))

    # Ensure minimum bbox size to prevent NaN during training
    if bwn < MIN_BBOX_DIM or bhn < MIN_BBOX_DIM:
        logger.warning(
            f"Bbox too small for {label_path}: w={bwn:.6f}, h={bhn:.6f}. Expanding to minimum."
        )
        bwn = max(bwn, MIN_BBOX_DIM)
        bhn = max(bhn, MIN_BBOX_DIM)

    vals = [str(int(cls)), f"{cxn:.6f}", f"{cyn:.6f}", f"{bwn:.6f}", f"{bhn:.6f}"]
    for kp in kpts_px:
        xn = _clamp01(kp.x / max(img_w, 1))
        yn = _clamp01(kp.y / max(img_h, 1))
        vals.append(f"{xn:.6f}")
        vals.append(f"{yn:.6f}")
        vals.append(str(int(kp.v)))

    # Use crash-safe atomic write
    content = " ".join(vals) + "\n"
    CrashSafeWriter.write_label(label_path, content)


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
    txts = sorted(labels_dir.glob("*.txt"))
    if not txts:
        return (0, 0)

    old_k = len(old_kpt_names)
    new_k = len(new_kpt_names)

    # build mapping new_index -> old_index
    mapping: Dict[int, Optional[int]] = {}
    if mode == "index":
        for ni in range(new_k):
            mapping[ni] = ni if ni < old_k else None
    else:
        # by name
        old_map = {name: i for i, name in enumerate(old_kpt_names)}
        for ni, name in enumerate(new_kpt_names):
            mapping[ni] = old_map.get(name)

    files_modified = 0
    for lp in txts:
        try:
            txt = lp.read_text(encoding="utf-8").strip()
            if not txt:
                continue
            line = txt.splitlines()[0].strip()
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            rest = parts[5:]

            old_kpts: List[Keypoint] = []
            if len(rest) == 2 * old_k:
                for i in range(old_k):
                    old_kpts.append(
                        Keypoint(float(rest[2 * i]), float(rest[2 * i + 1]), 2)
                    )
            elif len(rest) == 3 * old_k:
                for i in range(old_k):
                    old_kpts.append(
                        Keypoint(
                            float(rest[3 * i]),
                            float(rest[3 * i + 1]),
                            int(rest[3 * i + 2]),
                        )
                    )
            else:
                # skip malformed
                continue

            # build new
            new_kpts: List[Keypoint] = []
            for ni in range(new_k):
                oi = mapping[ni]
                if oi is not None and 0 <= oi < len(old_kpts):
                    new_kpts.append(old_kpts[oi])
                else:
                    new_kpts.append(Keypoint(0.0, 0.0, 0))

            # write back
            vals = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
            for kp in new_kpts:
                vals.append(f"{kp.x:.6f}")
                vals.append(f"{kp.y:.6f}")
                vals.append(str(int(kp.v)))
            content = " ".join(vals) + "\n"
            CrashSafeWriter.write_label(lp, content)
            files_modified += 1

        except Exception as e:
            logger.warning(f"Failed to migrate {lp}: {e}")

    return (files_modified, len(txts))


# -----------------------------
# Dataset utilities
# -----------------------------


def list_labeled_indices(image_paths: List[Path], labels_dir: Path) -> List[int]:
    """Return indices of images that have non-empty label files."""
    labeled = []
    for idx, img_path in enumerate(image_paths):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        try:
            if label_path.read_text(encoding="utf-8").strip():
                labeled.append(idx)
        except Exception:
            continue
    return labeled


def _yolo_unique_stem(stem: str, used: set, salt: str) -> str:
    """Return a unique stem, appending a hash suffix on collision."""
    if stem not in used:
        used.add(stem)
        return stem
    h = hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8]
    cand = f"{stem}_{h}"
    used.add(cand)
    return cand


def _validate_bbox(
    parts: List[str], label_name: str
) -> Optional[Tuple[float, float, float, float]]:
    """Parse and validate the bounding-box portion of a YOLO pose label.

    Returns ``(cx, cy, bw, bh)`` or ``None`` when the label is invalid.
    """
    MIN_BBOX_DIM = 0.001
    EPSILON = 1e-8
    try:
        cx, cy, bw, bh = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
    except Exception:
        return None

    for v in (cx, cy, bw, bh):
        if not np.isfinite(v):
            logger.warning(f"Non-finite bbox value in {label_name}: {v}")
            return None

    if bw < MIN_BBOX_DIM or bh < MIN_BBOX_DIM:
        logger.warning(f"Bbox too small in {label_name}: w={bw:.6f}, h={bh:.6f}")
        return None

    if not (-EPSILON <= cx <= 1.0 + EPSILON and -EPSILON <= cy <= 1.0 + EPSILON):
        logger.warning(
            f"Bbox center out of range in {label_name}: cx={cx:.6f}, cy={cy:.6f}"
        )
        return None
    if not (0.0 <= bw <= 1.0 + EPSILON and 0.0 <= bh <= 1.0 + EPSILON):
        logger.warning(
            f"Bbox dims out of range in {label_name}: w={bw:.6f}, h={bh:.6f}"
        )
        return None

    return (cx, cy, bw, bh)


def _validate_keypoints(parts: List[str], kpt_count: int, label_name: str) -> bool:
    """Return ``True`` when all keypoints in *parts* pass validation."""
    EPSILON = 1e-8
    for i in range(5, 5 + 3 * kpt_count, 3):
        try:
            x = float(parts[i])
            y = float(parts[i + 1])
            v = int(float(parts[i + 2]))
        except Exception:
            logger.warning(
                f"Invalid keypoint format in {label_name} at index {(i - 5) // 3}"
            )
            return False

        if not np.isfinite(x) or not np.isfinite(y):
            logger.warning(
                f"Non-finite keypoint in {label_name}: kpt={(i - 5) // 3}, x={x}, y={y}"
            )
            return False

        if v not in (0, 1, 2):
            logger.warning(
                f"Invalid visibility {v} in {label_name} at keypoint {(i - 5) // 3}"
            )
            return False

        if v > 0:
            if not (-EPSILON <= x <= 1.0 + EPSILON and -EPSILON <= y <= 1.0 + EPSILON):
                logger.warning(
                    f"Keypoint out of range in {label_name}: kpt={(i - 5) // 3}, "
                    f"x={x:.6f}, y={y:.6f}, vis={v}"
                )
                return False
    return True


def _parse_label_parts(label_path: Path, kpt_count: int) -> Optional[List[str]]:
    """Read and validate a YOLO pose label file.

    Returns the whitespace-split parts list, or ``None`` when the label is
    invalid or incomplete.
    """
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not text:
        return None
    parts = text.split()
    if len(parts) < 5 + 3 * kpt_count:
        return None
    if _validate_bbox(parts, label_path.name) is None:
        return None
    if not _validate_keypoints(parts, kpt_count, label_path.name):
        return None
    return parts


def _has_any_visible_kpt_from_parts(
    parts: List[str], kpt_count: int, include_occluded: bool
) -> bool:
    """Return ``True`` when at least one keypoint is visible."""
    for i in range(5, 5 + 3 * kpt_count, 3):
        v = int(float(parts[i + 2]))
        if include_occluded:
            if v > 0:
                return True
        else:
            if v == 2:
                return True
    return False


def _rewrite_label_for_training(parts: List[str], dst: Path):
    """Write label to *dst*, zeroing occluded (v==1) keypoints."""
    out = parts[:]
    for i in range(5, len(out), 3):
        if i + 2 >= len(out):
            break
        try:
            v = int(float(out[i + 2]))
        except Exception:
            continue
        if v == 1:
            out[i] = "0"
            out[i + 1] = "0"
            out[i + 2] = "0"
    dst.write_text(" ".join(out) + "\n", encoding="utf-8")


def _collect_labeled_items(
    image_paths: List[Path],
    labels_dir: Path,
    extra_datasets: Optional[List[Tuple[List[Path], Path]]],
    extra_items: Optional[List[Tuple[Path, Path]]],
) -> List[Tuple[Path, Path]]:
    """Gather all (image, label) pairs from primary + extra sources."""
    datasets = [(image_paths, labels_dir)]
    if extra_datasets:
        datasets.extend(extra_datasets)

    items: List[Tuple[Path, Path]] = []
    for imgs, lbl_dir in datasets:
        labeled_indices = list_labeled_indices(imgs, lbl_dir)
        for idx in labeled_indices:
            img_path = imgs[idx]
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                items.append((img_path, label_path))

    if extra_items:
        for img_path, label_path in extra_items:
            if img_path.exists() and label_path.exists():
                items.append((img_path, label_path))
    return items


def _copy_split(
    split_items: List[Tuple[Path, Path]],
    img_dir: Path,
    lbl_dir: Path,
    ignore_occluded: bool,
    kpt_count: int,
    used_stems: set,
    manifest_rows: list,
):
    """Copy a train or val split into the dataset directory structure."""
    skipped = 0
    kept: List[Tuple[Path, Path]] = []
    include_occluded = not bool(ignore_occluded)
    for img_path, label_path in split_items:
        parts = _parse_label_parts(label_path, kpt_count)
        if parts is None:
            skipped += 1
            continue
        if not _has_any_visible_kpt_from_parts(
            parts, kpt_count, include_occluded=include_occluded
        ):
            skipped += 1
            continue
        stem = _yolo_unique_stem(img_path.stem, used_stems, str(img_path))
        img_dst = img_dir / f"{stem}{img_path.suffix.lower()}"
        lbl_dst = lbl_dir / f"{stem}.txt"
        shutil.copy2(img_path, img_dst)
        if ignore_occluded:
            _rewrite_label_for_training(parts, lbl_dst)
        else:
            shutil.copy2(label_path, lbl_dst)
        manifest_rows.append(
            {
                "src_image": str(img_path),
                "src_label": str(label_path),
                "dst_image": str(img_dst),
                "dst_label": str(lbl_dst),
            }
        )
        kept.append((img_path, label_path))
    return len(kept), skipped, kept


def _ensure_val_fallback(
    train_items: List[Tuple[Path, Path]],
    images_val: Path,
    labels_val: Path,
    kpt_count: int,
    ignore_occluded_val: bool,
    used_stems: set,
    manifest_rows: list,
) -> int:
    """Copy one training sample into val when the val split ended up empty.

    Returns the number of validation samples added (0 or 1).
    """
    include_occluded = not bool(ignore_occluded_val)
    for img_path, label_path in train_items:
        parts = _parse_label_parts(label_path, kpt_count)
        if parts is None:
            continue
        if not _has_any_visible_kpt_from_parts(
            parts, kpt_count, include_occluded=include_occluded
        ):
            continue
        stem = _yolo_unique_stem(img_path.stem, used_stems, str(img_path))
        img_dst = images_val / f"{stem}{img_path.suffix.lower()}"
        lbl_dst = labels_val / f"{stem}.txt"
        shutil.copy2(img_path, img_dst)
        shutil.copy2(label_path, lbl_dst)
        manifest_rows.append(
            {
                "src_image": str(img_path),
                "src_label": str(label_path),
                "dst_image": str(img_dst),
                "dst_label": str(lbl_dst),
            }
        )
        logger.warning(
            "Validation set was empty after filtering; duplicated one train sample for validation."
        )
        return 1
    return 0


def _write_dataset_yaml(
    output_dir: Path,
    keypoint_names: List[str],
    class_names: List[str],
    manifest_rows: list,
) -> Path:
    """Write ``dataset.yaml`` and ``manifest.json``; return the yaml path."""
    import yaml

    k = len(keypoint_names)
    data = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [k, 3],
        "names": {i: n for i, n in enumerate(class_names)},
        "kpt_names": {0: keypoint_names},
    }
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_rows, indent=2), encoding="utf-8"
    )
    return yaml_path


def build_yolo_pose_dataset(
    image_paths: List[Path],
    labels_dir: Path,
    output_dir: Path,
    train_frac: float,
    seed: int,
    class_names: List[str],
    keypoint_names: List[str],
    extra_datasets: Optional[List[Tuple[List[Path], Path]]] = None,
    extra_items: Optional[List[Tuple[Path, Path]]] = None,
    train_items: Optional[List[Tuple[Path, Path]]] = None,
    val_items: Optional[List[Tuple[Path, Path]]] = None,
    ignore_occluded_train: bool = True,
    ignore_occluded_val: bool = False,
) -> Dict[str, object]:
    """
    Build a YOLO Pose dataset with copied images/labels under output_dir.

    Returns dict with paths and counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    items = _collect_labeled_items(image_paths, labels_dir, extra_datasets, extra_items)

    if len(items) < 2 and (not train_items or not val_items):
        raise ValueError("Need at least 2 labeled frames to build train/val split.")

    if train_items is None or val_items is None:
        rng = random.Random(int(seed))
        rng.shuffle(items)
        train_frac = max(0.05, min(0.95, float(train_frac)))
        n_train = int(len(items) * train_frac)
        n_train = max(1, min(n_train, len(items) - 1))
        train_items = items[:n_train]
        val_items = items[n_train:]

    used_stems: set = set()
    manifest_rows: list = []
    k = len(keypoint_names)

    kept_train, _, _ = _copy_split(
        train_items,
        images_train,
        labels_train,
        ignore_occluded_train,
        k,
        used_stems,
        manifest_rows,
    )
    kept_val, _, _ = _copy_split(
        val_items,
        images_val,
        labels_val,
        ignore_occluded_val,
        k,
        used_stems,
        manifest_rows,
    )

    if kept_train <= 0:
        raise ValueError("No training labels with visible keypoints after filtering.")

    if kept_val <= 0:
        kept_val = _ensure_val_fallback(
            train_items,
            images_val,
            labels_val,
            k,
            ignore_occluded_val,
            used_stems,
            manifest_rows,
        )

    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    train_txt.write_text("images/train\n", encoding="utf-8")
    val_txt.write_text("images/val\n", encoding="utf-8")

    yaml_path = _write_dataset_yaml(
        output_dir, keypoint_names, class_names, manifest_rows
    )

    return {
        "yaml_path": yaml_path,
        "train_list": train_txt,
        "val_list": val_txt,
        "train_count": kept_train,
        "val_count": kept_val,
        "labeled_count": len(items),
        "manifest": output_dir / "manifest.json",
    }


def build_coco_keypoints_dataset(
    image_paths: List[Path],
    labels_dir: Path,
    output_dir: Path,
    class_names: List[str],
    keypoint_names: List[str],
    skeleton_edges: List[Tuple[int, int]],
    extra_datasets: Optional[List[Tuple[List[Path], Path]]] = None,
    extra_items: Optional[List[Tuple[Path, Path]]] = None,
) -> Dict[str, object]:
    """
    Build a COCO keypoints dataset under output_dir.

    Copies labeled frames into output_dir/images and writes annotations.json.
    """

    def _unique_stem(stem: str, used: set, salt: str) -> str:
        if stem not in used:
            used.add(stem)
            return stem
        h = hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8]
        cand = f"{stem}_{h}"
        used.add(cand)
        return cand

    output_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    datasets = [(image_paths, labels_dir)]
    if extra_datasets:
        datasets.extend(extra_datasets)

    items: List[Tuple[Path, Path]] = []
    for imgs, lbl_dir in datasets:
        labeled_indices = list_labeled_indices(imgs, lbl_dir)
        for idx in labeled_indices:
            img_path = imgs[idx]
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                items.append((img_path, label_path))

    if extra_items:
        for img_path, label_path in extra_items:
            if img_path.exists() and label_path.exists():
                items.append((img_path, label_path))

    if not items:
        raise ValueError("No labeled frames found for SLEAP export.")

    k = len(keypoint_names)
    used_stems = set()
    manifest_rows = []

    images = []
    annotations = []
    image_map: Dict[str, Dict[str, object]] = {}
    img_id = 1
    ann_id = 1

    def _parse_label_lines(label_path: Path) -> List[List[float]]:
        try:
            text = label_path.read_text(encoding="utf-8").strip()
        except Exception:
            return []
        if not text:
            return []
        lines = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5 + 3 * k:
                continue
            try:
                vals = [float(p) for p in parts]
            except Exception:
                continue
            lines.append(vals)
        return lines

    for img_path, label_path in items:
        key = str(img_path)
        if key not in image_map:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            stem = _unique_stem(img_path.stem, used_stems, str(img_path))
            img_dst = images_out / f"{stem}{img_path.suffix.lower()}"
            shutil.copy2(img_path, img_dst)
            image_map[key] = {
                "id": img_id,
                "file_name": str(Path("images") / img_dst.name),
                "width": int(w),
                "height": int(h),
                "dst_image": str(img_dst),
            }
            images.append(
                {
                    "id": img_id,
                    "file_name": str(Path("images") / img_dst.name),
                    "width": int(w),
                    "height": int(h),
                }
            )
            manifest_rows.append(
                {
                    "src_image": str(img_path),
                    "src_label": str(label_path),
                    "dst_image": str(img_dst),
                }
            )
            img_id += 1

        entry = image_map[key]
        w = entry["width"]
        h = entry["height"]

        lines = _parse_label_lines(label_path)
        if not lines:
            continue

        for vals in lines:
            cls = int(vals[0])
            cx, cy, bw, bh = vals[1:5]
            if cls < 0 or cls >= len(class_names):
                continue
            # bbox in pixels (x,y,w,h)
            x = (cx - bw / 2.0) * w
            y = (cy - bh / 2.0) * h
            bw_px = bw * w
            bh_px = bh * h
            x = max(0.0, min(float(w - 1), x))
            y = max(0.0, min(float(h - 1), y))
            bw_px = max(0.0, min(float(w), bw_px))
            bh_px = max(0.0, min(float(h), bh_px))

            keypoints = []
            num_kpts = 0
            for i in range(5, 5 + 3 * k, 3):
                xk = vals[i]
                yk = vals[i + 1]
                v = int(vals[i + 2])
                # SLEAP COCO import treats v=1 as visible, so map occluded to missing.
                if v == 2:
                    num_kpts += 1
                    xk = max(0.0, min(float(w - 1), xk * w))
                    yk = max(0.0, min(float(h - 1), yk * h))
                    v_out = 2
                else:
                    xk, yk = 0.0, 0.0
                    v_out = 0
                keypoints.extend([float(xk), float(yk), int(v_out)])

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": int(entry["id"]),
                    "category_id": int(cls + 1),
                    "bbox": [float(x), float(y), float(bw_px), float(bh_px)],
                    "area": float(bw_px * bh_px),
                    "iscrowd": 0,
                    "num_keypoints": int(num_kpts),
                    "keypoints": keypoints,
                }
            )
            ann_id += 1

    if not annotations:
        raise ValueError("No valid annotations found for COCO export.")

    categories = []
    skeleton = [[int(a + 1), int(b + 1)] for a, b in skeleton_edges]
    for i, name in enumerate(class_names):
        categories.append(
            {
                "id": int(i + 1),
                "name": name,
                "supercategory": "animal",
                "keypoints": list(keypoint_names),
                "skeleton": skeleton,
            }
        )

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_path = output_dir / "annotations.json"
    coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_rows, indent=2), encoding="utf-8"
    )

    return {
        "dataset_dir": output_dir,
        "coco_path": coco_path,
        "labeled_count": len(annotations),
        "manifest": output_dir / "manifest.json",
    }


# -----------------------------
# Embedding Worker Thread
# -----------------------------


class EmbeddingWorker(QObject):
    """Worker thread for computing embeddings."""

    progress = Signal(int, int)  # done, total
    status = Signal(str)
    finished = Signal(
        object, object, object
    )  # embeddings(np.ndarray), eligible_indices(list[int]), meta(dict)
    failed = Signal(str)

    def __init__(
        self,
        image_paths: List[Path],
        eligible_indices: List[int],
        cache_dir: Path,
        model_name: str,
        device_pref: str,
        batch_size: int,
        use_enhance: bool,
        max_side: int,
        cache_ok: bool = True,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.eligible_indices = eligible_indices
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.device_pref = device_pref
        self.batch_size = batch_size
        self.use_enhance = use_enhance
        self.max_side = max_side
        self.cache_ok = cache_ok
        self._cancel = False

    def cancel(self) -> None:
        """Signal the worker loop to stop at the next safe checkpoint."""
        self._cancel = True

    @Slot()
    def run(self) -> None:
        """Compute/load embeddings and emit final embedding matrix for eligible frames."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "model_name": self.model_name,
                "device_pref": self.device_pref,
                "batch_size": int(self.batch_size),
                "use_enhance": bool(self.use_enhance),
                "max_side": int(self.max_side),
            }
            key = _stable_hash_dict(meta)
            cache = IncrementalEmbeddingCache(self.cache_dir, key)
            cache.set_metadata(meta)

            if self.cache_ok:
                needs_compute, _ = cache.needs_update(self.image_paths)
            else:
                needs_compute = list(range(len(self.image_paths)))

            if not needs_compute:
                self.status.emit("Loading cached embeddings…")
                emb_all = cache.get_embeddings(self.image_paths)
                if emb_all is None:
                    needs_compute = list(range(len(self.image_paths)))
                else:
                    emb = emb_all[self.eligible_indices]
                    cached_meta = cache.get_metadata() or {}
                    meta.update(cached_meta)
                    self.finished.emit(emb, self.eligible_indices, meta)
                    return

            self.status.emit("Loading model…")
            device = _choose_device(self.device_pref)

            import timm
            import torch
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform

            model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
            model.eval().to(device)
            data_config = resolve_data_config({}, model=model)
            transform = create_transform(**data_config)

            # embed missing
            feats = []
            total = len(needs_compute)
            done = 0

            for start in range(0, total, self.batch_size):
                if self._cancel:
                    self.failed.emit("Canceled.")
                    return

                batch_idx = needs_compute[start : start + self.batch_size]
                imgs = []
                for i in batch_idx:
                    img = _read_image_pil(self.image_paths[i])
                    img = _maybe_downscale_pil(img, self.max_side)
                    if self.use_enhance:
                        img = _enhance_pil_for_pose(img)
                    imgs.append(transform(img))

                batch = torch.stack(imgs).to(device)

                with torch.no_grad():
                    out = model(batch)
                    out = out.detach().float().cpu().numpy()

                # L2 normalize
                out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
                feats.append(out)

                done += len(batch_idx)
                self.progress.emit(done, total)
                self.status.emit(f"Embedding {done}/{total}")

            if feats:
                new_emb = np.vstack(feats).astype(np.float32)
                cache.update_embeddings(
                    [self.image_paths[i] for i in needs_compute], new_emb
                )

            emb_all = cache.get_embeddings(self.image_paths)
            if emb_all is None:
                self.failed.emit("Failed to load embeddings from cache.")
                return
            emb = emb_all[self.eligible_indices]
            cache.set_metadata(meta)
            self.finished.emit(emb, self.eligible_indices, meta)

        except Exception as e:
            self.failed.emit(str(e))


# -----------------------------
# Incremental Embedding Cache
# -----------------------------


class IncrementalEmbeddingCache:
    """Manages incremental embedding computation and caching."""

    def __init__(self, cache_root: Path, cache_key: str):
        self.cache_root = cache_root
        self.cache_key = cache_key
        self.cache_dir = (
            cache_root / "embeddings" / self._sanitize_model_name(cache_key)
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_path = self.cache_dir / "embeddings.npy"
        self.index_path = self.cache_dir / "index.csv"
        self.meta_path = self.cache_dir / "metadata.json"

        self._embeddings: Optional[np.ndarray] = None
        self._index: Dict[str, int] = {}  # path -> row index
        self._stats: Dict[str, Tuple[int, int]] = {}  # path -> (mtime, size)
        self._load()

    @staticmethod
    def _sanitize_model_name(name: str) -> str:
        """Convert model name to safe directory name."""
        import re

        return re.sub(r"[^\w\-.]", "_", name)

    def _load(self):
        """Load existing embeddings and index."""
        if self.embeddings_path.exists() and self.index_path.exists():
            try:
                self._embeddings = np.load(self.embeddings_path)

                # Load index CSV
                with self.index_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._index[row["path"]] = int(row["index"])
                        try:
                            self._stats[row["path"]] = (
                                int(row.get("mtime", 0)),
                                int(row.get("size", 0)),
                            )
                        except Exception:
                            self._stats[row["path"]] = (0, 0)

                logger.info(
                    f"Loaded {len(self._index)} cached embeddings for {self.cache_key}"
                )
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._embeddings = None
                self._index = {}

    def _save(self):
        """Save embeddings and index to disk."""
        if self._embeddings is None:
            return

        try:
            np.save(self.embeddings_path, self._embeddings)

            # Save index CSV
            with self.index_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "index", "mtime", "size"])
                for path, idx in sorted(self._index.items(), key=lambda x: x[1]):
                    p = Path(path)
                    if p.exists():
                        mtime = int(p.stat().st_mtime)
                        size = int(p.stat().st_size)
                        writer.writerow([path, idx, mtime, size])
                        self._stats[path] = (mtime, size)
                    else:
                        writer.writerow([path, idx, 0, 0])

            logger.info(f"Saved {len(self._index)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_metadata(self) -> Dict:
        """Get cache metadata."""
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def set_metadata(self: object, meta: Dict) -> object:
        """Save cache metadata."""
        try:
            self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def needs_update(self, image_paths: List[Path]) -> Tuple[List[int], List[int]]:
        """
        Check which images need embedding computation.
        Returns: (needs_compute_indices, cached_indices)
        """
        needs_compute = []
        cached = []

        for i, path in enumerate(image_paths):
            path_str = str(path)

            # Check if in cache and unchanged
            if path_str in self._index:
                try:
                    stat = path.stat()
                    prev = self._stats.get(path_str, (0, 0))
                    if int(stat.st_mtime) == prev[0] and int(stat.st_size) == prev[1]:
                        cached.append(i)
                    else:
                        needs_compute.append(i)
                except Exception:
                    needs_compute.append(i)
            else:
                needs_compute.append(i)

        return needs_compute, cached

    def get_embeddings(self, image_paths: List[Path]) -> Optional[np.ndarray]:
        """Get embeddings for given paths (if all are cached)."""
        if self._embeddings is None:
            return None

        indices = []
        for path in image_paths:
            path_str = str(path)
            if path_str not in self._index:
                return None
            indices.append(self._index[path_str])

        return self._embeddings[indices]

    def add_embeddings(
        self: object, image_paths: List[Path], embeddings: np.ndarray
    ) -> object:
        """Add new embeddings to cache."""
        if len(image_paths) != embeddings.shape[0]:
            raise ValueError("Mismatch between paths and embeddings")

        # Initialize if empty
        if self._embeddings is None:
            self._embeddings = embeddings
            for i, path in enumerate(image_paths):
                self._index[str(path)] = i
        else:
            # Append new embeddings
            start_idx = len(self._embeddings)
            self._embeddings = np.vstack([self._embeddings, embeddings])
            for i, path in enumerate(image_paths):
                self._index[str(path)] = start_idx + i

        self._save()

    def update_embeddings(
        self: object, image_paths: List[Path], embeddings: np.ndarray
    ) -> object:
        """Update or add embeddings for given paths."""
        if len(image_paths) != embeddings.shape[0]:
            raise ValueError("Mismatch between paths and embeddings")
        if self._embeddings is None:
            self.add_embeddings(image_paths, embeddings)
            return
        to_add_paths = []
        to_add_emb = []
        for i, path in enumerate(image_paths):
            key = str(path)
            if key in self._index:
                idx = self._index[key]
                self._embeddings[idx] = embeddings[i]
            else:
                to_add_paths.append(path)
                to_add_emb.append(embeddings[i])
        if to_add_paths:
            self.add_embeddings(to_add_paths, np.vstack(to_add_emb))
        else:
            self._save()

    def remove_paths(self, paths: Set[str]) -> int:
        """Remove embeddings for deleted paths. Returns number removed."""
        if not paths:
            return 0
        if self._embeddings is None:
            self._load()
        if self._embeddings is None or not self._index:
            return 0

        remove_set = {p for p in paths if p in self._index}
        if not remove_set:
            return 0

        keep_items = [
            (path, idx)
            for path, idx in sorted(self._index.items(), key=lambda x: x[1])
            if path not in remove_set
        ]
        if not keep_items:
            self.clear()
            return len(remove_set)

        keep_indices = [idx for _, idx in keep_items]
        self._embeddings = self._embeddings[keep_indices]
        new_index: Dict[str, int] = {}
        new_stats: Dict[str, Tuple[int, int]] = {}
        for new_i, (path, _old_idx) in enumerate(keep_items):
            new_index[path] = new_i
            if path in self._stats:
                new_stats[path] = self._stats[path]
        self._index = new_index
        self._stats = new_stats
        self._save()
        return len(remove_set)

    def clear(self: object) -> object:
        """Clear all cached embeddings."""
        self._embeddings = None
        self._index = {}
        for p in [self.embeddings_path, self.index_path, self.meta_path]:
            if p.exists():
                p.unlink()


# -----------------------------
# Utility: Filter Near-Duplicates
# -----------------------------


def filter_near_duplicates(
    embeddings: np.ndarray, indices: List[int], threshold: float = 0.95
) -> List[int]:
    """
    Remove near-duplicate frames based on cosine similarity.
    Returns filtered indices.

    Args:
        embeddings: All embeddings (N x D)
        indices: List of frame indices to filter
        threshold: Cosine similarity threshold (0-1). Higher = more strict filtering.

    Returns:
        Filtered list of indices with near-duplicates removed
    """
    if len(indices) <= 1:
        return indices

    sub_emb = embeddings[indices]

    # Compute pairwise similarities
    sims = _cosine_sim_matrix(sub_emb, sub_emb)

    # Greedy filtering: keep first frame, then only add frames dissimilar to kept ones
    keep = []
    for i in range(len(indices)):
        if not keep:
            keep.append(i)
            continue

        # Check similarity to all kept frames
        max_sim = max(sims[i, j] for j in keep)
        if max_sim < threshold:
            keep.append(i)

    return [indices[i] for i in keep]
