"""
Core logic for FilterKit tool.
Includes perceptual hashing, duplicate removal, and diversity sampling.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from hydra_suite.core.identity.dataset.naming import parse_identity_image_filename

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _iter_images_in_dir(directory: Path) -> List[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _collect_coco_images(root: Path) -> Optional[List[Path]]:
    """Return resolved image paths from a COCO dataset root, or None if not detected."""
    preferred_names = (
        "annotations.json",
        "instances.json",
        "instances_train.json",
        "instances_val.json",
    )
    candidates: List[Path] = []
    for name in preferred_names:
        p = root / name
        if p.is_file():
            candidates.append(p)
    ann_dir = root / "annotations"
    if ann_dir.is_dir():
        candidates.extend(sorted(ann_dir.glob("*.json")))
    candidates.extend(sorted(root.glob("*.coco.json")))
    candidates.extend(sorted(root.glob("*.json")))

    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (
            isinstance(payload, dict)
            and all(
                isinstance(payload.get(k), list)
                for k in ("images", "annotations", "categories")
            )
        ):
            continue
        paths: List[Path] = []
        for img_entry in payload.get("images", []):
            file_name = img_entry.get("file_name")
            if not file_name:
                continue
            raw = Path(str(file_name))
            for cand_path in (
                root / raw,
                root / "images" / raw,
                root / raw.name,
                root / "images" / raw.name,
            ):
                if cand_path.exists():
                    paths.append(cand_path.resolve())
                    break
        if paths:
            return paths
    return None


def _collect_yolo_images(root: Path) -> Optional[List[Path]]:
    """Return resolved image paths from a YOLO OBB/detect dataset root, or None."""
    try:
        from hydra_suite.training.dataset_inspector import inspect_obb_or_detect_dataset

        inspection = inspect_obb_or_detect_dataset(root)
        paths = [
            Path(item.image_path).resolve()
            for split_items in inspection.splits.values()
            for item in split_items
        ]
        return paths or None
    except Exception:
        return None


def collect_images_for_root(root: Path) -> Tuple[str, List[Path]]:
    """
    Detect dataset format under *root* and return (source_kind, image_paths).

    Detection priority: class_folders → coco → yolo_obb → images/ subdir → flat root.
    """
    root = Path(root).expanduser().resolve()

    # 1. Class-folder (train/val/test splits with per-class subdirs)
    split_dirs = [root / s for s in ("train", "val", "test") if (root / s).is_dir()]
    if split_dirs:
        paths: List[Path] = []
        for split_dir in split_dirs:
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    paths.extend(_iter_images_in_dir(class_dir))
        if paths:
            return "class_folders", paths

    # 2. COCO JSON
    coco = _collect_coco_images(root)
    if coco:
        return "coco", coco

    # 3. YOLO OBB / detect
    yolo = _collect_yolo_images(root)
    if yolo:
        return "yolo_obb", yolo

    # 4. Standard images/ subdirectory (MAT / ClassKit standard)
    images_dir = root / "images"
    if images_dir.is_dir():
        paths = _iter_images_in_dir(images_dir)
        if paths:
            return "images", paths

    # 5. Flat root fallback
    return "images", _iter_images_in_dir(root)


class _HashBKTreeNode:
    def __init__(self, signature: int, kept_index: int) -> None:
        self.signature = signature
        self.kept_indices = [kept_index]
        self.children: Dict[int, "_HashBKTreeNode"] = {}


class _HashBKTree:
    def __init__(self) -> None:
        self.root: Optional[_HashBKTreeNode] = None

    @staticmethod
    def _distance(sig1: int, sig2: int) -> int:
        return int((sig1 ^ sig2) & ((1 << 64) - 1)).bit_count()

    def add(self, signature: int, kept_index: int) -> None:
        if self.root is None:
            self.root = _HashBKTreeNode(signature, kept_index)
            return

        node = self.root
        while True:
            distance = self._distance(signature, node.signature)
            if distance == 0:
                node.kept_indices.append(kept_index)
                return

            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _HashBKTreeNode(signature, kept_index)
                return
            node = child

    def query(self, signature: int, max_distance: int) -> List[int]:
        if self.root is None:
            return []

        matches: List[int] = []

        def _walk(node: _HashBKTreeNode) -> None:
            distance = self._distance(signature, node.signature)
            if distance <= max_distance:
                matches.extend(node.kept_indices)

            lower = max(0, distance - max_distance)
            upper = distance + max_distance
            for child_distance, child in node.children.items():
                if lower <= child_distance <= upper:
                    _walk(child)

        _walk(self.root)
        return matches


class _HistogramSignatureIndex:
    def __init__(self, bins: int = 32, initial_capacity: int = 256) -> None:
        self._bins = bins
        self._matrix = np.empty((initial_capacity, bins), dtype=np.float32)
        self._means = np.empty((initial_capacity,), dtype=np.float32)
        self._size = 0

    def add(self, signature: np.ndarray) -> None:
        if self._size >= self._matrix.shape[0]:
            new_capacity = max(1, self._matrix.shape[0] * 2)
            new_matrix = np.empty((new_capacity, self._bins), dtype=np.float32)
            new_means = np.empty((new_capacity,), dtype=np.float32)
            new_matrix[: self._size] = self._matrix[: self._size]
            new_means[: self._size] = self._means[: self._size]
            self._matrix = new_matrix
            self._means = new_means

        self._matrix[self._size] = signature
        self._means[self._size] = float(signature.mean())
        self._size += 1

    def query(self, signature: np.ndarray, threshold: float) -> np.ndarray:
        if self._size == 0:
            return np.empty((0,), dtype=np.int64)

        signatures = self._matrix[: self._size]
        kept_means = self._means[: self._size].astype(np.float64, copy=False)
        signature = np.asarray(signature, dtype=np.float32)
        signature_mean = float(signature.mean())

        coeff = np.sqrt(signatures * signature).sum(axis=1, dtype=np.float64)
        denominator = np.sqrt(
            kept_means * signature_mean * float(self._bins * self._bins)
        )

        ratio = np.zeros((self._size,), dtype=np.float64)
        valid = denominator > 0
        ratio[valid] = np.clip(coeff[valid] / denominator[valid], 0.0, 1.0)

        if np.any(~valid):
            invalid_indices = np.flatnonzero(~valid)
            for idx in invalid_indices:
                ratio[idx] = 1.0 - float(
                    cv2.compareHist(
                        signatures[idx],
                        signature,
                        cv2.HISTCMP_BHATTACHARYYA,
                    )
                )

        distances = np.sqrt(np.maximum(0.0, 1.0 - ratio))
        return np.flatnonzero(distances <= threshold)


class FilterKitCore:
    @staticmethod
    def _should_report_progress(idx: int, total: int) -> bool:
        return idx == 1 or idx % 250 == 0 or idx == total

    def _maybe_report_progress(self, progress_callback, idx: int, total: int) -> None:
        if progress_callback and self._should_report_progress(idx, total):
            progress_callback(idx, total)

    @staticmethod
    def _is_exact_hash_dedup(
        method: str, threshold: float, color_threshold: Optional[float]
    ) -> bool:
        return (
            method in {"phash", "dhash", "ahash"}
            and threshold <= 0
            and color_threshold is None
        )

    @staticmethod
    def _is_hash_method(method: str) -> bool:
        return method in {"phash", "dhash", "ahash"}

    @staticmethod
    def _filter_duplicate_groups(
        duplicate_groups: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        filtered_groups = []
        for group in duplicate_groups:
            group["count"] = len(group["paths"])
            if group["count"] > 1:
                filtered_groups.append(group)
        filtered_groups.sort(key=lambda group: group["count"], reverse=True)
        return filtered_groups

    @staticmethod
    def _append_group_path(
        duplicate_groups: List[Dict[str, Any]],
        group_index: int,
        item: Dict[str, Any],
    ) -> None:
        if 0 <= group_index < len(duplicate_groups):
            duplicate_groups[group_index]["paths"].append(str(item["path"]))

    @staticmethod
    def _add_duplicate_group(
        duplicate_groups: List[Dict[str, Any]],
        signature,
        item: Dict[str, Any],
        method: str,
    ) -> None:
        duplicate_groups.append(
            {
                "hash": str(signature),
                "count": 1,
                "paths": [str(item["path"])],
                "method": method,
            }
        )

    def _prepare_dedup_item(
        self,
        item: Dict[str, Any],
        method: str,
        color_threshold: Optional[float],
    ) -> bool:
        try:
            if "dedup_signature" not in item:
                img = cv2.imread(item["path"])
                if img is None:
                    return False
                item["dedup_signature"] = self.compute_signature(img, method)
                if color_threshold is not None:
                    item["color_signature"] = self.compute_color_signature(img)
            elif color_threshold is not None and "color_signature" not in item:
                img = cv2.imread(item["path"])
                if img is not None:
                    item["color_signature"] = self.compute_color_signature(img)
        except Exception:
            return False
        return True

    def _record_exact_hash_item(
        self,
        item: Dict[str, Any],
        signature,
        method: str,
        seen_exact_signatures: set,
        sig_to_cluster_index: Dict[int, int],
        duplicate_groups: List[Dict[str, Any]],
        kept_items: List[Dict[str, Any]],
        kept_color_signatures: List[Optional[np.ndarray]],
    ) -> bool:
        sig_key = int(signature)
        if sig_key in seen_exact_signatures:
            self._append_group_path(
                duplicate_groups,
                sig_to_cluster_index.get(sig_key, -1),
                item,
            )
            return False

        seen_exact_signatures.add(sig_key)
        sig_to_cluster_index[sig_key] = len(duplicate_groups)
        kept_items.append(item)
        kept_color_signatures.append(None)
        self._add_duplicate_group(duplicate_groups, sig_key, item, method)
        return True

    def _find_matching_signature_index(
        self,
        signature,
        current_color_signature,
        kept_signatures: List,
        kept_color_signatures: List[Optional[np.ndarray]],
        threshold: float,
        method: str,
        color_threshold: Optional[float],
    ) -> int:
        for sig_idx, existing_signature in enumerate(kept_signatures):
            if not self.is_duplicate(signature, existing_signature, threshold, method):
                continue
            if color_threshold is None:
                return sig_idx

            existing_color_signature = kept_color_signatures[sig_idx]
            if (
                current_color_signature is not None
                and existing_color_signature is not None
                and self.color_distance(
                    current_color_signature, existing_color_signature
                )
                > color_threshold
            ):
                continue
            return sig_idx

        return -1

    def _find_matching_hash_index(
        self,
        signature: int,
        current_color_signature,
        hash_index: _HashBKTree,
        kept_color_signatures: List[Optional[np.ndarray]],
        threshold: float,
        color_threshold: Optional[float],
    ) -> int:
        candidate_indices = sorted(
            hash_index.query(int(signature), max(0, int(threshold)))
        )
        for sig_idx in candidate_indices:
            if color_threshold is None:
                return sig_idx

            existing_color_signature = kept_color_signatures[sig_idx]
            if (
                current_color_signature is not None
                and existing_color_signature is not None
                and self.color_distance(
                    current_color_signature, existing_color_signature
                )
                > color_threshold
            ):
                continue
            return sig_idx

        return -1

    def _find_matching_histogram_index(
        self,
        signature: np.ndarray,
        current_color_signature,
        histogram_index: _HistogramSignatureIndex,
        kept_color_signatures: List[Optional[np.ndarray]],
        threshold: float,
        color_threshold: Optional[float],
    ) -> int:
        candidate_indices = histogram_index.query(signature, threshold)
        for sig_idx in candidate_indices:
            if color_threshold is None:
                return int(sig_idx)

            existing_color_signature = kept_color_signatures[int(sig_idx)]
            if (
                current_color_signature is not None
                and existing_color_signature is not None
                and self.color_distance(
                    current_color_signature, existing_color_signature
                )
                > color_threshold
            ):
                continue
            return int(sig_idx)

        return -1

    def _record_signature_item(
        self,
        item: Dict[str, Any],
        signature,
        current_color_signature,
        matched_index: int,
        method: str,
        kept_signatures: List,
        kept_items: List[Dict[str, Any]],
        kept_color_signatures: List[Optional[np.ndarray]],
        duplicate_groups: List[Dict[str, Any]],
    ) -> None:
        if matched_index == -1:
            kept_signatures.append(signature)
            kept_items.append(item)
            kept_color_signatures.append(current_color_signature)
            self._add_duplicate_group(duplicate_groups, signature, item, method)
            return

        self._append_group_path(duplicate_groups, matched_index, item)

    def available_dedup_methods(self) -> List[str]:
        return ["phash", "dhash", "ahash", "histogram"]

    def compute_dhash(self, image: np.ndarray, hash_size: int = 8) -> int:
        """
        Compute dHash (difference hash) for an image.

        Args:
            image: BGR or grayscale image.
            hash_size: Size of hash (default 8).

        Returns:
            64-bit integer hash.
        """
        if image is None:
            return 0

        # Resize to (width, height) = (hash_size + 1, hash_size)
        resized = cv2.resize(image, (hash_size + 1, hash_size))

        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # Compare adjacent pixels
        diff = gray[:, 1:] > gray[:, :-1]

        # Convert to integer
        return sum([2**i for i, v in enumerate(diff.flatten()) if v])

    def compute_ahash(self, image: np.ndarray, hash_size: int = 8) -> int:
        """Compute average hash (aHash) as 64-bit integer."""
        if image is None:
            return 0

        resized = cv2.resize(image, (hash_size, hash_size))
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        avg_val = gray.mean()
        bits = gray > avg_val
        return sum([2**i for i, v in enumerate(bits.flatten()) if v])

    def compute_phash(self, image: np.ndarray, hash_size: int = 8) -> int:
        """Compute perceptual hash (pHash) using DCT as 64-bit integer."""
        if image is None:
            return 0

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        resized = cv2.resize(gray, (32, 32)).astype(np.float32)
        dct = cv2.dct(resized)
        dct_low_freq = dct[:hash_size, :hash_size]

        flat = dct_low_freq.flatten()
        median_val = np.median(flat[1:]) if len(flat) > 1 else np.median(flat)
        bits = dct_low_freq > median_val
        return sum([2**i for i, v in enumerate(bits.flatten()) if v])

    def compute_hist_signature(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        """Compute normalized grayscale histogram signature for similarity checks."""
        if image is None:
            return np.zeros((bins,), dtype=np.float32)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def compute_color_signature(self, image: np.ndarray) -> np.ndarray:
        """Compute normalized HSV color histogram signature for color diversity checks."""
        if image is None:
            return np.zeros((8 * 8 * 8,), dtype=np.float32)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    def color_distance(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Bhattacharyya distance over color histograms (0=similar, 1=different)."""
        return float(cv2.compareHist(sig1, sig2, cv2.HISTCMP_BHATTACHARYYA))

    def compute_signature(self, image: np.ndarray, method: str = "phash"):
        """Compute method-specific signature for deduplication."""
        method = (method or "phash").lower()
        if method == "phash":
            return self.compute_phash(image)
        if method == "dhash":
            return self.compute_dhash(image)
        if method == "ahash":
            return self.compute_ahash(image)
        if method == "histogram":
            return self.compute_hist_signature(image)
        return self.compute_phash(image)

    def signature_distance(self, sig1, sig2, method: str = "phash") -> float:
        """Method-specific distance (lower means more similar)."""
        method = (method or "phash").lower()
        if method in {"phash", "dhash", "ahash"}:
            return float(self.hamming_distance(int(sig1), int(sig2)))
        if method == "histogram":
            # Bhattacharyya distance: 0=identical, 1=different
            return float(cv2.compareHist(sig1, sig2, cv2.HISTCMP_BHATTACHARYYA))
        return float(self.hamming_distance(int(sig1), int(sig2)))

    def is_duplicate(self, sig1, sig2, threshold: float, method: str = "phash") -> bool:
        """Return True when signatures are considered duplicates for method+threshold."""
        distance = self.signature_distance(sig1, sig2, method)
        return distance <= threshold

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two 64-bit hashes."""
        return int((hash1 ^ hash2) & ((1 << 64) - 1)).bit_count()

    def load_images_from_root(
        self, root: "str | Path"
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Load images from any supported dataset format, returning (source_kind, items)."""
        source_kind, image_paths = collect_images_for_root(Path(root))
        items: List[Dict[str, Any]] = []
        for idx, path in enumerate(image_paths):
            items.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "det_id": idx,
                    "frame_idx": idx,
                    "det_idx": 0,
                    "annotations": [],
                    "source_type": source_kind,
                    "interpolated": False,
                }
            )
        return source_kind, items

    def load_dataset(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Load flat MAT identity dataset images and extract metadata from filenames.
        """
        dataset = []
        folder = Path(folder_path)
        if not folder.exists():
            return []

        # Check for metadata.json in parent directory
        metadata_path = folder.parent / "metadata.json"
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        files = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        for file_path in files:
            img_path = str(file_path.resolve())
            filename = file_path.name

            parsed = parse_identity_image_filename(filename)
            if parsed is None:
                continue

            item = {
                "path": img_path,
                "filename": filename,
                "det_id": int(parsed.get("det_id", 0)),
                "frame_idx": int(parsed.get("frame_idx", 0)),
                "det_idx": int(parsed.get("det_idx", 0)),
                "annotations": [],
                "source_type": str(parsed.get("source_type", "detected")),
                "interpolated": bool(parsed.get("interpolated", False)),
            }
            if parsed.get("detection_id") is not None:
                item["detection_id"] = int(parsed["detection_id"])
            if parsed.get("trajectory_id") is not None:
                item["trajectory_id"] = int(parsed["trajectory_id"])

            if metadata:
                for image_entry in metadata.get("images", metadata.get("crops", [])):
                    if str(image_entry.get("filename", "")).strip() == filename:
                        item["annotations"].append(image_entry)
                        break
                if not item["annotations"]:
                    for frame_data in metadata.get("frames", []):
                        if frame_data.get("frame_id") == item["frame_idx"]:
                            item["annotations"].extend(
                                frame_data.get("annotations", [])
                            )
                            break

            dataset.append(item)

        return dataset

    def temporal_subsample(
        self, dataset: List[Dict[str, Any]], interval: int
    ) -> List[Dict[str, Any]]:
        """Keep every Nth frame."""
        if interval <= 1:
            return dataset

        if not dataset:
            return []

        # Sort by frame_idx first
        sorted_data = sorted(dataset, key=lambda x: x["frame_idx"])

        # Get unique frames
        unique_frames = sorted({d["frame_idx"] for d in sorted_data})

        # Select every interval-th frame
        kept_frames = set(unique_frames[::interval])

        return [d for d in sorted_data if d["frame_idx"] in kept_frames]

    def deduplicate_by_hash(
        self,
        dataset: List[Dict[str, Any]],
        threshold: float = 0,
        method: str = "phash",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        return_groups: bool = False,
        color_threshold: Optional[float] = None,
    ):
        """
        Remove duplicates based on selected perceptual method.
        For hash methods (pHash/dHash/aHash), threshold is hamming bits.
        For histogram, threshold is Bhattacharyya distance in [0,1].

        Args:
            dataset: List of items (must contain 'path').
            threshold: Method-specific distance threshold.
            method: One of phash, dhash, ahash, histogram.
        """
        if not dataset:
            return []

        method = (method or "phash").lower()
        if method not in set(self.available_dedup_methods()):
            method = "phash"

        kept_items = []
        seen_exact_signatures = set()
        kept_signatures = []
        kept_color_signatures = []
        sig_to_cluster_index = {}
        duplicate_groups = []
        hash_index = _HashBKTree() if self._is_hash_method(method) else None
        histogram_index = _HistogramSignatureIndex() if method == "histogram" else None

        dataset_sorted = sorted(dataset, key=lambda x: (x["frame_idx"], x["det_id"]))

        total = len(dataset_sorted)
        for idx, item in enumerate(dataset_sorted, start=1):
            if not self._prepare_dedup_item(item, method, color_threshold):
                self._maybe_report_progress(progress_callback, idx, total)
                continue

            signature = item["dedup_signature"]
            current_color_signature = item.get("color_signature")

            if self._is_exact_hash_dedup(method, float(threshold), color_threshold):
                self._record_exact_hash_item(
                    item,
                    signature,
                    method,
                    seen_exact_signatures,
                    sig_to_cluster_index,
                    duplicate_groups,
                    kept_items,
                    kept_color_signatures,
                )
                self._maybe_report_progress(progress_callback, idx, total)
                continue

            if self._is_hash_method(method) and hash_index is not None:
                matched_index = self._find_matching_hash_index(
                    int(signature),
                    current_color_signature,
                    hash_index,
                    kept_color_signatures,
                    float(threshold),
                    color_threshold,
                )
            elif method == "histogram" and histogram_index is not None:
                matched_index = self._find_matching_histogram_index(
                    signature,
                    current_color_signature,
                    histogram_index,
                    kept_color_signatures,
                    float(threshold),
                    color_threshold,
                )
            else:
                matched_index = self._find_matching_signature_index(
                    signature,
                    current_color_signature,
                    kept_signatures,
                    kept_color_signatures,
                    float(threshold),
                    method,
                    color_threshold,
                )

            is_new_signature = matched_index == -1
            self._record_signature_item(
                item,
                signature,
                current_color_signature,
                matched_index,
                method,
                kept_signatures,
                kept_items,
                kept_color_signatures,
                duplicate_groups,
            )
            if is_new_signature:
                if self._is_hash_method(method) and hash_index is not None:
                    hash_index.add(int(signature), len(kept_items) - 1)
                elif method == "histogram" and histogram_index is not None:
                    histogram_index.add(signature)
            self._maybe_report_progress(progress_callback, idx, total)

        if return_groups:
            return kept_items, self._filter_duplicate_groups(duplicate_groups)

        return kept_items

    def diversity_sample(
        self, dataset: List[Dict[str, Any]], n_samples: int
    ) -> List[Dict[str, Any]]:
        """
        Select diverse samples using MiniBatchKMeans on resized images.
        """
        if not dataset or len(dataset) <= n_samples:
            return dataset

        # Extract features (32x32 resized image)
        feature_size = (32, 32)
        features = []
        valid_items = []

        for item in dataset:
            try:
                img = cv2.imread(item["path"])
                if img is None:
                    continue
                # Use grayscale for feature vector to save space/time
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img

                resized = cv2.resize(gray, feature_size)
                features.append(resized.flatten())
                valid_items.append(item)
            except Exception:
                continue

        if not features:
            return []

        X = np.array(features)

        # Use MiniBatchKMeans for speed
        kmeans = MiniBatchKMeans(
            n_clusters=n_samples, random_state=42, n_init=3, batch_size=1024
        )
        kmeans.fit(X)

        # Find closest sample to each centroid
        from sklearn.metrics import pairwise_distances_argmin_min

        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

        # Use a set to avoid duplicates if multiple centroids map to same sample
        selected_indices = sorted(set(closest))

        return [valid_items[i] for i in selected_indices]
