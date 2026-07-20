"""
Caching utilities for per-detection individual properties.

This cache is keyed by:
  - detection hash (inference identity + video/range fingerprint)
  - filter settings hash
  - extractor settings hash
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.3"


def _normalize_for_hash(value: Any) -> Any:
    """Convert values into deterministic JSON-serializable structures."""
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "type": "ndarray",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "digest": hashlib.md5(arr.tobytes()).hexdigest(),
        }
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return "NaN"
        if np.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(k): _normalize_for_hash(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    return value


def _hash_payload(payload: Dict[str, Any], length: int = 16) -> str:
    blob = json.dumps(_normalize_for_hash(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:length]


def _file_fingerprint(path_str: Any) -> Dict[str, Any]:
    configured = str(path_str or "")
    resolved = ""
    if configured:
        try:
            resolved = str(Path(configured).expanduser().resolve())
        except Exception:
            resolved = configured
    fp = {
        "configured_path": configured,
        "resolved_path": resolved,
        "exists": False,
        "size_bytes": None,
        "mtime_ns": None,
    }
    if resolved and Path(resolved).exists():
        try:
            stat = Path(resolved).stat()
            fp["exists"] = True
            fp["size_bytes"] = int(stat.st_size)
            fp["mtime_ns"] = int(stat.st_mtime_ns)
        except OSError:
            pass
    return fp


def compute_detection_hash(
    inference_model_id: str,
    video_path: str,
    start_frame: int,
    end_frame: int,
    detection_cache_version: str = "2.0",
) -> str:
    """Hash identity for raw detections over a specific video frame range."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "inference_model_id": str(inference_model_id or ""),
        "video": _file_fingerprint(video_path),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "detection_cache_version": str(detection_cache_version),
    }
    return _hash_payload(payload)


def compute_filter_settings_hash(params: Dict[str, Any]) -> str:
    """Hash filtering settings that determine filtered detections."""
    roi_mask = params.get("ROI_MASK", None)
    roi_digest = None
    if isinstance(roi_mask, np.ndarray):
        roi_arr = np.ascontiguousarray(roi_mask)
        roi_digest = hashlib.md5(roi_arr.tobytes()).hexdigest()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "detection_method": params.get("DETECTION_METHOD", "background_subtraction"),
        "yolo_confidence_threshold": params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25),
        "yolo_iou_threshold": params.get("YOLO_IOU_THRESHOLD", 0.7),
        "yolo_obb_mode": params.get("YOLO_OBB_MODE", "direct"),
        "pose_overrides_headtail": bool(params.get("POSE_OVERRIDES_HEADTAIL", True)),
        "yolo_seq_crop_pad_ratio": params.get("YOLO_SEQ_CROP_PAD_RATIO", 0.15),
        "yolo_seq_min_crop_size_px": params.get("YOLO_SEQ_MIN_CROP_SIZE_PX", 64),
        "yolo_seq_enforce_square_crop": bool(
            params.get("YOLO_SEQ_ENFORCE_SQUARE_CROP", True)
        ),
        "yolo_headtail_conf_threshold": params.get("YOLO_HEADTAIL_CONF_THRESHOLD", 0.6),
        "enable_size_filtering": bool(params.get("ENABLE_SIZE_FILTERING", False)),
        "min_object_size": params.get("MIN_OBJECT_SIZE", 0),
        "max_object_size": params.get("MAX_OBJECT_SIZE", None),
        "roi_mask_digest": roi_digest,
    }
    return _hash_payload(payload)


def _compute_pose_statistics(
    keypoints: Optional[np.ndarray], min_valid_conf: float = 0.2
) -> tuple[float, float, int, int]:
    """Compute pose summary statistics from raw keypoints.

    Returns:
        (mean_conf, valid_fraction, num_valid, num_keypoints)
    """
    if keypoints is None:
        return 0.0, 0.0, 0, 0
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return 0.0, 0.0, 0, 0

    conf_values = arr[:, 2]
    # Normalize confidence values to [0, 1]
    conf_values = np.clip(conf_values, 0.0, 1.0)

    mean_conf = float(np.mean(conf_values))
    valid_mask = conf_values >= float(min_valid_conf)
    valid_fraction = float(np.mean(valid_mask))
    num_valid = int(np.sum(valid_mask))
    num_keypoints = int(len(arr))

    return mean_conf, valid_fraction, num_valid, num_keypoints


def _resolve_tier_backend(params: Dict[str, Any], stage: str = "obb"):
    """Resolve ``RUNTIME_TIER`` to a ``ResolvedBackend`` for cache-key derivation.

    Core stays pure -- imports only from ``hydra_suite.runtime.resolver``, never
    an app layer. The resolver is stage-inert except for ``bgsub``, so ``"obb"``
    reproduces the string the legacy GUI wrote into ``COMPUTE_RUNTIME`` /
    ``POSE_SLEAP_DEVICE`` (both derived, pre-Gen-2, from
    ``resolve_compute_runtime(tier, platform, "obb")`` / the SLEAP pose stage,
    which resolve identically). Uses the default ``artifact_available=True`` to
    match the FT1 classify cache-key helper (``worker._classify_cache_runtime_string``).
    """
    from hydra_suite.runtime.resolver import RuntimeResolver, detect_platform

    tier = str(params.get("RUNTIME_TIER", "") or "").strip().lower()
    if tier not in {"cpu", "gpu", "gpu_fast"}:
        tier = "cpu"
    return RuntimeResolver(tier, detect_platform()).resolve(stage)


def _runtime_string_from_resolved(resolved) -> str:
    """Map a ``ResolvedBackend`` to the legacy compute-runtime cache string."""
    if resolved.backend == "tensorrt":
        return "tensorrt"
    if resolved.backend == "coreml":
        return "coreml"
    return resolved.device  # "cpu" / "cuda" / "mps"


def _sleap_device_from_resolved(resolved) -> str:
    """Reproduce the legacy ``POSE_SLEAP_DEVICE`` string from a resolved backend.

    The old GUI mapped the resolved SLEAP pose runtime to a SLEAP device string
    via ``derive_pose_runtime_settings`` (cpu->"cpu", mps->"mps", cuda->"cuda:0",
    tensorrt->"cuda:0"). That is exactly ``"cuda:0"`` on CUDA, else the device.
    """
    return "cuda:0" if resolved.device == "cuda" else resolved.device


def compute_extractor_hash(params: Dict[str, Any]) -> str:
    """Hash extractor settings that shape individual-property outputs.

    Note: pose_min_kpt_conf_valid is NOT included in the hash anymore.
    Summary statistics are computed on-demand when reading from cache,
    so changing the threshold doesn't invalidate the cache.

    Runtime Gen-2 (FT2): the ``compute_runtime`` / ``pose_sleap_device`` payload
    fields are a PERSISTED cache key (via ``compute_individual_properties_id``).
    They are now STABLE-derived from ``RUNTIME_TIER`` instead of the retired
    ``COMPUTE_RUNTIME`` / ``POSE_SLEAP_DEVICE`` params, reproducing the pre-Gen-2
    strings byte-for-byte so existing property caches stay valid. The payload
    key shape is unchanged.
    """
    pose_enabled = bool(params.get("ENABLE_POSE_EXTRACTOR", False))
    pose_model_type = str(params.get("POSE_MODEL_TYPE", "yolo")).strip().lower()
    pose_model_dir = str(params.get("POSE_MODEL_DIR", "")).strip()
    resolved_backend = _resolve_tier_backend(params, stage="obb")
    compute_runtime = _runtime_string_from_resolved(resolved_backend)
    pose_exported_model_path = str(params.get("POSE_EXPORTED_MODEL_PATH", "")).strip()
    pose_skeleton_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "enable_pose_extractor": pose_enabled,
        "pose_model_type": pose_model_type,
        "compute_runtime": compute_runtime,
        "pose_skeleton_file": (
            _file_fingerprint(pose_skeleton_file) if pose_skeleton_file else None
        ),
        "pose_exported_model": (
            _file_fingerprint(pose_exported_model_path)
            if pose_exported_model_path
            else None
        ),
    }
    if pose_model_type == "sleap":
        payload["pose_sleap_env"] = params.get("POSE_SLEAP_ENV", "sleap")
        payload["pose_sleap_device"] = _sleap_device_from_resolved(resolved_backend)
        payload["pose_sleap_batch"] = params.get("POSE_SLEAP_BATCH", 4)
        payload["pose_sleap_max_instances"] = 1
    if pose_enabled:
        payload["pose_model"] = _file_fingerprint(pose_model_dir)
    return _hash_payload(payload)


def compute_individual_properties_id(
    detection_hash: str,
    filter_settings_hash: str,
    extractor_hash: str,
) -> str:
    """Canonical identity key for individual-properties cache artifacts."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "detection_hash": str(detection_hash),
        "filter_settings_hash": str(filter_settings_hash),
        "extractor_hash": str(extractor_hash),
    }
    return _hash_payload(payload)


def compute_apriltag_cache_id(
    params: Dict[str, Any],
    inference_model_id: str,
) -> str:
    """Compute a settings hash for the AprilTag observation cache.

    Includes all AprilTag detector parameters that affect detection results
    plus the upstream detection model ID (since tags are detected from
    detection crops).  Thread count is excluded (performance-only).
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "inference_model_id": str(inference_model_id or ""),
        "apriltag_family": str(params.get("APRILTAG_FAMILY", "tag36h11")),
        "apriltag_max_hamming": int(params.get("APRILTAG_MAX_HAMMING", 1)),
        "apriltag_decimate": float(params.get("APRILTAG_DECIMATE", 1.0)),
        "apriltag_blur": float(params.get("APRILTAG_BLUR", 0.8)),
        "apriltag_refine_edges": bool(params.get("APRILTAG_REFINE_EDGES", True)),
        "apriltag_decode_sharpening": float(
            params.get("APRILTAG_DECODE_SHARPENING", 0.25)
        ),
        "apriltag_unsharp_kernel": list(params.get("APRILTAG_UNSHARP_KERNEL", (5, 5))),
        "apriltag_unsharp_sigma": float(params.get("APRILTAG_UNSHARP_SIGMA", 1.0)),
        "apriltag_unsharp_amount": float(params.get("APRILTAG_UNSHARP_AMOUNT", 1.5)),
        "apriltag_contrast_factor": float(params.get("APRILTAG_CONTRAST_FACTOR", 1.5)),
        "apriltag_max_tag_id": (
            int(params["APRILTAG_MAX_TAG_ID"])
            if params.get("APRILTAG_MAX_TAG_ID") is not None
            else None
        ),
        "padding_fraction": float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),
    }
    return _hash_payload(payload)


def compute_classify_cache_id(
    model_path: str,
    compute_runtime: str,
    inference_model_id: str,
    calibration_signature: str = "",
) -> str:
    """Compute a settings hash for the CNN classification cache.

    Includes the model fingerprint, compute runtime, and the upstream
    detection model ID.  Confidence threshold and batch size are excluded
    because the cache stores raw predictions — threshold filtering is
    applied at read time.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "inference_model_id": str(inference_model_id or ""),
        "model": _file_fingerprint(model_path),
        "compute_runtime": str(compute_runtime or "cpu").strip().lower(),
        "calibration_signature": str(calibration_signature or ""),
    }
    return _hash_payload(payload)


class IndividualPropertiesCache:
    """NPZ-backed cache for per-detection properties keyed by frame and detection ID."""

    CACHE_VERSION = "1.0"

    def __init__(
        self,
        cache_path: str,
        mode: str = "w",
    ):
        self.cache_path = Path(cache_path)
        self.mode = mode
        self._data: Dict[str, np.ndarray] = {}
        self._loaded_data = None
        self._compatible = True
        self._cached_frames = set()
        self.metadata: Dict[str, Any] = {}

        if self.mode == "r" and self.cache_path.exists():
            self._loaded_data = np.load(str(self.cache_path), allow_pickle=True)
            meta = self._loaded_data.get("metadata", None)
            if meta is None:
                self._compatible = False
                self._cached_frames = set()
                return
            try:
                self.metadata = dict(meta.item())
            except Exception:
                self.metadata = {}
            if str(self.metadata.get("version", "")) != self.CACHE_VERSION:
                self._compatible = False
                self._loaded_data.close()
                self._loaded_data = None
                return
            self._cached_frames = self._extract_cached_frames()

    def _extract_cached_frames(self) -> set:
        if self._loaded_data is None:
            return set()
        frames = set()
        for key in self._loaded_data.files:
            if key.startswith("frame_") and key.endswith("_detection_ids"):
                try:
                    frames.add(int(key.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        return frames

    def is_compatible(self) -> bool:
        """Return True if the loaded cache matches the expected schema version."""
        return self._compatible

    def get_cached_frames(self) -> Iterable[int]:
        """Return a sorted iterable of frame indices present in the cache."""
        return sorted(self._cached_frames)

    def add_frame(
        self,
        frame_idx: int,
        detection_ids: List[float],
        pose_mean_conf: Optional[List[float]] = None,
        pose_valid_fraction: Optional[List[float]] = None,
        pose_num_valid: Optional[List[int]] = None,
        pose_num_keypoints: Optional[List[int]] = None,
        pose_keypoints: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """Add a frame of detection data to cache.

        Note: pose_mean_conf, pose_valid_fraction, pose_num_valid, and pose_num_keypoints
        are deprecated and ignored. Only raw keypoints are stored.
        Summary statistics are computed on-demand when reading.
        """
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot write.")

        n = len(detection_ids)
        ids_arr = np.asarray(detection_ids, dtype=np.float64)

        if pose_keypoints is None:
            pose_keypoints = [None] * n
        pose_keypoints_arr = np.empty((n,), dtype=object)
        for i in range(n):
            kpts = pose_keypoints[i] if i < len(pose_keypoints) else None
            if kpts is None:
                pose_keypoints_arr[i] = None
            else:
                arr = np.asarray(kpts, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(
                        "pose_keypoints entries must be arrays of shape (K, 3)"
                    )
                pose_keypoints_arr[i] = arr

        frame_key = f"frame_{int(frame_idx):06d}"
        self._data[f"{frame_key}_detection_ids"] = ids_arr
        self._data[f"{frame_key}_pose_keypoints"] = pose_keypoints_arr

    def save(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Flush all accumulated frame data to a compressed .npz file on disk."""
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot save.")
        meta = {
            "version": self.CACHE_VERSION,
            "schema_version": SCHEMA_VERSION,
        }
        if metadata:
            meta.update(metadata)
        self._data["metadata"] = np.array(meta, dtype=object)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self.cache_path), **self._data)
        logger.info("Saved individual properties cache: %s", self.cache_path)
        self._data.clear()

    def get_frame(self, frame_idx: int, min_valid_conf: float = 0.2) -> Dict[str, Any]:
        """Get frame data with summary statistics computed on-demand.

        Args:
            frame_idx: Frame index to retrieve
            min_valid_conf: Minimum confidence threshold for keypoint validity (default: 0.2)

        Returns:
            Dict with detection_ids, pose_mean_conf, pose_valid_fraction,
            pose_num_valid, pose_num_keypoints, and pose_keypoints
        """
        if self.mode != "r":
            raise RuntimeError("Cache opened in write mode, cannot read.")
        if self._loaded_data is None:
            raise RuntimeError("No cache data loaded.")

        frame_key = f"frame_{int(frame_idx):06d}"
        ids_arr = self._loaded_data.get(f"{frame_key}_detection_ids", np.array([]))
        pose_arr = self._loaded_data.get(
            f"{frame_key}_pose_keypoints", np.array([], dtype=object)
        )

        # Compute summary statistics on-demand from raw keypoints
        n = len(ids_arr)
        mean_conf_list = []
        valid_fraction_list = []
        num_valid_list = []
        num_kpts_list = []

        for i in range(n):
            kpts = pose_arr[i] if i < len(pose_arr) else None
            mean_conf, valid_frac, num_valid, num_kpts = _compute_pose_statistics(
                kpts, min_valid_conf
            )
            mean_conf_list.append(mean_conf)
            valid_fraction_list.append(valid_frac)
            num_valid_list.append(num_valid)
            num_kpts_list.append(num_kpts)

        return {
            "detection_ids": ids_arr.tolist(),
            "pose_mean_conf": mean_conf_list,
            "pose_valid_fraction": valid_fraction_list,
            "pose_num_valid": num_valid_list,
            "pose_num_keypoints": num_kpts_list,
            "pose_keypoints": list(pose_arr.tolist()),
        }

    def get_detection(
        self, frame_idx: int, detection_id: float
    ) -> Optional[Dict[str, Any]]:
        """Get per-detection data for a specific detection ID in a frame.

        Returns a dict of pose properties for the matching detection, or None
        if the detection ID is not found in that frame.
        """
        frame = self.get_frame(frame_idx)
        ids = frame["detection_ids"]
        target = float(detection_id)
        for i, did in enumerate(ids):
            if float(did) == target:
                return {
                    "pose_mean_conf": frame["pose_mean_conf"][i],
                    "pose_valid_fraction": frame["pose_valid_fraction"][i],
                    "pose_num_valid": frame["pose_num_valid"][i],
                    "pose_num_keypoints": frame["pose_num_keypoints"][i],
                    "pose_keypoints": frame["pose_keypoints"][i],
                }
        return None

    def close(self) -> None:
        """Close the underlying NpzFile handle and release all in-memory cache data."""
        if self._loaded_data is not None:
            self._loaded_data.close()
            self._loaded_data = None
