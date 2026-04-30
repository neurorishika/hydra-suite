"""Shared utility functions for pose runtime backends.

Device resolution, skeleton parsing, prediction coercion, and
other helper functions used across runtime implementations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hydra_suite.core.identity.pose.types import PoseResult
from hydra_suite.utils.gpu_utils import (
    CUDA_AVAILABLE,
    MPS_AVAILABLE,
    ONNXRUNTIME_AVAILABLE,
    SLEAP_RUNTIME_ONNX_AVAILABLE,
    SLEAP_RUNTIME_TENSORRT_AVAILABLE,
    TENSORRT_AVAILABLE,
    TORCH_CUDA_AVAILABLE,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# DEVICE RESOLUTION
# ==============================================================================


def resolve_device(requested: str, backend_family: str) -> str:
    """Resolve device string to actual device identifier."""
    req = str(requested or "auto").strip().lower()
    cuda_like_available = CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE
    if req in ("", "auto"):
        if cuda_like_available:
            return "cuda:0"
        if MPS_AVAILABLE:
            return "mps"
        return "cpu"

    if req in ("cuda", "gpu"):
        req = "cuda:0"
    if req.startswith("cuda") and not cuda_like_available:
        logger.warning(
            "Requested CUDA for %s runtime but no compatible torch CUDA backend is available. Falling back to CPU.",
            backend_family,
        )
        return "cpu"
    if req == "mps" and not MPS_AVAILABLE:
        logger.warning(
            "Requested MPS for %s runtime but MPS is unavailable. Falling back to CPU.",
            backend_family,
        )
        return "cpu"
    return req


# ==============================================================================
# SKELETON AND KEYPOINT PARSING
# ==============================================================================


def load_skeleton_from_json(path_str: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Load keypoint names and skeleton edges from JSON file."""
    if not path_str:
        return [], []
    p = Path(path_str).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise RuntimeError(f"Skeleton file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))

    names_raw = data.get("keypoint_names", data.get("keypoints", []))
    edges_raw = data.get("skeleton_edges", data.get("edges", []))

    names = [str(v).strip() for v in names_raw if str(v).strip()]
    edges: List[Tuple[int, int]] = []
    for edge in edges_raw:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        try:
            edges.append((int(edge[0]), int(edge[1])))
        except Exception:
            continue
    return names, edges


def normalize_conf_values(conf: Any) -> np.ndarray:
    """Normalize confidence values to [0, 1] range."""
    arr = np.asarray(conf, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if np.any((arr < 0.0) | (arr > 1.0)):
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


# ==============================================================================
# PREDICTION PARSING
# ==============================================================================


def extract_metadata_attr(meta: Any, names: Sequence[str], default: Any = None) -> Any:
    """Extract attribute from metadata object by trying sequence of names."""
    for name in names:
        if isinstance(meta, dict) and name in meta:
            return meta.get(name)
        if hasattr(meta, name):
            return getattr(meta, name)
    return default


def as_array(value: Any) -> Optional[np.ndarray]:
    """Convert value to numpy array, handling torch/tf tensors."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            return value.cpu().numpy()
        except Exception:
            pass
    try:
        return np.asarray(value)
    except Exception:
        return None


def dict_first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    """Return first key present in mapping."""
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def pick_best_instance(
    xy: np.ndarray, conf: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Pick best instance from multi-instance predictions."""
    if xy.ndim == 2 and xy.shape[1] == 2:
        n_kpts = int(xy.shape[0])
        if conf is None:
            conf_vec = np.zeros((n_kpts,), dtype=np.float32)
        else:
            conf_vec = normalize_conf_values(conf).reshape((-1,))
            if conf_vec.size < n_kpts:
                conf_vec = np.pad(conf_vec, (0, n_kpts - conf_vec.size))
            elif conf_vec.size > n_kpts:
                conf_vec = conf_vec[:n_kpts]
        return np.column_stack((xy.astype(np.float32), conf_vec))

    if xy.ndim == 3 and xy.shape[-1] == 2:
        n_instances = int(xy.shape[0])
        n_kpts = int(xy.shape[1])
        if n_instances <= 0:
            return None
        if conf is None:
            conf = np.zeros((n_instances, n_kpts), dtype=np.float32)
        else:
            conf = normalize_conf_values(conf)
            if conf.ndim == 1:
                conf = np.tile(conf[None, :], (n_instances, 1))
            elif conf.ndim > 2:
                conf = conf.reshape((n_instances, -1))
            if conf.shape[0] < n_instances:
                pad_rows = n_instances - conf.shape[0]
                conf = np.pad(conf, ((0, pad_rows), (0, 0)))
            elif conf.shape[0] > n_instances:
                conf = conf[:n_instances, :]
            if conf.shape[1] < n_kpts:
                pad_cols = n_kpts - conf.shape[1]
                conf = np.pad(conf, ((0, 0), (0, pad_cols)))
            elif conf.shape[1] > n_kpts:
                conf = conf[:, :n_kpts]
        mean_scores = np.nanmean(conf, axis=1)
        idx = int(np.nanargmax(mean_scores)) if len(mean_scores) else 0
        xy_i = np.asarray(xy[idx], dtype=np.float32)
        conf_i = normalize_conf_values(conf[idx])
        return np.column_stack((xy_i, conf_i))
    return None


def _pad_to_batch(
    out: List[Optional[np.ndarray]], batch_size: int
) -> List[Optional[np.ndarray]]:
    """Pad output list to batch_size with None entries and truncate."""
    if len(out) < batch_size:
        out.extend([None] * (batch_size - len(out)))
    return out[:batch_size]


def _coerce_dict_prediction(
    pred_out: Dict[str, Any], batch_size: int
) -> List[Optional[np.ndarray]]:
    """Coerce dict-style predictor output (instance_peaks + instance_peak_vals)."""
    xy = as_array(
        dict_first_present(
            pred_out,
            ["instance_peaks", "pred_instance_peaks", "peaks", "keypoints"],
        )
    )
    conf = as_array(
        dict_first_present(
            pred_out,
            [
                "instance_peak_vals",
                "pred_instance_peak_vals",
                "peak_vals",
                "scores",
                "confidences",
            ],
        )
    )
    if xy is None:
        return [None] * batch_size

    # Normalize to [B, ...].
    xy, conf = _normalize_dict_xy_conf(xy, conf, batch_size)
    if xy is None:
        return [None] * batch_size

    min_conf_ndim = 2 if xy.ndim == 3 else 3
    out: List[Optional[np.ndarray]] = []
    for b in range(min(batch_size, xy.shape[0])):
        conf_b = conf[b] if conf is not None and conf.ndim >= min_conf_ndim else None
        out.append(pick_best_instance(xy[b], conf_b))
    return _pad_to_batch(out, batch_size)


def _normalize_dict_xy_conf(
    xy: np.ndarray, conf: Optional[np.ndarray], batch_size: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalize xy/conf arrays from dict output to batch-leading shape."""
    if xy.ndim == 2 and xy.shape[1] == 2:
        xy = xy[None, :, :]
        if conf is not None and conf.ndim == 1:
            conf = conf[None, :]
    elif xy.ndim == 3 and xy.shape[-1] == 2:
        if int(xy.shape[0]) != int(batch_size):
            xy = xy[None, :, :, :]
            if conf is not None and conf.ndim == 2:
                conf = conf[None, :, :]
    elif xy.ndim == 4 and xy.shape[-1] == 2:
        pass
    else:
        return None, None
    return xy, conf


def _coerce_list_prediction(
    pred_out: Sequence[Any], batch_size: int
) -> List[Optional[np.ndarray]]:
    """Coerce list-style predictor output (one entry per crop)."""
    out: List[Optional[np.ndarray]] = []
    for item in list(pred_out)[:batch_size]:
        out.append(_coerce_single_list_item(item))
    return _pad_to_batch(out, batch_size)


def _coerce_single_list_item(item: Any) -> Optional[np.ndarray]:
    """Coerce a single item from list-style predictor output."""
    if isinstance(item, dict):
        parsed = coerce_prediction_batch(item, 1)
        return parsed[0] if parsed else None
    arr = as_array(item)
    if arr is None:
        return None
    if arr.ndim == 2 and arr.shape[1] == 3:
        return np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 2:
        conf = np.zeros((arr.shape[0],), dtype=np.float32)
        return np.column_stack((arr.astype(np.float32), conf))
    return None


def _coerce_raw_array(pred_out: Any, batch_size: int) -> List[Optional[np.ndarray]]:
    """Coerce raw ndarray predictor output."""
    empty = [None] * batch_size
    arr = as_array(pred_out)
    if arr is None:
        return empty
    if arr.ndim == 3 and arr.shape[-1] == 3:
        out = [
            np.asarray(arr[i], dtype=np.float32)
            for i in range(min(batch_size, arr.shape[0]))
        ]
        return _pad_to_batch(out, batch_size)
    if arr.ndim == 2 and arr.shape[1] == 3 and batch_size == 1:
        return [np.asarray(arr, dtype=np.float32)]
    return empty


def coerce_prediction_batch(
    pred_out: Any, batch_size: int
) -> List[Optional[np.ndarray]]:
    """Convert predictor output into per-crop Nx3 arrays (x, y, conf)."""
    empty = [None] * int(max(0, batch_size))
    if pred_out is None or batch_size <= 0:
        return empty

    if isinstance(pred_out, dict):
        return _coerce_dict_prediction(pred_out, batch_size)

    if isinstance(pred_out, (list, tuple)):
        return _coerce_list_prediction(pred_out, batch_size)

    return _coerce_raw_array(pred_out, batch_size)


# ==============================================================================
# RUNTIME FLAVOR PARSING
# ==============================================================================


def parse_runtime_request(requested: str) -> Tuple[str, Optional[str]]:
    """Parse runtime request string into (flavor, device)."""
    req = str(requested or "auto").strip().lower()
    req = req.replace("-", "_").replace(" ", "_")
    if req in {"", "auto"}:
        return "auto", None

    if req in {"native", "onnx", "tensorrt"}:
        return req, None

    for prefix in ("native", "onnx", "tensorrt"):
        token = f"{prefix}_"
        if req.startswith(token):
            suffix = req[len(token) :]
            if suffix in {"cpu", "mps"}:
                return prefix, suffix
            if suffix in {"cuda"}:
                return prefix, "cuda:0"
            return prefix, None

    return "native", None


def normalize_runtime_flavor(backend_family: str, requested: str) -> str:
    """Normalize runtime flavor request to actual available runtime."""
    req, _device = parse_runtime_request(requested)
    backend = str(backend_family or "yolo").strip().lower()
    if req in {"native", "onnx", "tensorrt"}:
        return req
    if req != "auto":
        return "native"
    if backend == "sleap":
        if SLEAP_RUNTIME_TENSORRT_AVAILABLE and CUDA_AVAILABLE:
            return "tensorrt"
        if SLEAP_RUNTIME_ONNX_AVAILABLE:
            return "onnx"
        return "native"
    if TENSORRT_AVAILABLE and CUDA_AVAILABLE:
        return "tensorrt"
    if ONNXRUNTIME_AVAILABLE:
        return "onnx"
    return "native"


# ==============================================================================
# GENERIC HELPERS
# ==============================================================================


def nested_get(mapping: Dict[str, Any], path: Sequence[str]) -> Any:
    """Get nested value from dict using path sequence."""
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def safe_pos_int(value: Any) -> Optional[int]:
    """Convert value to positive integer or return None."""
    try:
        v = int(value)
    except Exception:
        return None
    return v if v > 0 else None


def align_up(value: int, stride: int) -> int:
    """Align value up to stride."""
    s = max(1, int(stride))
    return int(((int(value) + s - 1) // s) * s)


def align_hw_to_stride(
    hw: Tuple[int, int], stride: int, min_size: int = 64, max_size: int = 1024
) -> Tuple[int, int]:
    """Align height and width to stride with min/max constraints."""
    h = max(1, int(hw[0]))
    w = max(1, int(hw[1]))
    h = align_up(h, stride)
    w = align_up(w, stride)
    h = max(int(min_size), min(int(max_size), h))
    w = max(int(min_size), min(int(max_size), w))
    return int(h), int(w)


def load_structured_config(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON or YAML config file."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(text)
            return data if isinstance(data, dict) else None
    except Exception:
        return None

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def empty_pose_result() -> PoseResult:
    """Create an empty PoseResult."""
    return PoseResult(
        keypoints=None,
        mean_conf=0.0,
        valid_fraction=0.0,
        num_valid=0,
        num_keypoints=0,
    )


def summarize_keypoints(
    keypoints: Optional[np.ndarray], min_valid_conf: float
) -> PoseResult:
    """Summarize keypoints into a PoseResult."""
    if keypoints is None:
        return empty_pose_result()
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return empty_pose_result()
    arr = arr.copy()
    arr[:, 2] = normalize_conf_values(arr[:, 2])
    valid_mask = arr[:, 2] >= float(min_valid_conf)
    return PoseResult(
        keypoints=arr,
        mean_conf=float(np.nanmean(arr[:, 2])),
        valid_fraction=float(np.mean(valid_mask)),
        num_valid=int(np.sum(valid_mask)),
        num_keypoints=int(len(arr)),
    )
