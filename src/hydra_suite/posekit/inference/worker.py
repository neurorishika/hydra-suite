#!/usr/bin/env python3
"""Run SLEAP exported-model inference for a batch of images.

Executed inside the selected SLEAP conda environment.
Input: JSON path passed as argv[1]
Output: writes {"preds": {...}} JSON to cfg["out_json"].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hydra_suite.utils.hidden_files import is_hidden_file

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None


def _as_array(value: Any) -> Optional[np.ndarray]:
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


def _dict_first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _load_image(path_str: str) -> np.ndarray:
    path = str(path_str)
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    if Image is not None:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb)
            # Keep BGR-like convention used elsewhere in the pipeline.
            return arr[:, :, ::-1].copy()
    raise RuntimeError(
        "Unable to load image. Install either opencv-python or pillow in the SLEAP env."
    )


def _normalize_conf_values(conf: Any) -> np.ndarray:
    arr = np.asarray(conf, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if np.any((arr < 0.0) | (arr > 1.0)):
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


def _extract_xy_conf_from_raw(
    raw: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract raw xy and confidence arrays from various raw output formats."""
    if isinstance(raw, dict):
        xy_arr = _as_array(
            _dict_first_present(
                raw,
                (
                    "instance_peaks",
                    "pred_instance_peaks",
                    "keypoints",
                    "points",
                    "xy",
                ),
            )
        )
        conf_arr = _as_array(
            _dict_first_present(
                raw,
                (
                    "instance_peak_vals",
                    "pred_peak_vals",
                    "confidences",
                    "confidence",
                    "scores",
                ),
            )
        )
        return xy_arr, conf_arr
    if isinstance(raw, (tuple, list)) and len(raw) >= 1:
        xy_arr = _as_array(raw[0])
        conf_arr = _as_array(raw[1]) if len(raw) > 1 else None
        return xy_arr, conf_arr
    return _as_array(raw), None


def _reshape_xy(xy_arr: np.ndarray) -> Optional[np.ndarray]:
    """Reshape xy array to [B, K, 2], returning None on incompatible shapes."""
    xy_arr = np.asarray(xy_arr)
    if xy_arr.ndim == 4:
        xy_arr = xy_arr[:, 0, :, :]
    elif xy_arr.ndim == 2:
        xy_arr = xy_arr[None, :, :]
    elif xy_arr.ndim != 3:
        return None
    if xy_arr.shape[-1] < 2:
        return None
    return xy_arr[..., :2]


def _reshape_conf(conf_arr: np.ndarray) -> Optional[np.ndarray]:
    """Reshape confidence array to [B, K], returning None on incompatible shapes."""
    conf_arr = np.asarray(conf_arr)
    if conf_arr.ndim == 4:
        return conf_arr[:, 0, :, 0]
    if conf_arr.ndim == 3:
        if conf_arr.shape[-1] == 1:
            return conf_arr[:, :, 0]
        return conf_arr[:, 0, :]
    if conf_arr.ndim == 1:
        return conf_arr[None, :]
    if conf_arr.ndim == 2:
        return conf_arr
    return None


def _align_batch_dim(
    xy_arr: np.ndarray, conf_arr: Optional[np.ndarray], batch_size: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Ensure xy and conf arrays match the expected batch_size."""
    if xy_arr.shape[0] == batch_size:
        return xy_arr, conf_arr
    if xy_arr.shape[0] == 1 and batch_size > 1:
        xy_arr = np.repeat(xy_arr, batch_size, axis=0)
        if conf_arr is not None and conf_arr.shape[0] == 1:
            conf_arr = np.repeat(conf_arr, batch_size, axis=0)
        return xy_arr, conf_arr
    if batch_size == 1:
        xy_arr = xy_arr[:1]
        if conf_arr is not None:
            conf_arr = conf_arr[:1]
        return xy_arr, conf_arr
    return None, None


def _normalize_xy_conf(
    raw: Any, batch_size: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    xy_arr, conf_arr = _extract_xy_conf_from_raw(raw)

    if xy_arr is None:
        return None, None

    xy_arr = _reshape_xy(xy_arr)
    if xy_arr is None:
        return None, None

    if conf_arr is not None:
        conf_arr = _reshape_conf(conf_arr)

    if conf_arr is not None:
        conf_arr = _normalize_conf_values(conf_arr)

    xy_arr, conf_arr = _align_batch_dim(xy_arr, conf_arr, batch_size)
    if xy_arr is None:
        return None, None

    return xy_arr.astype(np.float32, copy=False), (
        conf_arr.astype(np.float32, copy=False) if conf_arr is not None else None
    )


def _extract_metadata_attr(meta: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(meta, dict) and name in meta:
            return meta.get(name)
        if hasattr(meta, name):
            return getattr(meta, name)
    return default


def _detect_input_spec(
    exported_model_path: str,
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    input_hw: Optional[Tuple[int, int]] = None
    input_channels: Optional[int] = None
    try:
        from sleap_nn.export.metadata import load_metadata  # type: ignore

        meta = load_metadata(str(exported_model_path))
    except Exception:
        meta = None

    if meta is None:
        return None, None

    hw = _extract_metadata_attr(
        meta,
        names=["crop_size", "crop_hw", "input_hw", "image_hw"],
        default=None,
    )
    if isinstance(hw, (list, tuple)) and len(hw) >= 2:
        try:
            input_hw = (int(hw[0]), int(hw[1]))
        except Exception:
            input_hw = None

    input_shape = _extract_metadata_attr(
        meta,
        names=["input_image_shape", "input_shape"],
        default=None,
    )
    if input_shape is not None:
        try:
            arr = np.asarray(input_shape).reshape((-1,))
            if arr.size >= 4:
                if int(arr[-1]) in (1, 3):
                    input_channels = int(arr[-1])
                    input_hw = (int(arr[-3]), int(arr[-2]))
                elif int(arr[1]) in (1, 3):
                    input_channels = int(arr[1])
                    input_hw = (int(arr[-2]), int(arr[-1]))
        except Exception:
            pass

    channels = _extract_metadata_attr(
        meta,
        names=["input_channels", "channels", "num_channels"],
        default=None,
    )
    if channels is not None:
        try:
            input_channels = int(channels)
        except Exception:
            pass

    return input_hw, input_channels


def _detect_input_spec_from_predictor(
    predictor: Any,
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    input_hw: Optional[Tuple[int, int]] = None
    input_channels: Optional[int] = None
    session = None
    for name in ("session", "_session", "ort_session"):
        cand = getattr(predictor, name, None)
        if cand is not None:
            session = cand
            break
    if session is None or not hasattr(session, "get_inputs"):
        return None, None
    try:
        inputs = session.get_inputs()
    except Exception:
        return None, None
    if not inputs:
        return None, None
    try:
        shape = list(getattr(inputs[0], "shape", []) or [])
    except Exception:
        shape = []
    dims: List[int] = []
    for d in shape:
        try:
            dims.append(int(d))
        except Exception:
            dims.append(-1)
    if len(dims) >= 4:
        if dims[-1] in (1, 3):
            input_channels = int(dims[-1])
            if dims[-3] > 0 and dims[-2] > 0:
                input_hw = (int(dims[-3]), int(dims[-2]))
        elif dims[1] in (1, 3):
            input_channels = int(dims[1])
            if dims[-2] > 0 and dims[-1] > 0:
                input_hw = (int(dims[-2]), int(dims[-1]))
    return input_hw, input_channels


def _prepare_crop_for_predictor(
    crop: np.ndarray,
    input_hw: Optional[Tuple[int, int]],
    input_channels: Optional[int],
) -> Tuple[np.ndarray, float, float]:
    arr = np.asarray(crop)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise RuntimeError(
            f"Invalid crop shape for SLEAP export predictor: {arr.shape}"
        )

    orig_h, orig_w = int(arr.shape[0]), int(arr.shape[1])
    if input_channels == 1 and arr.shape[2] != 1:
        if cv2 is not None:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)[:, :, None]
        else:
            arr = np.mean(arr[:, :, :3], axis=2, keepdims=True).astype(arr.dtype)
    elif (input_channels is None or input_channels == 3) and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if input_hw is not None:
        h, w = int(input_hw[0]), int(input_hw[1])
        if h > 0 and w > 0 and (orig_h != h or orig_w != w):
            if cv2 is None:
                raise RuntimeError(
                    "OpenCV is required for resized SLEAP exported inference."
                )
            if arr.shape[2] == 1:
                resized = cv2.resize(
                    arr[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR
                )
                arr = resized[:, :, None]
            else:
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

    pred_h, pred_w = int(arr.shape[0]), int(arr.shape[1])
    sx = float(orig_w) / float(pred_w) if pred_w > 0 else 1.0
    sy = float(orig_h) / float(pred_h) if pred_h > 0 else 1.0

    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return np.asarray(arr, dtype=np.uint8), sx, sy


def _prepare_batch_for_predictor(
    raw_crops: Sequence[np.ndarray],
    input_hw: Optional[Tuple[int, int]],
    input_channels: Optional[int],
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    crops: List[np.ndarray] = []
    scales: List[Tuple[float, float]] = []
    for c in raw_crops:
        prepared, sx, sy = _prepare_crop_for_predictor(c, input_hw, input_channels)
        crops.append(prepared)
        scales.append((sx, sy))
    return crops, scales


def _detect_predictor_input_format(predictor: Any) -> Optional[Dict[str, Any]]:
    """Detect expected dtype and layout (NCHW/NHWC) from an ONNX session."""
    session = None
    for attr in ("session", "_session", "ort_session", "_ort_session", "sess"):
        cand = getattr(predictor, attr, None)
        if cand is not None and hasattr(cand, "get_inputs"):
            session = cand
            break
    if session is None:
        return None
    try:
        inputs = session.get_inputs()
        if not inputs:
            return None
        inp = inputs[0]
        raw_type = str(getattr(inp, "type", "")).lower()
        shape = list(getattr(inp, "shape", []) or [])
    except Exception:
        return None
    is_float = "float" in raw_type
    dims: List[int] = []
    for d in shape:
        try:
            dims.append(int(d))
        except (TypeError, ValueError):
            dims.append(-1)
    layout = "nhwc"
    if len(dims) >= 4:
        if dims[1] in (1, 3):
            layout = "nchw"
        elif dims[-1] in (1, 3):
            layout = "nhwc"
    return {"is_float": is_float, "layout": layout}


def _min_batch_from_session(predictor: Any) -> Optional[int]:
    """Try to read minimum batch size from an ONNX Runtime session on the predictor."""
    for attr in ("session", "_session", "ort_session", "_ort_session", "sess"):
        cand = getattr(predictor, attr, None)
        if cand is not None and hasattr(cand, "get_inputs"):
            try:
                inputs = cand.get_inputs()
                if inputs:
                    shape = getattr(inputs[0], "shape", [])
                    if shape:
                        b = int(shape[0])
                        if b > 0:
                            return b
            except Exception:
                pass
    return None


def _min_batch_from_engine(predictor: Any) -> Optional[int]:
    """Try to read minimum batch size from a TensorRT engine on the predictor."""
    for attr in ("engine", "_engine", "trt_engine"):
        engine = getattr(predictor, attr, None)
        if engine is None:
            continue
        if hasattr(engine, "get_profile_shape"):
            try:
                min_s, _, _ = engine.get_profile_shape(0, 0)
                if min_s and int(min_s[0]) > 0:
                    return int(min_s[0])
            except Exception:
                pass
        if hasattr(engine, "get_binding_shape"):
            try:
                shape = list(engine.get_binding_shape(0))
                if shape and int(shape[0]) > 0:
                    return int(shape[0])
            except Exception:
                pass
    return None


def _detect_model_min_batch(predictor: Any) -> Optional[int]:
    """Detect the minimum required batch from the model session/engine."""
    result = _min_batch_from_session(predictor)
    if result is not None:
        return result
    return _min_batch_from_engine(predictor)


def _predict_batch(
    predictor: Any, crops: Sequence[np.ndarray], runtime_flavor: str
) -> Any:
    # Ensure all crops are 3D [H,W,C] for reliable stacking.
    crops_3d: List[np.ndarray] = []
    for c in crops:
        a = np.asarray(c, dtype=np.uint8)
        if a.ndim == 2:
            a = a[:, :, None]
        crops_3d.append(a)

    batch_uint8: Optional[np.ndarray] = None
    try:
        batch_uint8 = np.stack(crops_3d, axis=0)  # [B, H, W, C]
    except Exception:
        batch_uint8 = None

    # Detect model's expected input format from ONNX session if available.
    fmt = _detect_predictor_input_format(predictor)

    # If format detected, try it first (no guessing).
    if fmt is not None and batch_uint8 is not None:
        inp = (
            batch_uint8.astype(np.float32) / 255.0
            if fmt.get("is_float", True)
            else batch_uint8.copy()
        )
        if fmt.get("layout") == "nchw" and inp.ndim == 4:
            inp = np.transpose(inp, (0, 3, 1, 2))
        try:
            return predictor.predict(inp)
        except Exception:
            pass  # fall through to alternatives

    # Fallback: try numpy array formats first, list(crops) last.
    attempts: List[Any] = []
    if batch_uint8 is not None:
        if batch_uint8.ndim == 4:
            attempts.append(
                np.transpose(batch_uint8.astype(np.float32) / 255.0, (0, 3, 1, 2))
            )  # NCHW float
        attempts.append(batch_uint8.astype(np.float32) / 255.0)  # NHWC float
        attempts.append(batch_uint8)  # NHWC uint8
    attempts.append(list(crops))  # raw list – last resort

    last_err: Optional[Exception] = None
    for inp in attempts:
        try:
            return predictor.predict(inp)
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(
        f"SLEAP exported predictor failed for runtime '{runtime_flavor}': {last_err}"
    )


def _load_predictor(
    exported_model_path: str,
    runtime_flavor: str,
    device: str,
    batch: int,
    max_instances: int,
) -> Any:
    from sleap_nn.export.predictors import load_exported_model

    model_path = str(exported_model_path or "").strip()
    model_candidate = Path(model_path).expanduser().resolve()
    if model_candidate.is_dir():
        if runtime_flavor == "onnx":
            onnx_files = sorted(
                p for p in model_candidate.rglob("*.onnx") if not is_hidden_file(p)
            )
            if not onnx_files:
                raise RuntimeError(
                    f"No ONNX artifact found in export directory: {model_candidate}"
                )
            model_candidate = onnx_files[0]
        elif runtime_flavor == "tensorrt":
            engine_files = sorted(
                p
                for p in [
                    *model_candidate.rglob("*.engine"),
                    *model_candidate.rglob("*.trt"),
                ]
                if not is_hidden_file(p)
            )
            if not engine_files:
                raise RuntimeError(
                    f"No TensorRT artifact found in export directory: {model_candidate}"
                )
            model_candidate = engine_files[0]

    kwargs_base: Dict[str, Any] = {
        "batch_size": max(1, int(batch)),
        "max_instances": max(1, int(max_instances)),
    }
    if runtime_flavor == "onnx":
        if str(device).startswith("cuda"):
            kwargs_base["providers"] = ["CUDAExecutionProvider"]
        else:
            kwargs_base["providers"] = ["CPUExecutionProvider"]
    elif runtime_flavor == "tensorrt":
        if not str(device).startswith("cuda"):
            raise RuntimeError(
                f"SLEAP TensorRT runtime requires CUDA device, got: {device}"
            )
        kwargs_base["device"] = str(device)

    attempts = [
        {"runtime": runtime_flavor, **kwargs_base},
        {"inference_model": runtime_flavor, **kwargs_base},
        {"model_type": runtime_flavor, **kwargs_base},
        {"runtime": runtime_flavor},
        {},
    ]
    last_err: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return load_exported_model(str(model_candidate), **kwargs)
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Failed to initialize SLEAP exported predictor: {last_err}")


def main() -> int:
    cfg_path = Path(sys.argv[1]).expanduser().resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    images = [str(p) for p in (cfg.get("images") or [])]
    out_json = Path(str(cfg.get("out_json", ""))).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    runtime_flavor = str(cfg.get("runtime_flavor", "onnx")).strip().lower()
    exported_model_path = str(cfg.get("exported_model_path", "")).strip()
    device = str(cfg.get("device", "cpu")).strip().lower()
    batch_size = max(1, int(cfg.get("batch", 4)))
    max_instances = max(1, int(cfg.get("max_instances", 1)))
    num_kpts = max(1, int(cfg.get("num_keypoints", 1)))
    cfg_input_hw = cfg.get("input_hw")
    forced_input_hw: Optional[Tuple[int, int]] = None
    if isinstance(cfg_input_hw, (list, tuple)) and len(cfg_input_hw) >= 2:
        try:
            h = int(cfg_input_hw[0])
            w = int(cfg_input_hw[1])
            if h > 0 and w > 0:
                forced_input_hw = (h, w)
        except Exception:
            forced_input_hw = None

    predictor = _load_predictor(
        exported_model_path=exported_model_path,
        runtime_flavor=runtime_flavor,
        device=device,
        batch=batch_size,
        max_instances=max_instances,
    )
    model_min_batch = _detect_model_min_batch(predictor)
    if model_min_batch is not None and model_min_batch > batch_size:
        print(
            f"Model requires min batch={model_min_batch}, adjusting from {batch_size}",
            flush=True,
        )
        batch_size = model_min_batch
    input_hw, input_channels = _detect_input_spec(exported_model_path)
    if forced_input_hw is not None:
        input_hw = forced_input_hw
    if input_hw is None:
        hw2, ch2 = _detect_input_spec_from_predictor(predictor)
        input_hw = hw2 or input_hw
        input_channels = ch2 if ch2 is not None else input_channels
    if input_hw is None:
        raise RuntimeError(
            "SLEAP exported runtime requires fixed input shape metadata or an explicit input_hw. "
            "Re-export the model with fixed input height/width."
        )

    preds: Dict[str, List[Tuple[float, float, float]]] = {}
    total = len(images)
    done = 0

    for i in range(0, total, batch_size):
        chunk = images[i : i + batch_size]
        if not chunk:
            continue
        chunk_padded = list(chunk)
        if len(chunk_padded) < batch_size:
            chunk_padded.extend([chunk_padded[-1]] * (batch_size - len(chunk_padded)))

        raw_crops = [_load_image(p) for p in chunk_padded]
        crops, scales = _prepare_batch_for_predictor(
            raw_crops, input_hw, input_channels
        )
        raw = _predict_batch(predictor, crops, runtime_flavor)
        xy_arr, conf_arr = _normalize_xy_conf(raw, batch_size=len(chunk_padded))

        for j, path in enumerate(chunk):
            rows: List[Tuple[float, float, float]] = []
            if xy_arr is not None and j < xy_arr.shape[0]:
                xy = xy_arr[j]
                conf = (
                    conf_arr[j]
                    if conf_arr is not None and j < conf_arr.shape[0]
                    else np.zeros((xy.shape[0],), dtype=np.float32)
                )
                n = min(xy.shape[0], num_kpts)
                sx, sy = scales[j] if j < len(scales) else (1.0, 1.0)
                for k in range(n):
                    c = float(conf[k]) if k < len(conf) else 0.0
                    x = float(xy[k, 0]) * float(sx)
                    y = float(xy[k, 1]) * float(sy)
                    rows.append((x, y, float(np.clip(c, 0.0, 1.0))))
            if len(rows) < num_kpts:
                rows.extend([(0.0, 0.0, 0.0)] * (num_kpts - len(rows)))
            preds[str(Path(path))] = rows
            done += 1
            print(f"PROGRESS {done} {total}", flush=True)

    out_json.write_text(json.dumps({"preds": preds}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
