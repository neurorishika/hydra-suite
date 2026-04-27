"""Utilities for exporting per-detection analysis properties into wide CSV outputs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from hydra_suite.core.identity.classification.cnn import CNNIdentityCache

from .cache import IndividualPropertiesCache
from .detected_cache import DetectedPropertiesCache

POSE_SUMMARY_COLUMNS = [
    "PoseMeanConf",
    "PoseValidFraction",
    "PoseNumValid",
    "PoseNumKeypoints",
]

DETECTED_HEADING_COLUMNS = [
    "HeadingResolved",  # final disambiguated heading angle after head-tail
    "HeadingMethod",  # source used: "headtail", "pose", "velocity", "default"
    "HeadingIsDirected",  # True when head-vs-tail direction was successfully resolved
    "HeadTailAngleRad",  # angle from head-tail classifier (may differ from HeadingResolved)
    "HeadTailClassifierConf",  # raw confidence from the head-tail model
]


def _pose_value_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("Pose")]


def _cnn_value_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("CNN_")]


def _sanitize_cnn_factor_name(name: str, idx: int) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()
    if not token:
        token = f"factor{idx:02d}"
    return token


def _sanitize_class_name(name: str, idx: int) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()
    if not token:
        token = f"cls{idx:02d}"
    return token


def build_cnn_prob_columns(
    label: str,
    factor_names: Sequence[str] | None,
    class_names_per_factor: "tuple[tuple[str, ...], ...] | None",
) -> List[Tuple[int, int, str]]:
    """Return ``(factor_idx, class_idx, col_name)`` for per-class prob columns.

    Returns empty list when *class_names_per_factor* is ``None`` or empty.
    Single-factor → ``CNN_{label}_{class}_Prob``.
    Multi-factor  → ``CNN_{label}_{factor}_{class}_Prob``.
    """
    if not class_names_per_factor:
        return []
    fn_tuple = tuple(str(n) for n in (factor_names or ()))
    is_multihead = len(fn_tuple) > 1
    specs: List[Tuple[int, int, str]] = []
    used: dict[str, int] = {}
    for factor_idx, class_names in enumerate(class_names_per_factor):
        factor_tok = (
            _sanitize_cnn_factor_name(
                fn_tuple[factor_idx] if factor_idx < len(fn_tuple) else "",
                factor_idx,
            )
            if is_multihead
            else None
        )
        for class_idx, class_name in enumerate(class_names):
            class_tok = _sanitize_class_name(class_name, class_idx)
            base_col = (
                f"CNN_{label}_{factor_tok}_{class_tok}_Prob"
                if is_multihead
                else f"CNN_{label}_{class_tok}_Prob"
            )
            count = used.get(base_col, 0)
            col = base_col if count == 0 else f"{base_col}_{count}"
            used[base_col] = count + 1
            specs.append((factor_idx, class_idx, col))
    return specs


def build_cnn_output_columns(
    label: str,
    factor_names: Sequence[str] | None,
) -> List[Tuple[str | None, str, str]]:
    """Return ordered ``(factor, class_col, conf_col)`` specs for a classifier."""
    names = tuple(str(name) for name in (factor_names or ()) if str(name).strip())
    if len(names) <= 1:
        return [(None, f"CNN_{label}_Class", f"CNN_{label}_Conf")]

    specs: List[Tuple[str | None, str, str]] = []
    used: dict[str, int] = {}
    for idx, name in enumerate(names):
        base_factor = _sanitize_cnn_factor_name(name, idx)
        count = used.get(base_factor, 0)
        factor = base_factor if count == 0 else f"{base_factor}_{count}"
        used[base_factor] = count + 1
        specs.append(
            (
                factor,
                f"CNN_{label}_{factor}_Class",
                f"CNN_{label}_{factor}_Conf",
            )
        )
    return specs


def flatten_cnn_prediction_row(
    label: str,
    factor_names: Sequence[str] | None,
    class_names: Sequence[str | None] | None,
    confidences: Sequence[float] | None,
) -> Dict[str, Any]:
    """Flatten one CNN prediction into export-ready columns.

    Flat single-head models keep the legacy ``CNN_<label>_Class`` /
    ``CNN_<label>_Conf`` columns. Multi-head models expand to one class/conf pair
    per factor.
    """
    specs = build_cnn_output_columns(label, factor_names)
    class_names = tuple(class_names or ())
    confidences = tuple(confidences or ())
    row: Dict[str, Any] = {}
    for idx, (_factor, class_col, conf_col) in enumerate(specs):
        class_name = class_names[idx] if idx < len(class_names) else None
        row[class_col] = str(class_name) if class_name is not None else np.nan
        row[conf_col] = float(confidences[idx]) if idx < len(confidences) else np.nan
    return row


def _detected_heading_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c in DETECTED_HEADING_COLUMNS]


def _sanitize_keypoint_name(name: str, idx: int) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()
    if not token:
        token = f"kp{idx:03d}"
    return token


def build_pose_keypoint_labels(
    keypoint_names: Optional[Sequence[str]], max_keypoints: int
) -> List[str]:
    """Build a deduplicated, sanitized list of column-safe keypoint label strings.

    Uses provided names where available, appending auto-generated 'kp###' labels
    to reach at least *max_keypoints* entries.
    """
    names = list(keypoint_names or [])
    labels: List[str] = []
    used = set()
    for idx, name in enumerate(names):
        base = _sanitize_keypoint_name(name, idx)
        label = base
        if label in used:
            label = f"{base}_{idx:03d}"
        used.add(label)
        labels.append(label)

    total = max(int(max_keypoints), len(labels))
    for idx in range(len(labels), total):
        label = f"kp{idx:03d}"
        while label in used:
            label = f"{label}_{idx:03d}"
        used.add(label)
        labels.append(label)
    return labels


def pose_wide_columns_for_labels(labels: Sequence[str]) -> List[str]:
    """Return the ordered list of wide-format column names (X, Y, Conf) for each keypoint label."""
    cols: List[str] = []
    for label in labels:
        cols.extend(
            [
                f"PoseKpt_{label}_X",
                f"PoseKpt_{label}_Y",
                f"PoseKpt_{label}_Conf",
            ]
        )
    return cols


def _coerce_token(text: str) -> Any:
    """Try to parse text as int, otherwise return as string."""
    try:
        return int(text)
    except ValueError:
        return text


def _parse_ignore_keypoint_tokens(ignore_keypoints: Any) -> List[Any]:
    if ignore_keypoints is None:
        return []
    if isinstance(ignore_keypoints, str):
        raw_tokens = ignore_keypoints.split(",")
    elif isinstance(ignore_keypoints, (list, tuple, set)):
        raw_tokens = ignore_keypoints
    else:
        return []

    out: List[Any] = []
    for value in raw_tokens:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        out.append(_coerce_token(text))
    return out


def _resolve_ignored_keypoint_indices(
    ignore_keypoints: Any, keypoint_names: Optional[Sequence[str]]
) -> Set[int]:
    tokens = _parse_ignore_keypoint_tokens(ignore_keypoints)
    if not tokens:
        return set()
    names = [str(v) for v in (keypoint_names or [])]
    name_to_idx = {name.lower(): idx for idx, name in enumerate(names)}
    ignore_idxs: Set[int] = set()
    for token in tokens:
        if isinstance(token, int):
            ignore_idxs.add(int(token))
            continue
        idx = name_to_idx.get(str(token).strip().lower())
        if idx is not None:
            ignore_idxs.add(int(idx))
    return ignore_idxs


def _apply_ignore_to_keypoints(
    keypoints: np.ndarray, ignore_indices: Set[int]
) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    if not ignore_indices:
        return arr
    keep = [i for i in range(len(arr)) if i not in ignore_indices]
    if not keep:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(arr[keep], dtype=np.float32)


def _filter_labels_for_ignore(
    labels: Sequence[str],
    keypoint_names: Optional[Sequence[str]],
    ignore_indices: Set[int],
) -> Tuple[List[str], Set[int]]:
    if not labels:
        return [], set()
    if not ignore_indices:
        return list(labels), set()

    # Indices align with pose keypoint index order.
    dropped_label_indices = {
        idx for idx in ignore_indices if 0 <= int(idx) < len(labels)
    }
    filtered = [
        label for idx, label in enumerate(labels) if idx not in dropped_label_indices
    ]
    return filtered, dropped_label_indices


def flatten_pose_keypoints_row(
    keypoints: Any, labels: Sequence[str]
) -> Dict[str, float]:
    """Expand a (K, 3) keypoints array into a flat dict of PoseKpt_<label>_X/Y/Conf columns."""
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        arr = np.zeros((0, 3), dtype=np.float32)

    row: Dict[str, float] = {}
    for idx, label in enumerate(labels):
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"
        if idx < len(arr):
            row[x_col] = float(arr[idx, 0])
            row[y_col] = float(arr[idx, 1])
            row[c_col] = float(np.clip(arr[idx, 2], 0.0, 1.0))
        else:
            row[x_col] = np.nan
            row[y_col] = np.nan
            row[c_col] = np.nan
    return row


def build_pose_lookup_dataframe(
    cache: IndividualPropertiesCache,
    keypoint_names: Optional[Sequence[str]] = None,
    ignore_keypoints: Any = None,
    min_valid_conf: float = 0.2,
) -> pd.DataFrame:
    """Flatten cache entries into frame+detection keyed wide pose rows.

    Args:
        cache: Cache to read from
        keypoint_names: Optional keypoint names for labeling
        ignore_keypoints: Keypoints to ignore
        min_valid_conf: Minimum confidence threshold for keypoint validity
    """
    entries: List[Dict[str, Any]] = []
    max_keypoints = 0
    ignore_indices = _resolve_ignored_keypoint_indices(ignore_keypoints, keypoint_names)

    for frame_idx in cache.get_cached_frames():
        # Pass min_valid_conf to compute stats on-demand
        frame = cache.get_frame(int(frame_idx), min_valid_conf=min_valid_conf)
        ids = frame.get("detection_ids", [])
        mean_conf = frame.get("pose_mean_conf", [])
        valid_fraction = frame.get("pose_valid_fraction", [])
        num_valid = frame.get("pose_num_valid", [])
        num_keypoints = frame.get("pose_num_keypoints", [])
        keypoints = frame.get("pose_keypoints", [])
        count = min(
            len(ids),
            len(mean_conf),
            len(valid_fraction),
            len(num_valid),
            len(num_keypoints),
            len(keypoints),
        )
        for idx in range(count):
            try:
                det_id_int = int(ids[idx])
            except Exception:
                continue
            kpts = np.asarray(keypoints[idx], dtype=np.float32)
            kpts = _apply_ignore_to_keypoints(kpts, ignore_indices)
            max_keypoints = max(max_keypoints, int(len(kpts)))
            entries.append(
                {
                    "_pose_frame_id": int(frame_idx),
                    "_pose_detection_id": det_id_int,
                    "PoseMeanConf": float(np.clip(mean_conf[idx], 0.0, 1.0)),
                    "PoseValidFraction": float(np.clip(valid_fraction[idx], 0.0, 1.0)),
                    "PoseNumValid": int(num_valid[idx]),
                    "PoseNumKeypoints": int(num_keypoints[idx]),
                    "_pose_keypoints": kpts,
                }
            )

    labels = build_pose_keypoint_labels(keypoint_names, max_keypoints)
    labels, _ = _filter_labels_for_ignore(labels, keypoint_names, ignore_indices)
    keypoint_cols = pose_wide_columns_for_labels(labels)
    if not entries:
        return pd.DataFrame(
            columns=[
                "_pose_frame_id",
                "_pose_detection_id",
                *POSE_SUMMARY_COLUMNS,
                *keypoint_cols,
            ]
        )

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        row = {
            "_pose_frame_id": entry["_pose_frame_id"],
            "_pose_detection_id": entry["_pose_detection_id"],
            "PoseMeanConf": entry["PoseMeanConf"],
            "PoseValidFraction": entry["PoseValidFraction"],
            "PoseNumValid": entry["PoseNumValid"],
            "PoseNumKeypoints": entry["PoseNumKeypoints"],
        }
        row.update(flatten_pose_keypoints_row(entry["_pose_keypoints"], labels))
        rows.append(row)

    return pd.DataFrame(rows)


def augment_trajectories_with_pose_df(
    trajectories_df: pd.DataFrame,
    pose_lookup_df: pd.DataFrame,
    coordinate_scale: float = 1.0,
) -> pd.DataFrame:
    """Merge pose properties into trajectory rows using (FrameID, DetectionID).

    Args:
        trajectories_df: Trajectories dataframe to augment.
        pose_lookup_df: Wide pose lookup dataframe from build_pose_lookup_dataframe.
        coordinate_scale: Scale factor applied to all PoseKpt_*_X and PoseKpt_*_Y
            columns after merging.  Set to ``1.0 / RESIZE_FACTOR`` when the pose
            cache was built on down-scaled frames but the trajectory CSV already
            contains original-resolution coordinates.
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if (
        "FrameID" not in trajectories_df.columns
        or "DetectionID" not in trajectories_df.columns
    ):
        return trajectories_df.copy()

    out = trajectories_df.copy()
    out = out.drop(columns=_pose_value_columns(out), errors="ignore")

    pose_cols = (
        _pose_value_columns(pose_lookup_df) if pose_lookup_df is not None else []
    )
    if pose_lookup_df is None or pose_lookup_df.empty:
        for col in POSE_SUMMARY_COLUMNS:
            out[col] = np.nan
        return out

    out["_frame_join"] = (
        pd.to_numeric(out["FrameID"], errors="coerce").round().astype("Int64")
    )
    out["_detection_join"] = (
        pd.to_numeric(out["DetectionID"], errors="coerce").round().astype("Int64")
    )

    lookup = pose_lookup_df.copy()
    lookup["_pose_frame_id"] = (
        pd.to_numeric(lookup["_pose_frame_id"], errors="coerce").round().astype("Int64")
    )
    lookup["_pose_detection_id"] = (
        pd.to_numeric(lookup["_pose_detection_id"], errors="coerce")
        .round()
        .astype("Int64")
    )

    merged = out.merge(
        lookup[["_pose_frame_id", "_pose_detection_id", *pose_cols]],
        how="left",
        left_on=["_frame_join", "_detection_join"],
        right_on=["_pose_frame_id", "_pose_detection_id"],
        sort=False,
    )
    merged.drop(
        columns=[
            "_frame_join",
            "_detection_join",
            "_pose_frame_id",
            "_pose_detection_id",
        ],
        inplace=True,
        errors="ignore",
    )
    for col in POSE_SUMMARY_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    # Rescale keypoint X/Y when the cache was built on down-scaled frames
    if coordinate_scale != 1.0 and coordinate_scale > 0.0:
        for col in merged.columns:
            col_str = str(col)
            if col_str.startswith("PoseKpt_") and (
                col_str.endswith("_X") or col_str.endswith("_Y")
            ):
                merged[col] = (
                    pd.to_numeric(merged[col], errors="coerce") * coordinate_scale
                )

    return merged


def augment_trajectories_with_pose_cache(
    trajectories_df: pd.DataFrame,
    cache_path: str,
    ignore_keypoints: Any = None,
    min_valid_conf: float = 0.2,
    coordinate_scale: float = 1.0,
) -> pd.DataFrame:
    """Load properties cache and merge wide pose columns into trajectory rows.

    Args:
        trajectories_df: Trajectories dataframe to augment
        cache_path: Path to individual properties cache
        ignore_keypoints: Keypoints to ignore
        min_valid_conf: Minimum confidence threshold for keypoint validity
        coordinate_scale: Scale factor applied to PoseKpt_*_X / _Y columns.
            Set to ``1.0 / RESIZE_FACTOR`` when the cache was built on
            down-scaled frames.
    """
    cache = IndividualPropertiesCache(cache_path, mode="r")
    try:
        if not cache.is_compatible():
            raise RuntimeError(
                f"Incompatible individual-properties cache: {cache_path}"
            )
        names = cache.metadata.get("pose_keypoint_names", [])
        if not isinstance(names, (list, tuple)):
            names = []
        lookup = build_pose_lookup_dataframe(
            cache,
            keypoint_names=names,
            ignore_keypoints=ignore_keypoints,
            min_valid_conf=min_valid_conf,
        )
    finally:
        cache.close()
    return augment_trajectories_with_pose_df(
        trajectories_df, lookup, coordinate_scale=coordinate_scale
    )


def build_detected_properties_lookup_dataframe(
    cache: DetectedPropertiesCache,
) -> pd.DataFrame:
    """Flatten detected-properties cache entries into frame+detection keyed rows."""
    entries: List[Dict[str, Any]] = []
    for frame_idx in cache.get_cached_frames():
        frame = cache.get_frame(int(frame_idx))
        detection_ids = frame.get("detection_ids", [])
        heading_resolved = frame.get("HeadingResolved", [])
        heading_method = frame.get("HeadingMethod", [])
        heading_is_directed = frame.get("HeadingIsDirected", [])
        headtail_angle = frame.get("HeadTailAngleRad", [])
        headtail_conf = frame.get("HeadTailClassifierConf", [])
        count = min(
            len(detection_ids),
            len(heading_resolved),
            len(heading_method),
            len(heading_is_directed),
            len(headtail_angle),
            len(headtail_conf),
        )
        for idx in range(count):
            try:
                det_id = int(detection_ids[idx])
            except Exception:
                continue
            entries.append(
                {
                    "_detprop_frame_id": int(frame_idx),
                    "_detprop_detection_id": det_id,
                    "HeadingResolved": heading_resolved[idx],
                    "HeadingMethod": heading_method[idx],
                    "HeadingIsDirected": bool(heading_is_directed[idx]),
                    "HeadTailAngleRad": headtail_angle[idx],
                    "HeadTailClassifierConf": headtail_conf[idx],
                }
            )
    return pd.DataFrame(
        entries,
        columns=[
            "_detprop_frame_id",
            "_detprop_detection_id",
            *DETECTED_HEADING_COLUMNS,
        ],
    )


def augment_trajectories_with_detected_properties_df(
    trajectories_df: pd.DataFrame,
    detected_lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge detected-frame heading metadata into trajectory rows by detection."""
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if (
        "FrameID" not in trajectories_df.columns
        or "DetectionID" not in trajectories_df.columns
    ):
        return trajectories_df.copy()

    out = trajectories_df.copy()
    out = out.drop(columns=_detected_heading_columns(out), errors="ignore")
    if detected_lookup_df is None or detected_lookup_df.empty:
        return _ensure_interp_columns(out, DETECTED_HEADING_COLUMNS)

    out["_frame_join"] = (
        pd.to_numeric(out["FrameID"], errors="coerce").round().astype("Int64")
    )
    out["_detection_join"] = (
        pd.to_numeric(out["DetectionID"], errors="coerce").round().astype("Int64")
    )

    lookup = detected_lookup_df.copy()
    lookup["_detprop_frame_id"] = (
        pd.to_numeric(lookup["_detprop_frame_id"], errors="coerce")
        .round()
        .astype("Int64")
    )
    lookup["_detprop_detection_id"] = (
        pd.to_numeric(lookup["_detprop_detection_id"], errors="coerce")
        .round()
        .astype("Int64")
    )

    merged = out.merge(
        lookup[
            ["_detprop_frame_id", "_detprop_detection_id", *DETECTED_HEADING_COLUMNS]
        ],
        how="left",
        left_on=["_frame_join", "_detection_join"],
        right_on=["_detprop_frame_id", "_detprop_detection_id"],
        sort=False,
    )
    merged.drop(
        columns=[
            "_frame_join",
            "_detection_join",
            "_detprop_frame_id",
            "_detprop_detection_id",
        ],
        inplace=True,
        errors="ignore",
    )
    return _ensure_interp_columns(merged, DETECTED_HEADING_COLUMNS)


def augment_trajectories_with_detected_properties_cache(
    trajectories_df: pd.DataFrame,
    cache_path: str,
) -> pd.DataFrame:
    """Load detected-properties cache and merge its canonical heading columns."""
    cache = DetectedPropertiesCache(cache_path, mode="r")
    try:
        if not cache.is_compatible():
            raise RuntimeError(f"Incompatible detected-properties cache: {cache_path}")
        lookup = build_detected_properties_lookup_dataframe(cache)
    finally:
        cache.close()
    return augment_trajectories_with_detected_properties_df(trajectories_df, lookup)


def build_detected_cnn_lookup_dataframe(
    cache: CNNIdentityCache,
    label: str = "cnn_identity",
) -> pd.DataFrame:
    """Flatten detected-frame CNN predictions into frame+detection keyed rows.

    When the cache stores per-class probability vectors (v3 schema), one
    ``CNN_{label}_{class}_Prob`` column is added per class per factor so that
    the full output distribution is available in the exported CSV.
    """
    specs = build_cnn_output_columns(label, cache.factor_names)
    output_cols = [
        col for _factor, class_col, conf_col in specs for col in (class_col, conf_col)
    ]
    prob_specs = build_cnn_prob_columns(
        label, cache.factor_names, cache.class_names_per_factor
    )
    prob_cols = [col for _fi, _ci, col in prob_specs]

    rows: List[Dict[str, Any]] = []
    for frame_idx in cache.get_cached_frames():
        preds = cache.load(int(frame_idx))
        probs_list = cache.load_probs(int(frame_idx)) if prob_specs else None
        for pred_idx, pred in enumerate(preds):
            row: Dict[str, Any] = {
                "_cnn_frame_id": int(frame_idx),
                "_cnn_detection_id": int(frame_idx) * 10000 + int(pred.det_index),
            }
            row.update(
                flatten_cnn_prediction_row(
                    label,
                    pred.factor_names,
                    pred.class_names,
                    pred.confidences,
                )
            )
            if prob_specs:
                per_det_probs = (
                    probs_list[pred_idx]
                    if probs_list is not None and pred_idx < len(probs_list)
                    else None
                )
                for factor_idx, class_idx, col in prob_specs:
                    val: float = np.nan
                    if per_det_probs is not None and factor_idx < len(per_det_probs):
                        fprobs = per_det_probs[factor_idx]
                        if fprobs is not None and class_idx < len(fprobs):
                            val = float(fprobs[class_idx])
                    row[col] = val
            rows.append(row)
    return pd.DataFrame(
        rows,
        columns=["_cnn_frame_id", "_cnn_detection_id", *output_cols, *prob_cols],
    )


def augment_trajectories_with_detected_cnn_df(
    trajectories_df: pd.DataFrame,
    detected_cnn_df: pd.DataFrame,
    label: str = "cnn_identity",
) -> pd.DataFrame:
    """Merge detected-frame CNN predictions into trajectory rows by detection."""
    output_cols = (
        _cnn_value_columns(detected_cnn_df) if detected_cnn_df is not None else []
    )
    if not output_cols:
        output_cols = [f"CNN_{label}_Class", f"CNN_{label}_Conf"]
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if (
        "FrameID" not in trajectories_df.columns
        or "DetectionID" not in trajectories_df.columns
    ):
        return trajectories_df.copy()

    out = trajectories_df.copy()
    if detected_cnn_df is None or detected_cnn_df.empty:
        return _ensure_interp_columns(out, output_cols)

    out["_frame_join"] = (
        pd.to_numeric(out["FrameID"], errors="coerce").round().astype("Int64")
    )
    out["_detection_join"] = (
        pd.to_numeric(out["DetectionID"], errors="coerce").round().astype("Int64")
    )

    lookup = detected_cnn_df.copy()
    lookup["_cnn_frame_id"] = (
        pd.to_numeric(lookup["_cnn_frame_id"], errors="coerce").round().astype("Int64")
    )
    lookup["_cnn_detection_id"] = (
        pd.to_numeric(lookup["_cnn_detection_id"], errors="coerce")
        .round()
        .astype("Int64")
    )

    merged = out.merge(
        lookup[["_cnn_frame_id", "_cnn_detection_id", *output_cols]],
        how="left",
        left_on=["_frame_join", "_detection_join"],
        right_on=["_cnn_frame_id", "_cnn_detection_id"],
        sort=False,
    )
    merged.drop(
        columns=[
            "_frame_join",
            "_detection_join",
            "_cnn_frame_id",
            "_cnn_detection_id",
        ],
        inplace=True,
        errors="ignore",
    )
    return _ensure_interp_columns(merged, output_cols)


def augment_trajectories_with_detected_cnn_cache(
    trajectories_df: pd.DataFrame,
    cache_path: str,
    label: str = "cnn_identity",
) -> pd.DataFrame:
    """Load detected-frame CNN cache and merge class/confidence columns."""
    cache = CNNIdentityCache(cache_path)
    lookup = build_detected_cnn_lookup_dataframe(cache, label=label)
    return augment_trajectories_with_detected_cnn_df(
        trajectories_df, lookup, label=label
    )


def _ensure_pose_columns(
    df: pd.DataFrame, extra_cols: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    out = df.copy()
    for col in list(POSE_SUMMARY_COLUMNS) + list(extra_cols or []):
        if col.startswith("Pose") and col not in out.columns:
            out[col] = np.nan
    return out


def merge_interpolated_pose_df(
    trajectories_df: pd.DataFrame, interpolated_pose_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Fill missing pose rows from interpolated analysis keyed by (FrameID, TrajectoryID).

    Existing detection-keyed pose values are preserved.
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if (
        interpolated_pose_df is None
        or interpolated_pose_df.empty
        or "FrameID" not in trajectories_df.columns
        or "TrajectoryID" not in trajectories_df.columns
    ):
        return _ensure_pose_columns(trajectories_df)

    pose_cols_interp = [
        c for c in interpolated_pose_df.columns if str(c).startswith("Pose")
    ]
    if not pose_cols_interp or not {"frame_id", "trajectory_id"}.issubset(
        interpolated_pose_df.columns
    ):
        return _ensure_pose_columns(trajectories_df)

    out = _ensure_pose_columns(trajectories_df, extra_cols=pose_cols_interp)
    out["_frame_join"] = (
        pd.to_numeric(out["FrameID"], errors="coerce").round().astype("Int64")
    )
    out["_traj_join"] = (
        pd.to_numeric(out["TrajectoryID"], errors="coerce").round().astype("Int64")
    )

    interp = interpolated_pose_df.copy()
    interp["_frame_join"] = (
        pd.to_numeric(interp["frame_id"], errors="coerce").round().astype("Int64")
    )
    interp["_traj_join"] = (
        pd.to_numeric(interp["trajectory_id"], errors="coerce").round().astype("Int64")
    )
    interp_lookup = interp[
        ["_frame_join", "_traj_join", *pose_cols_interp]
    ].drop_duplicates(subset=["_frame_join", "_traj_join"], keep="first")

    merged = out.merge(
        interp_lookup,
        how="left",
        on=["_frame_join", "_traj_join"],
        suffixes=("", "_interp"),
        sort=False,
    )
    for col in pose_cols_interp:
        interp_col = f"{col}_interp"
        if interp_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = merged[col].where(merged[col].notna(), merged[interp_col])
    merged.drop(
        columns=[
            "_frame_join",
            "_traj_join",
            *[f"{c}_interp" for c in pose_cols_interp],
        ],
        inplace=True,
        errors="ignore",
    )
    return _ensure_pose_columns(merged)


# ---------------------------------------------------------------------------
# Shared interpolated merge helpers
# ---------------------------------------------------------------------------


def _ensure_interp_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Ensure columns exist in the dataframe, filling missing ones with NaN."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _prepare_interp_join_keys(
    trajectories_df: pd.DataFrame, interp_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add _frame_join and _traj_join keys to both dataframes."""
    out = trajectories_df.copy()
    out["_frame_join"] = (
        pd.to_numeric(out["FrameID"], errors="coerce").round().astype("Int64")
    )
    out["_traj_join"] = (
        pd.to_numeric(out["TrajectoryID"], errors="coerce").round().astype("Int64")
    )
    interp = interp_df.copy()
    interp["_frame_join"] = (
        pd.to_numeric(interp["frame_id"], errors="coerce").round().astype("Int64")
    )
    interp["_traj_join"] = (
        pd.to_numeric(interp["trajectory_id"], errors="coerce").round().astype("Int64")
    )
    return out, interp


def _backfill_interp_columns(
    merged: pd.DataFrame,
    column_map: Dict[str, str],
) -> pd.DataFrame:
    """Fill target columns from source columns and drop source columns."""
    for src_col, tgt_col in column_map.items():
        if src_col in merged.columns:
            if tgt_col not in merged.columns:
                merged[tgt_col] = np.nan
            merged[tgt_col] = merged[tgt_col].where(
                merged[tgt_col].notna(), merged[src_col]
            )
            merged.drop(columns=[src_col], inplace=True, errors="ignore")
    return merged


def _can_merge_interp(
    trajectories_df: pd.DataFrame,
    interp_df: Optional[pd.DataFrame],
    required_cols: Set[str],
) -> bool:
    """Check whether an interpolated merge is feasible."""
    if trajectories_df is None or trajectories_df.empty:
        return False
    if interp_df is None or interp_df.empty:
        return False
    if "FrameID" not in trajectories_df.columns:
        return False
    if "TrajectoryID" not in trajectories_df.columns:
        return False
    if not required_cols.issubset(interp_df.columns):
        return False
    return True


# ---------------------------------------------------------------------------
# AprilTag interpolated merge
# ---------------------------------------------------------------------------

APRILTAG_INTERP_COLUMNS = ["InterpTagID", "InterpTagHamming", "InterpTagConf"]


def merge_interpolated_apriltag_df(
    trajectories_df: pd.DataFrame,
    interp_tag_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge interpolated AprilTag observations into final trajectories.

    Existing ``TagID`` values (from pre-tracking) are preserved.
    Interpolated rows receive ``InterpTagID`` / ``InterpTagHamming``.
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if not _can_merge_interp(
        trajectories_df, interp_tag_df, {"frame_id", "trajectory_id", "tag_id"}
    ):
        return trajectories_df

    out, interp = _prepare_interp_join_keys(trajectories_df, interp_tag_df)
    out = _ensure_interp_columns(out, APRILTAG_INTERP_COLUMNS)

    tag_cols = ["tag_id"]
    if "hamming" in interp.columns:
        tag_cols.append("hamming")
    if "confidence" in interp.columns:
        tag_cols.append("confidence")

    interp_lookup = interp[["_frame_join", "_traj_join", *tag_cols]].drop_duplicates(
        subset=["_frame_join", "_traj_join"], keep="first"
    )

    merged = out.merge(
        interp_lookup,
        how="left",
        on=["_frame_join", "_traj_join"],
        suffixes=("", "_itag"),
        sort=False,
    )

    merged = _backfill_interp_columns(
        merged,
        {
            "tag_id": "InterpTagID",
            "hamming": "InterpTagHamming",
            "confidence": "InterpTagConf",
        },
    )

    merged.drop(
        columns=["_frame_join", "_traj_join"],
        inplace=True,
        errors="ignore",
    )
    return merged


# ---------------------------------------------------------------------------
# CNN identity interpolated merge
# ---------------------------------------------------------------------------


def merge_interpolated_cnn_df(
    trajectories_df: pd.DataFrame,
    interp_cnn_df: Optional[pd.DataFrame],
    label: str = "cnn_identity",
) -> pd.DataFrame:
    """Merge interpolated CNN identity predictions into final trajectories.

    Flat classifiers use ``CNN_{label}_Class`` / ``CNN_{label}_Conf``.
    Multi-head classifiers use one class/conf pair per factor.
    """
    interp_value_cols = (
        _cnn_value_columns(interp_cnn_df) if interp_cnn_df is not None else []
    )
    uses_wide_columns = bool(interp_value_cols)
    if uses_wide_columns:
        output_cols = list(interp_value_cols)
        required_cols = {"frame_id", "trajectory_id"}
    else:
        col_class = f"CNN_{label}_Class"
        col_conf = f"CNN_{label}_Conf"
        output_cols = [col_class, col_conf]
        required_cols = {"frame_id", "trajectory_id", "class_name", "confidence"}

    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if not _can_merge_interp(trajectories_df, interp_cnn_df, required_cols):
        return _ensure_interp_columns(trajectories_df, output_cols)

    out, interp = _prepare_interp_join_keys(trajectories_df, interp_cnn_df)
    out = _ensure_interp_columns(out, output_cols)

    if uses_wide_columns:
        rename_map = {col: f"{col}_icnn" for col in output_cols}
        interp_lookup = interp[["_frame_join", "_traj_join", *output_cols]].rename(
            columns=rename_map
        )
        column_map = {src_col: tgt_col for tgt_col, src_col in rename_map.items()}
    else:
        interp_lookup = interp[
            ["_frame_join", "_traj_join", "class_name", "confidence"]
        ].drop_duplicates(subset=["_frame_join", "_traj_join"], keep="first")
        column_map = {"class_name": col_class, "confidence": col_conf}

    interp_lookup = interp_lookup.drop_duplicates(
        subset=["_frame_join", "_traj_join"], keep="first"
    )

    merged = out.merge(
        interp_lookup,
        how="left",
        on=["_frame_join", "_traj_join"],
        suffixes=("", "_icnn"),
        sort=False,
    )

    merged = _backfill_interp_columns(merged, column_map)

    merged.drop(
        columns=["_frame_join", "_traj_join"],
        inplace=True,
        errors="ignore",
    )
    return merged


# ---------------------------------------------------------------------------
# Head-tail interpolated merge
# ---------------------------------------------------------------------------

HEADTAIL_INTERP_COLUMNS = [
    "InterpHeadingRad",
    "InterpHeadingConf",
    "InterpHeadingDirected",
]


def merge_interpolated_headtail_df(
    trajectories_df: pd.DataFrame,
    interp_ht_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge interpolated head-tail direction into final trajectories."""
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if not _can_merge_interp(
        trajectories_df, interp_ht_df, {"frame_id", "trajectory_id", "heading_rad"}
    ):
        return _ensure_interp_columns(trajectories_df, HEADTAIL_INTERP_COLUMNS)

    out, interp = _prepare_interp_join_keys(trajectories_df, interp_ht_df)
    out = _ensure_interp_columns(out, HEADTAIL_INTERP_COLUMNS)

    ht_cols = ["heading_rad"]
    if "heading_conf" in interp.columns:
        ht_cols.append("heading_conf")
    if "heading_directed" in interp.columns:
        ht_cols.append("heading_directed")

    interp_lookup = interp[["_frame_join", "_traj_join", *ht_cols]].drop_duplicates(
        subset=["_frame_join", "_traj_join"], keep="first"
    )

    merged = out.merge(
        interp_lookup,
        how="left",
        on=["_frame_join", "_traj_join"],
        suffixes=("", "_iht"),
        sort=False,
    )

    merged = _backfill_interp_columns(
        merged,
        {
            "heading_rad": "InterpHeadingRad",
            "heading_conf": "InterpHeadingConf",
            "heading_directed": "InterpHeadingDirected",
        },
    )

    merged.drop(
        columns=["_frame_join", "_traj_join"],
        inplace=True,
        errors="ignore",
    )
    return merged
