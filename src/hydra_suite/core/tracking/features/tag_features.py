"""
Per-frame tag feature helpers for the tracking loop.

These functions translate tag observations from the :class:`TagObservationCache`
into the ``association_data`` dictionaries consumed by the assigner.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel for "no tag observed"
NO_TAG: int = -1

_NAN = float("nan")


def build_tag_detection_map(
    tag_cache: Any,
    frame_idx: int,
) -> Dict[int, int]:
    """Map detection-slot index → tag_id for a single frame.

    Parameters
    ----------
    tag_cache:
        An open :class:`TagObservationCache` in read mode (or *None*).
    frame_idx:
        The video frame index to query.

    Returns
    -------
    dict mapping ``det_index`` → ``tag_id`` for every tag observed in this
    frame.  Empty dict if the cache is *None* or the frame has no tags.
    """
    if tag_cache is None:
        return {}
    try:
        obs = tag_cache.get_frame(frame_idx)
    except Exception:
        return {}

    tag_ids = obs.get("tag_ids", np.array([], dtype=np.int32))
    det_indices = obs.get("det_indices", np.array([], dtype=np.int32))

    if len(tag_ids) == 0:
        return {}

    result: Dict[int, int] = {}
    for tid, didx in zip(tag_ids.tolist(), det_indices.tolist()):
        # If multiple tags land in the same detection slot, keep the first.
        if didx not in result:
            result[didx] = int(tid)
    return result


def build_tag_detection_hamming_map(
    tag_cache: Any,
    frame_idx: int,
) -> Dict[int, int]:
    """Map detection-slot index → hamming distance for a single frame.

    Parameters
    ----------
    tag_cache:
        An open :class:`TagObservationCache` in read mode (or *None*).
    frame_idx:
        The video frame index to query.

    Returns
    -------
    dict mapping ``det_index`` → ``hamming`` for every tag observed in this
    frame.  Empty dict if the cache is *None* or the frame has no tags.
    """
    if tag_cache is None:
        return {}
    try:
        obs = tag_cache.get_frame(frame_idx)
    except Exception:
        return {}

    det_indices = obs.get("det_indices", np.array([], dtype=np.int32))
    hammings = obs.get("hammings", np.array([], dtype=np.int32))

    if len(det_indices) == 0:
        return {}

    result: Dict[int, int] = {}
    for didx, hamming in zip(det_indices.tolist(), hammings.tolist()):
        if didx not in result:
            result[didx] = int(hamming)
    return result


def build_detection_tag_id_list(
    tag_det_map: Dict[int, int],
    num_detections: int,
) -> List[int]:
    """Return a list aligned with detections: ``tag_id`` or :data:`NO_TAG`.

    This is what goes into ``association_data["detection_tag_ids"]``.
    """
    return [tag_det_map.get(j, NO_TAG) for j in range(num_detections)]


def get_detection_tag_csv_values(
    det_index: int,
    tag_det_map: Dict[int, int],
    tag_hamming_map: Dict[int, int],
    tag_label_map: Dict[int, str],
) -> Tuple[object, object, object, object]:
    """Return ``(tag_id, label, conf, hamming)`` for *det_index*, or four NaNs.

    Parameters
    ----------
    det_index:
        Local detection index within the frame.
    tag_det_map:
        Output of :func:`build_tag_detection_map` for the current frame.
    tag_hamming_map:
        Output of :func:`build_tag_detection_hamming_map` for the current frame.
    tag_label_map:
        Mapping from integer AprilTag ID to catalog label string.

    Returns
    -------
    tuple of ``(DetectedTagID, DetectedTagLabel, DetectedTagConf, DetectedTagHamming)``
    or ``(nan, nan, nan, nan)`` when no tag was observed for this detection.
    """
    tag_id = tag_det_map.get(det_index, NO_TAG)
    if tag_id == NO_TAG:
        return (_NAN, _NAN, _NAN, _NAN)
    hamming = tag_hamming_map.get(det_index, 0)
    conf = 1.0 / (1.0 + max(0, hamming))
    label: Optional[str] = tag_label_map.get(tag_id)
    return (
        float(tag_id),
        label if label is not None else _NAN,
        float(conf),
        float(hamming),
    )
