"""Synthetic OBBResult construction for interpolated-crop inference.

Builds an ``OBBResult`` (the same struct real OBB detections produce) from
one frame's interpolated-gap tasks, so the SAME batched stage functions
``Pipeline`` calls for real detections
(``extract_canonical_crops_batch``/``run_pose_batch``, ``run_cnn_batch``,
``run_headtail_batch``, ``extract_aabb_crops``/``run_apriltag``) can run on
interpolated geometry unmodified. See design spec "Architecture" and
"Key architectural finding".
"""

from __future__ import annotations

import logging

import numpy as np

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    ClippingStats,
    canonical_affine,
)
from hydra_suite.core.individual.dataset.naming import synthetic_interpolated_det_id
from hydra_suite.core.individual.geometry import ellipse_to_obb_corners
from hydra_suite.core.inference.result import OBBResult

logger = logging.getLogger(__name__)


def filter_degenerate_tasks(
    tasks: list[dict],
    geometry: CanonicalGeometry,
    clipping_stats: "ClippingStats | None",
) -> list[dict]:
    """Drop tasks whose ellipse-derived OBB is degenerate, tallying the drop.

    ``extract_canonical_crops``/``_batch`` (``stages/crops.py``) does NOT
    raise or skip on a degenerate OBB -- it silently fudges an identity
    affine (crops.py:97-98) and has no ``ClippingStats`` plumbing at all
    (design spec "Error handling", adversarial-review G2/G3). This function
    is what restores today's loud-skip-and-tally behavior: it must run
    BEFORE any task reaches ``build_synthetic_obb_result``/the batch stage
    functions, exactly mirroring what
    ``interpolated_crops.py::_compute_frame_corners_and_affines`` used to do
    inline. For kept tasks it also records the real overflow via
    ``canonical_affine``, matching what ``Pipeline`` does for real
    detections (``pipeline.py:331-338``).
    """
    kept: list[dict] = []
    for task in tasks:
        corners = ellipse_to_obb_corners(
            task["cx"], task["cy"], task["w"], task["h"], task["theta"]
        )
        try:
            canonical_affine(corners, geometry)
        except ValueError:
            if clipping_stats is not None:
                clipping_stats.record_degenerate()
            logger.warning(
                "Interp pose/CNN/tag/headtail: skipping frame_id=%s traj_id=%s "
                "-- degenerate OBB has no Layer 1 canonical transform "
                "(canonical_affine raised); the stage layer would otherwise "
                "silently fudge an identity-affine crop instead of skipping.",
                task["frame_id"],
                task["traj_id"],
            )
            continue
        if clipping_stats is not None:
            clipping_stats.record(corners, geometry)
        kept.append(task)
    return kept


def build_synthetic_obb_result(frame_idx: int, tasks: list[dict]) -> OBBResult:
    """Build an ``OBBResult`` for one frame's (already-filtered) interpolated tasks.

    ``tasks`` is ``frame_tasks[f]`` (or the ``filter_degenerate_tasks``
    output of it) -- each dict has ``cx``/``cy``/``w``/``h``/``theta``/
    ``frame_id``/``traj_id``/``interp_index``. Detection ids are negative
    and stable per (frame_id, trajectory_id, interp_index) via
    ``synthetic_interpolated_det_id`` -- the SAME scheme
    ``parse_identity_image_filename`` already uses for interpolated crop
    filenames (``naming.py``), so a synthetic id can never collide with the
    positive real-detection id space (``OBBResult.make_detection_ids``).
    """
    n = len(tasks)
    corners = np.zeros((n, 4, 2), dtype=np.float32)
    centroids = np.zeros((n, 2), dtype=np.float32)
    angles = np.zeros(n, dtype=np.float32)
    sizes = np.zeros(n, dtype=np.float32)
    shapes = np.zeros((n, 2), dtype=np.float32)
    confidences = np.ones(n, dtype=np.float32)
    det_ids = np.zeros(n, dtype=np.int64)

    for i, task in enumerate(tasks):
        corners[i] = ellipse_to_obb_corners(
            task["cx"], task["cy"], task["w"], task["h"], task["theta"]
        )
        centroids[i] = (task["cx"], task["cy"])
        angles[i] = task["theta"]
        area = float(np.pi / 4.0 * task["w"] * task["h"])
        sizes[i] = area
        aspect = float(task["w"] / task["h"]) if task["h"] else 0.0
        shapes[i] = (area, aspect)
        det_ids[i] = synthetic_interpolated_det_id(
            task["frame_id"], task["traj_id"], task["interp_index"]
        )

    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=angles,
        sizes=sizes,
        shapes=shapes,
        confidences=confidences,
        corners=corners,
        detection_ids=det_ids,
    )
