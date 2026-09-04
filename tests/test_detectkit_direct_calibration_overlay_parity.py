"""The overlay must show the SELECTED ROW'S OWN predictions.

Non-tautological by construction: the expected value is NOT a hand-built
preview, it is a fresh production rescore of the SAME pre-merge parts through
``rescore_parts`` + ``config_for_point`` at the row's own settings. Every
earlier overlay test compared the dialog against a preview the test itself
fabricated, which could never catch a collection-time cap mismatch.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hydra_suite.core.inference.direct_calibration_sweep import (  # noqa: E402
    MergeSettings,
    config_for_point,
    detections_from_result,
    rescore_parts,
)
from hydra_suite.core.inference.result import OBBResult  # noqa: E402
from hydra_suite.core.inference.runtime import RuntimeContext  # noqa: E402
from hydra_suite.detectkit.gui.dialogs.direct_calibration_results import (  # noqa: E402
    DirectCalibrationResultsDialog,
)
from hydra_suite.detectkit.jobs import direct_calibration as job  # noqa: E402

SLICE_PARAMS = {
    "geometry_mode": "auto_object",
    "imgsz": 640,
    "object_tile_fraction": 0.4,
    "overlap": 0.2,
}
MERGE = MergeSettings("greedy_nmm", "ios", 0.5)


class _PlainSource:
    """A RegionSource whose regions are disjoint -- no tile plan needed."""

    merge_policy = "plain"

    def merge_plan(self, _frame_idx):
        return None


def _parts(
    n_high: int,
    n_low: int,
    *,
    high_size=(100.0, 500.0),
    low_size=(1000.0, 2000.0),
    low_conf_range=(0.31, 0.33),
):
    """One frame of pre-merge parts: many high-conf SMALL + few low-conf LARGE.

    The raw cap (2 * max_targets) selects by CONFIDENCE; the final cap selects
    by SIZE. Making the low-confidence detections the largest ones is exactly
    the configuration in which lifting the raw cap changes which detections
    the final cap can even see.
    """
    total = n_high + n_low
    confs = np.concatenate(
        [
            np.linspace(0.53, 0.92, n_high, dtype=np.float32),
            np.linspace(low_conf_range[0], low_conf_range[1], n_low, dtype=np.float32),
        ]
    )
    sizes = np.concatenate(
        [
            np.linspace(high_size[0], high_size[1], n_high, dtype=np.float32),
            np.linspace(low_size[0], low_size[1], n_low, dtype=np.float32),
        ]
    )
    centroids = np.zeros((total, 2), dtype=np.float32)
    corners = np.zeros((total, 4, 2), dtype=np.float32)
    for i in range(total):
        # Far apart on a grid so no IoU/dedup step can couple them.
        cx = 40.0 + 60.0 * (i % 32)
        cy = 40.0 + 60.0 * (i // 32)
        centroids[i] = (cx, cy)
        half = float(np.sqrt(sizes[i])) / 2.0
        corners[i] = [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ]
    return [
        OBBResult(
            frame_idx=0,
            centroids=centroids,
            angles=np.zeros(total, dtype=np.float32),
            sizes=sizes,
            shapes=np.stack([sizes, np.ones(total, dtype=np.float32)], axis=1),
            confidences=confs,
            corners=corners,
            detection_ids=OBBResult.make_detection_ids(0, total),
        )
    ]


def _request(tmp_path, *, max_targets, confidences):
    from hydra_suite.core.inference.direct_calibration_grid import build_candidate_grid
    from hydra_suite.data.al.escalation import LabelRecord
    from hydra_suite.utils.geometry_levels import GeometryLevel

    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    path = tmp_path / "f0.png"
    cv2.imwrite(str(path), np.zeros((2000, 2000, 3), np.uint8))
    label = LabelRecord(
        class_id=0,
        confidence=1.0,
        points=np.array([[4, 4], [20, 4], [20, 20], [4, 20]], dtype=np.float32),
        level=GeometryLevel.POLYGON,
    )
    evidence = job.EvidenceSet(
        frames=[(path, [label])],
        split="val",
        instances=1,
        size_range=((2000, 2000), (2000, 2000)),
        sampled_from=1,
        fingerprint="deadbeef",
    )
    return job.DirectCalibrationRequest(
        model_path=model,
        task="obb",
        evidence=evidence,
        candidates=build_candidate_grid(SLICE_PARAMS)[:1],
        confidences=confidences,
        merge_settings=(MERGE,),
        runtime_tier="cpu",
        max_targets=max_targets,
        evidence_dir=tmp_path / "evidence",
    )


def _polygon_set(polygons):
    return {
        tuple(np.round(np.asarray(p, dtype=np.float64).reshape(-1), 3).tolist())
        for p in polygons
    }


def _overlay_vs_production(
    tmp_path,
    *,
    max_targets,
    n_high,
    n_low,
    row_confidence,
    low_conf_range=(0.31, 0.33),
    low_size=(1000.0, 2000.0),
):
    request = _request(
        tmp_path, max_targets=max_targets, confidences=(0.10, row_confidence)
    )
    candidate = request.candidates[0]
    base_config = config_for_point(
        str(request.model_path),
        slice_params=candidate.slice_params(),
        merge=MERGE,
        confidence=request.confidences[0],
        max_targets=request.max_targets,
        runtime_tier="cpu",
        model_task="obb",
    )
    runtime = RuntimeContext.from_config(base_config)
    source = _PlainSource()
    parts_per_frame = [
        _parts(n_high, n_low, low_conf_range=low_conf_range, low_size=low_size)
    ]

    preview = job._preview_for(
        request,
        candidate,
        0,
        MERGE,
        row_confidence,
        parts_per_frame,
        source,
        base_config,
        runtime,
    )
    point = job._point_for(
        candidate,
        request=request,
        merge=MERGE,
        confidence=row_confidence,
        tiles=1,
        seconds=0.0,
        score=job._zero_score(),
        candidate_index=0,
    )
    rendered = DirectCalibrationResultsDialog._row_predictions(preview, point, 0)

    row_config = config_for_point(
        str(request.model_path),
        slice_params=candidate.slice_params(),
        merge=MERGE,
        confidence=row_confidence,
        max_targets=request.max_targets,
        runtime_tier="cpu",
        model_task="obb",
    )
    produced = detections_from_result(
        rescore_parts(parts_per_frame[0], source, row_config, runtime, frame_idx=0)
    )
    return _polygon_set(rendered), _polygon_set([d.polygon_px for d in produced])


def test_overlay_equals_a_fresh_production_rescore_when_the_raw_cap_binds(tmp_path):
    """60 candidates, N=20 (raw cap 40), low-confidence row: the reported bug."""
    rendered, produced = _overlay_vs_production(
        tmp_path, max_targets=20, n_high=40, n_low=20, row_confidence=0.30
    )
    assert (
        produced
    ), "fixture must produce detections for the comparison to mean anything"
    assert rendered == produced


def test_overlay_equals_production_above_the_downstream_crop_clamp(tmp_path):
    """max_targets > 64, where ``_effective_max_detections`` clamps at 128.

    100 high-confidence SMALL detections the row keeps, plus 100 BELOW-gate
    detections that are the largest in the frame. The row emits 100; a
    preview whose own final cap is the 128 clamp would spend most of its 128
    slots on detections the row's gate then throws away.
    """
    rendered, produced = _overlay_vs_production(
        tmp_path,
        max_targets=100,
        n_high=100,
        n_low=100,
        row_confidence=0.30,
        low_conf_range=(0.12, 0.20),
        low_size=(3000.0, 5000.0),
    )
    assert len(produced) == 100
    assert rendered == produced
