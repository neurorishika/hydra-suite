"""The export detector must match the detector that produced the tracking.

`_init_detection_runner` builds the detection pass that active-learning export
uses to recover label geometry for the frames the selector picked. That pass is
only meaningful if it runs the SAME detector the tracking run did.

`build_obb_only_config` builds its params dict from scratch with a small fixed
set of keys, so whole families of geometry-determining params were dropped on
the way to the export config -- notably the `SLICE_*` (SAHI) family, which made
export run unsliced against a tracking run that was sliced.

The oracle here is agreement: for the same params, the export config and the
tracking config must not disagree about the geometry that decides what gets
detected.
"""

from hydra_suite.core.inference.config import build_inference_config_from_params
from hydra_suite.data import dataset_generation


def _sliced_segment_params():
    """Params shaped like a sliced direct-segment tracking run."""
    return {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_TASK": "segment",
        "YOLO_OBB_DIRECT_MODEL_PATH": "seg.pt",
        "RUNTIME_TIER": "gpu",
        "MAX_TARGETS": 25,
        "REFERENCE_BODY_SIZE": 49.83,
        "RESIZE_FACTOR": 1.0,
        "DATASET_EXPORT_LEVELS": ["polygon", "obb", "aabb"],
        # SAHI geometry, exactly the flat keys build_engine_params emits.
        "SLICE_ENABLED": True,
        "SLICE_GEOMETRY_MODE": "auto_object",
        "SLICE_OVERLAP": 0.2,
        "SLICE_HEIGHT": 0,
        "SLICE_WIDTH": 0,
        "SLICE_OBJECT_TILE_FRACTION": 0.05,
        "SLICE_TRAINED_BODY_PX": 96.5,
        # Segment-extraction geometry.
        "YOLO_OBB_SEG_NUM_ANGLES": 36,
        "YOLO_OBB_SEG_CROP_SIZE": 128,
        "YOLO_OBB_SEG_PAD_RATIO": 0.25,
        "YOLO_OBB_SEG_MASK_THRESHOLD": 0.4,
    }


def _export_cfg(monkeypatch, params):
    captured = {}

    class _FakeRunner:
        def __init__(self, cfg, cache_dir=None, video_path=None):
            captured["cfg"] = cfg

    monkeypatch.setattr(
        "hydra_suite.core.inference.runner.InferenceRunner", _FakeRunner
    )
    assert dataset_generation._init_detection_runner(params, "/tmp/x.mp4") is not None
    return captured["cfg"]


def test_export_detector_preserves_sahi_slicing(monkeypatch):
    """The bug: export ran unsliced against a sliced tracking run."""
    params = _sliced_segment_params()
    export = _export_cfg(monkeypatch, params)
    tracking = build_inference_config_from_params(params)

    assert tracking.obb.direct.slice.enabled is True, "fixture must be sliced"
    assert export.obb.direct.slice.enabled is True, "export dropped SAHI slicing"


def test_export_detector_slice_geometry_matches_tracking(monkeypatch):
    """Not just enabled -- the same tile grid, or export sees different objects."""
    params = _sliced_segment_params()
    export = _export_cfg(monkeypatch, params).obb.direct.slice
    tracking = build_inference_config_from_params(params).obb.direct.slice

    for field in (
        "enabled",
        "geometry_mode",
        "overlap_height_ratio",
        "overlap_width_ratio",
        "object_tile_fraction",
        "reference_body_px",
        "slice_width",
        "slice_height",
        "merge_policy",
        "merge_metric",
        "merge_threshold",
    ):
        assert getattr(export, field) == getattr(tracking, field), field


def test_export_detector_segment_geometry_matches_tracking(monkeypatch):
    """The segment-to-OBB kernel knobs must match too."""
    params = _sliced_segment_params()
    export = _export_cfg(monkeypatch, params).obb.direct
    tracking = build_inference_config_from_params(params).obb.direct

    for field in (
        "seg_num_angles",
        "seg_crop_size",
        "seg_pad_ratio",
        "seg_mask_threshold",
    ):
        assert getattr(export, field) == getattr(tracking, field), field


def test_export_detector_unsliced_when_tracking_unsliced(monkeypatch):
    """Faithfulness cuts both ways: don't invent slicing that tracking lacked."""
    params = _sliced_segment_params()
    params["SLICE_ENABLED"] = False
    export = _export_cfg(monkeypatch, params)
    assert export.obb.direct.slice.enabled is False


def test_export_detector_still_requests_native_polygons(monkeypatch):
    """The polygon opt-in must survive the added param passthrough."""
    export = _export_cfg(monkeypatch, _sliced_segment_params())
    assert export.obb.emit_native_geometry is True


# ── the export detection cap must not be the tracking animal count ─────────


def test_export_detector_does_not_cap_at_tracking_max_targets(monkeypatch):
    """MAX_TARGETS is the user's declared animal count, a tracking knob.

    Applying it to the export pass bakes a fabricated "only N animals here"
    ground truth into the labels for exactly the crowded frames active
    learning exists to find. The post-filter cap keeps the LARGEST detections
    rather than the most confident, so the truncation is biased as well as
    lossy. `AL_DEFAULT_MAX_TARGETS` exists for this; DetectKit's AL worker
    already uses it (see `64b8c7cd`), and this path must too.
    """
    from hydra_suite.data.al.inference_adapter import AL_DEFAULT_MAX_TARGETS

    params = _sliced_segment_params()
    params["MAX_TARGETS"] = 25
    export = _export_cfg(monkeypatch, params).obb

    assert export.max_detections >= AL_DEFAULT_MAX_TARGETS, (
        f"export truncates to {export.max_detections} detections/frame; "
        "crowded frames would be exported with invented ground truth"
    )


def test_export_detector_cap_has_headroom_over_declared_count(monkeypatch):
    """A colony larger than the default ceiling still gets headroom.

    Mirrors DetectKit's `max(AL_DEFAULT_MAX_TARGETS, 2 * expected_count)` so
    the ceiling can never bite before the count signals can measure.
    """
    from hydra_suite.data.al.inference_adapter import AL_DEFAULT_MAX_TARGETS

    params = _sliced_segment_params()
    params["MAX_TARGETS"] = 400
    export = _export_cfg(monkeypatch, params).obb

    assert export.max_detections >= 2 * 400
    assert export.max_detections > AL_DEFAULT_MAX_TARGETS


def test_export_does_not_reuse_a_cache_capped_below_its_own_ceiling(monkeypatch):
    """A cache written under the tracking cap must not satisfy the export pass.

    Tracking's raw detection cache holds at most `2 * MAX_TARGETS` rows. Export
    deliberately runs a much higher ceiling so crowded frames are not exported
    with invented ground truth. Serving export from tracking's cache would hand
    it exactly the truncated set the higher ceiling exists to avoid, so the two
    must land on DIFFERENT cache keys -- `max_detections` and
    `raw_detection_cap` are in the key for precisely this reason.
    """
    from hydra_suite.core.inference.cache.keys import detection_cache_key
    from hydra_suite.core.inference.config import build_obb_only_config

    params = _sliced_segment_params()
    params["MAX_TARGETS"] = 25
    export = _export_cfg(monkeypatch, params).obb

    tracking = build_obb_only_config(
        params["YOLO_OBB_DIRECT_MODEL_PATH"],
        confidence_threshold=0.05,
        iou_threshold=0.5,
        max_targets=25,
        mode="direct",
        model_task="segment",
    ).obb

    assert export.max_detections > tracking.max_detections
    assert detection_cache_key(export, None) != detection_cache_key(tracking, None)


def test_export_does_not_inherit_tracking_tile_memory_budget(monkeypatch):
    """Tile memory budget is an EXECUTION control, not detection geometry.

    Slice geometry must match tracking (what gets detected); the tile memory
    budget must not, because it only decides how many tiles ride in one model
    call. Export runs a much higher detection ceiling than tracking, and the
    dense segment-mask term scales with that ceiling, so inheriting tracking's
    budget made a sliced segment export inadmissible -- refused outright, with
    zero labels, rather than throttled to smaller chunks.
    """
    from hydra_suite.core.inference.stages.slicing import (
        MAX_TILE_BATCH_BYTES,
        estimated_prediction_job_bytes,
    )

    params = _sliced_segment_params()
    params["SLICE_MEMORY_BUDGET_MIB"] = 256
    export = _export_cfg(monkeypatch, params).obb

    budget = min(MAX_TILE_BATCH_BYTES, export.direct.slice.tile_memory_budget_bytes)
    # A single 1024px segment tile at export's own ceiling must be admissible.
    per_tile = estimated_prediction_job_bytes(
        imgsz=1024,
        task="segment",
        max_detections=export.raw_detection_cap,
        source_bytes=1931 * 1931 * 3,
    )
    assert per_tile <= budget, (
        f"one tile needs {per_tile} bytes but only {budget} are admitted; "
        "sliced segment export would be refused outright"
    )
