"""Single source of truth for TrackerKit's "advanced config" defaults.

Both entry points -- the headless CLI (``cli_config.py``) and the GUI
(``gui/orchestrators/config.py``) -- load the same advanced-config JSON file
(via ``hydra_suite.paths.get_advanced_config_path``) and previously carried
their own hand-written default table to seed it. Those two literal tables had
already diverged (the GUI's had four ``obb_seg_*`` keys the CLI's lacked), and
neither of them had a ``reference_aspect_ratio`` entry at all -- which is why
the CLI invented its own (wrong, 4.0) default for it inline instead of
sharing one with the GUI's widget default (2.0).

This module is that one shared table. Any key referenced by either loader
lives here; both loaders return a copy of ``DEFAULT_ADVANCED_CONFIG`` merged
with the user's saved JSON on top.
"""

from __future__ import annotations

from typing import Any

#: Union of the two previously-divergent default tables
#: (``trackerkit/cli_config.py::_default_advanced_config`` and
#: ``trackerkit/gui/orchestrators/config.py::ConfigOrchestrator._load_advanced_config``),
#: plus the two canonicalization knobs neither table had:
#: ``canonical_margin`` (the inference-time crop margin, previously read from
#: a key nothing ever wrote) and ``reference_aspect_ratio`` (previously only
#: a GUI widget default / a CLI-only inline override).
DEFAULT_ADVANCED_CONFIG: dict[str, Any] = {
    # ROI crop suggestion / video crop encode settings
    "roi_crop_warning_threshold": 0.6,  # Warn if ROI is <60% of frame
    "roi_crop_auto_suggest": True,  # Auto-suggest cropping
    "roi_crop_remind_every_session": False,  # Remind every time or once
    "roi_crop_padding_fraction": 0.05,  # Padding as fraction of min(width, height) - typically 5%
    "video_crop_codec": "libx264",  # Codec for cropped videos (libx264 for quality)
    "video_crop_crf": 18,  # CRF quality (lower = better, 18 = visually lossless)
    "video_crop_preset": "medium",  # ffmpeg preset (ultrafast, fast, medium, slow, veryslow)
    # YOLO Batching - Memory Fractions (device-specific optimization)
    "mps_memory_fraction": 0.3,  # Conservative 30% of unified memory for MPS (Apple Silicon)
    "cuda_memory_fraction": 0.7,  # 70% of VRAM for CUDA (NVIDIA GPUs)
    "tensorrt_build_workspace_gb": 4.0,  # TensorRT builder workspace limit in GB
    "tensorrt_build_batch_size": None,  # Optional fixed TensorRT build batch override
    "yolo_headtail_detect_conf_threshold": 0.25,  # Minimum detection confidence before head-tail inference runs; lower-confidence detections remain unknown
    "headtail_batch_size": 64,  # Canonical crop batch size for head-tail classifier inference
    "realtime_visualization_emit_stride": 1,  # Emit GUI overlays every Nth frame during realtime tracking while preserving full-speed tracking/video output
    "visualization_emit_stride": 1,  # Optional GUI overlay decimation for non-realtime runs
    # Dataset Generation - YOLO Detection Parameters (separate from tracking)
    "dataset_yolo_confidence_threshold": 0.05,  # Very low - detect all animals including uncertain ones for annotation
    "dataset_yolo_iou_threshold": 0.5,  # Moderate - remove obvious duplicates but keep borderline cases for manual review
    # Identity swap-correction & rejoin gate (advanced; UI-exposed defaults
    # cover most cases — tune here only if defaults are inappropriate).
    "identity_swap_conf_margin": 0.2,  # prob margin to count a frame as mutual mismatch
    "identity_rejoin_velocity_budget": 1.5,  # safety factor on (frames_lost * v_max) for identity rejoin distance
    "identity_rejoin_dist_floor": None,  # absolute min rejoin distance (None = 2 * body_size)
    # Segment-as-OBB rotated-rect kernel (advanced; only read when
    # YOLO_OBB_DIRECT_TASK == "segment" -- tune here only if the
    # defaults are too slow/inaccurate for your footage).
    "obb_seg_num_angles": 24,  # coarse angle-search steps over [0, pi); linear cost
    "obb_seg_crop_size": 64,  # mask resample resolution (crop_size^2 pixels); quadratic cost
    "obb_seg_pad_ratio": 0.15,  # fractional padding around the box before cropping (clip safety)
    "obb_seg_mask_threshold": 0.5,  # foreground cutoff on the resampled soft mask
    # Global canonicalization geometry (fixed canvas = REFERENCE_BODY_SIZE *
    # sqrt(reference_aspect_ratio) * canonical_margin on the long edge). The
    # operator's only dial for avoiding clipped animals is canonical_margin;
    # reference_aspect_ratio ALSO centres the detection aspect-ratio filter
    # when that filter is enabled (core/inference/config.py).
    "reference_aspect_ratio": 2.0,  # species-typical major/minor axis ratio
    "canonical_margin": 1.3,  # canonical crop canvas long-edge margin over the reference major axis
}


def default_advanced_config() -> dict[str, Any]:
    """Return a fresh mutable copy of :data:`DEFAULT_ADVANCED_CONFIG`."""
    return dict(DEFAULT_ADVANCED_CONFIG)
