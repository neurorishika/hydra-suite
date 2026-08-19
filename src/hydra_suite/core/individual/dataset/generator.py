"""
Individual-level analysis for identity classification and pose tracking.
Extracts regions around detections for downstream processing.
"""

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from hydra_suite.core.canonicalization.geometry import (
    CanonicalGeometry,
    ClippingStats,
    canonical_affine,
)
from hydra_suite.core.individual.dataset.naming import (
    build_detection_image_filename,
    build_interpolated_image_filename,
)
from hydra_suite.core.individual.geometry import (
    ellipse_to_obb_corners,
    resolve_directed_angle,
)

logger = logging.getLogger(__name__)


class IndividualDatasetGenerator:
    """
    Real-time individual dataset generator that runs during tracking.

    Exports OBB-masked crops for each detection, where only the detected
    animal's OBB region is visible and the rest is masked. This provides
    clean, isolated training data for individual-level analysis.

    Key features:
    - Runs in parallel with tracking (called per-frame)
    - Uses actual OBB polygon to mask out other animals
    - Saves crops with track/trajectory ID labels
    - Uses already-filtered detections (ROI + size filtering done by tracking)
    """

    def __init__(
        self,
        params: Dict[str, Any],
        output_dir: Optional[str],
        video_name: str,
        dataset_name: Optional[str] = None,
    ):
        """
        Initialize the individual dataset generator.

        Args:
            params: Parameter dictionary
            output_dir: Base directory for saving crops
            video_name: Name of the source video (for organizing output)
            dataset_name: Optional custom name for the dataset
        """
        self.params = params
        self.enabled = params.get("ENABLE_INDIVIDUAL_DATASET", False)
        self.save_images = bool(
            params.get("ENABLE_INDIVIDUAL_IMAGE_SAVE", self.enabled)
        )

        # Output configuration
        self.output_dir = Path(output_dir) if output_dir else None
        self.video_name = video_name
        self.dataset_name = (
            params.get("INDIVIDUAL_DATASET_NAME", "individual_dataset")
            if dataset_name is None
            else dataset_name
        )

        # Background color as BGR tuple (default: black)
        bg_color = params.get("INDIVIDUAL_BACKGROUND_COLOR", (0, 0, 0))
        if isinstance(bg_color, (list, tuple)) and len(bg_color) == 3:
            self.background_color = tuple(bg_color)
        else:
            self.background_color = (0, 0, 0)

        # Fill other animals' OBB regions with background color before saving crops
        self.suppress_foreign_obb = bool(
            params.get("SUPPRESS_FOREIGN_OBB_DATASET", False)
        )

        # Crop extraction goes through the project-wide Layer 1
        # CanonicalGeometry (one fixed canvas shared by every crop-consuming
        # stage). There is no second framing knob and no non-canonical
        # fallback (spec 2026-08-18).
        _adv = params.get("ADVANCED_CONFIG", {})
        self._canonical_ref_ar = float(_adv.get("reference_aspect_ratio", 2.0))
        if self._canonical_ref_ar <= 0:
            raise ValueError(
                "ADVANCED_CONFIG.reference_aspect_ratio must be > 0: the crop "
                "dataset is extracted through the project-wide canonical canvas, "
                "the same one inference, head-tail, and the classifiers use. There "
                "is no non-canonical fallback (spec 2026-08-18)."
            )
        self._geometry: CanonicalGeometry = CanonicalGeometry.from_reference(
            reference_body_px=float(params.get("REFERENCE_BODY_SIZE", 20.0))
            * float(params.get("RESIZE_FACTOR", 1.0)),
            aspect_ratio=self._canonical_ref_ar,
            margin=float(_adv.get("canonical_margin", 1.3)),
        )

        # Provenance/QA counters accumulated across the run: how many crops
        # were clipped by the fixed canvas, and the worst overflow_ratio seen
        # (<= 1.0 means "fit"; written into metadata.json's parameters.canonical
        # block, see _write_metadata).
        self._clipping_stats = ClippingStats()

        # Save interval (use detections as-is, just control frequency)
        self.save_every_n_frames = params.get("INDIVIDUAL_SAVE_INTERVAL", 1)

        # Output format — crop-dataset export is MODEL INPUT (training data
        # for the identity/pose/classification pipelines), and it must be
        # pixel-identical to the uncompressed canonical crop used at
        # inference time. JPEG's DCT recompression introduces lossy
        # quantization artifacts that a canonical crop extracted straight
        # from the source frame at inference never sees, silently breaking
        # train/infer consistency. So this exporter only accepts lossless
        # formats (default: png); jpg/jpeg is rejected outright. The
        # human-facing oriented-video exporter (oriented_video.py) is not
        # model input and is unaffected by this restriction.
        self.output_format = params.get("INDIVIDUAL_OUTPUT_FORMAT", "png")
        if str(self.output_format).lower() in ("jpg", "jpeg"):
            raise ValueError(
                "IndividualDatasetGenerator: INDIVIDUAL_OUTPUT_FORMAT="
                f"{self.output_format!r} is not allowed. Crop-dataset export "
                "is MODEL INPUT for training and must match the lossless, "
                "uncompressed canonical crop used at inference time -- JPEG "
                "recompression introduces DCT quantization loss that breaks "
                "train/infer consistency. Use a lossless format (e.g. 'png') "
                "instead."
            )
        self.jpg_quality = params.get("INDIVIDUAL_JPG_QUALITY", 100)

        # Statistics
        self.total_saved = 0
        self.crops_dir = None
        self.run_dir = None
        self.metadata = []

        # Async write thread — keeps disk I/O off the tracking/interpolation thread.
        # A sentinel value of None is pushed to the queue to signal shutdown.
        self._write_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._write_thread: Optional[threading.Thread] = None

        # Initialize output directory
        if self.enabled and self.output_dir:
            self._setup_output_directory()

        logger.info(
            f"Individual dataset generator initialized: enabled={self.enabled}, "
            f"output_dir={self.output_dir}"
        )

    def _setup_output_directory(self):
        """Create output directory structure."""
        run_id = self.params.get("INDIVIDUAL_DATASET_RUN_ID")

        name_part = str(self.dataset_name).strip() if self.dataset_name else ""

        if run_id:
            if name_part:
                dataset_folder_name = f"{name_part}_{run_id}"
            else:
                dataset_folder_name = run_id
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if name_part:
                dataset_folder_name = f"{name_part}_{timestamp}"
            else:
                dataset_folder_name = timestamp
        self.run_dir = self.output_dir / dataset_folder_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if self.save_images:
            self.crops_dir = self.run_dir / "images"
            self.crops_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.crops_dir = None

        # Create metadata file path
        self.metadata_path = self.run_dir / "metadata.json"

        # Load existing metadata if present (append mode)
        if self.save_images and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    data = json.load(f)
                self.metadata = data.get("images", data.get("crops", []))
                self.total_saved = len(self.metadata)
            except Exception:
                pass

        logger.info(
            "Individual dataset output directory: %s",
            self.crops_dir if self.crops_dir is not None else self.run_dir,
        )
        if self.save_images:
            self._start_write_thread()

    def _start_write_thread(self) -> None:
        """Start the background thread that drains the write queue."""
        if self._write_thread is not None and self._write_thread.is_alive():
            return
        self._write_thread = threading.Thread(
            target=self._write_worker, daemon=True, name="IndivDatasetWriter"
        )
        self._write_thread.start()

    def _write_worker(self) -> None:
        """Background worker: pop (crop, filepath, fmt, quality) tuples and write."""
        while True:
            item = self._write_queue.get()
            if item is None:  # sentinel — shut down
                break
            crop, filepath, fmt, quality = item
            try:
                cv2.imwrite(str(filepath), crop)
            except Exception as exc:
                logger.warning("Async crop write failed (%s): %s", filepath, exc)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        meas: Sequence[Sequence[float]],
        obb_corners: Optional[Sequence[np.ndarray]] = None,
        ellipse_params: Optional[Sequence[Sequence[float]]] = None,
        confidences: Optional[Sequence[float]] = None,
        track_ids: Optional[Sequence[int]] = None,
        trajectory_ids: Optional[Sequence[int]] = None,
        coord_scale_factor: float = 1.0,
        detection_ids: Optional[Sequence[int]] = None,
        heading_hints: Optional[Sequence[float]] = None,
        directed_mask: Optional[Sequence[int]] = None,
        velocities: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
        canonical_affines: Optional[Sequence[Optional[np.ndarray]]] = None,
        canonical_canvas_dims: Optional[Sequence[Optional[Tuple[int, int]]]] = None,
        canonical_M_inverse: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> int:
        """
        Process a frame and save masked crops for each detection.

        Supports both YOLO OBB detections and ellipse detections from
        background subtraction. For ellipses, OBB corners are computed
        from the ellipse parameters.

        Detections passed here are already filtered by ROI and size filtering
        in the tracking pipeline - no additional filtering needed.

        Args:
            frame: Input frame (BGR) - should be ORIGINAL resolution
            frame_id: Current frame number
            meas: List of measurements [cx, cy, theta, ...] for each detection (in detection resolution)
            obb_corners: List of OBB corner arrays (4 points each) for YOLO detections (in detection resolution)
            ellipse_params: List of ellipse params [major_axis, minor_axis] for BG sub detections (in detection resolution)
                           (center and theta are taken from meas)
            confidences: Optional list of confidence scores
            track_ids: Optional list of track IDs for each detection
            trajectory_ids: Optional list of trajectory IDs for each detection
            coord_scale_factor: Scale factor to convert detection coords to original resolution (1/resize_factor)
            detection_ids: Optional list of unique Detection IDs for each detection
            heading_hints: Optional directed heading angles (radians) per detection from head-tail model.
            directed_mask: Optional boolean mask (0/1) per detection indicating whether heading_hints is valid.
            velocities: Optional (vx, vy) tuples per detection for motion-based fallback orientation.
            canonical_affines: Optional M_canonical (2x3) affine matrices per detection.
                When provided and canonical crop dimensions > 0, extract rotation-normalised
                canonical crops instead of AABB-masked crops.  Falls back to legacy AABB
                extraction when None or when a specific detection has no affine.

        Returns:
            num_saved: Number of crops saved from this frame
        """
        if not self.enabled or self.crops_dir is None:
            return 0

        if not meas:
            return 0

        # Must have either obb_corners or ellipse_params
        if not obb_corners and not ellipse_params:
            return 0

        # Check save interval
        if frame_id % self.save_every_n_frames != 0:
            return 0

        num_saved = 0
        h, w = frame.shape[:2]

        # Determine detection source type
        use_obb = obb_corners is not None and len(obb_corners) > 0
        use_ellipse = ellipse_params is not None and len(ellipse_params) > 0

        for i in range(len(meas)):
            # Get detection info and scale to original resolution
            m = meas[i]
            cx = float(m[0]) * coord_scale_factor
            cy = float(m[1]) * coord_scale_factor
            theta = float(m[2]) if len(m) > 2 else 0.0

            # Get OBB corners - either directly or computed from ellipse
            # Scale corners/params to match original resolution
            if use_obb and i < len(obb_corners):
                # Scale OBB corners to original resolution
                corners = (
                    np.asarray(obb_corners[i], dtype=np.float32) * coord_scale_factor
                )  # Shape: (4, 2)
                source_type = "yolo_obb"
            elif use_ellipse and i < len(ellipse_params):
                # Get ellipse parameters and scale to original resolution
                ep = ellipse_params[i]
                major_axis = float(ep[0]) * coord_scale_factor
                minor_axis = float(ep[1]) * coord_scale_factor
                corners = ellipse_to_obb_corners(cx, cy, major_axis, minor_axis, theta)
                source_type = "ellipse"
            else:
                # Skip if no geometry data available for this detection
                continue

            # Get confidence
            conf = confidences[i] if confidences and i < len(confidences) else 1.0

            # Build foreign OBB list for contamination suppression (YOLO OBB only).
            other_corners = None
            if self.suppress_foreign_obb and use_obb and obb_corners is not None:
                scaled_all = [
                    np.asarray(obb_corners[j], dtype=np.float32) * coord_scale_factor
                    for j in range(len(obb_corners))
                    if j != i
                ]
                other_corners = scaled_all if scaled_all else None

            # --- Extract crop through the session Layer 1 CanonicalGeometry ---
            # An externally supplied M_canonical (upstream already computed the
            # affine) is honoured verbatim when given; otherwise this recomputes
            # it from the OBB corners against the one project-wide geometry --
            # there is no per-caller canvas and no non-canonical fallback.
            crop = None
            crop_info = None
            M_canonical_i = None
            M_inverse_i = None

            from hydra_suite.core.canonicalization.crop import extract_canonical_crop

            _pre_affine = (
                canonical_affines[i]
                if (canonical_affines is not None and i < len(canonical_affines))
                else None
            )
            if _pre_affine is not None:
                M_can = np.asarray(_pre_affine, dtype=np.float64)
            else:
                try:
                    M_can, _axis_theta, _ = canonical_affine(corners, self._geometry)
                except ValueError:
                    # Degenerate OBB (zero-length edge): it cannot be
                    # canonicalized and there is no non-canonical fallback
                    # crop any more, so this detection is dropped. Counted so
                    # the drop is visible in the run summary rather than
                    # showing up only as a shorter dataset.
                    M_can = None
                    self._clipping_stats.record_degenerate()

            if M_can is not None:
                _cw, _ch = self._geometry.canvas_w, self._geometry.canvas_h

                self._clipping_stats.record(corners, self._geometry)

                M_canonical_i = M_can
                crop = extract_canonical_crop(
                    frame,
                    M_can,
                    geometry=self._geometry,
                    bg_color=self.background_color,
                )

                # Use pre-computed M_inverse when available, else compute
                _pre_inv = (
                    canonical_M_inverse[i]
                    if (
                        canonical_M_inverse is not None
                        and i < len(canonical_M_inverse)
                        and canonical_M_inverse[i] is not None
                    )
                    else None
                )
                M_inverse_i = (
                    _pre_inv
                    if _pre_inv is not None
                    else cv2.invertAffineTransform(M_can).astype(np.float32)
                )

                # Foreign OBB suppression in canonical space
                if other_corners and self.suppress_foreign_obb:
                    for oc in other_corners:
                        # Transform foreign corners into canonical crop space
                        oc_h = np.hstack([oc, np.ones((4, 1))]).T  # 3 x 4
                        oc_canon = (M_can @ oc_h).T.astype(np.int32)  # 4 x 2
                        cv2.fillPoly(crop, [oc_canon], self.background_color)

                crop_info = {
                    "crop_size": (_cw, _ch),
                    "crop_bbox": None,  # not applicable for canonical crops
                    "obb_corners_local": None,
                    "obb_corners_expanded_local": None,
                    "canonical_center_px": [
                        _cw / 2.0,
                        _ch / 2.0,
                    ],
                    "canonical_size_px": [
                        float(_cw),
                        float(_ch),
                    ],
                    "expansion_factor": None,
                }

            if crop is not None:
                # Build metadata for this crop
                track_id = track_ids[i] if track_ids and i < len(track_ids) else -1
                traj_id = (
                    trajectory_ids[i]
                    if trajectory_ids and i < len(trajectory_ids)
                    else -1
                )
                det_id = (
                    detection_ids[i]
                    if detection_ids and i < len(detection_ids)
                    else None
                )

                # Resolve directed orientation: head-tail model > motion > OBB axis.
                _hint = (
                    float(heading_hints[i])
                    if heading_hints is not None and i < len(heading_hints)
                    else None
                )
                _is_directed = (
                    bool(directed_mask[i])
                    if directed_mask is not None and i < len(directed_mask)
                    else False
                )
                _vel = (
                    velocities[i]
                    if velocities is not None and i < len(velocities)
                    else None
                )
                _vx = float(_vel[0]) if _vel is not None else None
                _vy = float(_vel[1]) if _vel is not None else None
                canon_angle, canon_directed, canon_source = resolve_directed_angle(
                    theta, _hint, _is_directed, _vx, _vy
                )

                _canon_block = {
                    "center_px": crop_info["canonical_center_px"],
                    "size_px": crop_info["canonical_size_px"],
                    "angle_rad": float(canon_angle),
                    "major_axis_theta_rad": float(theta),
                    "minor_axis_theta_rad": float(theta + (np.pi / 2.0)),
                    "directed": bool(canon_directed),
                    "orientation_source": canon_source,
                    "M_canonical": M_canonical_i.tolist(),
                    "M_inverse": M_inverse_i.tolist(),
                    "crop_type": "canonical",
                }

                crop_metadata = {
                    "frame_id": int(frame_id),
                    "detection_idx": i,
                    "track_id": int(track_id),
                    "trajectory_id": int(traj_id),
                    "detection_id": int(det_id) if det_id is not None else None,
                    "confidence": float(conf),
                    "center": [cx, cy],
                    "theta": theta,
                    "crop_size": crop_info["crop_size"],
                    "obb_corners": corners.tolist(),
                    "obb_corners_local": crop_info["obb_corners_local"],
                    "obb_corners_expanded_local": crop_info[
                        "obb_corners_expanded_local"
                    ],
                    "canonicalization": _canon_block,
                    "source_type": source_type,
                }

                # Save crop
                saved = self._save_crop(
                    crop, frame_id, i, track_id, crop_metadata, detection_id=det_id
                )
                if saved:
                    num_saved += 1

        return num_saved

    def save_interpolated_crop(
        self,
        frame: np.ndarray,
        frame_id: int,
        cx: float,
        cy: float,
        w: float,
        h: float,
        theta: float,
        traj_id: int,
        interp_from: Tuple[int, int],
        interp_index: int,
        interp_total: int,
        heading_angle: Optional[float] = None,
        heading_directed: bool = False,
        canonical_affine: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Save one interpolated crop for trajectory gap-filling supervision."""
        if not self.enabled or self.crops_dir is None:
            return None

        corners = ellipse_to_obb_corners(cx, cy, w, h, theta)

        # Prefer the session Layer 1 canonical crop; recompute the affine from
        # the corners against self._geometry when the caller didn't already
        # supply one (an external affine is honoured verbatim when given).
        M_inv = None
        M_can_used = None
        crop = None
        crop_info = None
        from hydra_suite.core.canonicalization.crop import extract_canonical_crop
        from hydra_suite.core.canonicalization.geometry import (
            canonical_affine as _canonical_affine_fn,
        )

        if canonical_affine is not None:
            M_can = np.asarray(canonical_affine, dtype=np.float64)
        else:
            try:
                M_can, _axis_theta, _ = _canonical_affine_fn(corners, self._geometry)
            except ValueError:
                # See process_frame: degenerate OBBs are dropped, not faked.
                M_can = None
                self._clipping_stats.record_degenerate()

        if M_can is not None:
            _cw, _ch = self._geometry.canvas_w, self._geometry.canvas_h

            self._clipping_stats.record(corners, self._geometry)

            crop = extract_canonical_crop(
                frame,
                M_can,
                geometry=self._geometry,
                bg_color=self.background_color,
            )
            M_inv = cv2.invertAffineTransform(M_can).astype(np.float32)
            M_can_used = M_can
            crop_info = {
                "crop_size": (_cw, _ch),
                "obb_corners_local": None,
                "obb_corners_expanded_local": None,
                "canonical_center_px": [_cw / 2.0, _ch / 2.0],
                "canonical_size_px": [float(_cw), float(_ch)],
            }

        if crop is None:
            return None

        filename = build_interpolated_image_filename(
            frame_id=frame_id,
            trajectory_id=traj_id,
            interp_from=(int(interp_from[0]), int(interp_from[1])),
            interp_index=interp_index,
            interp_total=interp_total,
            extension=self.output_format,
        )

        interp_canon_angle, interp_canon_directed, interp_canon_source = (
            resolve_directed_angle(theta, heading_angle, heading_directed)
        )
        _interp_canon_block = {
            "center_px": crop_info["canonical_center_px"],
            "size_px": crop_info["canonical_size_px"],
            "angle_rad": float(interp_canon_angle),
            "major_axis_theta_rad": float(theta),
            "minor_axis_theta_rad": float(theta + (np.pi / 2.0)),
            "directed": bool(interp_canon_directed),
            "orientation_source": interp_canon_source,
            "M_canonical": np.asarray(M_can_used).tolist(),
            "M_inverse": M_inv.tolist(),
            "crop_type": "canonical",
        }

        metadata = {
            "frame_id": int(frame_id),
            "detection_idx": -1,
            "track_id": -1,
            "trajectory_id": int(traj_id),
            "detection_id": None,
            "confidence": None,
            "center": [float(cx), float(cy)],
            "theta": float(theta),
            "crop_size": crop_info["crop_size"],
            "obb_corners": corners.tolist(),
            "obb_corners_local": crop_info["obb_corners_local"],
            "obb_corners_expanded_local": crop_info["obb_corners_expanded_local"],
            "canonicalization": _interp_canon_block,
            "source_type": "interpolated",
            "interpolated": True,
            "interp_from_frames": [int(interp_from[0]), int(interp_from[1])],
            "interp_index": int(interp_index),
            "interp_total": int(interp_total),
        }

        return self._save_crop(
            crop,
            frame_id,
            0,
            -1,
            metadata,
            detection_id=None,
            name_prefix="interp_",
            filename_override=filename,
        )

    def _save_crop(
        self,
        crop,
        frame_id,
        det_idx,
        track_id,
        metadata,
        detection_id=None,
        name_prefix="",
        filename_override=None,
    ):
        """
        Save a crop to disk.

        Args:
            crop: Image to save
            frame_id: Frame number
            det_idx: Detection index within frame
            track_id: Track ID
            metadata: Metadata dict for this crop
            detection_id: Unique Detection ID (optional)

        Returns:
            str | None: filename if saved successfully
        """
        try:
            # Generate filename
            if filename_override:
                filename = filename_override
            elif detection_id is not None:
                # Use DetectionID if available (preferred/unique)
                filename = (
                    f"{name_prefix}"
                    f"{build_detection_image_filename(detection_id, self.output_format)}"
                )
            elif track_id >= 0:
                filename = (
                    f"{name_prefix}f{frame_id:06d}_t{track_id:04d}_d{det_idx:02d}."
                    f"{self.output_format}"
                )
            else:
                filename = (
                    f"{name_prefix}f{frame_id:06d}_d{det_idx:02d}."
                    f"{self.output_format}"
                )

            filepath = self.crops_dir / filename

            # Enqueue the write instead of blocking the calling thread.
            # The background worker (_write_worker) performs the actual imwrite.
            self._write_queue.put(
                (crop.copy(), filepath, self.output_format, self.jpg_quality)
            )

            # Store metadata
            metadata["filename"] = filename
            self.metadata.append(metadata)
            self.total_saved += 1

            return filename

        except Exception as e:
            logger.warning(f"Failed to save crop: {e}")
            return None

    def finalize(self) -> Optional[str]:
        """
        Finalize the dataset and save metadata.
        Called when tracking completes.

        Returns:
            str: Path to the dataset directory, or None if not enabled
        """
        if not self.enabled or self.run_dir is None:
            return None

        if not self.save_images or self.crops_dir is None:
            return str(self.run_dir)

        # Drain the async write queue before writing metadata so that all crops
        # are guaranteed to be on disk when the JSON is finalized.
        if self._write_thread is not None and self._write_thread.is_alive():
            self._write_queue.put(None)  # sentinel
            self._write_thread.join()
            self._write_thread = None

        # Save metadata JSON
        dataset_info = {
            "video_name": self.video_name,
            "total_crops": self.total_saved,
            "parameters": {
                "save_interval": self.save_every_n_frames,
                "output_format": self.output_format,
                "canonical": {
                    **self._geometry.to_dict(),
                    "clipped_count": self._clipping_stats.clipped_count,
                    "worst_overflow_ratio": self._clipping_stats.worst_overflow_ratio,
                    "degenerate_skipped_count": (
                        self._clipping_stats.degenerate_skipped_count
                    ),
                },
            },
            "images": self.metadata,
            "crops": self.metadata,  # Backward compatibility for older consumers.
        }

        _clip_summary = self._clipping_stats.summary()
        if _clip_summary:
            logger.warning("Canonicalization summary: %s", _clip_summary)
        dataset_info["clipping_summary"] = _clip_summary

        try:
            with open(self.metadata_path, "w") as f:
                json.dump(dataset_info, f, indent=2)

            logger.info(
                f"Individual dataset complete: {self.total_saved} crops saved to "
                f"{self.crops_dir.parent}"
            )

            # Create README
            readme_path = self.crops_dir.parent / "README.md"
            with open(readme_path, "w") as f:
                f.write("# Individual Detection Dataset\n\n")
                f.write(f"Generated from: {self.video_name}\n\n")
                f.write("## Contents\n\n")
                f.write(f"- **{self.total_saved} OBB-masked crops**\n")
                f.write("- Each crop contains only the detected animal (OBB region)\n")
                f.write("- Background outside OBB is masked to black\n")
                f.write("- Detections are pre-filtered by ROI and size in tracking\n\n")
                f.write("## File Naming\n\n")
                f.write(f"- `did<ID>.{self.output_format}` for detected rows\n")
                f.write(
                    f"- `interp_fXXXXXX_trajYYYY_segAAAAAA-BBBBBB_pCCCofDDD.{self.output_format}` for interpolated rows\n\n"
                )
                f.write("## Usage\n\n")
                f.write("These crops can be used for:\n")
                f.write("- Training individual identity classifiers\n")
                f.write("- Pose estimation model training\n")
                f.write("- Behavior classification\n")
                if _clip_summary:
                    f.write("\n## Clipping Warning\n\n")
                    f.write(f"{_clip_summary}\n")

            return str(self.crops_dir.parent)

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return None
