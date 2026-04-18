"""YOLO Oriented Bounding Box (OBB) detector.

Supports direct and sequential detection modes with optional TensorRT/ONNX
acceleration and head-tail orientation classification.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from ._obb_geometry import OBBGeometryMixin
from ._runtime_artifacts import RuntimeArtifactMixin

logger = logging.getLogger(__name__)


class YOLOOBBDetector(OBBGeometryMixin, RuntimeArtifactMixin):
    """
    Detects objects using a pretrained YOLO OBB (Oriented Bounding Box) model.
    Compatible interface with ObjectDetector for seamless integration.
    """

    def __init__(self, params):
        self.params = params
        self.model = None
        self.detect_model = None
        self._headtail_analyzer = None  # HeadTailAnalyzer instance
        self._onnx_predict_device = None
        self.obb_predict_device = None
        self.detect_predict_device = None
        self.device = self._detect_device()
        self.use_tensorrt = False
        self.use_onnx = False
        self.onnx_imgsz = None
        self.onnx_batch_size = 1
        self.tensorrt_batch_size = 1
        self.obb_mode = str(self.params.get("YOLO_OBB_MODE", "direct")).strip().lower()
        if self.obb_mode not in {"direct", "sequential"}:
            self.obb_mode = "direct"
        self.direct_model_path = str(
            self.params.get(
                "YOLO_OBB_DIRECT_MODEL_PATH",
                self.params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt"),
            )
            or "yolo26s-obb.pt"
        )
        self.detect_model_path = str(
            self.params.get("YOLO_DETECT_MODEL_PATH", "") or ""
        ).strip()
        self.crop_obb_model_path = str(
            self.params.get("YOLO_CROP_OBB_MODEL_PATH", "") or ""
        ).strip()
        self.headtail_model_path = str(
            self.params.get("YOLO_HEADTAIL_MODEL_PATH", "") or ""
        ).strip()
        self.active_obb_model_path = (
            self.direct_model_path
            if self.obb_mode == "direct"
            else (self.crop_obb_model_path or self.direct_model_path)
        )
        # Keep legacy field in sync for downstream code that still reads YOLO_MODEL_PATH.
        self.params["YOLO_MODEL_PATH"] = self.active_obb_model_path
        self._load_model()
        self._load_aux_models()

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    def _detect_device(self):
        """Detect and configure the optimal device for inference."""
        from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE

        # Check user preference
        device_preference = self.params.get("YOLO_DEVICE", "auto")

        if device_preference != "auto":
            logger.info(f"Using user-specified device: {device_preference}")
            return device_preference

        # Auto-detect best available device using centralized gpu_utils
        if TORCH_CUDA_AVAILABLE:
            device = "cuda:0"
            logger.info(f"CUDA GPU detected, using {device}")
        elif MPS_AVAILABLE:
            device = "mps"  # Apple Silicon GPU
            logger.info("Apple Metal Performance Shaders (MPS) detected, using mps")
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")

        return device

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _current_obb_mode(self) -> str:
        """Return the active OBB mode even for partially initialized instances."""
        mode = (
            str(
                getattr(self, "obb_mode", self.params.get("YOLO_OBB_MODE", "direct"))
                or "direct"
            )
            .strip()
            .lower()
        )
        if mode not in {"direct", "sequential"}:
            return "direct"
        return mode

    def _load_model(self):
        """Load the YOLO OBB model with optional TensorRT optimization."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(
                "ultralytics package not found. Please install it: pip install ultralytics"
            )
            raise ImportError(
                "ultralytics package required for YOLO detection. Install with: pip install ultralytics"
            )
        self._configure_ultralytics_logging()

        model_path_str = str(
            self.params.get(
                "YOLO_MODEL_PATH",
                getattr(self, "active_obb_model_path", "yolo26s-obb.pt"),
            )
            or "yolo26s-obb.pt"
        )
        enable_tensorrt = self.params.get("ENABLE_TENSORRT", False)
        enable_onnx_runtime = self.params.get("ENABLE_ONNX_RUNTIME", False)
        model_path = Path(model_path_str).expanduser().resolve()
        local_model_file = model_path.exists() and model_path.is_file()

        # Check if TensorRT is requested and available
        from hydra_suite.utils.gpu_utils import (
            ONNXRUNTIME_AVAILABLE,
            TENSORRT_AVAILABLE,
        )

        if enable_onnx_runtime and ONNXRUNTIME_AVAILABLE and local_model_file:
            self._try_load_onnx_model(model_path_str)
            if self.use_onnx:
                self.obb_predict_device = self._onnx_predict_device or self.device
                return

        if (
            enable_tensorrt
            and TENSORRT_AVAILABLE
            and self.device.startswith("cuda")
            and local_model_file
        ):
            self._try_load_tensorrt_model(model_path_str)
            if self.use_tensorrt:
                self.obb_predict_device = self.device
                return
            else:
                logger.info("Falling back to standard PyTorch inference")

        # For pretrained model names (yolo26s-obb.pt, etc.), pass directly to YOLO
        # These will be auto-downloaded by ultralytics
        if model_path_str.startswith(("yolov8", "yolov11", "yolo26")):
            try:
                self.model = YOLO(model_path_str)
                # Move model to the appropriate device
                self.model.to(self.device)
                self.obb_predict_device = None
                logger.info(
                    f"YOLO OBB model loaded successfully: {model_path_str} on device: {self.device}"
                )
                return
            except Exception as e:
                logger.error(f"Failed to load YOLO model '{model_path_str}': {e}")
                raise

        # Check if the file exists
        if not model_path.exists():
            logger.error(
                f"YOLO model file not found: {model_path}\n"
                f"Original path: {model_path_str}\n"
                f"Working directory: {Path.cwd()}"
            )
            raise FileNotFoundError(
                f"YOLO model file not found: {model_path}. "
                f"Please check the path and ensure the file exists."
            )

        try:
            # Use the resolved absolute path as a string
            self.model = YOLO(str(model_path))
            # Move model to the appropriate device
            self.model.to(self.device)
            self.obb_predict_device = None
            logger.info(
                f"YOLO OBB model loaded successfully from {model_path} on device: {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{model_path}': {e}")
            raise

    def _load_model_for_task(self, model_path_str: str, task: str):
        """Load an auxiliary YOLO model for detect/classify tasks."""
        if not model_path_str:
            return None, None
        from ultralytics import YOLO

        self._configure_ultralytics_logging()

        runtime_model_path_str = self._prepare_runtime_artifact_for_task(
            model_path_str, task
        )

        model_path = Path(runtime_model_path_str).expanduser().resolve()
        use_builtin = runtime_model_path_str.startswith(("yolo26", "yolov8", "yolov11"))
        if use_builtin:
            model = YOLO(runtime_model_path_str, task=task)
        else:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"YOLO {task} model file not found: {runtime_model_path_str}"
                )
            model = YOLO(str(model_path), task=task)
        self._attach_runtime_artifact_path(model, model_path)
        is_pytorch_checkpoint = use_builtin or model_path.suffix.lower() == ".pt"
        predict_device = self.device
        if (
            not is_pytorch_checkpoint
            and model_path.suffix.lower() == ".onnx"
            and str(self.device).strip().lower() == "mps"
            and self._should_force_onnx_cpu_fallback(model_path)
        ):
            predict_device = "cpu"
        if is_pytorch_checkpoint:
            try:
                model.to(self.device)
                # Avoid passing device per inference call when model is already placed.
                # This prevents repeated select_device() logs in preview loops.
                predict_device = None
            except Exception:
                # Fallback to per-call device argument for compatibility.
                predict_device = self.device
        return model, predict_device

    def _load_headtail_model(self, model_path_str: str):
        """Load optional head-tail model via HeadTailAnalyzer."""
        from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

        ref_ar = float(self._advanced_config_value("reference_aspect_ratio", 2.0))
        margin = float(
            self._advanced_config_value("yolo_headtail_canonical_margin", 1.3)
        )
        conf_threshold = float(self.params.get("YOLO_HEADTAIL_CONF_THRESHOLD", 0.50))
        batch_size = max(1, int(self.params.get("HEADTAIL_BATCH_SIZE", 64)))
        headtail_runtime = str(
            self.params.get(
                "HEADTAIL_COMPUTE_RUNTIME",
                self.params.get("COMPUTE_RUNTIME", self.device),
            )
            or self.device
        )

        analyzer = HeadTailAnalyzer(
            model_path=model_path_str,
            compute_runtime=headtail_runtime,
            conf_threshold=conf_threshold,
            batch_size=batch_size,
            reference_aspect_ratio=ref_ar,
            canonical_margin=margin,
        )
        if analyzer.class_names:
            HeadTailAnalyzer._validate_class_names(
                analyzer.class_names,
                strict=True,
                source=f"head-tail checkpoint {Path(model_path_str).name}",
            )
        self._headtail_analyzer = analyzer
        logger.info(
            "Loaded %s head-tail classifier from %s.",
            analyzer.backend,
            Path(model_path_str).name,
        )

    def _load_aux_models(self):
        """Load optional sequential + head-tail models."""
        if self._current_obb_mode() == "sequential":
            if not self.detect_model_path:
                raise ValueError(
                    "Sequential YOLO OBB mode requires YOLO_DETECT_MODEL_PATH."
                )
            self.detect_model, self.detect_predict_device = self._load_model_for_task(
                self.detect_model_path, task="detect"
            )
            logger.info("YOLO detect model loaded for sequential mode.")

        if self.headtail_model_path:
            self._load_headtail_model(self.headtail_model_path)
            if self._headtail_analyzer is not None:
                backend = self._headtail_analyzer.backend
                if backend not in ("yolo", "none"):
                    logger.info(
                        "Head-tail tiny classifier model loaded (%s).",
                        backend,
                    )
                else:
                    logger.info("YOLO head-tail classify model loaded.")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _runtime_fixed_batch_size(self) -> int:
        """Return fixed runtime batch size when backend enforces static batch dims."""
        if self.use_tensorrt and int(getattr(self, "tensorrt_batch_size", 1)) > 1:
            return int(self.tensorrt_batch_size)
        if self.use_onnx and int(getattr(self, "onnx_batch_size", 1)) > 1:
            return int(self.onnx_batch_size)
        return 1

    @staticmethod
    def _is_coreml_failure(exc) -> bool:
        msg = str(exc)
        return (
            "CoreMLExecutionProvider" in msg
            or "Unable to compute the prediction using a neural network model" in msg
        )

    def _predict_with_coreml_fallback(self, model, predict_kwargs, context: str):
        try:
            return model.predict(**predict_kwargs)
        except Exception as exc:
            predict_device = (
                str(
                    predict_kwargs.get("device")
                    or getattr(self, "obb_predict_device", None)
                    or self.device
                )
                .strip()
                .lower()
            )
            if (
                not self.use_onnx
                or predict_device != "mps"
                or not self._is_coreml_failure(exc)
            ):
                raise
            logger.warning(
                "YOLO ONNX %s failed on mps/CoreML path. Retrying on CPU ORT provider.",
                context,
            )
            self._mark_onnx_artifact_for_cpu_fallback(
                getattr(model, "_hydra_runtime_artifact_path", None)
                or getattr(self, "onnx_model_path", None)
            )
            self._onnx_predict_device = "cpu"
            self.obb_predict_device = "cpu"
            try:
                if hasattr(model, "predictor"):
                    model.predictor = None
            except Exception:
                pass
            retry_kwargs = dict(predict_kwargs)
            retry_kwargs["device"] = "cpu"
            return model.predict(**retry_kwargs)

    def _predict_obb_results(
        self, source, target_classes, raw_conf_floor, max_det, imgsz=None
    ):
        """Run OBB model prediction with backend-specific constraints."""
        fixed_batch = self._runtime_fixed_batch_size()
        # obb_predict_device is None when the model was placed via .to(device).
        # Always fall back to self.device so the explicit device= argument is always
        # passed to predict(), preventing Ultralytics from auto-selecting a wrong device.
        predict_device = getattr(self, "obb_predict_device", None) or self.device

        if isinstance(source, list):
            if len(source) == 0:
                return []

            # Static-batch runtimes require exact batch size. Chunk and pad to avoid
            # invalid-batch errors while still leveraging one predict() call per chunk.
            if fixed_batch > 1:
                all_results = []
                for chunk_start in range(0, len(source), fixed_batch):
                    chunk = list(source[chunk_start : chunk_start + fixed_batch])
                    actual_chunk = len(chunk)
                    if actual_chunk < fixed_batch:
                        chunk.extend([chunk[0]] * (fixed_batch - actual_chunk))
                    predict_kwargs = dict(
                        source=chunk,
                        conf=raw_conf_floor,
                        iou=1.0,  # Always use custom OBB IOU filtering after inference
                        classes=target_classes,
                        max_det=max_det,
                        verbose=False,
                    )
                    if predict_device is not None:
                        predict_kwargs["device"] = predict_device
                    if self.use_onnx and self.onnx_imgsz:
                        predict_kwargs["imgsz"] = int(self.onnx_imgsz)
                    elif imgsz is not None:
                        predict_kwargs["imgsz"] = imgsz
                    chunk_results = self._predict_with_coreml_fallback(
                        self.model,
                        predict_kwargs,
                        context="chunked OBB inference",
                    )
                    all_results.extend(chunk_results[:actual_chunk])
                return all_results

            source_input = source
        elif fixed_batch > 1:
            source_input = [source] * fixed_batch
        else:
            source_input = source

        predict_kwargs = dict(
            source=source_input,
            conf=raw_conf_floor,
            iou=1.0,  # Always use custom OBB IOU filtering after inference
            classes=target_classes,
            max_det=max_det,
            verbose=False,
        )
        if predict_device is not None:
            predict_kwargs["device"] = predict_device
        if self.use_onnx and self.onnx_imgsz:
            predict_kwargs["imgsz"] = int(self.onnx_imgsz)
        elif imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        results = self._predict_with_coreml_fallback(
            self.model,
            predict_kwargs,
            context="OBB inference",
        )
        if not isinstance(source, list) and fixed_batch > 1:
            results = results[:1]
        return results

    # ------------------------------------------------------------------
    # Sequential / crop helpers
    # ------------------------------------------------------------------

    def _clip_crop_box(self, x1, y1, x2, y2, frame_w, frame_h):
        xi1 = int(np.floor(max(0.0, x1)))
        yi1 = int(np.floor(max(0.0, y1)))
        xi2 = int(np.ceil(min(float(frame_w), x2)))
        yi2 = int(np.ceil(min(float(frame_h), y2)))
        if xi2 <= xi1 or yi2 <= yi1:
            return None
        return xi1, yi1, xi2, yi2

    def _build_sequential_crop(self, frame, bbox_xyxy):
        """Create padded crop from stage-1 detection bbox."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        pad_ratio = float(self.params.get("YOLO_SEQ_CROP_PAD_RATIO", 0.15))
        min_crop_size = float(self.params.get("YOLO_SEQ_MIN_CROP_SIZE_PX", 64))
        enforce_square = bool(self.params.get("YOLO_SEQ_ENFORCE_SQUARE_CROP", True))

        crop_w = bw * (1.0 + 2.0 * max(0.0, pad_ratio))
        crop_h = bh * (1.0 + 2.0 * max(0.0, pad_ratio))
        if enforce_square:
            side = max(crop_w, crop_h)
            crop_w = side
            crop_h = side
        crop_w = max(min_crop_size, crop_w)
        crop_h = max(min_crop_size, crop_h)

        xx1 = cx - crop_w * 0.5
        yy1 = cy - crop_h * 0.5
        xx2 = cx + crop_w * 0.5
        yy2 = cy + crop_h * 0.5

        clipped = self._clip_crop_box(xx1, yy1, xx2, yy2, w, h)
        if clipped is None:
            return None, None
        xi1, yi1, xi2, yi2 = clipped
        crop = frame[yi1:yi2, xi1:xi2]
        if crop is None or crop.size == 0:
            return None, None
        return crop, (float(xi1), float(yi1))

    # ------------------------------------------------------------------
    # Head-tail classification
    # ------------------------------------------------------------------

    def _compute_headtail_hints(self, frame, obb_corners_list, profiler=None):
        """Infer directed heading hints from optional head-tail classifier.

        Delegates to the cross-frame batched implementation so both the
        single-frame and multi-frame paths share the same logic.
        """
        results = self._compute_headtail_hints_cross_frame(
            [frame],
            [obb_corners_list],
            profiler=profiler,
        )
        return results[0]

    def _should_compute_canonical_affines(self):
        """Return whether native-scale canonical affines are needed downstream."""
        params = self.params or {}
        return bool(
            params.get("ENABLE_POSE_EXTRACTOR", False)
            or params.get("ENABLE_INDIVIDUAL_DATASET", False)
            or params.get("ENABLE_INDIVIDUAL_IMAGE_SAVE", False)
            or params.get("EXPORT_FINAL_CANONICAL_IMAGES", False)
            or params.get("FINAL_MEDIA_EXPORT_VIDEOS_ENABLED", False)
            or params.get("GENERATE_ORIENTED_TRACK_VIDEOS", False)
        )

    def _select_headtail_candidate_indices(
        self,
        raw_meas,
        raw_sizes,
        raw_shapes,
        raw_confidences,
        raw_obb_corners,
        *,
        roi_mask=None,
    ):
        """Return raw detection indices that survive current detector filtering."""
        if not raw_meas:
            return []
        (
            _meas,
            _sizes,
            _shapes,
            _confidences,
            _obb_corners,
            candidate_ids,
        ) = self.filter_raw_detections(
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            roi_mask=roi_mask,
            detection_ids=list(range(len(raw_meas))),
        )
        selected_ids = [int(idx) for idx in candidate_ids]
        detect_conf_threshold = float(
            self.params.get(
                "YOLO_HEADTAIL_DETECT_CONF_THRESHOLD",
                self.params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25),
            )
        )
        if detect_conf_threshold <= 0.0:
            return selected_ids

        gated_ids = []
        for idx, confidence in zip(selected_ids, _confidences):
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                continue
            if confidence_value >= detect_conf_threshold:
                gated_ids.append(idx)
        return gated_ids

    def _compute_headtail_hints_for_indices(
        self,
        frame,
        raw_obb_corners,
        candidate_indices,
        *,
        include_canonical_affines=True,
        profiler=None,
    ):
        """Run head-tail inference on a subset of detections and scatter back."""
        total = len(raw_obb_corners)
        heading_hints = [float("nan")] * total
        heading_confidences = [0.0] * total
        directed_mask = [0] * total
        canonical_affines = [None] * total if include_canonical_affines else None

        if not candidate_indices:
            return heading_hints, heading_confidences, directed_mask, canonical_affines

        selected_corners = [raw_obb_corners[idx] for idx in candidate_indices]
        (
            selected_hints,
            selected_confidences,
            selected_directed,
            selected_affines,
        ) = self._compute_headtail_hints(
            frame,
            selected_corners,
            profiler=profiler,
        )

        for slot, raw_idx in enumerate(candidate_indices):
            if raw_idx >= total:
                continue
            if slot < len(selected_hints):
                heading_hints[raw_idx] = selected_hints[slot]
            if slot < len(selected_confidences):
                heading_confidences[raw_idx] = float(selected_confidences[slot])
            if slot < len(selected_directed):
                directed_mask[raw_idx] = int(selected_directed[slot])
            if canonical_affines is not None and slot < len(selected_affines):
                canonical_affines[raw_idx] = selected_affines[slot]

        return heading_hints, heading_confidences, directed_mask, canonical_affines

    def _compute_headtail_hints_cross_frame(
        self,
        frames,
        per_frame_obb_corners,
        *,
        include_canonical_affines=True,
        profiler=None,
    ):
        """Batch head-tail classification across multiple frames in one GPU call.

        Delegates crop canonicalization and inference to
        :class:`~hydra_suite.core.identity.headtail_analyzer.HeadTailAnalyzer`,
        then re-derives native-scale affines from OBB corners for downstream
        consumers.

        Args:
            frames: list of *N* video frames (BGR ndarray).
            per_frame_obb_corners: list of *N* lists, where each inner list
                contains the OBB corners for detections in the corresponding
                frame.

        Returns:
            list of *N* tuples ``(heading_hints, heading_confidences,
            directed_mask, canonical_affines)``
            where ``canonical_affines`` is a list of (2, 3) float32 arrays or None.
        """
        n_frames = len(frames)
        # Pre-allocate per-frame result arrays (default: no orientation).
        results_per_frame = []
        for corners in per_frame_obb_corners:
            n = len(corners)
            results_per_frame.append(
                (
                    [float("nan")] * n,
                    [0.0] * n,
                    [0] * n,
                    ([None] * n if include_canonical_affines else None),
                )
            )

        analyzer = self._headtail_analyzer
        if analyzer is None or not analyzer.is_available:
            return results_per_frame

        # ----- Phases 1-3: delegate to HeadTailAnalyzer --------------------
        ht_results = analyzer.analyze_crops(
            frames, per_frame_obb_corners, profiler=profiler
        )

        # Unpack (heading, confidence, directed_flag) tuples into result arrays
        for fi in range(n_frames):
            for di, (heading, conf, directed) in enumerate(ht_results[fi]):
                results_per_frame[fi][0][di] = heading
                results_per_frame[fi][1][di] = float(conf)
                results_per_frame[fi][2][di] = directed

        if not include_canonical_affines:
            return results_per_frame

        # ----- Phase 4: replace stored affines with native-scale variants --
        # Head-tail used a fixed-size canvas (e.g. 128px) for batched GPU
        # inference. Re-derive native-scale affines from OBB corners so
        # downstream consumers (individual dataset, oriented video) get
        # crops at the source video's native pixel resolution.
        #
        # Vectorised: pre-compute all edge norms and canvas dims in bulk,
        # then loop only for cv2.getAffineTransform (inherently per-element).
        try:
            from hydra_suite.core.canonicalization.crop import compute_alignment_affine

            ref_ar = float(self._advanced_config_value("reference_aspect_ratio", 2.0))
            padding = float(self.params.get("INDIVIDUAL_CROP_PADDING", 0.1))
            _margin = 1.0 + max(0.0, padding)
            _ar = max(1.0, ref_ar)

            # Flatten all corners for vectorised edge-norm computation
            _all_corners = []
            _all_indices = []
            for fi in range(n_frames):
                for di, corners in enumerate(per_frame_obb_corners[fi]):
                    _all_corners.append(
                        np.asarray(corners, dtype=np.float32).reshape(4, 2)
                    )
                    _all_indices.append((fi, di))

            if _all_corners:
                _stacked = np.stack(_all_corners)  # (N, 4, 2)
                # Edge lengths: e01 = ||c1-c0||, e12 = ||c2-c1||
                _e01 = np.linalg.norm(_stacked[:, 1] - _stacked[:, 0], axis=1)
                _e12 = np.linalg.norm(_stacked[:, 2] - _stacked[:, 1], axis=1)
                _major = np.maximum(_e01, _e12)
                # Canvas dimensions (vectorised)
                _raw_w = _major * _margin
                _canvas_w = np.maximum(8, (np.ceil(_raw_w / 2.0) * 2).astype(np.int32))
                _canvas_h = np.maximum(
                    8, np.round(_canvas_w / _ar / 2.0).astype(np.int32) * 2
                )

                for idx, (fi, di) in enumerate(_all_indices):
                    try:
                        cw_i = int(_canvas_w[idx])
                        ch_i = int(_canvas_h[idx])
                        M_align, _ = compute_alignment_affine(
                            _all_corners[idx], cw_i, ch_i, padding
                        )
                        results_per_frame[fi][3][di] = M_align.astype(np.float32)
                    except (ValueError, Exception):
                        pass  # keep whatever was there (or None)
        except ImportError:
            pass  # graceful fallback if canonical_crop not available

        return results_per_frame

    # ------------------------------------------------------------------
    # Raw detection runners (direct / sequential)
    # ------------------------------------------------------------------

    def _run_direct_raw_detection(
        self,
        frame,
        target_classes,
        raw_conf_floor,
        max_det,
        return_class_ids: bool = False,
    ):
        results = self._predict_obb_results(
            frame, target_classes, raw_conf_floor, max_det
        )
        if len(results) == 0:
            if return_class_ids:
                return [], [], [], [], [], [], None
            return [], [], [], [], [], None
        result0 = results[0]
        if result0.obb is None or len(result0.obb) == 0:
            if return_class_ids:
                return [], [], [], [], [], [], result0
            return [], [], [], [], [], result0
        if return_class_ids:
            (
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confidences,
                raw_obb_corners,
                raw_class_ids,
            ) = self._extract_raw_detections(result0.obb, return_class_ids=True)
            return (
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confidences,
                raw_obb_corners,
                raw_class_ids,
                result0,
            )
        raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners = (
            self._extract_raw_detections(result0.obb)
        )
        return (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            result0,
        )

    def _seq_stage1_predict(self, frame, target_classes, raw_conf_floor, max_det):
        detect_target_classes = self.params.get(
            "YOLO_DETECT_TARGET_CLASSES", target_classes
        )
        seq_detect_conf = float(
            self.params.get("YOLO_SEQ_DETECT_CONF_THRESHOLD", raw_conf_floor)
        )
        seq_detect_conf = max(1e-4, seq_detect_conf)
        detect_kwargs = dict(
            source=frame,
            conf=seq_detect_conf,
            iou=1.0,
            classes=detect_target_classes,
            max_det=max_det,
            verbose=False,
        )
        detect_predict_device = (
            getattr(self, "detect_predict_device", None) or self.device
        )
        if detect_predict_device is not None:
            detect_kwargs["device"] = detect_predict_device
        seq_detect_imgsz = int(self.params.get("YOLO_SEQ_DETECT_IMGSZ", 0))
        if seq_detect_imgsz > 0:
            detect_kwargs["imgsz"] = seq_detect_imgsz
        return self._predict_detect_stage1(detect_kwargs, detect_predict_device)

    def _predict_detect_stage1(self, detect_kwargs, detect_predict_device):
        try:
            return self.detect_model.predict(**detect_kwargs)
        except Exception as exc:
            if str(
                detect_predict_device
            ).strip().lower() != "mps" or not self._is_coreml_failure(exc):
                raise
            logger.warning(
                "YOLO detect stage-1 ONNX inference failed on mps/CoreML path. Retrying on CPU ORT provider."
            )
            self._mark_onnx_artifact_for_cpu_fallback(
                getattr(self.detect_model, "_hydra_runtime_artifact_path", None)
            )
            self.detect_predict_device = "cpu"
            try:
                if hasattr(self.detect_model, "predictor"):
                    self.detect_model.predictor = None
            except Exception:
                pass
            retry_kwargs = dict(detect_kwargs)
            retry_kwargs["device"] = "cpu"
            return self.detect_model.predict(**retry_kwargs)

    def _seq_stage1_fixed_batch_size(self) -> int | None:
        artifact_path = getattr(self.detect_model, "_hydra_runtime_artifact_path", None)
        if not artifact_path:
            return None
        suffix = Path(str(artifact_path)).suffix.lower()
        if suffix not in {".onnx", ".engine", ".trt"}:
            return None
        raw_value = self.params.get("YOLO_DETECT_RUNTIME_BUILD_BATCH_SIZE", None)
        if raw_value in (None, "", 0, "0"):
            raw_value = self.params.get("TENSORRT_MAX_BATCH_SIZE", 1)
        try:
            return max(1, int(raw_value or 1))
        except (TypeError, ValueError):
            return 1

    def _seq_stage1_predict_batched(
        self,
        frames,
        target_classes,
        raw_conf_floor,
        max_det,
    ):
        fixed_batch_size = self._seq_stage1_fixed_batch_size()
        if fixed_batch_size is not None:
            all_results = []
            for chunk_start in range(0, len(frames), fixed_batch_size):
                chunk = list(frames[chunk_start : chunk_start + fixed_batch_size])
                actual_chunk = len(chunk)
                if actual_chunk < fixed_batch_size and actual_chunk > 0:
                    chunk.extend([chunk[0]] * (fixed_batch_size - actual_chunk))
                chunk_results = self._seq_stage1_predict(
                    chunk,
                    target_classes,
                    raw_conf_floor,
                    max_det,
                )
                all_results.extend(list(chunk_results)[:actual_chunk])
            return all_results
        return self._seq_stage1_predict(
            list(frames),
            target_classes,
            raw_conf_floor,
            max_det,
        )

    def _seq_build_crops(self, frame, xyxy, order, max_det):
        crops = []
        crop_offsets = []
        crop_original_sizes = []
        for idx in order:
            crop, offset = self._build_sequential_crop(frame, xyxy[idx])
            if crop is None or offset is None:
                continue
            crop_original_sizes.append((crop.shape[1], crop.shape[0]))  # (w, h)
            crops.append(crop)
            crop_offsets.append(offset)
        return crops, crop_offsets, crop_original_sizes

    def _seq_resize_crops_for_stage2(self, crops):
        stage2_imgsz = int(self.params.get("YOLO_SEQ_STAGE2_IMGSZ", 160))
        if stage2_imgsz <= 0:
            return crops, None
        resize_interp = getattr(cv2, "INTER_LINEAR", getattr(cv2, "INTER_AREA", 1))
        resized_crops = []
        for crop in crops:
            h_c, w_c = crop.shape[:2]
            if h_c != stage2_imgsz or w_c != stage2_imgsz:
                resized_crops.append(
                    cv2.resize(
                        crop,
                        (stage2_imgsz, stage2_imgsz),
                        interpolation=resize_interp,
                    )
                )
            else:
                resized_crops.append(crop)
        return resized_crops, stage2_imgsz

    def _seq_pad_crops_to_pow2(self, crops_for_stage2, predict_imgsz):
        n_real = len(crops_for_stage2)
        pow2_pad = self.params.get("YOLO_SEQ_STAGE2_POW2_PAD", 0)
        if not (pow2_pad and predict_imgsz is not None and n_real > 0):
            return crops_for_stage2, n_real
        p2 = 1
        while p2 < n_real:
            p2 *= 2
        if p2 > n_real:
            pad_img = np.zeros((predict_imgsz, predict_imgsz, 3), dtype=np.uint8)
            crops_for_stage2 = list(crops_for_stage2) + [pad_img] * (p2 - n_real)
        return crops_for_stage2, n_real

    def _seq_individual_batch_size(self) -> int | None:
        raw_value = self.params.get("YOLO_SEQ_INDIVIDUAL_BATCH_SIZE", None)
        if raw_value in (None, "", 0, "0"):
            return None
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _seq_stage2_pad_crop(predict_imgsz, reference_crop):
        if predict_imgsz is not None and int(predict_imgsz) > 0:
            size = int(predict_imgsz)
            return np.zeros((size, size, 3), dtype=np.uint8)
        return np.zeros_like(reference_crop)

    def _seq_run_stage2_obb_batched(
        self,
        crops_for_stage2,
        target_classes,
        raw_conf_floor,
        max_det,
        predict_imgsz,
    ):
        if not crops_for_stage2:
            return []
        batch_size = self._seq_individual_batch_size()
        if batch_size is None:
            padded_crops, n_real_crops = self._seq_pad_crops_to_pow2(
                crops_for_stage2, predict_imgsz
            )
            stage2_results = self._seq_run_stage2_obb(
                padded_crops,
                target_classes,
                raw_conf_floor,
                max_det,
                predict_imgsz,
            )
            return list(stage2_results)[:n_real_crops]

        all_results = []
        for chunk_start in range(0, len(crops_for_stage2), batch_size):
            chunk = list(crops_for_stage2[chunk_start : chunk_start + batch_size])
            actual_chunk = len(chunk)
            if actual_chunk < batch_size and actual_chunk > 0:
                pad_crop = self._seq_stage2_pad_crop(predict_imgsz, chunk[0])
                chunk.extend([pad_crop] * (batch_size - actual_chunk))
            chunk_results = self._seq_run_stage2_obb(
                chunk,
                target_classes,
                raw_conf_floor,
                max_det,
                predict_imgsz,
            )
            all_results.extend(list(chunk_results)[:actual_chunk])
        return all_results

    def _seq_run_stage2_obb(
        self, crops_for_stage2, target_classes, raw_conf_floor, max_det, predict_imgsz
    ):
        try:
            return self._predict_obb_results(
                crops_for_stage2,
                target_classes=target_classes,
                raw_conf_floor=raw_conf_floor,
                max_det=max_det,
                imgsz=predict_imgsz,
            )
        except TypeError as exc:
            # Backward-compat for monkeypatched test doubles without imgsz kwarg.
            if "imgsz" not in str(exc):
                raise
            return self._predict_obb_results(
                crops_for_stage2,
                target_classes=target_classes,
                raw_conf_floor=raw_conf_floor,
                max_det=max_det,
            )

    def _seq_accumulate_crop_detections(
        self,
        stage2_results,
        crop_offsets,
        crop_original_sizes,
        n_real_crops,
        predict_imgsz,
        return_class_ids,
    ):
        merged_meas = []
        merged_sizes = []
        merged_shapes = []
        merged_conf = []
        merged_corners = []
        merged_class_ids = []
        n_stage2 = min(len(stage2_results), len(crop_offsets), n_real_crops)
        for i in range(n_stage2):
            result = stage2_results[i]
            x0, y0 = crop_offsets[i]
            if result is None or result.obb is None or len(result.obb) == 0:
                continue
            if return_class_ids:
                (
                    crop_meas,
                    crop_sizes,
                    crop_shapes,
                    crop_conf,
                    crop_corners,
                    crop_class_ids,
                ) = self._extract_raw_detections(result.obb, return_class_ids=True)
            else:
                crop_class_ids = []
                (
                    crop_meas,
                    crop_sizes,
                    crop_shapes,
                    crop_conf,
                    crop_corners,
                ) = self._extract_raw_detections(result.obb)
            if not crop_meas:
                continue
            if predict_imgsz is not None and i < len(crop_original_sizes):
                orig_w, orig_h = crop_original_sizes[i]
                sx = orig_w / float(predict_imgsz)
                sy = orig_h / float(predict_imgsz)
            else:
                sx, sy = 1.0, 1.0
            for j in range(len(crop_meas)):
                m = np.asarray(crop_meas[j], dtype=np.float32).copy()
                m[0] = m[0] * np.float32(sx) + np.float32(x0)
                m[1] = m[1] * np.float32(sy) + np.float32(y0)
                c = np.asarray(crop_corners[j], dtype=np.float32).copy()
                c[:, 0] = c[:, 0] * np.float32(sx) + np.float32(x0)
                c[:, 1] = c[:, 1] * np.float32(sy) + np.float32(y0)
                merged_meas.append(m)
                # Scale area back to original-frame pixel space (area scales by sx*sy)
                merged_sizes.append(float(crop_sizes[j]) * sx * sy)
                merged_shapes.append(tuple(crop_shapes[j]))
                merged_conf.append(float(crop_conf[j]))
                merged_corners.append(c)
                if return_class_ids:
                    merged_class_ids.append(int(crop_class_ids[j]))
        return (
            merged_meas,
            merged_sizes,
            merged_shapes,
            merged_conf,
            merged_corners,
            merged_class_ids,
        )

    def _seq_sort_and_return(
        self,
        merged_meas,
        merged_sizes,
        merged_shapes,
        merged_conf,
        merged_corners,
        merged_class_ids,
        max_det,
        det0,
        return_class_ids,
    ):
        if not merged_meas:
            if return_class_ids:
                return [], [], [], [], [], [], det0
            return [], [], [], [], [], det0
        conf_arr = np.asarray(merged_conf, dtype=np.float32)
        order_final = np.argsort(conf_arr)[::-1]
        if len(order_final) > max_det:
            order_final = order_final[:max_det]
        raw_meas = [merged_meas[i] for i in order_final]
        raw_sizes = [merged_sizes[i] for i in order_final]
        raw_shapes = [merged_shapes[i] for i in order_final]
        raw_confidences = [merged_conf[i] for i in order_final]
        raw_obb_corners = [merged_corners[i] for i in order_final]
        if return_class_ids:
            raw_class_ids = [merged_class_ids[i] for i in order_final]
            return (
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confidences,
                raw_obb_corners,
                raw_class_ids,
                det0,
            )
        return (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            det0,
        )

    def _run_sequential_raw_detection(
        self,
        frame,
        target_classes,
        raw_conf_floor,
        max_det,
        return_class_ids: bool = False,
        profiler: object = None,
    ):
        if self.detect_model is None:
            if return_class_ids:
                return [], [], [], [], [], [], None
            return [], [], [], [], [], None

        try:
            detect_results = self._seq_stage1_predict(
                frame, target_classes, raw_conf_floor, max_det
            )
        except Exception as exc:
            logger.error("YOLO sequential detect stage failed: %s", exc)
            return [], [], [], [], [], None

        if not detect_results:
            return [], [], [], [], [], None
        det0 = detect_results[0]
        boxes = getattr(det0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], [], [], [], [], det0

        xyxy = np.ascontiguousarray(boxes.xyxy.cpu().numpy(), dtype=np.float32)
        det_conf = np.ascontiguousarray(boxes.conf.cpu().numpy(), dtype=np.float32)
        order = np.argsort(det_conf)[::-1]
        if len(order) > max_det:
            order = order[:max_det]

        if profiler is not None:
            profiler.phase_start("sequential_obb_crop")
        crops, crop_offsets, crop_original_sizes = self._seq_build_crops(
            frame, xyxy, order, max_det
        )
        if profiler is not None:
            profiler.phase_end("sequential_obb_crop", work_units=len(crops))

        if not crops:
            if return_class_ids:
                return [], [], [], [], [], [], det0
            return [], [], [], [], [], det0

        crops_for_stage2, predict_imgsz = self._seq_resize_crops_for_stage2(crops)
        n_real_crops = len(crops_for_stage2)

        if profiler is not None:
            profiler.phase_start("sequential_obb_inference")
        stage2_results = self._seq_run_stage2_obb_batched(
            crops_for_stage2, target_classes, raw_conf_floor, max_det, predict_imgsz
        )
        if profiler is not None:
            profiler.phase_end(
                "sequential_obb_inference",
                work_units=n_real_crops,
            )

        (
            merged_meas,
            merged_sizes,
            merged_shapes,
            merged_conf,
            merged_corners,
            merged_class_ids,
        ) = self._seq_accumulate_crop_detections(
            stage2_results,
            crop_offsets,
            crop_original_sizes,
            n_real_crops,
            predict_imgsz,
            return_class_ids,
        )

        return self._seq_sort_and_return(
            merged_meas,
            merged_sizes,
            merged_shapes,
            merged_conf,
            merged_corners,
            merged_class_ids,
            max_det,
            det0,
            return_class_ids,
        )

    # ------------------------------------------------------------------
    # Public detection API
    # ------------------------------------------------------------------

    def detect_objects(
        self: object,
        frame: object,
        frame_count: object,
        return_raw: bool = False,
        profiler: object = None,
    ) -> object:
        """
        Detects objects in a frame using YOLO OBB.

        Args:
            frame: Input frame (grayscale or BGR)
            frame_count: Current frame number for logging

        Returns:
            Default mode:
                meas, sizes, shapes, yolo_results, confidences
            If return_raw=True:
                raw_meas, raw_sizes, raw_shapes, yolo_results, raw_confidences,
                raw_obb_corners, raw_heading_hints, raw_heading_confidences,
                raw_directed_mask, raw_canonical_affines
        """
        if self.model is None:
            logger.error("YOLO model not initialized")
            if return_raw:
                return [], [], [], None, [], [], [], [], [], None
            return [], [], [], None, []

        p = self.params
        target_classes = p.get("YOLO_TARGET_CLASSES", None)  # None means all classes
        raw_conf_floor = max(1e-4, float(p.get("RAW_YOLO_CONFIDENCE_FLOOR", 1e-3)))
        max_det = self._raw_detection_cap()

        if profiler is not None:
            profiler.phase_start("yolo_obb_inference")
        try:
            if self._current_obb_mode() == "sequential":
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    yolo_results,
                ) = self._run_sequential_raw_detection(
                    frame,
                    target_classes=target_classes,
                    raw_conf_floor=raw_conf_floor,
                    max_det=max_det,
                    profiler=profiler,
                )
            else:
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    yolo_results,
                ) = self._run_direct_raw_detection(
                    frame,
                    target_classes=target_classes,
                    raw_conf_floor=raw_conf_floor,
                    max_det=max_det,
                )
        except Exception as e:
            logger.error(f"YOLO inference failed on frame {frame_count}: {e}")
            if return_raw:
                return [], [], [], None, [], [], [], [], [], None
            return [], [], [], None, []
        finally:
            if profiler is not None:
                profiler.phase_end("yolo_obb_inference")

        if not raw_meas:
            raw_heading_hints, raw_heading_confidences, raw_directed_mask = [], [], []
            if return_raw:
                return (
                    [],
                    [],
                    [],
                    yolo_results,
                    [],
                    [],
                    raw_heading_hints,
                    raw_heading_confidences,
                    raw_directed_mask,
                    None,
                )
            return [], [], [], yolo_results, []

        if bool(self.params.get("TRACKING_REALTIME_MODE", False)):
            include_canonical_affines = bool(
                return_raw and self._should_compute_canonical_affines()
            )
            candidate_indices = self._select_headtail_candidate_indices(
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confidences,
                raw_obb_corners,
            )
            (
                raw_heading_hints,
                raw_heading_confidences,
                raw_directed_mask,
                _raw_affines,
            ) = self._compute_headtail_hints_for_indices(
                frame,
                raw_obb_corners,
                candidate_indices,
                include_canonical_affines=include_canonical_affines,
                profiler=profiler,
            )
        else:
            include_canonical_affines = bool(
                return_raw and self._should_compute_canonical_affines()
            )
            (
                raw_heading_hints,
                raw_heading_confidences,
                raw_directed_mask,
                _raw_affines,
            ) = self._compute_headtail_hints_cross_frame(
                [frame],
                [raw_obb_corners],
                include_canonical_affines=include_canonical_affines,
                profiler=profiler,
            )[
                0
            ]

        if return_raw:
            if yolo_results is not None:
                pass
            return (
                raw_meas,
                raw_sizes,
                raw_shapes,
                yolo_results,
                raw_confidences,
                raw_obb_corners,
                raw_heading_hints,
                raw_heading_confidences,
                raw_directed_mask,
                _raw_affines,
            )

        (
            meas,
            sizes,
            shapes,
            confidences,
            obb_corners_list,
            _,
            filtered_heading_hints,
            _filtered_heading_confidences,
            filtered_directed_mask,
        ) = self.filter_raw_detections(
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            roi_mask=None,
            detection_ids=None,
            heading_hints=raw_heading_hints,
            heading_confidences=raw_heading_confidences,
            directed_mask=raw_directed_mask,
        )

        if meas:
            logger.debug(f"Frame {frame_count}: YOLO detected {len(meas)} objects")

        # Return filtered OBB corners alongside other data
        # Store in results object for access by individual dataset generator
        if yolo_results is not None:
            pass

        return meas, sizes, shapes, yolo_results, confidences

    def _batched_sequential_mode(
        self, frames, start_frame_idx, return_raw, progress_callback, profiler=None
    ):
        actual_frame_count = len(frames)
        p = self.params
        target_classes = p.get("YOLO_TARGET_CLASSES", None)
        raw_conf_floor = max(1e-4, float(p.get("RAW_YOLO_CONFIDENCE_FLOOR", 1e-3)))
        max_det = self._raw_detection_cap()

        if profiler is not None:
            profiler.phase_start("yolo_seq_stage1_inference")
        try:
            stage1_results = self._seq_stage1_predict_batched(
                frames,
                target_classes,
                raw_conf_floor,
                max_det,
            )
        except Exception as exc:
            logger.error("YOLO batched sequential detect stage failed: %s", exc)
            empty_raw = ([], [], [], [], [], [], [], [], None)
            empty_filtered = ([], [], [], [], [])
            return [empty_raw if return_raw else empty_filtered for _ in frames]
        if profiler is not None:
            profiler.phase_end(
                "yolo_seq_stage1_inference",
                work_units=actual_frame_count,
            )

        per_frame_stage1 = [None] * actual_frame_count
        per_frame_merged = [
            {
                "meas": [],
                "sizes": [],
                "shapes": [],
                "conf": [],
                "corners": [],
            }
            for _ in range(actual_frame_count)
        ]
        all_crops = []
        all_offsets = []
        all_original_sizes = []
        all_frame_indices = []

        if profiler is not None:
            profiler.phase_start("sequential_obb_crop")
        for idx in range(actual_frame_count):
            det0 = stage1_results[idx] if idx < len(stage1_results) else None
            per_frame_stage1[idx] = det0
            boxes = getattr(det0, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = np.ascontiguousarray(boxes.xyxy.cpu().numpy(), dtype=np.float32)
            det_conf = np.ascontiguousarray(boxes.conf.cpu().numpy(), dtype=np.float32)
            order = np.argsort(det_conf)[::-1]
            if len(order) > max_det:
                order = order[:max_det]
            crops, crop_offsets, crop_original_sizes = self._seq_build_crops(
                frames[idx], xyxy, order, max_det
            )
            for crop, crop_offset, crop_size in zip(
                crops, crop_offsets, crop_original_sizes
            ):
                all_crops.append(crop)
                all_offsets.append(crop_offset)
                all_original_sizes.append(crop_size)
                all_frame_indices.append(idx)
        if profiler is not None:
            profiler.phase_end("sequential_obb_crop", work_units=len(all_crops))

        if all_crops:
            crops_for_stage2, predict_imgsz = self._seq_resize_crops_for_stage2(
                all_crops
            )
            if profiler is not None:
                profiler.phase_start("sequential_obb_inference")
            stage2_results = self._seq_run_stage2_obb_batched(
                crops_for_stage2,
                target_classes,
                raw_conf_floor,
                max_det,
                predict_imgsz,
            )
            if profiler is not None:
                profiler.phase_end(
                    "sequential_obb_inference",
                    work_units=len(all_crops),
                )

            for crop_index, result in enumerate(stage2_results):
                frame_index = all_frame_indices[crop_index]
                x0, y0 = all_offsets[crop_index]
                if result is None or result.obb is None or len(result.obb) == 0:
                    continue
                (
                    crop_meas,
                    crop_sizes,
                    crop_shapes,
                    crop_conf,
                    crop_corners,
                ) = self._extract_raw_detections(result.obb)
                if not crop_meas:
                    continue
                if predict_imgsz is not None:
                    orig_w, orig_h = all_original_sizes[crop_index]
                    sx = orig_w / float(predict_imgsz)
                    sy = orig_h / float(predict_imgsz)
                else:
                    sx, sy = 1.0, 1.0
                bucket = per_frame_merged[frame_index]
                for detection_index in range(len(crop_meas)):
                    meas = np.asarray(
                        crop_meas[detection_index], dtype=np.float32
                    ).copy()
                    meas[0] = meas[0] * np.float32(sx) + np.float32(x0)
                    meas[1] = meas[1] * np.float32(sy) + np.float32(y0)
                    corners = np.asarray(
                        crop_corners[detection_index], dtype=np.float32
                    ).copy()
                    corners[:, 0] = corners[:, 0] * np.float32(sx) + np.float32(x0)
                    corners[:, 1] = corners[:, 1] * np.float32(sy) + np.float32(y0)
                    bucket["meas"].append(meas)
                    bucket["sizes"].append(float(crop_sizes[detection_index]) * sx * sy)
                    bucket["shapes"].append(tuple(crop_shapes[detection_index]))
                    bucket["conf"].append(float(crop_conf[detection_index]))
                    bucket["corners"].append(corners)

        per_frame_raw = []
        for idx in range(actual_frame_count):
            bucket = per_frame_merged[idx]
            if not bucket["meas"]:
                per_frame_raw.append(None)
                continue
            conf_arr = np.asarray(bucket["conf"], dtype=np.float32)
            order_final = np.argsort(conf_arr)[::-1]
            if len(order_final) > max_det:
                order_final = order_final[:max_det]
            per_frame_raw.append(
                (
                    [bucket["meas"][i] for i in order_final],
                    [bucket["sizes"][i] for i in order_final],
                    [bucket["shapes"][i] for i in order_final],
                    [bucket["conf"][i] for i in order_final],
                    [bucket["corners"][i] for i in order_final],
                )
            )

        if self._headtail_analyzer is not None and self._headtail_analyzer.is_available:
            include_canonical_affines = bool(
                return_raw and self._should_compute_canonical_affines()
            )
            per_frame_corners = [
                raw[4] if raw is not None else [] for raw in per_frame_raw
            ]
            headtail_per_frame = self._compute_headtail_hints_cross_frame(
                frames[:actual_frame_count],
                per_frame_corners,
                include_canonical_affines=include_canonical_affines,
                profiler=profiler,
            )
        else:
            headtail_per_frame = None

        batch_detections = []
        for idx in range(actual_frame_count):
            raw = per_frame_raw[idx]
            if return_raw:
                if raw is None:
                    batch_detections.append(([], [], [], [], [], [], [], [], None))
                else:
                    (
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confidences,
                        raw_obb_corners,
                    ) = raw
                    if headtail_per_frame is not None:
                        (
                            raw_heading_hints,
                            raw_heading_confidences,
                            raw_directed_mask,
                            raw_canonical_affines,
                        ) = headtail_per_frame[idx]
                    else:
                        raw_heading_hints = [float("nan")] * len(raw_meas)
                        raw_heading_confidences = [0.0] * len(raw_meas)
                        raw_directed_mask = [0] * len(raw_meas)
                        raw_canonical_affines = None
                    batch_detections.append(
                        (
                            raw_meas,
                            raw_sizes,
                            raw_shapes,
                            raw_confidences,
                            raw_obb_corners,
                            raw_heading_hints,
                            raw_heading_confidences,
                            raw_directed_mask,
                            raw_canonical_affines,
                        )
                    )
            else:
                if raw is None:
                    batch_detections.append(([], [], [], [], []))
                else:
                    batch_detections.append(
                        self._assemble_batched_frame_result(
                            raw,
                            headtail_per_frame,
                            idx,
                            return_raw=False,
                        )
                    )

            if progress_callback and (idx + 1) % 10 == 0:
                progress_callback(
                    idx + 1,
                    actual_frame_count,
                    f"Processing batch frame {idx + 1}/{actual_frame_count}",
                )

        return batch_detections

    def _resolve_fixed_batch_params(self):
        fixed_batch_size = None
        fixed_backend = None
        if self.use_tensorrt and hasattr(self, "tensorrt_batch_size"):
            fixed_batch_size = max(1, int(self.tensorrt_batch_size))
            fixed_backend = "TensorRT"
        elif self.use_onnx:
            fixed_batch_size = max(1, int(getattr(self, "onnx_batch_size", 1)))
            fixed_backend = "ONNX"
        return fixed_batch_size, fixed_backend

    def _run_fixed_batch_obb_inference(
        self,
        frames,
        actual_frame_count,
        fixed_batch_size,
        fixed_backend,
        target_classes,
        raw_conf_floor,
        max_det,
    ):
        all_results = []
        obb_predict_device = getattr(self, "obb_predict_device", None) or self.device
        for chunk_start in range(0, actual_frame_count, fixed_batch_size):
            chunk_end = min(chunk_start + fixed_batch_size, actual_frame_count)
            chunk_frames = frames[chunk_start:chunk_end]
            chunk_size = len(chunk_frames)
            if chunk_size < fixed_batch_size:
                padding_needed = fixed_batch_size - chunk_size
                dummy_frame = chunk_frames[0]
                chunk_frames = list(chunk_frames) + [dummy_frame] * padding_needed
                logger.debug(
                    f"Padded final chunk from {chunk_size} to {fixed_batch_size} for {fixed_backend}"
                )
            try:
                predict_kwargs = dict(
                    source=chunk_frames,
                    conf=raw_conf_floor,
                    iou=1.0,  # Always use custom OBB IOU filtering after inference
                    classes=target_classes,
                    max_det=max_det,
                    verbose=False,
                )
                if obb_predict_device is not None:
                    predict_kwargs["device"] = obb_predict_device
                if self.use_onnx and self.onnx_imgsz:
                    predict_kwargs["imgsz"] = int(self.onnx_imgsz)
                chunk_results = self._predict_with_coreml_fallback(
                    self.model,
                    predict_kwargs,
                    context="batched OBB inference",
                )
                # Only keep results for actual frames (not padding)
                all_results.extend(chunk_results[:chunk_size])
            except Exception as e:
                logger.error(f"YOLO batched inference failed on chunk: {e}")
                # Return empty results for this chunk
                all_results.extend([None] * chunk_size)
        return all_results

    def _onnx_per_frame_fallback(
        self,
        frames,
        start_frame_idx,
        target_classes,
        raw_conf_floor,
        max_det,
        obb_predict_device,
    ):
        results_batch = []
        for idx, frame in enumerate(frames):
            try:
                single_kwargs = dict(
                    source=frame,
                    conf=raw_conf_floor,
                    iou=1.0,
                    classes=target_classes,
                    max_det=max_det,
                    verbose=False,
                )
                if obb_predict_device is not None:
                    single_kwargs["device"] = obb_predict_device
                if self.onnx_imgsz:
                    single_kwargs["imgsz"] = int(self.onnx_imgsz)
                    single_results = self._predict_with_coreml_fallback(
                        self.model,
                        single_kwargs,
                        context="single-frame OBB fallback inference",
                    )
                results_batch.append(
                    single_results[0] if len(single_results) > 0 else None
                )
            except Exception as frame_err:
                logger.error(
                    "YOLO ONNX single-frame fallback failed at batch frame %d: %s",
                    start_frame_idx + idx,
                    frame_err,
                )
                results_batch.append(None)
        return results_batch

    def _run_standard_obb_batch_inference(
        self,
        frames,
        start_frame_idx,
        target_classes,
        raw_conf_floor,
        max_det,
    ):
        obb_predict_device = getattr(self, "obb_predict_device", None) or self.device
        try:
            predict_kwargs = dict(
                source=frames,
                conf=raw_conf_floor,
                iou=1.0,  # Always use custom OBB IOU filtering after inference
                classes=target_classes,
                max_det=max_det,
                verbose=False,
            )
            if obb_predict_device is not None:
                predict_kwargs["device"] = obb_predict_device
            if self.use_onnx and self.onnx_imgsz:
                predict_kwargs["imgsz"] = int(self.onnx_imgsz)
            return self._predict_with_coreml_fallback(
                self.model,
                predict_kwargs,
                context="standard batched OBB inference",
            )
        except Exception as e:
            logger.error(f"YOLO batched inference failed: {e}")
            if not self.use_onnx:
                return None
            logger.warning(
                "ONNX batched inference unavailable, falling back to per-frame ONNX inference."
            )
            return self._onnx_per_frame_fallback(
                frames,
                start_frame_idx,
                target_classes,
                raw_conf_floor,
                max_det,
                obb_predict_device,
            )

    def _extract_per_frame_raw(self, results_batch, actual_frame_count):
        per_frame_raw = []
        for idx in range(actual_frame_count):
            results = results_batch[idx]
            if results is None or results.obb is None or len(results.obb) == 0:
                per_frame_raw.append(None)
            else:
                per_frame_raw.append(self._extract_raw_detections(results.obb))
        return per_frame_raw

    def _assemble_batched_frame_result(
        self,
        raw,
        headtail_per_frame,
        idx,
        return_raw,
    ):
        if raw is None:
            return ([], [], [], [], [])
        raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners = raw
        if headtail_per_frame is not None:
            (
                raw_heading_hints,
                raw_heading_confidences,
                raw_directed_mask,
                _raw_affines,
            ) = headtail_per_frame[idx]
        else:
            raw_heading_hints = [float("nan")] * len(raw_meas)
            raw_heading_confidences = [0.0] * len(raw_meas)
            raw_directed_mask = [0] * len(raw_meas)
            _raw_affines = None
        if return_raw:
            return (
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confidences,
                raw_obb_corners,
                raw_heading_hints,
                raw_heading_confidences,
                raw_directed_mask,
                _raw_affines,
            )
        (
            meas,
            sizes,
            shapes,
            confidences,
            obb_corners_list,
            _,
            _heading_hints,
            _heading_confidences,
            _directed_mask,
        ) = self.filter_raw_detections(
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            roi_mask=None,
            detection_ids=None,
            heading_hints=raw_heading_hints,
            heading_confidences=raw_heading_confidences,
            directed_mask=raw_directed_mask,
        )
        return (meas, sizes, shapes, confidences, obb_corners_list)

    def detect_objects_batched(
        self: object,
        frames: object,
        start_frame_idx: object,
        progress_callback: object = None,
        return_raw: bool = False,
        profiler: object = None,
    ) -> object:
        """
        Detect objects in a batch of frames using YOLO OBB.

        Args:
            frames: List of frames (numpy arrays)
            start_frame_idx: Starting frame index for this batch
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            List of tuples per frame:
              - return_raw=False: (meas, sizes, shapes, confidences, obb_corners)
              - return_raw=True:  (
                    raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners,
                  raw_heading_hints, raw_heading_confidences,
                  raw_directed_mask, raw_canonical_affines
                )
        """
        if self.model is None:
            logger.error("YOLO model not initialized")
            return [([], [], [], [], []) for _ in frames]

        p = self.params
        target_classes = p.get("YOLO_TARGET_CLASSES", None)
        raw_conf_floor = max(1e-4, float(p.get("RAW_YOLO_CONFIDENCE_FLOOR", 1e-3)))
        max_det = self._raw_detection_cap()

        # Sequential mode requires per-frame OBB processing because each frame
        # generates variable crop counts for the stage-2 OBB model.
        if self._current_obb_mode() == "sequential":
            return self._batched_sequential_mode(
                frames, start_frame_idx, return_raw, progress_callback, profiler
            )

        # -------------------------------------------------------------------
        # Direct OBB mode: batch the OBB inference, then run head-tail
        # classification per-frame afterwards.  This avoids the critical
        # performance pitfall where TensorRT/ONNX with fixed batch dims would
        # pad every single-frame call (e.g. 1 frame → 16 copies).
        # -------------------------------------------------------------------

        actual_frame_count = len(frames)
        fixed_batch_size, fixed_backend = self._resolve_fixed_batch_params()

        if profiler is not None:
            profiler.phase_start("yolo_obb_inference")

        if fixed_batch_size is not None:
            results_batch = self._run_fixed_batch_obb_inference(
                frames,
                actual_frame_count,
                fixed_batch_size,
                fixed_backend,
                target_classes,
                raw_conf_floor,
                max_det,
            )
        else:
            # Standard PyTorch inference - no chunking needed
            results_batch = self._run_standard_obb_batch_inference(
                frames, start_frame_idx, target_classes, raw_conf_floor, max_det
            )
            if results_batch is None:
                return [([], [], [], [], []) for _ in frames]

        if profiler is not None:
            profiler.phase_end("yolo_obb_inference")

        # ===================================================================
        # Post-process: extract raw detections, cross-frame head-tail, assemble
        # ===================================================================

        # Phase 1 — extract raw detections from each frame's OBB result
        per_frame_raw = self._extract_per_frame_raw(results_batch, actual_frame_count)

        # Phase 2 — cross-frame head-tail classification (single GPU call
        # batching canonical crops from ALL frames together).
        if self._headtail_analyzer is not None and self._headtail_analyzer.is_available:
            include_canonical_affines = bool(
                return_raw and self._should_compute_canonical_affines()
            )
            per_frame_corners = [
                raw[4] if raw is not None else [] for raw in per_frame_raw
            ]
            headtail_per_frame = self._compute_headtail_hints_cross_frame(
                frames[:actual_frame_count],
                per_frame_corners,
                include_canonical_affines=include_canonical_affines,
                profiler=profiler,
            )
        else:
            headtail_per_frame = None

        # Phase 3 — assemble final batch detections
        batch_detections = []
        for idx in range(actual_frame_count):
            batch_detections.append(
                self._assemble_batched_frame_result(
                    per_frame_raw[idx], headtail_per_frame, idx, return_raw
                )
            )
            if progress_callback and (idx + 1) % 10 == 0:
                progress_callback(
                    idx + 1,
                    actual_frame_count,
                    f"Processing batch frame {idx + 1}/{actual_frame_count}",
                )

        return batch_detections

    def apply_conservative_split(self, fg_mask, gray=None, background=None):
        """
        Placeholder method for compatibility with ObjectDetector interface.
        YOLO doesn't use foreground masks, so this is a no-op.
        """
        return fg_mask
