"""Quick Test dialog: run a trained YOLO model on sample images and display results."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)

# Maximum number of sample images to test
_MAX_SAMPLES = 8

# OBB drawing colour (green) and thickness
_OBB_COLOR = (0, 255, 0)
_OBB_THICKNESS = 2


#: Confidence / IOU thresholds used for the quick-test's own NMS pass.
#: These are intentionally fixed (not user-configurable) -- this dialog's
#: purpose is a fast visual sanity check, not a tunable inference run.
_TEST_CONFIDENCE_THRESHOLD = 0.25
_TEST_IOU_THRESHOLD = 0.45

#: Max detections fed to ``load_obb_executor``'s NMS. Matches the legacy
#: ``YOLOOBBDetector`` path's hardcoded ``YOLO_MAX_TARGETS`` (100) -- this is
#: a visual sanity-check tool for a multi-animal tracker, and
#: ``load_obb_executor``'s own default (20) is too low for busy scenes and
#: would silently hide real detections.
_TEST_MAX_DET = 100


def training_device_to_compute_runtime(device: str) -> str:
    """Map a literal training-device string onto a ``load_obb_executor`` compute runtime.

    Both callers of this dialog (``trackerkit``'s ``train_yolo_dialog`` and
    ``detectkit``'s project-level training config) expose a free-text training
    device selector (fed straight into YOLO's own ``device=`` training
    parameter) -- a genuinely different, simpler concept than the app's tiered
    ``compute_runtime`` system (``cpu``/``gpu``/``gpu_fast`` resolved via
    the runtime-tier resolver). Quick Test runs a freshly trained model that
    has no exported TensorRT/CoreML artifact yet, so gpu_fast-style tier
    resolution doesn't apply here -- this is a literal device sanity check,
    not a tier selection. Map the literal device string directly onto the
    closest ``load_obb_executor``-accepted runtime instead of routing it
    through the tier resolver.

    ``"auto"`` (the training-device combo's default, first entry) is not a
    literal device -- it means "pick the best available device", mirroring
    the legacy ``YOLOOBBDetector._detect_device`` priority (CUDA > MPS > CPU),
    so it is resolved here via the same centralized availability flags
    instead of regressing to a hardcoded CPU runtime.
    """
    dev = str(device or "").strip().lower()
    if dev.startswith("cuda"):
        return "cuda"
    if dev == "mps":
        return "mps"
    if dev == "auto" or not dev:
        from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE

        if TORCH_CUDA_AVAILABLE:
            return "cuda"
        if MPS_AVAILABLE:
            return "mps"
        return "cpu"
    return "cpu"


def build_test_params(
    model_path: str,
    role: str,
    compute_runtime: str,
    imgsz: int,
    crop_pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
    detect_model_path: str = "",
) -> dict:
    """Build a minimal parameter dict for ``load_obb_executor``-based inference.

    Parameters
    ----------
    model_path:
        Path to the trained ``.pt`` model weights under test.
    role:
        One of ``"obb_direct"``, ``"seq_detect"``, ``"seq_crop_obb"``.
    compute_runtime:
        Compute runtime string accepted by
        ``core.inference.runtime_artifacts.load_obb_executor``
        (``"cpu"``, ``"cuda"``, ``"mps"``, ...).
    imgsz:
        Inference image size (used as ``imgsz_override`` for
        ``load_obb_executor``, and as the stage-2 crop size for
        ``"seq_crop_obb"``).
    crop_pad_ratio:
        Crop padding ratio for sequential crop-OBB mode.
    min_crop_size_px:
        Minimum crop size in pixels for sequential mode.
    enforce_square:
        Whether to enforce square crops in sequential mode.
    detect_model_path:
        Stage-1 detection model used to build crops when testing a
        sequential crop-OBB model (``role == "seq_crop_obb"``). Ignored for
        other roles.

    Returns
    -------
    dict
        ``{"model_path", "compute_runtime", "imgsz", "task"}`` plus, for
        ``role == "seq_crop_obb"``, ``"detect_model_path"`` and the crop
        parameters above.
    """
    task = "detect" if role == "seq_detect" else "obb"
    params: dict = {
        "model_path": model_path,
        "compute_runtime": compute_runtime,
        "imgsz": imgsz,
        "task": task,
    }

    if role == "seq_crop_obb":
        params["detect_model_path"] = detect_model_path or model_path
        params["crop_pad_ratio"] = crop_pad_ratio
        params["min_crop_size_px"] = min_crop_size_px
        params["enforce_square"] = enforce_square

    return params


class _SeqCropSpec:
    """Minimal stand-in for ``OBBSequentialConfig`` fields ``build_crops`` reads."""

    def __init__(self, crop_pad_ratio, min_crop_size_px, enforce_square_crop):
        self.crop_pad_ratio = crop_pad_ratio
        self.min_crop_size_px = min_crop_size_px
        self.enforce_square_crop = enforce_square_crop


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class _TestWorker(BaseWorker):
    """Runs YOLO inference on sample images in a background thread."""

    image_ready = Signal(np.ndarray)  # annotated BGR frame
    finished_all = Signal()

    def __init__(self, params: dict, image_paths: list[str]) -> None:
        super().__init__()
        self.params = params
        self.image_paths = image_paths

    def execute(self):
        """Load production OBB executor(s) and run inference on each sample image.

        Uses ``load_obb_executor`` (the same production factory the
        preview-detection path uses) instead of the legacy
        ``YOLOOBBDetector``. Geometry is extracted via the shared
        ``core/inference/stages/obb.py`` helpers so this dialog draws exactly
        the OBB quads the production pipeline would produce.
        """
        from hydra_suite.core.inference.runtime_artifacts import load_obb_executor

        self.status.emit("Loading model...")
        task = self.params.get("task", "obb")
        compute_runtime = self.params.get("compute_runtime", "cpu")
        imgsz = self.params.get("imgsz") or None

        executor = load_obb_executor(
            self.params["model_path"],
            compute_runtime,
            task=task,
            imgsz_override=imgsz,
            max_det=_TEST_MAX_DET,
        )

        detect_model_path = self.params.get("detect_model_path")
        detect_executor = None
        if detect_model_path:
            detect_executor = load_obb_executor(
                detect_model_path,
                compute_runtime,
                task="detect",
                max_det=_TEST_MAX_DET,
            )

        crop_spec = _SeqCropSpec(
            crop_pad_ratio=float(self.params.get("crop_pad_ratio", 0.15)),
            min_crop_size_px=float(self.params.get("min_crop_size_px", 64)),
            enforce_square_crop=bool(self.params.get("enforce_square", True)),
        )

        for idx, img_path in enumerate(self.image_paths):
            self.status.emit(
                f"Running inference on image {idx + 1}/{len(self.image_paths)}..."
            )
            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning("Could not read image: %s", img_path)
                continue

            if detect_executor is not None:
                corners = self._run_sequential(
                    frame, idx, detect_executor, executor, crop_spec, imgsz
                )
            elif task == "detect":
                corners = self._run_detect_only(frame, executor)
            else:
                corners = self._run_direct(frame, idx, executor)

            annotated = frame.copy()
            for quad in corners:
                pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated,
                    [pts],
                    isClosed=True,
                    color=_OBB_COLOR,
                    thickness=_OBB_THICKNESS,
                )

            self.image_ready.emit(annotated)

        self.finished_all.emit()

    @staticmethod
    def _run_direct(frame: np.ndarray, idx: int, executor) -> list:
        """Direct-mode OBB inference: run the executor and extract quads."""
        from hydra_suite.core.inference.stages.obb import extract_obb_result

        results = executor.predict(
            [frame],
            conf=_TEST_CONFIDENCE_THRESHOLD,
            iou=_TEST_IOU_THRESHOLD,
            verbose=False,
        )
        if not results:
            return []
        result = extract_obb_result(results[0], idx)
        return list(result.corners)

    @staticmethod
    def _run_detect_only(frame: np.ndarray, executor) -> list:
        """Stage-1-detect-only test: draw axis-aligned boxes as quads."""
        results = executor.predict(
            [frame],
            conf=_TEST_CONFIDENCE_THRESHOLD,
            iou=_TEST_IOU_THRESHOLD,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        quads = []
        for x1, y1, x2, y2 in boxes.xyxy.cpu().numpy():
            quads.append(
                np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            )
        return quads

    @staticmethod
    def _run_sequential(
        frame: np.ndarray,
        idx: int,
        detect_executor,
        obb_executor,
        crop_spec: "_SeqCropSpec",
        stage2_imgsz,
    ) -> list:
        """Sequential detect-then-crop-OBB test: build crops off stage-1 boxes."""
        from hydra_suite.core.inference.stages.obb import (
            build_crops,
            extract_obb_result,
            merge_obb_results,
            resize_crops_for_stage2,
        )

        detect_results = detect_executor.predict(
            [frame],
            conf=_TEST_CONFIDENCE_THRESHOLD,
            iou=_TEST_IOU_THRESHOLD,
            verbose=False,
        )
        if not detect_results:
            return []
        boxes = getattr(detect_results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        crops, offsets = build_crops(frame, boxes, crop_spec, None)
        if not crops:
            return []

        orig_sizes = [(c.shape[1], c.shape[0]) for c in crops]
        stage2_size = int(stage2_imgsz) if stage2_imgsz else 0
        crops_for_stage2 = (
            resize_crops_for_stage2(crops, stage2_size) if stage2_size > 0 else crops
        )

        obb_results = obb_executor.predict(
            crops_for_stage2,
            conf=_TEST_CONFIDENCE_THRESHOLD,
            iou=_TEST_IOU_THRESHOLD,
            verbose=False,
        )

        sub = []
        for i, r in enumerate(obb_results):
            orig_w, orig_h = orig_sizes[i]
            scale = (
                (orig_w / stage2_size, orig_h / stage2_size)
                if stage2_size > 0
                else (1.0, 1.0)
            )
            sub.append(extract_obb_result(r, idx, offset=offsets[i], scale=scale))

        merged = merge_obb_results(idx, sub)
        return list(merged.corners)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


def _collect_sample_images(
    dataset_dir: str, max_count: int = _MAX_SAMPLES
) -> list[str]:
    """Collect sample images from a YOLO-format dataset directory.

    Preference order: ``val/images`` > ``train/images`` > any ``images/`` subdir
    > top-level image files.
    """
    root = Path(dataset_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def _image_files(d: Path) -> list[str]:
        if not d.is_dir():
            return []
        return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in exts)[
            :max_count
        ]

    # Try val/images first, then train/images
    for split in ("val", "valid", "test", "train"):
        imgs = _image_files(root / split / "images")
        if imgs:
            return imgs

    # Try top-level images/ dir
    imgs = _image_files(root / "images")
    if imgs:
        return imgs

    # Fall back to any images in the root
    imgs = _image_files(root)
    return imgs


class ModelTestDialog(BaseDialog):
    """Dialog that runs a trained YOLO model on sample dataset images and displays
    annotated results so the user can visually verify detection quality."""

    def __init__(
        self,
        model_path: str,
        role: str,
        dataset_dir: str,
        compute_runtime: str = "cpu",
        imgsz: int = 640,
        crop_pad_ratio: float = 0.15,
        min_crop_size_px: int = 64,
        enforce_square: bool = True,
        detect_model_path: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(
            title="Quick Model Test",
            parent=parent,
            buttons=QDialogButtonBox.Close,
            apply_dark_style=False,
        )
        self.setMinimumSize(800, 500)
        self.resize(1000, 600)

        self._model_path = model_path
        self._role = role
        self._dataset_dir = dataset_dir
        self._compute_runtime = compute_runtime
        self._imgsz = imgsz
        self._crop_pad_ratio = crop_pad_ratio
        self._min_crop_size_px = min_crop_size_px
        self._enforce_square = enforce_square
        self._detect_model_path = detect_model_path

        self._worker: _TestWorker | None = None
        self._build_ui()
        self._start_test()

    # ---- UI ----

    def _build_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)

        self.status_label = QLabel("Collecting sample images...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        layout.addWidget(self.progress_bar)

        # Scrollable horizontal image strip
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.image_container = QWidget()
        self.image_layout = QHBoxLayout(self.image_container)
        self.image_layout.setContentsMargins(4, 4, 4, 4)
        self.image_layout.setSpacing(8)
        self.image_layout.addStretch()
        self.scroll_area.setWidget(self.image_container)
        layout.addWidget(self.scroll_area, stretch=1)

        self.add_content(container)

    # ---- Run ----

    def _start_test(self):
        images = _collect_sample_images(self._dataset_dir)
        if not images:
            self.status_label.setText("No sample images found in dataset directory.")
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
            return

        params = build_test_params(
            model_path=self._model_path,
            role=self._role,
            compute_runtime=self._compute_runtime,
            imgsz=self._imgsz,
            crop_pad_ratio=self._crop_pad_ratio,
            min_crop_size_px=self._min_crop_size_px,
            enforce_square=self._enforce_square,
            detect_model_path=self._detect_model_path,
        )

        self._worker = _TestWorker(params, images)
        self._worker.status.connect(self._on_status)
        self._worker.image_ready.connect(self._on_image_ready)
        self._worker.error.connect(self._on_error)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.start()

    # ---- Slots ----

    def _on_status(self, text: str):
        self.status_label.setText(text)

    def _on_image_ready(self, frame: np.ndarray):
        """Convert a BGR numpy frame to QPixmap and add to the horizontal strip."""
        if frame.ndim == 2:
            h, w = frame.shape
            qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        # Scale to a reasonable display height while keeping aspect ratio
        display_height = 400
        scaled = pixmap.scaledToHeight(display_height, Qt.SmoothTransformation)

        label = QLabel()
        label.setPixmap(scaled)
        label.setAlignment(Qt.AlignCenter)
        # Insert before the stretch item
        count = self.image_layout.count()
        self.image_layout.insertWidget(count - 1, label)

    def _on_error(self, message: str):
        self.status_label.setText(f"Error: {message}")

    def _on_finished(self):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        current = self.status_label.text()
        if not current.startswith("Error"):
            self.status_label.setText("Inference complete.")

    def closeEvent(self, event) -> None:
        """Terminate any running inference worker before closing the dialog."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(3000)
        super().closeEvent(event)
