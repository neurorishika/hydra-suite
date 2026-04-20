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


def build_test_params(
    model_path: str,
    role: str,
    device: str,
    imgsz: int,
    crop_pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
    detect_model_path: str = "",
) -> dict:
    """Build a YOLO detector parameter dict suitable for ``YOLOOBBDetector``.

    Parameters
    ----------
    model_path:
        Path to the trained ``.pt`` model weights.
    role:
        One of ``"obb_direct"``, ``"seq_detect"``, ``"seq_crop_obb"``.
    device:
        Compute device string (``"cpu"``, ``"cuda"``, ``"mps"``, ...).
    imgsz:
        Inference image size.
    crop_pad_ratio:
        Crop padding ratio for sequential crop-OBB mode.
    min_crop_size_px:
        Minimum crop size in pixels for sequential mode.
    enforce_square:
        Whether to enforce square crops in sequential mode.
    detect_model_path:
        Optional separate detection model for sequential mode stage 1.
    """
    params: dict = {
        "YOLO_MODEL_PATH": model_path,
        "YOLO_DEVICE": device,
        "YOLO_IMGSZ": imgsz,
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
        "YOLO_MAX_TARGETS": 100,
        "USE_TENSORRT": False,
        "USE_ONNX": False,
    }

    if role in ("seq_detect", "seq_crop_obb"):
        params["YOLO_OBB_MODE"] = "sequential"
        params["YOLO_DETECT_MODEL_PATH"] = detect_model_path or model_path
    else:
        params["YOLO_OBB_MODE"] = "direct"
        params["YOLO_OBB_DIRECT_MODEL_PATH"] = model_path

    if role == "seq_crop_obb":
        params["YOLO_CROP_OBB_MODEL_PATH"] = model_path
        params["YOLO_SEQ_STAGE2_IMGSZ"] = imgsz
        params["YOLO_SEQ_CROP_PAD_RATIO"] = crop_pad_ratio
        params["YOLO_SEQ_MIN_CROP_SIZE_PX"] = min_crop_size_px
        params["YOLO_SEQ_ENFORCE_SQUARE_CROP"] = enforce_square
        if detect_model_path:
            params["YOLO_MODEL_PATH"] = detect_model_path

    return params


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
        """Load the YOLO OBB detector and run inference on each provided image, emitting detections per frame."""
        from hydra_suite.core.detectors import YOLOOBBDetector

        self.status.emit("Loading model...")
        detector = YOLOOBBDetector(self.params)

        for idx, img_path in enumerate(self.image_paths):
            self.status.emit(
                f"Running inference on image {idx + 1}/{len(self.image_paths)}..."
            )
            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning("Could not read image: %s", img_path)
                continue

            result = detector.detect_objects(frame, frame_count=idx, return_raw=True)
            # result when return_raw=True:
            #   (meas, sizes, shapes, yolo_results, confidences,
            #    obb_corners, heading_hints, heading_confidences,
            #    directed_mask, canonical_affines)
            obb_corners = result[5] if len(result) > 5 else []

            annotated = frame.copy()
            for corners in obb_corners:
                pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    annotated,
                    [pts],
                    isClosed=True,
                    color=_OBB_COLOR,
                    thickness=_OBB_THICKNESS,
                )

            self.image_ready.emit(annotated)

        self.finished_all.emit()


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
        device: str = "cpu",
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
        self._device = device
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
            device=self._device,
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
