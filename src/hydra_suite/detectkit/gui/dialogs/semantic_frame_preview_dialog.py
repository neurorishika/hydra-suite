"""Visual result of a complete-frame SAM3 escalation check."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.detectkit.gui.canvas import OBBCanvas
from hydra_suite.widgets.dialogs import BaseDialog


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60.0:.1f} h"


class SemanticFramePreviewDialog(BaseDialog):
    """Zoomable predictions and optional ground truth for one test frame."""

    def __init__(
        self,
        result,
        *,
        selected_frames: int,
        project_frames: int,
        parent=None,
    ) -> None:
        super().__init__(
            "SAM3 complete-frame check",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self.result = result
        self.setMinimumSize(820, 600)
        self.resize(1100, 760)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        predictions = len(result.predictions)
        tiling = (
            "full-frame model input"
            if result.tile_px is None
            else f"{result.tiles_per_frame} tile(s), {result.tile_px} px each"
        )
        summary = QLabel(
            f"Random test image: {result.image_path.name} from "
            f"{result.source_name}. SAM3 found {predictions} segment(s) in "
            f"{result.seconds:.1f} s using {tiling}."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        estimates: list[str] = []
        if selected_frames:
            estimates.append(
                f"Selected run ({selected_frames} images): approximately "
                f"{_duration(result.seconds * selected_frames)}"
            )
        if project_frames and project_frames != selected_frames:
            estimates.append(
                f"Entire project ({project_frames} images): approximately "
                f"{_duration(result.seconds * project_frames)}"
            )
        estimate = QLabel(" · ".join(estimates))
        estimate.setWordWrap(True)
        layout.addWidget(estimate)

        controls = QHBoxLayout()
        self._show_gt = QCheckBox("Ground truth")
        self._show_gt.setChecked(True)
        self._show_gt.setEnabled(bool(result.ground_truth))
        self._show_gt.toggled.connect(self._refresh_visibility)
        controls.addWidget(self._show_gt)
        self._show_predictions = QCheckBox("Predictions")
        self._show_predictions.setChecked(True)
        self._show_predictions.toggled.connect(self._refresh_visibility)
        controls.addWidget(self._show_predictions)
        controls.addStretch(1)
        fit = QPushButton("Fit image")
        fit.clicked.connect(self._fit_image)
        controls.addWidget(fit)
        layout.addLayout(controls)

        self._canvas = OBBCanvas()
        self._canvas.setObjectName("semanticFramePreviewCanvas")
        layout.addWidget(self._canvas, 1)

        hint = QLabel(
            "Green solid fill = existing ground truth · Blue dashed fill = "
            "SAM3 prediction. Ctrl+wheel or +/- zooms, drag pans, and "
            "double-click fits. Estimates extrapolate this measured image and "
            "will vary with image content and hardware load. The measured "
            "image time excludes one-time model loading."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.add_content(container)
        self._load_result()

    def _load_result(self) -> None:
        if not self._canvas.load_image(str(self.result.image_path)):
            return
        ground_truth = [
            {"class_id": 0, "polygon_px": item.points.tolist()}
            for item in self.result.ground_truth
        ]
        predictions = [
            {
                "class_id": 2,
                "polygon_px": item.polygon_px.tolist(),
                "confidence": item.confidence,
            }
            for item in self.result.predictions
        ]
        names = {0: "Ground truth", 2: "Prediction"}
        self._canvas.set_gt_detections(ground_truth, names, fill_alpha=65)
        self._canvas.set_pred_detections(predictions, names, fill_alpha=55)
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        self._canvas.set_overlay_visibility(
            self._show_gt.isChecked(), self._show_predictions.isChecked()
        )

    def _fit_image(self) -> None:
        self._canvas.fit_in_view()
