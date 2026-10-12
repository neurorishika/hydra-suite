"""Smart Select options: what to pick from, the batch mix and the embeddings.

Builds the controls and owns their settings round-trip; the dialog reads the
values through :meth:`SmartSelectOptionsPanel.batch_config` and friends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.selection.clustering import auto_cluster_count
from hydra_suite.posekit.core.selection import PoseBatchConfig

EMBEDDING_MODELS = [
    # DINO v2 models (recommended for general image understanding)
    "timm/vit_base_patch14_dinov2.lvd142m",
    "timm/vit_small_patch14_dinov2.lvd142m",
    "timm/vit_large_patch14_dinov2.lvd142m",
    "timm/vit_giant_patch14_dinov2.lvd142m",
    # CLIP models
    "timm/vit_base_patch32_clip_224.openai",
    "timm/vit_bigG_14_clip_224.laion400M_e32",
    "timm/convnext_base_w_clip.laion2b_s29B_b131k_ft_in1k",
    # ResNet models
    "timm/resnet50.a1_in1k",
    "timm/resnet18.a1_in1k",
    "timm/resnet101.a1_in1k",
    # EfficientNet models
    "timm/efficientnet_b0.ra_in1k",
    "timm/efficientnet_b3.ra2_in1k",
    "timm/efficientnet_b5.sw_in12k_ft_in1k",
    # MobileNet models
    "timm/mobilenetv3_small_100.lamb_in1k",
    "timm/mobilenetv3_large_100.ra_in1k",
    # ConvNeXt models
    "timm/convnext_tiny.fb_in1k",
    "timm/convnext_base.fb_in1k",
]

CLUSTER_METHODS = [
    ("K-means", "kmeans"),
    ("MiniBatch K-means", "minibatch"),
    ("HDBSCAN (slow above ~10,000 frames)", "hdbscan"),
]

SCOPES = ["Unlabeled only", "All frames", "Labeling set"]

# (attribute, label, default %, tooltip)
MIX_FIELDS = [
    (
        "error_spin",
        "Near model errors",
        20,
        "Unlabeled frames that look like labeled frames the model gets wrong "
        "(pose error above the threshold). Needs model predictions.",
    ),
    (
        "uncertainty_spin",
        "Model unsure",
        25,
        "Frames where the model's keypoint confidence is lowest. Needs model "
        "predictions.",
    ),
    (
        "rare_spin",
        "Rare clusters",
        10,
        "Frames from small or barely-labeled clusters, sampled with weight "
        "size^-alpha.",
    ),
    (
        "diversity_spin",
        "Fill gaps",
        30,
        "Frames in the biggest uncovered regions: each pick most reduces the "
        "distance from unlabeled frames to their nearest labeled one, so the "
        "labeled set covers the whole dataset.",
    ),
    (
        "representative_spin",
        "Typical",
        10,
        "Frames from dense clusters that have few labels.",
    ),
]

# Widgets the dialog exposes under the same names.
WIDGET_NAMES = [
    "cb_scope",
    "cb_exclude_in_labeling",
    "n_spin",
    "k_spin",
    "cb_auto_k",
    "cluster_method_combo",
    *[attr for attr, *_ in MIX_FIELDS],
    "remainder_label",
    "alpha_spin",
    "error_thr_spin",
    "cb_dedup",
    "dedup_spin",
    "model_label",
    "model_combo",
    "dev_combo",
    "batch_spin",
    "max_side_spin",
    "cb_use_enhance",
]


class SmartSelectOptionsPanel(QWidget):
    """Batch, Mix and Embeddings controls of the Smart Select dialog."""

    def __init__(
        self,
        unit_count: int,
        frame_mode: bool,
        enhance_default: bool,
        devices: List[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._unit_count = int(unit_count)
        self._frame_mode = bool(frame_mode)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        layout.addLayout(top)
        top.addWidget(self._build_batch_box(), 3)
        top.addWidget(self._build_mix_box(), 2)

        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        layout.addWidget(self._build_embedding_box(enhance_default, devices))

        self._on_auto_k_toggled(self.cb_auto_k.isChecked())
        self._update_remainder()

    # ── construction ─────────────────────────────────────────────────────

    def _build_batch_box(self) -> QGroupBox:
        box = QGroupBox("Batch")
        form = QFormLayout(box)

        scope_row = QHBoxLayout()
        self.cb_scope = QComboBox()
        self.cb_scope.addItems(SCOPES)
        self.cb_scope.setToolTip(
            "Which frames may be picked:\n"
            "• Unlabeled only: frames without a saved label\n"
            "• All frames: any frame (labeled ones are re-checked)\n"
            "• Labeling set: reorder the frames already in the labeling set"
        )
        scope_row.addWidget(self.cb_scope, 1)
        self.cb_exclude_in_labeling = QCheckBox("Skip frames in labeling set")
        self.cb_exclude_in_labeling.setChecked(True)
        self.cb_exclude_in_labeling.setToolTip(
            "Don't pick frames that are already in the labeling set."
        )
        scope_row.addWidget(self.cb_exclude_in_labeling)
        form.addRow("Pick from", scope_row)

        size_row = QHBoxLayout()
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 50000)
        self.n_spin.setValue(50)
        self.n_spin.setToolTip(
            "Number of frames in the batch"
            + (" (source frames in Frame Mode)." if self._frame_mode else ".")
        )
        size_row.addWidget(self.n_spin)
        size_row.addSpacing(12)
        size_row.addWidget(QLabel("Clusters"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 5000)
        self.k_spin.setValue(50)
        size_row.addWidget(self.k_spin)
        self.cb_auto_k = QCheckBox("Auto")
        self.cb_auto_k.setChecked(True)
        self.cb_auto_k.setToolTip(
            "Use about √N clusters (N = frames), like ClassKit's over-clustering."
        )
        self.cb_auto_k.toggled.connect(self._on_auto_k_toggled)
        size_row.addWidget(self.cb_auto_k)
        size_row.addSpacing(12)
        self.cluster_method_combo = QComboBox()
        for text, value in CLUSTER_METHODS:
            self.cluster_method_combo.addItem(text, value)
        self.cluster_method_combo.setToolTip(
            "K-means is deterministic and recommended (MiniBatch is used "
            "automatically above 5,000 frames). HDBSCAN finds dense groups on "
            "its own: the cluster count then only sets its minimum cluster "
            "size (frames / clusters), and it slows down sharply above "
            "~10,000 frames."
        )
        size_row.addWidget(self.cluster_method_combo)
        size_row.addStretch(1)
        form.addRow("Frames", size_row)
        return box

    def _build_mix_box(self) -> QGroupBox:
        box = QGroupBox("Mix")
        mix = QGridLayout(box)
        for row, (attr, label, default, tip) in enumerate(MIX_FIELDS):
            lbl = QLabel(label)
            lbl.setToolTip(tip)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix(" %")
            spin.setValue(default)
            spin.setToolTip(tip)
            spin.valueChanged.connect(self._update_remainder)
            setattr(self, attr, spin)
            mix.addWidget(lbl, row, 0)
            mix.addWidget(spin, row, 1)
        self.remainder_label = QLabel()
        self.remainder_label.setToolTip(
            "Whatever the mix leaves goes to random spot checks."
        )
        mix.addWidget(self.remainder_label, len(MIX_FIELDS), 0, 1, 2)

        mix.addWidget(QLabel("Rarity α"), 0, 2)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.75)
        self.alpha_spin.setToolTip(
            "0 = clusters in proportion to their size, 1 = every cluster equal."
        )
        mix.addWidget(self.alpha_spin, 0, 3)
        mix.addWidget(QLabel("Error above"), 1, 2)
        self.error_thr_spin = QDoubleSpinBox()
        self.error_thr_spin.setRange(0.0, 1.0)
        self.error_thr_spin.setDecimals(3)
        self.error_thr_spin.setSingleStep(0.01)
        self.error_thr_spin.setValue(0.05)
        self.error_thr_spin.setToolTip(
            "A labeled frame counts as a model error when its mean keypoint "
            "error, divided by the object size, is above this (0.05 = the "
            "Evaluation dialog's PCK threshold)."
        )
        mix.addWidget(self.error_thr_spin, 1, 3)
        self.cb_dedup = QCheckBox("Skip near-duplicates")
        self.cb_dedup.setToolTip(
            "Never put two frames this similar (cosine) in the same batch."
        )
        mix.addWidget(self.cb_dedup, 2, 2)
        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.5, 1.0)
        self.dedup_spin.setSingleStep(0.01)
        self.dedup_spin.setValue(0.97)
        self.dedup_spin.setEnabled(False)
        self.cb_dedup.toggled.connect(self.dedup_spin.setEnabled)
        mix.addWidget(self.dedup_spin, 2, 3)
        return box

    def _build_embedding_box(self, enhance_default: bool, devices) -> QGroupBox:
        box = QGroupBox("Embeddings")
        form = QFormLayout(box)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(EMBEDDING_MODELS)
        form.addRow("Model", self.model_combo)
        dev_row = QHBoxLayout()
        self.dev_combo = QComboBox()
        self.dev_combo.addItems(list(devices))
        dev_row.addWidget(self.dev_combo)
        dev_row.addWidget(QLabel("Batch size"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        dev_row.addWidget(self.batch_spin)
        dev_row.addWidget(QLabel("Max side"))
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(0, 2048)
        self.max_side_spin.setValue(512)
        self.max_side_spin.setToolTip("Downscale images to this size first (0 = off).")
        dev_row.addWidget(self.max_side_spin)
        self.cb_use_enhance = QCheckBox("Enhance (CLAHE + unsharp)")
        self.cb_use_enhance.setChecked(bool(enhance_default))
        dev_row.addWidget(self.cb_use_enhance)
        dev_row.addStretch(1)
        form.addRow("Device", dev_row)
        return box

    # ── behaviour ────────────────────────────────────────────────────────

    def _on_auto_k_toggled(self, on: bool) -> None:
        """Auto shows the cluster count it will use (≈ √N, clamped)."""
        self.k_spin.setEnabled(not on)
        if on:
            self.k_spin.setValue(
                max(self.k_spin.minimum(), auto_cluster_count(self._unit_count))
            )

    def mix_total(self) -> int:
        return sum(getattr(self, attr).value() for attr, *_ in MIX_FIELDS)

    def _update_remainder(self) -> None:
        total = self.mix_total()
        if total > 100:
            self.remainder_label.setText(
                f"Mix adds up to {total}% — scaled down to fit; no random checks."
            )
        else:
            self.remainder_label.setText(f"Random checks: {100 - total}%")

    def show_prediction_source(self, source, reason: str, image_paths) -> None:
        """Explain the model signals and enable their controls when present."""
        has_preds = bool(source and source.predictions)
        for attr in ("error_spin", "uncertainty_spin", "error_thr_spin"):
            getattr(self, attr).setEnabled(has_preds)
        if not has_preds:
            why = reason or "No cached pose predictions."
            self.model_label.setText(
                f"<b>Model signals off.</b> {why} Run <i>Predict Dataset…</i> "
                "with a pose model to add error and uncertainty picks; their "
                "share goes to <i>Fill gaps</i> until then."
            )
            return
        preds = source.predictions
        n_cached = sum(
            1 for p in image_paths if str(p) in preds or str(Path(p).resolve()) in preds
        )
        n_all = len(image_paths)
        self.model_label.setText(
            f"<b>Model signals:</b> {source.label} — predictions cached for "
            f"{n_cached:,} of {n_all:,} frames."
            + (
                ""
                if n_cached == n_all
                else " Frames without a prediction can still be picked by "
                "the other slots."
            )
        )

    # ── values ───────────────────────────────────────────────────────────

    def embedding_key(self) -> tuple:
        """What the cached embeddings depend on (not device or batch size)."""
        return (
            self.model_combo.currentText().strip(),
            bool(self.cb_use_enhance.isChecked()),
            int(self.max_side_spin.value()),
        )

    def n_clusters(self) -> int:
        """0 = auto."""
        return 0 if self.cb_auto_k.isChecked() else int(self.k_spin.value())

    def cluster_method(self) -> str:
        return self.cluster_method_combo.currentData() or "kmeans"

    def batch_config(self) -> PoseBatchConfig:
        # The model slots keep their share even without predictions: with no
        # signal they pick nothing and the core hands their budget to the
        # "Fill gaps" top-up (zeroing them here would turn it into random
        # checks instead).
        return PoseBatchConfig(
            batch_size=int(self.n_spin.value()),
            error_fraction=self.error_spin.value() / 100.0,
            uncertainty_fraction=self.uncertainty_spin.value() / 100.0,
            rare_fraction=self.rare_spin.value() / 100.0,
            rare_alpha=float(self.alpha_spin.value()),
            diversity_fraction=self.diversity_spin.value() / 100.0,
            representative_fraction=self.representative_spin.value() / 100.0,
            error_threshold=float(self.error_thr_spin.value()),
            dedup_threshold=(
                float(self.dedup_spin.value()) if self.cb_dedup.isChecked() else None
            ),
        )

    # ── settings ─────────────────────────────────────────────────────────

    def apply_settings(self, s: Optional[Dict[str, Any]]) -> None:
        """Restore saved values; missing or malformed ones keep the defaults."""
        if not isinstance(s, dict) or not s:
            return

        def num(key, spin, cast=int, src=None):
            src = s if src is None else src
            try:
                spin.setValue(cast(src[key]))
            except (KeyError, TypeError, ValueError, OverflowError):
                pass

        def flag(key, box, default):
            value = s.get(key, default)
            if isinstance(value, bool):
                box.setChecked(value)

        if s.get("scope") in SCOPES:
            self.cb_scope.setCurrentText(s["scope"])
        flag("exclude_labeling", self.cb_exclude_in_labeling, True)
        model = s.get("model")
        if isinstance(model, str) and model.strip():
            self.model_combo.setCurrentText(model)
        devices = [self.dev_combo.itemText(i) for i in range(self.dev_combo.count())]
        if s.get("device") in devices:
            self.dev_combo.setCurrentText(s["device"])
        num("batch", self.batch_spin)
        num("max_side", self.max_side_spin)
        flag("use_enhance", self.cb_use_enhance, self.cb_use_enhance.isChecked())
        num("n", self.n_spin)
        num("k", self.k_spin)
        flag("auto_k", self.cb_auto_k, True)
        i = self.cluster_method_combo.findData(s.get("cluster_method_v2"))
        if i >= 0:
            self.cluster_method_combo.setCurrentIndex(i)
        mix = s.get("mix")
        if isinstance(mix, dict):
            for attr, *_ in MIX_FIELDS:
                num(attr, getattr(self, attr), src=mix)
        num("rare_alpha", self.alpha_spin, float)
        num("error_threshold", self.error_thr_spin, float)
        flag("dedup", self.cb_dedup, False)
        num("dedup_threshold", self.dedup_spin, float)
        self.dedup_spin.setEnabled(self.cb_dedup.isChecked())
        self._on_auto_k_toggled(self.cb_auto_k.isChecked())

    def settings(self) -> Dict[str, Any]:
        return {
            "scope": self.cb_scope.currentText(),
            "exclude_labeling": bool(self.cb_exclude_in_labeling.isChecked()),
            "model": self.model_combo.currentText().strip(),
            "device": self.dev_combo.currentText(),
            "batch": int(self.batch_spin.value()),
            "max_side": int(self.max_side_spin.value()),
            "use_enhance": bool(self.cb_use_enhance.isChecked()),
            "n": int(self.n_spin.value()),
            "k": int(self.k_spin.value()),
            "auto_k": bool(self.cb_auto_k.isChecked()),
            "cluster_method_v2": self.cluster_method_combo.currentData(),
            "mix": {attr: int(getattr(self, attr).value()) for attr, *_ in MIX_FIELDS},
            "rare_alpha": float(self.alpha_spin.value()),
            "error_threshold": float(self.error_thr_spin.value()),
            "dedup": bool(self.cb_dedup.isChecked()),
            "dedup_threshold": float(self.dedup_spin.value()),
        }
