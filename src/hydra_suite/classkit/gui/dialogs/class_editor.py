"""ClassEditorDialog — polished two-panel labeling class / multi-factor scheme editor."""

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.classkit.config.presets import (
    describe_scheme,
    get_available_scheme_presets,
    save_scheme_preset,
)
from hydra_suite.classkit.gui.dialogs._helpers import _LabelRow
from hydra_suite.widgets.dialogs import (
    HYDRA_DIALOG_MUTED_TEXT_COLOR,
    HYDRA_DIALOG_STYLE,
)

_BTN_ADD = (
    "QPushButton { background-color:#1a4a1a; padding:4px 12px; border-radius:4px; }"
    "QPushButton:hover { background-color:#235a23; }"
)
_BTN_DEL = (
    "QPushButton { background-color:#4a1a1a; padding:4px 12px; border-radius:4px; }"
    "QPushButton:hover { background-color:#6b2424; }"
)
_BTN_NEUTRAL = (
    "QPushButton { background-color:#3a3a3a; color:#d6d6d6; padding:4px 12px; border-radius:4px; }"
    "QPushButton:hover { background-color:#4a4a4a; }"
)


class ClassEditorDialog(QDialog):
    """Polished two-panel labeling class / multi-factor scheme editor."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        scheme_dict: Optional[dict] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit Classes and Labeling Scheme")
        self.resize(820, 540)
        self.setMinimumWidth(680)
        self.setMinimumHeight(400)
        self.setStyleSheet(HYDRA_DIALOG_STYLE)

        self._factors: List[dict] = []
        self._preset_lookup: dict[str, object] = {}

        if scheme_dict and scheme_dict.get("factors"):
            self._factors = [
                {
                    "name": f.get("name", f"factor_{i}"),
                    "labels": list(f.get("labels", [])),
                    "shortcuts": list(f.get("shortcut_keys", [])),
                }
                for i, f in enumerate(scheme_dict["factors"])
            ]
        elif classes:
            self._factors = [
                {"name": "class", "labels": list(classes), "shortcuts": []}
            ]
        else:
            self._factors = [
                {"name": "class", "labels": ["class_1", "class_2"], "shortcuts": []}
            ]

        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        preset_bar = QHBoxLayout()
        lbl_preset = QLabel("Quick preset:")
        lbl_preset.setStyleSheet(
            f"color:{HYDRA_DIALOG_MUTED_TEXT_COLOR}; font-size:11px;"
        )
        preset_bar.addWidget(lbl_preset)
        self._preset_combo = QComboBox()
        self._preset_combo.setMaximumWidth(320)
        self._reload_presets()
        preset_bar.addWidget(self._preset_combo)
        btn_apply_preset = QPushButton("Apply")
        btn_apply_preset.setMaximumWidth(70)
        btn_apply_preset.clicked.connect(self._apply_preset)
        preset_bar.addWidget(btn_apply_preset)
        btn_save_preset = QPushButton("Save As Preset…")
        btn_save_preset.setStyleSheet(_BTN_NEUTRAL)
        btn_save_preset.clicked.connect(self._save_current_as_preset)
        preset_bar.addWidget(btn_save_preset)
        preset_bar.addStretch(1)
        outer.addLayout(preset_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background-color:#3e3e42; }")
        outer.addWidget(splitter, 1)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)

        left_header = QLabel("<b>Factors</b>")
        left_layout.addWidget(left_header)

        self._factor_list = QListWidget()
        self._factor_list.setMinimumWidth(150)
        self._factor_list.setMaximumWidth(220)
        self._factor_list.setStyleSheet(
            "QListWidget { background:#252526; alternate-background-color:#2d2d30; border:1px solid #3e3e42; border-radius:4px; }"
            "QListWidget::item { padding:6px 10px; color:#ffffff; }"
            "QListWidget::item:selected { background-color:#094771; color:#ffffff; }"
        )
        self._factor_list.currentRowChanged.connect(self._on_factor_selected)
        left_layout.addWidget(self._factor_list, 1)

        left_btn_row = QHBoxLayout()
        btn_add_factor = QPushButton("+ Factor")
        btn_add_factor.setStyleSheet(_BTN_ADD)
        btn_add_factor.clicked.connect(self._add_factor)
        btn_add_factor.setToolTip("Add a new factor")
        left_btn_row.addWidget(btn_add_factor)
        self._btn_del_factor = QPushButton("\u2212 Factor")
        self._btn_del_factor.setStyleSheet(_BTN_DEL)
        self._btn_del_factor.clicked.connect(self._remove_selected_factor)
        self._btn_del_factor.setToolTip("Remove selected factor")
        left_btn_row.addWidget(self._btn_del_factor)
        left_layout.addLayout(left_btn_row)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(6)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Factor name:"))
        self._factor_name_edit = QLineEdit()
        self._factor_name_edit.setPlaceholderText("e.g. direction")
        self._factor_name_edit.setMaximumWidth(200)
        self._factor_name_edit.textChanged.connect(self._on_name_edited)
        name_row.addWidget(self._factor_name_edit)
        name_row.addStretch(1)
        right_layout.addLayout(name_row)

        lbl_header = QHBoxLayout()
        lbl_header.addWidget(QLabel("<b>Labels</b>"))
        lbl_header.addWidget(
            QLabel(
                "<i style='color:#ffffff; font-size:11px;'>"
                "  Key: letter, digit, arrow (\u2191\u2193\u2190\u2192), symbol (+  \u2212  Space \u2026)</i>"
            )
        )
        lbl_header.addStretch(1)
        btn_add_label = QPushButton("+ Label")
        btn_add_label.setStyleSheet(_BTN_ADD)
        btn_add_label.setMaximumWidth(80)
        btn_add_label.clicked.connect(self._add_label_row)
        lbl_header.addWidget(btn_add_label)
        right_layout.addLayout(lbl_header)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            "QScrollArea { border:1px solid #3e3e42; border-radius:4px; background:#252526; }"
        )
        self._labels_container = QWidget()
        self._labels_layout = QVBoxLayout(self._labels_container)
        self._labels_layout.setContentsMargins(8, 8, 8, 8)
        self._labels_layout.setSpacing(4)
        self._labels_layout.addStretch(1)
        scroll_area.setWidget(self._labels_container)
        right_layout.addWidget(scroll_area, 1)

        splitter.addWidget(right_widget)
        splitter.setSizes([180, 600])

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._do_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._current_factor_idx: int = -1
        self._label_rows: List[_LabelRow] = []
        self._populating = False
        self._populate_factor_list()
        if self._factors:
            self._factor_list.setCurrentRow(0)

    def _populate_factor_list(self):
        self._factor_list.blockSignals(True)
        self._factor_list.clear()
        for f in self._factors:
            self._factor_list.addItem(f.get("name", "factor"))
        self._factor_list.blockSignals(False)

    def _on_factor_selected(self, row: int):
        if row < 0 or row >= len(self._factors):
            self._factor_name_edit.clear()
            self._clear_label_rows()
            self._current_factor_idx = -1
            return
        self._flush_current_factor()
        self._current_factor_idx = row
        self._load_factor_into_panel(self._factors[row])

    def _flush_current_factor(self):
        idx = self._current_factor_idx
        if idx < 0 or idx >= len(self._factors):
            return
        self._factors[idx]["name"] = self._factor_name_edit.text().strip() or "factor"
        self._factors[idx]["labels"] = [
            r.label() for r in self._label_rows if r.label()
        ]
        self._factors[idx]["shortcuts"] = [
            r.shortcut() for r in self._label_rows if r.label()
        ]
        item = self._factor_list.item(idx)
        if item:
            item.setText(self._factors[idx]["name"])

    def _load_factor_into_panel(self, factor: dict):
        self._populating = True
        self._factor_name_edit.setText(factor.get("name", ""))
        self._clear_label_rows()
        labels = factor.get("labels", [])
        shortcuts = factor.get("shortcuts", [])
        for i, lbl in enumerate(labels):
            key = shortcuts[i] if i < len(shortcuts) else ""
            self._append_label_row(lbl, key)
        self._populating = False

    def _on_name_edited(self, text: str):
        idx = self._current_factor_idx
        if idx < 0 or self._populating:
            return
        if idx < len(self._factors):
            self._factors[idx]["name"] = text.strip() or "factor"
        item = self._factor_list.item(idx)
        if item:
            item.setText(text.strip() or "factor")

    def _add_factor(self):
        self._flush_current_factor()
        new_name = f"factor_{len(self._factors) + 1}"
        self._factors.append(
            {"name": new_name, "labels": ["label_1"], "shortcuts": [""]}
        )
        self._populate_factor_list()
        self._factor_list.setCurrentRow(len(self._factors) - 1)

    def _remove_selected_factor(self):
        idx = self._factor_list.currentRow()
        if idx < 0 or len(self._factors) <= 1:
            QMessageBox.information(
                self, "Cannot Remove", "A scheme must have at least one factor."
            )
            return
        self._flush_current_factor()
        self._factors.pop(idx)
        self._current_factor_idx = -1
        self._populate_factor_list()
        new_row = min(idx, len(self._factors) - 1)
        self._factor_list.setCurrentRow(new_row)

    def _clear_label_rows(self):
        for row in self._label_rows:
            row.setParent(None)
        self._label_rows = []

    def _append_label_row(self, label: str = "", shortcut: str = ""):
        row = _LabelRow(label, shortcut, self._labels_container)
        insert_pos = max(0, self._labels_layout.count() - 1)
        self._labels_layout.insertWidget(insert_pos, row)
        self._label_rows.append(row)
        row.delete_button().clicked.connect(lambda: self._remove_label_row(row))

    def _add_label_row(self):
        self._append_label_row("", "")

    def _remove_label_row(self, row: "_LabelRow"):
        if len(self._label_rows) <= 1:
            QMessageBox.information(
                self, "Cannot Remove", "A factor must have at least one label."
            )
            return
        self._label_rows.remove(row)
        row.setParent(None)

    def _apply_preset(self):
        key = self._preset_combo.currentData()
        preset = self._preset_lookup.get(key)
        if preset is None:
            return
        reply = QMessageBox.question(
            self,
            "Apply Preset",
            f"Replace current scheme with the '{preset.label}' preset?",
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        self._current_factor_idx = -1
        self._factors = [
            {
                "name": factor.name,
                "labels": list(factor.labels),
                "shortcuts": list(factor.shortcut_keys),
            }
            for factor in preset.scheme.factors
        ]
        self._populate_factor_list()
        self._clear_label_rows()
        if self._factors:
            self._factor_list.setCurrentRow(0)

    def _reload_presets(self, selected_key: str | None = None) -> None:
        current_key = selected_key or self._preset_combo.currentData()
        self._preset_lookup = {}
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        self._preset_combo.addItem("— choose preset —", None)
        for preset in get_available_scheme_presets():
            self._preset_lookup[preset.key] = preset
            self._preset_combo.addItem(preset.label, preset.key)
        if current_key is not None:
            index = self._preset_combo.findData(current_key)
            if index >= 0:
                self._preset_combo.setCurrentIndex(index)
        self._preset_combo.blockSignals(False)

    def _save_current_as_preset(self):
        self._flush_current_factor()
        scheme_dict = self.get_scheme_dict()
        if not scheme_dict:
            QMessageBox.warning(
                self,
                "Nothing to Save",
                "Define at least one factor and label before saving a preset.",
            )
            return

        default_name = scheme_dict.get("name", "custom_scheme")
        preset_name, ok = QInputDialog.getText(
            self,
            "Save Scheme Preset",
            "Preset name:",
            text=default_name,
        )
        if not ok:
            return

        normalized_name = preset_name.strip()
        if not normalized_name:
            QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")
            return

        overwrite = False
        while True:
            try:
                preset_path = save_scheme_preset(
                    normalized_name,
                    scheme_dict,
                    overwrite=overwrite,
                )
                break
            except FileExistsError:
                reply = QMessageBox.question(
                    self,
                    "Overwrite Preset?",
                    f"A preset named '{normalized_name}' already exists. Overwrite it?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                )
                if reply != QMessageBox.Yes:
                    return
                overwrite = True
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid Preset", str(exc))
                return

        selected_key = f"custom:{preset_path.stem}"
        self._reload_presets(selected_key=selected_key)
        QMessageBox.information(
            self,
            "Preset Saved",
            f"Saved preset '{normalized_name}'.\n\n{describe_scheme(scheme_dict)}",
        )

    def _do_accept(self):
        self._flush_current_factor()
        for i, f in enumerate(self._factors):
            if not f.get("labels"):
                QMessageBox.warning(
                    self,
                    "Empty Labels",
                    f"Factor '{f.get('name', i + 1)}' has no labels. "
                    "Add at least one label or remove the factor.",
                )
                return
        self.accept()

    @property
    def flat_classes(self) -> List[str]:
        if not self._factors:
            return ["class_1", "class_2"]
        return self._factors[0].get("labels", [])

    def get_scheme_dict(self) -> Optional[dict]:
        factors = self._factors
        if len(factors) == 1:
            f = factors[0]
            return {
                "name": f.get("name", "custom"),
                "description": "Custom 1-factor labeling scheme",
                "factors": [
                    {
                        "name": f.get("name", "class"),
                        "labels": f.get("labels", []),
                        "shortcut_keys": f.get("shortcuts", []),
                    }
                ],
                "training_modes": ["flat_custom", "flat_yolo"],
            }
        modes = ["flat_custom", "flat_yolo", "multihead_custom", "multihead_yolo"]
        return {
            "name": "_".join(f.get("name", "f") for f in factors),
            "description": f"{len(factors)}-factor labeling scheme",
            "factors": [
                {
                    "name": f.get("name", f"factor_{i}"),
                    "labels": f.get("labels", []),
                    "shortcut_keys": f.get("shortcuts", []),
                }
                for i, f in enumerate(factors)
            ],
            "training_modes": modes,
        }
