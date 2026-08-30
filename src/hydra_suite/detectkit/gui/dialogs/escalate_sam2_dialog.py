"""Dialog to pick sources + SAM2 variant for escalate-all."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.inference.sam2.checkpoints import (
    DEFAULT_VARIANT,
    available_variants,
)
from hydra_suite.widgets.dialogs import BaseDialog


class EscalateSam2Dialog(BaseDialog):
    """Pick which OBB/keypoint sources to escalate to SAM2 segmentation."""

    def __init__(self, sources, parent=None) -> None:
        super().__init__("Escalate to segment (SAM2)", parent)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("SAM2 version:"))
        self._variant = QComboBox()
        for v in available_variants():
            self._variant.addItem(v)
        self._variant.setCurrentText(DEFAULT_VARIANT)
        layout.addWidget(self._variant)

        layout.addWidget(QLabel("Sources to escalate (click to toggle selection):"))
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for s in sources:
            eligible = s.level != "polygon"
            label = s.name if eligible else f"{s.name}  (already segment)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, s.name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            if s.level == "polygon":
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip("This source already contains segmentation polygons.")
            self._list.addItem(item)
            item.setSelected(eligible)
        layout.addWidget(self._list)

        container = QWidget()
        container.setLayout(layout)
        self.add_content(container)

    def _variant_combo_items(self) -> list[str]:
        return [self._variant.itemText(i) for i in range(self._variant.count())]

    def selectable_source_names(self) -> list[str]:
        return [
            self._list.item(i).text()
            for i in range(self._list.count())
            if self._list.item(i).flags() & Qt.ItemFlag.ItemIsEnabled
        ]

    def preselect_source(self, name: str) -> None:
        """Select only the named source (used when launched from a role block)."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
                continue
            item.setSelected(item.data(Qt.ItemDataRole.UserRole) == name)

    def selected_variant(self) -> str:
        return self._variant.currentText()

    def selected_sources(self) -> list[str]:
        return [
            self._list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._list.count())
            if self._list.item(i).isSelected()
        ]
