"""Left-side PoseKit source browser and labeling panel."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PoseSourceBrowserPanel(QWidget):
    """PoseKit left panel with cross-source labeling and current-source browsing."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Labeling Set"))
        self.labeling_list = QListWidget()
        self.labeling_list.setHorizontalScrollBarPolicy(
            self.labeling_list.horizontalScrollBarPolicy()
        )
        layout.addWidget(self.labeling_list, 1)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Find frame…")
        layout.addWidget(self.search_edit)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(
            [
                "Default",
                "Pred conf (high to low)",
                "Pred conf (low to high)",
                "Detected kpts (high to low)",
                "Detected kpts (low to high)",
                "Cluster id (low to high)",
                "Cluster id (high to low)",
            ]
        )
        sort_row.addWidget(self.sort_combo, 1)
        layout.addLayout(sort_row)

        source_group = QGroupBox("Current Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setContentsMargins(8, 8, 8, 8)
        source_layout.setSpacing(6)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source"))
        self.source_combo = QComboBox()
        source_row.addWidget(self.source_combo, 1)
        self.btn_manage_sources = QPushButton("Manage…")
        source_row.addWidget(self.btn_manage_sources)
        source_layout.addLayout(source_row)

        self.source_summary = QLabel("No sources added yet.")
        self.source_summary.setWordWrap(True)
        self.source_summary.setStyleSheet("QLabel { color: #9f9f9f; font-size: 11px; }")
        source_layout.addWidget(self.source_summary)
        layout.addWidget(source_group, 0)

        layout.addWidget(QLabel("Source Frames"))
        self.frame_list = QListWidget()
        layout.addWidget(self.frame_list, 1)

        frame_btns = QGridLayout()
        frame_btns.setHorizontalSpacing(6)
        frame_btns.setVerticalSpacing(6)

        self.btn_unlabeled_to_labeling = QPushButton("Unlabeled → Labeling")
        self.btn_unlabeled_to_labeling.setToolTip(
            "Move all unlabeled frames to labeling list"
        )
        frame_btns.addWidget(self.btn_unlabeled_to_labeling, 0, 0)

        self.btn_unlabeled_to_all = QPushButton("Unlabeled → All")
        self.btn_unlabeled_to_all.setToolTip(
            "Move unlabeled frames from labeling to source frames list"
        )
        frame_btns.addWidget(self.btn_unlabeled_to_all, 0, 1)

        self.btn_random_to_labeling = QPushButton("Random")
        self.btn_random_to_labeling.setToolTip(
            "Add random unlabeled frames from the current source to labeling"
        )
        self.spin_random_count = QSpinBox()
        self.spin_random_count.setRange(1, 1000)
        self.spin_random_count.setValue(10)
        frame_btns.addWidget(self.btn_random_to_labeling, 1, 0)
        frame_btns.addWidget(self.spin_random_count, 1, 1)

        self.btn_smart_select = QPushButton("Smart Select…")
        self.btn_smart_select.setToolTip(
            "Select diverse frames using embeddings + clustering"
        )
        frame_btns.addWidget(self.btn_smart_select, 2, 0)

        self.btn_delete_frames = QPushButton("Delete Selected…")
        self.btn_delete_frames.setToolTip(
            "Permanently delete selected images (and labels) from the dataset"
        )
        frame_btns.addWidget(self.btn_delete_frames, 2, 1)

        layout.addLayout(frame_btns)
