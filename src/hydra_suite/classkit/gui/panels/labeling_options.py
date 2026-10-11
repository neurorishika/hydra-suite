"""Labeling options: candidate sources and the Start Labeling button.

In labeling mode the panel sits beside the explorer plot (stacked top to
bottom) on landscape screens, so the plot keeps its height, and falls back to
a band below the plot (sections side by side) when the window is too narrow
for that.  :class:`PlacementController` picks whichever leaves the larger
square plot area.  Either way the options scroll while the candidate summary
and **Start Labeling** stay pinned and visible.

The panel only builds widgets; ``MainWindow`` wires their signals.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QBoxLayout,
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

_HINT_STYLE = "color: #aaaaaa; font-size: 11px;"
_SPIN_STYLE = "background: #252526;"
_MAX = 16777215  # QWIDGETSIZE_MAX

# Widgets MainWindow reads and wires by name.
WIDGET_NAMES = (
    "labeling_options_hint",
    "random_group",
    "sample_spin",
    "btn_sample_next",
    "al_group",
    "al_hint_label",
    "al_disabled_label",
    "al_status_label",
    "al_batch_spin",
    "al_build_btn",
    "al_candidates_badge",
    "al_balance_check",
    "al_error_spin",
    "al_combo_spin",
    "al_rare_spin",
    "al_rare_alpha_spin",
    "al_grid_unknown_check",
    "al_grid_cap_spin",
    "al_grid_balance_btn",
    "prepared_candidates_group",
    "prepared_candidates_summary",
    "candidate_table",
    "btn_start_labeling",
    "btn_clear_candidates",
)


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(_HINT_STYLE)
    return label


def _small_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(_HINT_STYLE)
    return label


def _percent_spin(maximum: int, value: int, tip: str) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(0, maximum)
    spin.setValue(value)
    spin.setSuffix("%")
    spin.setStyleSheet(_SPIN_STYLE)
    spin.setToolTip(tip)
    return spin


class LabelingOptionsPanel(QWidget):
    """Random labeling, active learning and the prepared candidate set."""

    MIN_WIDTH = 300
    DEFAULT_WIDTH = 340
    BAND_HEIGHT = 315  # height when placed below the plot

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._beside = True
        self.setMinimumWidth(self.MIN_WIDTH)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        content = QWidget()
        column = QBoxLayout(QBoxLayout.TopToBottom, content)
        column.setContentsMargins(0, 0, 6, 0)
        column.setSpacing(10)
        self._column = column

        self.labeling_options_hint = _hint(
            "Build a candidate set (random or active learning), then start labeling."
        )
        column.addWidget(self.labeling_options_hint)
        column.addWidget(self._build_random_group())
        column.addWidget(self._build_al_group())
        column.addWidget(self._build_candidate_table_group())
        column.addStretch(1)

        self.options_scroll = QScrollArea()
        self.options_scroll.setWidgetResizable(True)
        self.options_scroll.setFrameShape(QFrame.NoFrame)
        self.options_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.options_scroll.setWidget(content)
        outer.addWidget(self.options_scroll, 1)

        self._footer = self._build_start_footer()
        outer.addWidget(self._footer)

    # ── placement ────────────────────────────────────────────────────────
    @property
    def beside(self) -> bool:
        return self._beside

    def set_beside(self, beside: bool) -> None:
        """Stack top to bottom (beside the plot) or side by side (below it)."""
        beside = bool(beside)
        if beside == self._beside:
            return
        self._beside = beside
        # Sections stack in the column and sit side by side in the band;
        # the start bar stays pinned at the bottom either way.
        self._column.setDirection(
            QBoxLayout.TopToBottom if beside else QBoxLayout.LeftToRight
        )
        self._footer_layout.setDirection(
            QBoxLayout.TopToBottom if beside else QBoxLayout.LeftToRight
        )
        # The band is short: drop the explanatory hints there.
        self.labeling_options_hint.setVisible(beside)
        self.al_hint_label.setVisible(beside)
        self.options_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff if beside else Qt.ScrollBarAsNeeded
        )
        self.random_group.setMaximumWidth(_MAX if beside else 260)
        # Below the plot the Active Learning group takes the spare width.
        self._column.setStretchFactor(self.al_group, 0 if beside else 1)
        self._column.setStretch(self._column.count() - 1, 1 if beside else 0)
        self.setMinimumWidth(self.MIN_WIDTH if beside else 0)
        self.setMaximumHeight(_MAX if beside else self.BAND_HEIGHT)

    def _place_al_controls(self) -> None:
        """Mix shares as two pairs per row; Grid Balance controls below."""
        for i, (label, spin) in enumerate(self._mix_items):
            row, col = i // 2, (i % 2) * 2
            self._mix.addWidget(label, row, col)
            self._mix.addWidget(spin, row, col + 1)
        self._mix.setColumnStretch(1, 1)
        self._mix.setColumnStretch(3, 1)
        opts = self._opts
        opts.addWidget(self.al_balance_check, 0, 0, 1, 4)
        opts.addWidget(self.al_grid_unknown_check, 1, 0)
        opts.addWidget(self._cap_label, 1, 2)
        opts.addWidget(self.al_grid_cap_spin, 1, 3)
        opts.addWidget(self.al_grid_balance_btn, 2, 0, 1, 4)
        opts.setColumnStretch(1, 1)

    # ── sections ─────────────────────────────────────────────────────────
    def _build_random_group(self) -> QGroupBox:
        self.random_group = QGroupBox("Random Labeling")
        layout = QVBoxLayout(self.random_group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(
            _hint("Sample a lightweight batch from each cluster to start labeling.")
        )
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("<b>Per cluster:</b>"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 100)
        self.sample_spin.setValue(6)
        self.sample_spin.setToolTip("Unlabeled images to sample from each cluster")
        row.addWidget(self.sample_spin)
        self.btn_sample_next = QPushButton("Sample Next")
        self.btn_sample_next.setStyleSheet(
            "background: #0e639c; font-weight: bold; padding: 6px 12px;"
        )
        row.addWidget(self.btn_sample_next, 1)
        layout.addLayout(row)
        layout.addStretch(1)  # top-aligned when the band makes the group tall
        return self.random_group

    def _build_al_group(self) -> QGroupBox:
        self.al_group = QGroupBox("Active Learning")
        self.al_group.setStyleSheet("""
            QGroupBox { color: #777; border: 1px solid #3e3e42; border-radius: 4px;
                        margin-top: 8px; font-size: 11px; padding-top: 2px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)
        layout = QVBoxLayout(self.al_group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.al_hint_label = _hint(
            "Use model predictions to build a higher-value labeling batch."
        )
        layout.addWidget(self.al_hint_label)
        self.al_disabled_label = QLabel(
            "Load a trained model to enable active-learning batch building."
        )
        self.al_disabled_label.setWordWrap(True)
        self.al_disabled_label.setStyleSheet("color: #888; font-size: 11px;")
        self.al_disabled_label.hide()
        layout.addWidget(self.al_disabled_label)

        self.al_status_label = QLabel(
            "Model: none  |  Labeled: 0 / 0  |  Predictions: none"
        )
        self.al_status_label.setWordWrap(True)
        self.al_status_label.setStyleSheet("color: #777; font-size: 11px;")
        layout.addWidget(self.al_status_label)

        batch_row = QHBoxLayout()
        batch_row.setSpacing(8)
        batch_row.addWidget(QLabel("<b>Batch:</b>"))
        self.al_batch_spin = QSpinBox()
        self.al_batch_spin.setRange(5, 500)
        self.al_batch_spin.setValue(50)
        self.al_batch_spin.setStyleSheet(_SPIN_STYLE)
        batch_row.addWidget(self.al_batch_spin)
        self.al_build_btn = QPushButton("⚙  Build Batch")
        self.al_build_btn.setStyleSheet(
            "background: #0e639c; font-weight: bold; padding: 6px 12px;"
        )
        self.al_build_btn.setToolTip(
            "Select the highest-value unlabeled images for labeling —\n"
            "25% uncertain · 20% diverse · 20% similar to model errors · "
            "15% rare/missing combinations · 10% rare clusters · "
            "5% representative (or balance) · 5% audit\n"
            "(the error, combination and rare-cluster shares are set below)"
        )
        batch_row.addWidget(self.al_build_btn, 1)
        self.al_candidates_badge = QLabel("")  # "building…" / "50 selected"
        self.al_candidates_badge.setStyleSheet(
            "color: #ccc; font-size: 11px; font-weight: bold;"
        )
        batch_row.addWidget(self.al_candidates_badge)
        layout.addLayout(batch_row)

        # Batch mix: two label/spin pairs per row.
        mix = QGridLayout()
        mix.setHorizontalSpacing(8)
        mix.setVerticalSpacing(4)
        self.al_error_spin = _percent_spin(
            60,
            20,
            "Share of the batch taken from unlabeled images most similar to images\n"
            "the model currently gets WRONG (human-verified label != prediction),\n"
            "taken round-robin over the errors, most confidently wrong first.\n"
            "Needs verified labels the model has been run on; otherwise this\n"
            "share goes to uncertainty. 0 = off.",
        )
        self.al_combo_spin = _percent_spin(
            50,
            15,
            "Share of the batch aimed at factor COMBINATIONS that are rare or\n"
            "missing in the labeled set (e.g. blue_pink with 0 labels). Works on\n"
            "per-factor predictions, so it can find combinations the model has\n"
            "never output. Picks spread over the under-represented combinations.\n"
            "Needs a trained model. 0 = off.",
        )
        self.al_rare_spin = _percent_spin(
            50,
            10,
            "Share of the batch sampled with a bias toward SMALL clusters (and\n"
            "clusters with little labeled coverage), so under-represented groups\n"
            "are not starved by large ones. Needs clusters. 0 = off.",
        )
        self.al_rare_alpha_spin = _percent_spin(
            100,
            75,
            "How strongly small clusters are favoured.\n"
            "0% = proportional to cluster size (no bias),\n"
            "100% = every cluster gets the same expected share.",
        )
        self._mix_items = [
            (_small_label(text), spin)
            for text, spin in (
                ("Errors:", self.al_error_spin),
                ("Combos:", self.al_combo_spin),
                ("Rare clusters:", self.al_rare_spin),
                ("Rarity:", self.al_rare_alpha_spin),
            )
        ]
        self._mix = QGridLayout()
        self._mix.setHorizontalSpacing(8)
        self._mix.setVerticalSpacing(4)
        layout.addLayout(self._mix)

        self.al_balance_check = QCheckBox("Balance labels")
        self.al_balance_check.setToolTip(
            "When enabled, the 15% representative slot prioritises unlabeled samples\n"
            "predicted to belong to underrepresented label classes,\n"
            "steering the labeled set toward a more uniform distribution."
        )
        self.al_balance_check.setStyleSheet(_HINT_STYLE)
        self.al_grid_unknown_check = QCheckBox("incl. unknown")
        self.al_grid_unknown_check.setToolTip(
            "Grid Balance also visits combinations that contain 'unknown'."
        )
        self.al_grid_unknown_check.setStyleSheet(_HINT_STYLE)
        self._cap_label = _small_label("Cap:")
        self.al_grid_cap_spin = QSpinBox()
        self.al_grid_cap_spin.setRange(0, 100000)
        self.al_grid_cap_spin.setValue(0)
        self.al_grid_cap_spin.setSpecialValueText("none")
        self.al_grid_cap_spin.setStyleSheet(_SPIN_STYLE)
        self.al_grid_cap_spin.setToolTip(
            "Upper bound on how many labels Grid Balance aims for per combination.\n"
            "By default each combination is filled up to the most common one."
        )
        self.al_grid_balance_btn = QPushButton("\u25a6  Grid Balance\u2026")
        self.al_grid_balance_btn.setStyleSheet(
            "background: #6f42c1; font-weight: bold; padding: 6px 12px;"
        )
        self.al_grid_balance_btn.setToolTip(
            "CAPTCHA-style balancing: for each under-represented label combination\n"
            "(rarest first) click every matching image in a grid of likely and\n"
            "borderline candidates. Clicked images are labeled and replaced;\n"
            "unclicked images get no label. Runs until each combination reaches\n"
            "the most common one's count (or the cap)."
        )
        self._opts = QGridLayout()
        self._opts.setHorizontalSpacing(8)
        self._opts.setVerticalSpacing(6)
        layout.addLayout(self._opts)
        self._place_al_controls()
        layout.addStretch(1)
        return self.al_group

    def _build_candidate_table_group(self) -> QGroupBox:
        self.prepared_candidates_group = QGroupBox("Prepared Candidate Set")
        layout = QVBoxLayout(self.prepared_candidates_group)
        layout.setContentsMargins(10, 10, 10, 10)
        self.candidate_table = QTableWidget(0, 4)
        self.candidate_table.setHorizontalHeaderLabels(
            ["Source", "Index", "File", "Details"]
        )
        self.candidate_table.verticalHeader().setVisible(False)
        self.candidate_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.candidate_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.candidate_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.candidate_table.setAlternatingRowColors(True)
        self.candidate_table.setShowGrid(False)
        self.candidate_table.setMinimumHeight(160)
        header = self.candidate_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.candidate_table.setColumnHidden(1, True)  # file name says enough here
        self.candidate_table.setStyleSheet(
            "QTableWidget { background:#1a1a1a; color:#d0d0d0; font-size:11px; }"
        )
        self.candidate_table.hide()
        layout.addWidget(self.candidate_table)
        # The group itself is only useful while a candidate table is shown.
        self.prepared_candidates_group.setVisible(False)
        return self.prepared_candidates_group

    def _build_start_footer(self) -> QWidget:
        """Summary + Start Labeling, pinned below the scrolling options."""
        footer = QFrame()
        footer.setObjectName("labelingStartFooter")
        footer.setStyleSheet("#labelingStartFooter { border-top: 1px solid #3e3e42; }")
        # Summary above the buttons in the column, beside them in the band.
        layout = QBoxLayout(QBoxLayout.TopToBottom, footer)
        layout.setContentsMargins(0, 8, 6, 0)
        layout.setSpacing(8)
        self._footer_layout = layout
        self.prepared_candidates_summary = _hint(
            "No candidate set prepared yet. Build one from Random Labeling or "
            "Active Learning."
        )
        layout.addWidget(self.prepared_candidates_summary, 1)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_start_labeling = QPushButton("▶  Start Labeling")
        self.btn_start_labeling.setMinimumHeight(36)
        self.btn_start_labeling.setStyleSheet(
            "QPushButton { background: #28a745; color: white; font-weight: bold;"
            " font-size: 13px; padding: 6px 12px; }"
            "QPushButton:disabled { background: #2f3b31; color: #7a8a7c; }"
        )
        self.btn_start_labeling.setEnabled(False)
        self.btn_start_labeling.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_start_labeling.setMinimumWidth(170)
        row.addWidget(self.btn_start_labeling, 1)
        self.btn_clear_candidates = QPushButton("Clear")
        self.btn_clear_candidates.setToolTip("Clear the prepared candidate set")
        self.btn_clear_candidates.setMinimumHeight(36)
        self.btn_clear_candidates.setStyleSheet(
            "background: #3e3e42; padding: 6px 12px;"
        )
        row.addWidget(self.btn_clear_candidates)
        layout.addLayout(row)
        return footer


class PlacementController(QObject):
    """Put the panel beside or below the plot, whichever leaves a bigger plot.

    The explorer plot is roughly square, so the score is the side of the
    largest square it gets.  A 10% margin keeps the layout from flapping
    while a window is being resized.
    """

    HYSTERESIS = 1.1

    def __init__(self, splitter, panel: LabelingOptionsPanel) -> None:
        super().__init__(splitter)
        self._splitter = splitter
        self._panel = panel
        self._placed = False
        splitter.installEventFilter(self)
        panel.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if watched is self._splitter and event.type() == QEvent.Resize:
            self.update_placement()
        elif watched is self._panel and event.type() == QEvent.Show:
            self._placed = False  # fresh decision each time labeling mode opens
            self.update_placement()
        return False

    def scores(self, width: int, height: int):
        panel = self._panel
        beside = min(width - panel.DEFAULT_WIDTH, height)
        below = min(width, height - panel.BAND_HEIGHT)
        return beside, below

    def update_placement(self) -> None:
        width, height = self._splitter.width(), self._splitter.height()
        if width <= 0 or height <= 0 or not self._panel.isVisible():
            return
        beside, below = self.scores(width, height)
        if not self._placed:  # first layout: just take the better one
            want = beside >= below
        elif self._panel.beside:
            want = not (below > beside * self.HYSTERESIS)
        else:
            want = beside > below * self.HYSTERESIS
        self._placed = True
        if want == self._panel.beside and self._splitter.orientation() == (
            Qt.Horizontal if want else Qt.Vertical
        ):
            return
        self._panel.set_beside(want)
        if want:
            self._splitter.setOrientation(Qt.Horizontal)
            panel_w = self._panel.DEFAULT_WIDTH
            self._splitter.setSizes([max(1, width - panel_w), panel_w])
        else:
            self._splitter.setOrientation(Qt.Vertical)
            band = self._panel.BAND_HEIGHT
            self._splitter.setSizes([max(1, height - band), band])
