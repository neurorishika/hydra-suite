"""Two source lists for escalation dialogs: what to calibrate on, what to escalate.

Calibration and escalation are different jobs over different sources:
calibration needs real ground truth (frames labelled entirely with polygons),
escalation targets the sources that lack it. One list for both made the
calibration silently score against boxes. Shared by the SAM3 and SAM2
escalation dialogs so the two stay laid out and behave the same.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.detectkit.jobs.calibration_frames import (
    SCALE_MISMATCH_RATIO,
    has_polygon_frames,
    quick_median_body_px,
    scale_mismatch,
)

_INDEX_ROLE = Qt.ItemDataRole.UserRole

NO_POLYGON_SOURCES = (
    "No polygon ground truth in this project. Label polygons on a few frames "
    "(every animal in the frame) to calibrate."
)


def _median_body_px(cache: dict, sources) -> float:
    """Median body size of *sources*, cached by their paths.

    Runs on every selection click, so it must not decode images: it reads
    label files and image headers only (``quick_median_body_px``), and is
    cached per distinct selection.
    """
    key = tuple(sorted(_source_key(s) for s in sources))
    if not key:
        return 0.0
    if key not in cache:
        try:
            cache[key] = quick_median_body_px(sources)
        except Exception:  # pragma: no cover - unreadable labels
            cache[key] = 0.0
    return cache[key]


def scale_warning_text(cache: dict, calibration_sources, escalation_sources) -> str:
    """A warning when calibration and escalation animals differ in size.

    A tile fraction is relative to the animal, so a calibration only
    transfers to targets of a similar size. Targets without labels have no
    measurable size and never warn.
    """
    cal = _median_body_px(cache, calibration_sources)
    esc = _median_body_px(cache, escalation_sources)
    if not scale_mismatch(cal, esc):
        return ""
    return (
        f"⚠ The calibration frames' animals (~{cal:.0f} px) differ in size from "
        f"the escalation targets' (~{esc:.0f} px) by more than "
        f"{SCALE_MISMATCH_RATIO:g}×; a calibrated tile fraction may not transfer."
    )


def _key(path) -> str:
    try:
        return str(Path(str(path)).expanduser().resolve())
    except (OSError, RuntimeError):  # pragma: no cover - unresolvable path
        return str(path)


def _keys(saved) -> set[str]:
    """Saved identities, matching both resolved paths and bare names."""
    return {_key(p) for p in saved} | {str(p) for p in saved}


def _source_key(source) -> str:
    """Stable identity: the resolved path, or the name for path-less sources."""
    path = getattr(source, "path", "")
    return _key(path) if path else str(source.name)


def _row(source, index: int) -> QListWidgetItem:
    """A click-to-toggle row: selection, never a checkbox."""
    item = QListWidgetItem(f"{source.name}  ({source.level})")
    item.setData(_INDEX_ROLE, index)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
    return item


class CalibrationSourceSelector(QWidget):
    """Top: "Calibrate on" (polygon sources). Bottom: "Escalate" (all sources)."""

    calibration_changed = Signal()
    escalation_changed = Signal()

    def __init__(
        self,
        sources: Sequence,
        *,
        escalation_eligible: Callable[[object], bool] = lambda _s: True,
        ineligible_reason: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._sources = list(sources)
        self._eligible = escalation_eligible
        self._calibration_rows = [
            i for i, s in enumerate(self._sources) if has_polygon_frames(s)
        ]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        cal_group = QGroupBox("Calibrate on (polygon ground truth)")
        cal_layout = QVBoxLayout(cal_group)
        self.calibration_list = self._make_list()
        for i in self._calibration_rows:
            src = self._sources[i]
            item = _row(src, i)
            item.setToolTip(str(getattr(src, "path", "") or src.name))
            self.calibration_list.addItem(item)
        cal_layout.addWidget(self.calibration_list)
        self.calibration_empty_label = QLabel(NO_POLYGON_SOURCES)
        self.calibration_empty_label.setWordWrap(True)
        self.calibration_empty_label.setVisible(not self._calibration_rows)
        self.calibration_list.setVisible(bool(self._calibration_rows))
        cal_layout.addWidget(self.calibration_empty_label)
        layout.addWidget(cal_group, 1)

        esc_group = QGroupBox("Escalate")
        esc_layout = QVBoxLayout(esc_group)
        self.escalation_list = self._make_list()
        for i, src in enumerate(self._sources):
            item = _row(src, i)
            if not self._eligible(src):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setToolTip(ineligible_reason)
            self.escalation_list.addItem(item)
        esc_layout.addWidget(self.escalation_list)
        hint = QLabel("Click entries to toggle one or more sources.")
        hint.setWordWrap(True)
        esc_layout.addWidget(hint)
        layout.addWidget(esc_group, 2)

        self.restore(None, None, None)
        self.calibration_list.itemSelectionChanged.connect(self.calibration_changed)
        self.escalation_list.itemSelectionChanged.connect(self.escalation_changed)

    @staticmethod
    def _make_list() -> QListWidget:
        widget = QListWidget()
        widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        widget.setMinimumSize(220, 90)
        return widget

    # -- reading -------------------------------------------------------------

    def _selected(self, widget: QListWidget) -> list:
        rows = sorted(
            int(widget.item(r).data(_INDEX_ROLE))
            for r in range(widget.count())
            if widget.item(r).isSelected()
        )
        return [self._sources[i] for i in rows]

    def calibration_sources(self) -> list:
        return self._selected(self.calibration_list)

    def escalation_sources(self) -> list:
        return self._selected(self.escalation_list)

    def has_calibration_sources(self) -> bool:
        return bool(self._calibration_rows)

    def escalation_items(self) -> list[QListWidgetItem]:
        return [
            self.escalation_list.item(r) for r in range(self.escalation_list.count())
        ]

    def source_for(self, item: QListWidgetItem):
        return self._sources[int(item.data(_INDEX_ROLE))]

    def state(self) -> dict:
        esc = self.escalation_sources()
        return {
            "calibration_source_paths": [
                _source_key(s) for s in self.calibration_sources()
            ],
            "escalation_source_paths": [_source_key(s) for s in esc],
            "escalation_source_names": [s.name for s in esc],
        }

    # -- writing -------------------------------------------------------------

    def restore(
        self,
        calibration_paths: Sequence[str] | None,
        escalation_names: Sequence[str] | None,
        escalation_paths: Sequence[str] | None,
    ) -> None:
        """Re-apply a saved selection.

        ``calibration_paths=None`` (nothing saved) selects every polygon
        source. Escalation matches by path first; ``escalation_names`` is the
        legacy ``source_names`` key older projects saved. Ineligible rows are
        never selected.
        """
        cal_keys = None if calibration_paths is None else _keys(calibration_paths)
        for r in range(self.calibration_list.count()):
            item = self.calibration_list.item(r)
            src = self.source_for(item)
            item.setSelected(cal_keys is None or _source_key(src) in cal_keys)

        if escalation_paths:
            wanted = _keys(escalation_paths)
            match = lambda s: _source_key(s) in wanted  # noqa: E731
        else:
            names = set(escalation_names or [])
            match = lambda s: s.name in names  # noqa: E731
        for item in self.escalation_items():
            src = self.source_for(item)
            enabled = bool(item.flags() & Qt.ItemFlag.ItemIsEnabled)
            item.setSelected(enabled and match(src))

    def select_all_eligible_escalation(self) -> None:
        for item in self.escalation_items():
            item.setSelected(bool(item.flags() & Qt.ItemFlag.ItemIsEnabled))

    def select_escalation_source(self, name: str) -> None:
        """Select only the named, eligible source (launch from a role block)."""
        for item in self.escalation_items():
            enabled = bool(item.flags() & Qt.ItemFlag.ItemIsEnabled)
            item.setSelected(enabled and self.source_for(item).name == name)
