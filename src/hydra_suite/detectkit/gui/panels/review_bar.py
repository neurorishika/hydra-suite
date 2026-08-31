"""The frame-granular review bar, shown above the canvas.

Visible only when the current source has a staged review. It renders state
and emits intent; it touches neither the project nor the filesystem, which
is what keeps it testable without a project on disk.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class ReviewBar(QWidget):
    """Accept/reject the frame on screen, or the whole review."""

    accept_overwrite_requested = Signal()
    accept_add_new_requested = Signal()
    reject_requested = Signal()
    accept_all_requested = Signal()
    reject_all_requested = Signal()
    next_undecided_requested = Signal()
    revert_requested = Signal()
    rethreshold_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("detectkitRole", "reviewBar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self._summary = QLabel("")
        self._summary.setProperty("detectkitRole", "sectionHint")
        layout.addWidget(self._summary, 1)

        self._btn_overwrite = QPushButton("Replace")
        self._btn_overwrite.setToolTip(
            "Replace this frame's labels with the staged ones."
        )
        self._btn_add_new = QPushButton("Add New")
        self._btn_add_new.setToolTip(
            "Keep this frame's labels and add only the staged instances that do "
            "not overlap one already there."
        )
        self._btn_reject = QPushButton("Reject")
        self._btn_reject.setToolTip("Discard the staged labels for this frame.")
        self._btn_next = QPushButton("Next Undecided")
        self._btn_accept_all = QPushButton("Accept All…")
        self._btn_reject_all = QPushButton("Reject All…")
        self._btn_revert = QPushButton("Revert Review…")
        self._btn_revert.setToolTip(
            "Restore this source to its state before the review started. "
            "Available only while the review is open -- finishing it deletes "
            "the snapshot."
        )
        self._btn_rethreshold = QPushButton("Re-threshold…")
        self._btn_rethreshold.setToolTip(
            "Rewrite the staged result at a different confidence, using the "
            "cached candidates. No inference -- seconds, not hours."
        )

        for button in (
            self._btn_overwrite,
            self._btn_add_new,
            self._btn_reject,
            self._btn_next,
            self._btn_accept_all,
            self._btn_reject_all,
            self._btn_revert,
            self._btn_rethreshold,
        ):
            layout.addWidget(button)

        self._progress = QLabel("")
        layout.addWidget(self._progress)

        self._btn_overwrite.clicked.connect(self.accept_overwrite_requested)
        self._btn_add_new.clicked.connect(self.accept_add_new_requested)
        self._btn_reject.clicked.connect(self.reject_requested)
        self._btn_next.clicked.connect(self.next_undecided_requested)
        self._btn_accept_all.clicked.connect(self.accept_all_requested)
        self._btn_reject_all.clicked.connect(self.reject_all_requested)
        self._btn_revert.clicked.connect(self.revert_requested)
        self._btn_rethreshold.clicked.connect(self.rethreshold_requested)

        self.hide()

    # -- state ---------------------------------------------------------

    def set_review_state(
        self,
        producer: str,
        detail: str,
        decided: int,
        total: int,
        can_rethreshold: bool,
    ) -> None:
        """Show the bar for a staged review and render its progress."""
        self._summary.setText(f"Staged review — {producer}: {detail}")
        self._progress.setText(
            f"{decided}/{total} decided — review complete"
            if total and decided >= total
            else f"{decided}/{total} decided"
        )
        self._btn_rethreshold.setEnabled(bool(can_rethreshold))
        self._btn_rethreshold.setVisible(bool(can_rethreshold))
        self.show()

    def clear_review_state(self) -> None:
        """Hide the bar; the current source has no staged review."""
        self._summary.setText("")
        self._progress.setText("")
        self.hide()

    # -- accessors used by MainWindow and by tests ----------------------

    def summary_text(self) -> str:
        return self._summary.text()

    def progress_text(self) -> str:
        return self._progress.text()

    def accept_overwrite_button(self) -> QPushButton:
        return self._btn_overwrite

    def accept_add_new_button(self) -> QPushButton:
        return self._btn_add_new

    def reject_button(self) -> QPushButton:
        return self._btn_reject

    def accept_all_button(self) -> QPushButton:
        return self._btn_accept_all

    def reject_all_button(self) -> QPushButton:
        return self._btn_reject_all

    def next_undecided_button(self) -> QPushButton:
        return self._btn_next

    def revert_button(self) -> QPushButton:
        return self._btn_revert

    def rethreshold_button(self) -> QPushButton:
        return self._btn_rethreshold
