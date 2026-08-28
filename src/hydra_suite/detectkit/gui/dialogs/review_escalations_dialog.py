"""ReviewEscalationsDialog — accept/reject staged SAM2 escalation results."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog

from ...jobs.sam2_escalation import accept_pending_escalation, reject_pending_escalation


class ReviewEscalationsDialog(BaseDialog):
    """Review sources with a pending SAM2 escalation: accept or reject each.

    Each checked row is actioned immediately when its button is clicked (not
    deferred to dialog close) and removed from the list on success -- this is
    a working queue, not a form.
    """

    def __init__(self, pending_sources: list, parent=None, project_dir=None) -> None:
        super().__init__(
            "Review Escalations",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self.accepted_names: list[str] = []
        self.rejected_names: list[str] = []
        # Bounds the staging-directory deletes accept/reject perform.
        self._project_dir = project_dir

        container = QWidget()
        layout = QVBoxLayout(container)
        intro = QLabel(
            "These sources have a staged SAM2 segmentation result awaiting "
            "review. Accept to replace the source's labels with the staged "
            "result; reject to discard it.\n\n"
            "Accepted sources are marked unreviewed and are excluded from "
            'training until you use "Mark reviewed…" for them.'
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._list = QListWidget()
        for src in pending_sources:
            pending = src.pending_escalation
            if pending is None:
                continue
            item = QListWidgetItem(
                f"{src.name}  ->  {pending.target_level} "
                f"({pending.sam2_variant}, staged {pending.created_at})"
            )
            item.setData(Qt.UserRole, src)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._list.addItem(item)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._btn_accept = QPushButton("Accept Checked")
        self._btn_accept.clicked.connect(lambda: self._apply_checked(accept=True))
        self._btn_reject = QPushButton("Reject Checked")
        self._btn_reject.clicked.connect(lambda: self._apply_checked(accept=False))
        btn_row.addWidget(self._btn_accept)
        btn_row.addWidget(self._btn_reject)
        layout.addLayout(btn_row)

        self.add_content(container)

    def _checked_rows(self) -> list[int]:
        return [
            i
            for i in range(self._list.count())
            if self._list.item(i).checkState() == Qt.Checked
        ]

    def _apply_checked(self, *, accept: bool) -> None:
        rows = self._checked_rows()
        if not rows:
            QMessageBox.information(self, "Review Escalations", "No sources checked.")
            return
        for row in sorted(rows, reverse=True):
            item = self._list.item(row)
            src = item.data(Qt.UserRole)
            try:
                if accept:
                    accept_pending_escalation(src, self._project_dir)
                    self.accepted_names.append(src.name)
                else:
                    reject_pending_escalation(src, self._project_dir)
                    self.rejected_names.append(src.name)
            except Exception as exc:
                QMessageBox.warning(self, "Review Escalations", str(exc))
                continue
            self._list.takeItem(row)
