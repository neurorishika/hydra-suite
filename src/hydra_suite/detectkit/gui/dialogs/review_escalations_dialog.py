"""ReviewEscalationsDialog — accept/reject staged escalation results."""

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
from ...jobs.semantic_escalation import (
    accept_pending_semantic_escalation,
    rethreshold_floor_for,
    rethreshold_staged,
)


class ReviewEscalationsDialog(BaseDialog):
    """Review sources with a pending escalation: accept or reject each.

    Each checked row is actioned immediately when its button is clicked (not
    deferred to dialog close) and removed from the list on success -- this is
    a working queue, not a form.
    """

    def __init__(
        self, pending_sources: list, parent=None, project=None, project_dir=None
    ) -> None:
        super().__init__(
            "Review Escalations",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self.accepted_names: list[str] = []
        self.rejected_names: list[str] = []
        # Bounds the staging-directory deletes accept/reject perform.
        self._project_dir = project_dir
        # Needed to register the SAM3 sibling source on accept.
        self._project = project

        container = QWidget()
        layout = QVBoxLayout(container)
        intro = QLabel(
            "These sources have a staged segmentation result awaiting review.\n\n"
            "Geometry (SAM2) results REPLACE the source's own labels — same "
            "instances, upgraded to masks. Semantic (SAM3) results become a NEW "
            "SIBLING source and leave the original untouched, because they are a "
            "different instance set at a different geometry convention.\n\n"
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
            detail = (
                f"prompt '{pending.primer_prompt}'"
                if pending.primer_kind == "sam3"
                else pending.primer_variant or pending.sam2_variant
            )
            item = QListWidgetItem(
                f"{src.name}  ->  {pending.target_level} "
                f"[{pending.primer_kind}: {detail}, staged {pending.created_at}]"
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
        self._btn_rethreshold = QPushButton("Re-threshold Checked…")
        self._btn_rethreshold.setToolTip(
            "Rewrite a staged SAM3 result at a different confidence, using the "
            "cached candidates. No inference — seconds, not hours."
        )
        self._btn_rethreshold.clicked.connect(self._rethreshold_checked)
        btn_row.addWidget(self._btn_rethreshold)
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
                    if src.pending_escalation.primer_kind == "sam3":
                        accept_pending_semantic_escalation(
                            src, self._project, self._project_dir
                        )
                    else:
                        accept_pending_escalation(src, self._project_dir)
                    self.accepted_names.append(src.name)
                else:
                    reject_pending_escalation(src, self._project_dir)
                    self.rejected_names.append(src.name)
            except Exception as exc:
                QMessageBox.warning(self, "Review Escalations", str(exc))
                continue
            self._list.takeItem(row)

    def _rethreshold_checked(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        rows = self._checked_rows()
        targets = [
            self._list.item(r).data(Qt.UserRole)
            for r in rows
            if self._list.item(r).data(Qt.UserRole).pending_escalation.primer_kind
            == "sam3"
        ]
        if not targets:
            QMessageBox.information(
                self, "Re-threshold", "Check a staged SAM3 result first."
            )
            return
        current = float(
            targets[0].pending_escalation.primer_params.get("confidence", 0.35)
        )
        # The MINIMUM is the candidate cache's own floor: anything below it
        # is refused by rethreshold_staged, so offering it here would only
        # let the user pick an error message.
        minimum = rethreshold_floor_for(targets)
        value, ok = QInputDialog.getDouble(
            self,
            "Re-threshold",
            f"New confidence (cache floor {minimum:.2f}):",
            max(current, minimum),
            minimum,
            0.99,
            2,
        )
        if not ok:
            return
        for src in targets:
            merge_iou = float(
                src.pending_escalation.primer_params.get("merge_iou", 0.5)
            )
            try:
                kept = rethreshold_staged(src, confidence=value, merge_iou=merge_iou)
            except Exception as exc:
                QMessageBox.warning(self, "Re-threshold", str(exc))
                continue
            QMessageBox.information(
                self,
                "Re-threshold",
                f"{src.name}: {kept} instance(s) at confidence {value:.2f}.",
            )
