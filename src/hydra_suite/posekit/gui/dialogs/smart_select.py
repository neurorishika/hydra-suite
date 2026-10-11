"""Smart Select: pick the next frames to label, the way ClassKit picks candidates.

Embeddings of every frame are clustered (K-means, as in ClassKit) and a batch
is built from a mix of signals — frames that look like known model errors,
frames the model is unsure about, rare clusters, gaps the labeled set leaves,
typical under-labeled frames and random checks (see
:mod:`hydra_suite.posekit.core.selection`).  The result is an ordered candidate
list; *Start Labeling* walks it in that order.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QSize, QThread
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame
from hydra_suite.posekit.core.selection import (  # noqa: F401  (re-exported)
    REASON_AUDIT,
    REASON_DIVERSITY,
    REASON_ERROR,
    REASON_RARE,
    REASON_REPRESENTATIVE,
    REASON_UNCERTAINTY,
)
from hydra_suite.posekit.core.smart_select import (
    SmartSelectOutcome,
    SmartSelectRequest,
)
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog

from ...core.extensions import EmbeddingWorker
from ..panels import smart_select_candidates as candidates_panel
from ..panels import smart_select_options as options_panel
from ..panels.smart_select_candidates import (  # noqa: F401  (re-exported)
    REASON_EXPLORER,
    SmartSelectCandidatesPanel,
)
from ..panels.smart_select_options import (  # noqa: F401  (re-exported)
    CLUSTER_METHODS,
    EMBEDDING_MODELS,
    SCOPES,
    SmartSelectOptionsPanel,
)
from ..workers import SmartSelectWorker
from .utils import (
    _load_dialog_settings,
    _save_dialog_settings,
    get_available_devices,
    keep_thread_until_finished,
)

logger = logging.getLogger("pose_label.dialogs.smart_select")


@dataclass
class PredictionSource:
    """Cached pose predictions of the main window's current model."""

    label: str  # e.g. "best.pt (yolo)"
    predictions: Dict[str, list]


class SmartSelectDialog(QDialog):
    """Build an ordered batch of frames to label next."""

    def __init__(
        self,
        parent,
        project,
        image_paths: List[Path],
        is_labeled_fn: Callable[[Path], bool],
        frame_mode: bool = False,
        source_ids: Optional[List[Any]] = None,
        labeling_frames: Optional[set] = None,
        label_path_fn: Optional[Callable[[Path], Path]] = None,
        prediction_source: Optional[PredictionSource] = None,
        no_prediction_reason: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Smart Select")
        self.setMinimumSize(QSize(980, 720))
        self.project = project
        self.image_paths = list(image_paths)
        self.is_labeled_fn = is_labeled_fn or (lambda p: False)
        self.frame_mode = bool(frame_mode)
        self.labeling_frames = set(
            labeling_frames
            if labeling_frames is not None
            else getattr(parent, "labeling_frames", set())
        )
        self.label_path_fn = label_path_fn
        self.prediction_source = prediction_source
        self.no_prediction_reason = no_prediction_reason
        self._source_ids = (
            list(source_ids) if source_ids is not None else [None] * len(image_paths)
        )
        self._frame_groups = group_indices_by_frame(
            [p.name for p in self.image_paths], self._source_ids
        )
        self._idx_to_frame_key = {
            i: key for key, idxs in self._frame_groups.items() for i in idxs
        }

        self._emb_all: Optional[np.ndarray] = None
        self._emb_key: Optional[tuple] = None
        self._eligible_indices: List[int] = []
        self._outcome: Optional[SmartSelectOutcome] = None
        self.batch_units: List[int] = []
        self._unit_reason: Dict[int, str] = {}
        self._emb_thread: Optional[QThread] = None
        self._emb_worker: Optional[EmbeddingWorker] = None
        self._sel_worker: Optional[SmartSelectWorker] = None
        self._busy = False
        self._cancel_requested = False
        self._closed = False
        self._unreadable: set = set()
        self._last_labeled = np.zeros(len(self.image_paths), dtype=bool)
        self._last_candidates = np.zeros(len(self.image_paths), dtype=bool)

        # Results read by the caller after accept().
        self.selected_indices: List[int] = []
        self.start_labeling = False
        self._did_add = False

        self._build_ui()
        self.options.apply_settings(_load_dialog_settings("smart_select"))
        self.options.show_prediction_source(
            prediction_source, no_prediction_reason, self.image_paths
        )
        self._set_busy(False)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        if self.frame_mode:
            grouped = any(len(m) > 1 for m in self._frame_groups.values())
            banner = QLabel(
                "Frame Mode is on — Smart Select picks whole source frames; "
                "every individual on a picked frame is added."
                if grouped
                else "Frame Mode is on, but these file names don't say which "
                "source frame each image came from (Frame Mode reads "
                "did<detection id> crop names), so every image counts as its "
                "own frame."
            )
            banner.setWordWrap(True)
            banner.setStyleSheet("font-weight: bold;")
            layout.addWidget(banner)

        self.options = SmartSelectOptionsPanel(
            unit_count=(
                len(self._frame_groups) if self.frame_mode else len(self.image_paths)
            ),
            frame_mode=self.frame_mode,
            enhance_default=bool(getattr(self.project, "enhance_enabled", False)),
            devices=get_available_devices(),
        )
        layout.addWidget(self.options)
        for name in options_panel.WIDGET_NAMES:
            setattr(self, name, getattr(self.options, name))

        build_row = QHBoxLayout()
        self.btn_build = QPushButton("Build Batch")
        self.btn_build.setDefault(True)
        self.btn_build.setMinimumHeight(32)
        self.btn_build.setToolTip(
            "Compute (or load cached) embeddings, cluster every frame and pick "
            "the batch."
        )
        self.btn_cancel = QPushButton("Cancel")
        build_row.addWidget(self.btn_build, 1)
        build_row.addWidget(self.btn_cancel)
        layout.addLayout(build_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.candidates = SmartSelectCandidatesPanel(self.image_paths, self.frame_mode)
        layout.addWidget(self.candidates, 1)
        for name in candidates_panel.WIDGET_NAMES:
            setattr(self, name, getattr(self.candidates, name))

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_close = QPushButton("Close")
        self.btn_add = QPushButton("Add to Labeling Set")
        self.btn_add.setToolTip("Add the batch to the labeling set and close.")
        self.btn_start = QPushButton("Start Labeling")
        self.btn_start.setToolTip(
            "Add the batch to the labeling set, sort the list in batch order "
            "and open the first frame."
        )
        self.btn_start.setMinimumHeight(32)
        self.btn_start.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; font-weight:bold; "
            "padding: 4px 16px; } QPushButton:disabled { background:#3a3a3a; "
            "color:#777; }"
        )
        bottom.addWidget(self.btn_close)
        bottom.addWidget(self.btn_add)
        bottom.addWidget(self.btn_start)
        layout.addLayout(bottom)

        self.btn_build.clicked.connect(self._build)
        self.btn_cancel.clicked.connect(self._cancel)
        self.candidates.remove_requested.connect(self._remove_selected_rows)
        self.btn_explorer.clicked.connect(self._open_explorer)
        self.btn_save_csv.clicked.connect(self._save_csv)
        self.btn_close.clicked.connect(self.reject)
        self.btn_add.clicked.connect(lambda: self._finish(start=False))
        self.btn_start.clicked.connect(lambda: self._finish(start=True))

    def _set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.btn_build.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        has_batch = bool(self.batch_units) and not busy
        self.btn_add.setEnabled(has_batch)
        self.btn_start.setEnabled(has_batch)
        self.btn_remove.setEnabled(has_batch)
        self.btn_explorer.setEnabled(self._outcome is not None and not busy)
        self.btn_save_csv.setEnabled(self._outcome is not None and not busy)

    def _embedding_key(self) -> tuple:
        return self.options.embedding_key()

    def done(self, result: int) -> None:  # accept, reject and Esc all land here
        """Stop background work and remember the settings, however it closes."""
        # Signals already queued by a worker are still delivered after its
        # connections are dropped; every handler checks this first.
        self._closed = True
        self._stop_workers()
        try:
            _save_dialog_settings("smart_select", self.options.settings())
        except Exception:  # settings are a convenience; never block closing
            logger.debug("Could not save Smart Select settings", exc_info=True)
        super().done(result)

    def closeEvent(self, event) -> None:
        """Route the title-bar close through ``done`` (stops workers, saves)."""
        event.ignore()
        self.reject()

    # ── eligibility ───────────────────────────────────────────────────────

    def _build_eligible_indices(self) -> List[int]:
        scope = self.cb_scope.currentText()
        if scope == "All frames":
            idxs = list(range(len(self.image_paths)))
        elif scope == "Labeling set":
            idxs = sorted(
                i for i in self.labeling_frames if 0 <= i < len(self.image_paths)
            )
        else:
            idxs = [
                i for i, p in enumerate(self.image_paths) if not self.is_labeled_fn(p)
            ]
        if self.cb_exclude_in_labeling.isChecked() and scope != "Labeling set":
            idxs = [i for i in idxs if i not in self.labeling_frames]
        return idxs

    # ── build: embeddings, then the selection job ─────────────────────────

    def _build(self) -> None:
        if self._busy or not self.image_paths:
            return
        self._cancel_requested = False
        eligible = self._build_eligible_indices()
        if not eligible:
            QMessageBox.information(
                self, "Nothing to pick", "No frames match the 'Pick from' choice."
            )
            return
        self._eligible_indices = eligible
        if self._emb_all is not None and self._emb_key == self._embedding_key():
            self._start_selection()
            return
        self._compute_embeddings()

    def _compute_embeddings(self) -> None:
        self._set_busy(True)
        self.lbl_status.setText("Loading embeddings…")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._emb_thread = QThread(self)
        self._emb_worker = EmbeddingWorker(
            image_paths=self.image_paths,
            eligible_indices=self._eligible_indices,
            cache_dir=Path(self.project.out_root) / "posekit",
            model_name=self.model_combo.currentText().strip(),
            device_pref=self.dev_combo.currentText(),
            batch_size=int(self.batch_spin.value()),
            use_enhance=bool(self.cb_use_enhance.isChecked()),
            max_side=int(self.max_side_spin.value()),
            cache_ok=True,
        )
        self._emb_key_pending = self._embedding_key()
        self._emb_worker.moveToThread(self._emb_thread)
        self._emb_thread.started.connect(self._emb_worker.run)
        self._emb_worker.progress.connect(self._on_emb_progress)
        self._emb_worker.status.connect(self.lbl_status.setText)
        self._emb_worker.failed.connect(self._on_emb_failed)
        self._emb_worker.finished.connect(self._on_emb_finished)
        self._emb_thread.start()

    def _on_emb_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.progress.setFormat(f"Embedding %v / {max(1, total)}")

    def _end_emb_thread(self) -> None:
        thread = self._emb_thread
        if thread is not None:
            thread.quit()
            if thread.wait(3000):
                thread.deleteLater()
            else:
                keep_thread_until_finished(thread, self._emb_worker)
        self._emb_thread = None
        self._emb_worker = None

    def _is_current(self, worker) -> bool:
        """Only the live worker's signals count, and none once the dialog closed.

        A queued signal delivered after its connection was dropped arrives
        with ``sender()`` None; that is only trusted while the dialog is open.
        """
        if self._closed:
            return False
        sender = self.sender()
        return sender is None or sender is worker

    def _on_emb_failed(self, msg: str) -> None:
        if not self._is_current(self._emb_worker):
            return
        self._end_emb_thread()
        self._set_busy(False)
        self.progress.setFormat("%p%")
        self.progress.setValue(0)
        if msg == "Canceled.":
            self.lbl_status.setText("Canceled.")
            return
        self.lbl_status.setText("Embedding failed.")
        QMessageBox.critical(self, "Embedding failed", msg)

    def _on_emb_finished(self, emb_all, _eligible, meta) -> None:
        if not self._is_current(self._emb_worker):
            return
        self._end_emb_thread()
        self._remove_legacy_embedding_cache()
        emb_all = np.asarray(emb_all)
        if emb_all.ndim != 2 or len(emb_all) != len(self.image_paths):
            self._set_busy(False)
            QMessageBox.critical(
                self, "Embedding failed", "Embeddings do not match the frame list."
            )
            return
        unreadable = set(int(i) for i in (meta or {}).get("unreadable", []))
        if emb_all.shape[1] == 0 or len(unreadable) >= len(self.image_paths):
            self._set_busy(False)
            self.progress.setValue(0)
            self.lbl_status.setText("None of the images could be read.")
            QMessageBox.critical(
                self,
                "Smart Select",
                "None of the images could be read, so there is nothing to embed. "
                "Check that the image files exist and open.",
            )
            return
        self._emb_all = emb_all
        self._emb_key = self._emb_key_pending
        self._unreadable = unreadable
        self.progress.setFormat("%p%")
        if self._cancel_requested:
            # Canceled after the last check inside the worker: keep the
            # embeddings (the next Build reuses them) but stop here.
            self.progress.setValue(0)
            self.lbl_status.setText("Canceled.")
            self._set_busy(False)
            return
        self._start_selection()

    def _remove_legacy_embedding_cache(self) -> None:
        """Drop caches written by the previous Smart Select.

        They sat one folder deeper (``posekit/embeddings/embeddings``) under
        a key that included device and batch size, so nothing reads or
        cleans them any more.
        """
        legacy = Path(self.project.out_root) / "posekit" / "embeddings" / "embeddings"
        if not legacy.is_dir():
            return
        import shutil

        try:
            shutil.rmtree(legacy)
            logger.info("Removed the old Smart Select embedding cache: %s", legacy)
        except OSError:
            logger.debug("Could not remove %s", legacy, exc_info=True)

    def _start_selection(self) -> None:
        n = len(self.image_paths)
        labeled = np.array([bool(self.is_labeled_fn(p)) for p in self.image_paths])
        cand = np.zeros(n, dtype=bool)
        cand[self._eligible_indices] = True
        unusable = np.zeros(n, dtype=bool)
        if self._unreadable:  # no embedding: never clustered or picked
            unusable[sorted(self._unreadable)] = True
        self._last_labeled = labeled
        self._last_candidates = cand
        preds = self.prediction_source.predictions if self.prediction_source else None
        label_paths: Dict[int, Path] = {}
        if preds and self.label_path_fn is not None:
            for i in np.where(labeled)[0]:
                try:
                    label_paths[int(i)] = Path(self.label_path_fn(self.image_paths[i]))
                except Exception:
                    continue
        cfg = self.options.batch_config()
        unit_keys = (
            [self._idx_to_frame_key.get(i, ("_idx", i)) for i in range(n)]
            if self.frame_mode
            else None
        )
        request = SmartSelectRequest(
            embeddings=self._emb_all,
            image_paths=self.image_paths,
            candidate_mask=cand,
            labeled_mask=labeled,
            config=cfg,
            unit_keys=unit_keys,
            n_clusters=self.options.n_clusters(),
            cluster_method=self.options.cluster_method(),
            predictions=preds,
            label_paths=label_paths,
            num_keypoints=len(getattr(self.project, "keypoint_names", []) or []),
            unusable_mask=unusable,
        )
        self._set_busy(True)
        self.progress.setRange(0, 0)  # busy
        self._sel_worker = SmartSelectWorker(request, self)
        self._sel_worker.status.connect(self.lbl_status.setText)
        self._sel_worker.result.connect(self._on_selection_ready)
        self._sel_worker.error.connect(self._on_selection_failed)
        self._sel_worker.canceled.connect(self._on_selection_canceled)
        self._sel_worker.finished.connect(self._on_selection_thread_done)
        self._sel_worker.start()

    def _on_selection_thread_done(self) -> None:
        worker = self.sender()
        current = worker is None or worker is self._sel_worker
        if worker is not None:
            worker.deleteLater()
        if not current:
            return  # an older run; the current one still owns the UI
        self._sel_worker = None
        if self._emb_thread is None:
            self.progress.setRange(0, 100)
            self._set_busy(False)

    def _on_selection_failed(self, msg: str) -> None:
        if not self._is_current(self._sel_worker):
            return
        self.lbl_status.setText("Smart Select failed.")
        QMessageBox.critical(self, "Smart Select failed", msg)

    def _on_selection_canceled(self) -> None:
        if not self._is_current(self._sel_worker):
            return
        self.lbl_status.setText("Canceled.")

    def _on_selection_ready(self, outcome: SmartSelectOutcome) -> None:
        if not self._is_current(self._sel_worker):
            return
        if self._cancel_requested:  # canceled after the worker's last check
            self.lbl_status.setText("Canceled.")
            return
        self._outcome = outcome
        self.batch_units = list(outcome.result.indices)
        self._unit_reason = dict(outcome.result.reasons)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self._autosave_clusters()
        self._refresh_table()
        info = outcome.result.info
        parts = [
            f"{len(outcome.units):,} {'frames' if self.frame_mode else 'images'} "
            f"in {outcome.n_clusters} clusters"
        ]
        if outcome.n_predicted:
            inst = "individuals" if self.frame_mode else "images"
            unit = "frames" if self.frame_mode else "images"
            parts.append(f"{outcome.n_predicted:,} {inst} with predictions")
            parts.append(
                f"{outcome.n_error_scored:,} labeled {inst} compared to the "
                f"model, {int(info.get('n_errors', 0)):,} {unit} above the "
                "error threshold"
            )
        if info.get("suppressed"):
            parts.append(f"{int(info['suppressed']):,} near-duplicates skipped")
        if self._unreadable:
            parts.append(f"{len(self._unreadable):,} unreadable images skipped")
        self.lbl_status.setText("Done — " + "; ".join(parts) + ".")

    def _cancel(self) -> None:
        if self._busy:
            self._cancel_requested = True
            self.lbl_status.setText("Canceling…")
        if self._emb_worker is not None:
            self._emb_worker.cancel()
        if self._sel_worker is not None:
            self._sel_worker.cancel()

    def _stop_workers(self) -> None:
        """Cancel and detach background work so the dialog can close at once."""
        self._cancel()
        if self._emb_worker is not None:
            for sig in (
                self._emb_worker.finished,
                self._emb_worker.failed,
                self._emb_worker.progress,
                self._emb_worker.status,
            ):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
        if self._emb_thread is not None:
            thread, worker = self._emb_thread, self._emb_worker
            # The worker stops at its next batch; the thread's event loop
            # then ends.  Keep both alive until it has.
            thread.quit()
            if not thread.wait(200):
                keep_thread_until_finished(thread, worker, cancel=worker.cancel)
            self._emb_thread = None
            self._emb_worker = None
        if self._sel_worker is not None:
            worker = self._sel_worker
            for sig in (
                worker.result,
                worker.error,
                worker.canceled,
                worker.status,
                worker.finished,
            ):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            if not worker.wait(200):
                keep_thread_until_finished(worker, cancel=worker.cancel)
            self._sel_worker = None

    # ── table ─────────────────────────────────────────────────────────────

    def _unit_path(self, unit: int) -> Path:
        return self.candidates.unit_path(unit)

    def _refresh_table(self) -> None:
        self.candidates.show_batch(
            self._outcome,
            self.batch_units,
            self._unit_reason,
            self._unreadable,
            self._last_labeled,
        )
        self._set_busy(self._busy)

    def _remove_selected_rows(self) -> None:
        if self._busy:
            return
        rows = self.candidates.selected_rows()
        if not rows:
            return
        drop = {self.batch_units[r] for r in rows}
        self.batch_units = [u for u in self.batch_units if u not in drop]
        self._refresh_table()
        if self.batch_units:
            self.table.selectRow(min(rows[0], len(self.batch_units) - 1))

    # ── explorer / CSV ────────────────────────────────────────────────────

    def _open_explorer(self) -> None:
        if self._busy or self._outcome is None:
            return
        from .exploration import EmbeddingExplorerDialog

        out = self._outcome
        # Only frames with an embedding are shown (unreadable ones have none).
        shown = [int(u) for u in np.flatnonzero(out.usable_units)]
        pos = {u: k for k, u in enumerate(shown)}
        dialog = EmbeddingExplorerDialog(
            embeddings=out.unit_embeddings[shown],
            image_paths=[self._unit_path(u) for u in shown],
            cluster_ids=out.unit_clusters[shown],
            labeled_mask=[bool(out.unit_labeled[u]) for u in shown],
            initial_selection=[pos[u] for u in self.batch_units if u in pos],
            accept_text="Use Selection as Batch",
            parent=self,
        )
        try:
            accepted = dialog.exec() == QDialog.Accepted
            chosen = list(dict.fromkeys(shown[int(i)] for i in dialog.selected_indices))
        finally:
            # Each Explorer holds a full scatter scene; don't keep it around.
            if hasattr(dialog, "deleteLater"):
                dialog.deleteLater()
        if not accepted:
            return
        chosen_set = set(chosen)
        kept = [u for u in self.batch_units if u in chosen_set]
        new = [u for u in chosen if u not in set(kept)]
        # Only frames the "Pick from" choice allows (e.g. not labeled ones
        # under "Unlabeled only").
        added = [u for u in new if self._last_candidates[out.units[u]].any()]
        if len(added) < len(new):
            self.lbl_status.setText(
                f"Skipped {len(new) - len(added)} point(s) outside "
                f"'{self.cb_scope.currentText()}'."
            )
        for u in added:
            self._unit_reason[u] = REASON_EXPLORER
        self.batch_units = kept + added
        self._refresh_table()

    def _write_clusters_csv(self, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "cluster_id"])
            for i, path in enumerate(self.image_paths):
                w.writerow([str(path), int(self._outcome.instance_clusters[i])])

    def _autosave_clusters(self) -> None:
        """Keep ``clusters.csv`` (cluster sort, cluster-stratified split) current."""
        if self._outcome is None:
            return
        try:
            self._write_clusters_csv(
                Path(self.project.out_root) / "posekit" / "clusters" / "clusters.csv"
            )
        except OSError:
            logger.warning("Could not write clusters.csv", exc_info=True)

    def _save_csv(self) -> None:
        if self._outcome is None:
            return
        default = Path(self.project.out_root) / "posekit" / "clusters" / "clusters.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save cluster assignments", str(default), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._write_clusters_csv(Path(path))
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Wrote cluster CSV:\n{path}")

    # ── finish ────────────────────────────────────────────────────────────

    def batch_instances(self) -> List[int]:
        """Instance indices of the batch in order (every individual of a frame).

        Unreadable individuals of a picked frame are left out.
        """
        if self._outcome is None:
            return []
        out: List[int] = []
        seen = set()
        for u in self.batch_units:
            for i in self._outcome.units[u]:
                if i in self._unreadable:
                    continue
                if i not in seen:
                    seen.add(i)
                    out.append(i)
        return out

    def _finish(self, start: bool) -> None:
        if self._busy:
            return
        self.selected_indices = self.batch_instances()
        if not self.selected_indices:
            return
        self.start_labeling = bool(start)
        self._did_add = True
        self.accept()
