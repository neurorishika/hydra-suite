"""Dataset panel -- source management and image browser (left panel)."""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from ..evaluation import build_dataset_analysis_report
from ..utils import (
    ensure_detectkit_source_structure,
    list_images_in_source,
    source_class_id_map,
)

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


class DatasetPanel(QWidget):
    """Left panel: source selector, image browser, X-AnyLabeling launch."""

    manage_sources_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._main_window = None
        self._project: DetectKitProject | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel("Dataset Browser")
        header.setProperty("detectkitRole", "sectionTitle")
        layout.addWidget(header)

        intro = QLabel(
            "Manage source datasets, browse images, and round-trip annotations through X-AnyLabeling."
        )
        intro.setWordWrap(True)
        intro.setProperty("detectkitRole", "sectionHint")
        layout.addWidget(intro)

        sources_group = QGroupBox("Sources")
        sources_layout = QVBoxLayout(sources_group)
        sources_layout.setSpacing(8)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.source_combo.setMinimumContentsLength(18)
        self.source_combo.currentIndexChanged.connect(self._on_source_combo_changed)
        src_row.addWidget(self.source_combo, 1)
        sources_layout.addLayout(src_row)

        self.btn_manage_sources = QPushButton("Manage Sources…")
        self.btn_manage_sources.setProperty("detectkitVariant", "secondary")
        self.btn_manage_sources.clicked.connect(self.manage_sources_requested)
        sources_layout.addWidget(self.btn_manage_sources)

        self._source_summary = QLabel("No sources connected yet.")
        self._source_summary.setWordWrap(True)
        self._source_summary.setProperty("detectkitRole", "sectionHint")
        sources_layout.addWidget(self._source_summary)

        layout.addWidget(sources_group)

        images_group = QGroupBox("Images")
        images_layout = QVBoxLayout(images_group)
        images_layout.setSpacing(8)

        self._image_summary = QLabel("Select a source to browse its images.")
        self._image_summary.setWordWrap(True)
        self._image_summary.setProperty("detectkitRole", "sectionHint")
        images_layout.addWidget(self._image_summary)

        self.image_list = QListWidget()
        self.image_list.setAlternatingRowColors(True)
        self.image_list.setUniformItemSizes(True)
        self.image_list.currentRowChanged.connect(self._on_image_changed)
        images_layout.addWidget(self.image_list)

        layout.addWidget(images_group, 1)

        xany_group = QGroupBox("External Annotation")
        xany_layout = QVBoxLayout(xany_group)
        xany_layout.setSpacing(8)

        self._xal_hint = QLabel(
            "Open the current source in X-AnyLabeling, then refresh the source to pull edited labels back into DetectKit."
        )
        self._xal_hint.setWordWrap(True)
        self._xal_hint.setProperty("detectkitRole", "sectionHint")
        xany_layout.addWidget(self._xal_hint)

        env_row = QHBoxLayout()
        self.combo_xal_env = QComboBox()
        self.combo_xal_env.setToolTip("Conda environment with X-AnyLabeling installed.")
        self.btn_refresh_envs = QPushButton("⟳")
        self.btn_refresh_envs.setProperty("detectkitVariant", "quiet")
        self.btn_refresh_envs.setFixedWidth(30)
        self.btn_refresh_envs.setToolTip("Rescan conda environments")
        self.btn_refresh_envs.clicked.connect(self._refresh_xal_envs)
        env_row.addWidget(self.combo_xal_env, 1)
        env_row.addWidget(self.btn_refresh_envs)
        xany_layout.addLayout(env_row)

        xal_btn_row = QHBoxLayout()
        self.btn_xanylabeling = QPushButton("Open in X-AnyLabeling")
        self.btn_xanylabeling.clicked.connect(self._open_xanylabeling)
        self.btn_refresh = QPushButton("Refresh Labels")
        self.btn_refresh.setProperty("detectkitVariant", "secondary")
        self.btn_refresh.clicked.connect(self._refresh_labels)
        xal_btn_row.addWidget(self.btn_xanylabeling)
        xal_btn_row.addWidget(self.btn_refresh)
        xany_layout.addLayout(xal_btn_row)

        layout.addWidget(xany_group)

        analysis_group = QGroupBox("Dataset Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(8)

        analysis_hint = QLabel(
            "Inspect all connected sources together to catch class-mapping or crop-size issues before training."
        )
        analysis_hint.setWordWrap(True)
        analysis_hint.setProperty("detectkitRole", "sectionHint")
        analysis_layout.addWidget(analysis_hint)

        self.btn_analyze_dataset = QPushButton("Analyze Dataset")
        self.btn_analyze_dataset.clicked.connect(self._run_dataset_analysis)
        analysis_layout.addWidget(self.btn_analyze_dataset)

        self._analysis_view = QTextEdit()
        self._analysis_view.setReadOnly(True)
        self._analysis_view.setPlaceholderText(
            "Run dataset analysis to inspect merged source statistics and warnings."
        )
        self._analysis_view.setMinimumHeight(180)
        analysis_layout.addWidget(self._analysis_view)

        layout.addWidget(analysis_group)

        self._refresh_xal_envs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, proj: DetectKitProject, main_window) -> None:
        """Populate from project state."""
        self._project = proj
        self._main_window = main_window
        self.refresh_sources(proj)

        # Restore last selection
        if 0 <= proj.last_source_index < self.source_combo.count():
            self.source_combo.setCurrentIndex(proj.last_source_index)
        elif self.source_combo.count() > 0:
            self.source_combo.setCurrentIndex(0)

    def refresh_sources(self, proj: DetectKitProject) -> None:
        """Repopulate the source combo from *proj.sources*."""
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for src in proj.sources:
            display = src.name if src.name else src.path
            self.source_combo.addItem(display, userData=src.path)
        self.source_combo.blockSignals(False)
        source_count = len(proj.sources)
        if source_count == 0:
            self.image_list.clear()
            self._source_summary.setText("No sources connected yet.")
            self._image_summary.setText("Select a source to browse its images.")
            self._analysis_view.clear()
            return
        self._source_summary.setText(
            f"{source_count} source(s) available for browsing and export."
        )
        if self.source_combo.count() > 0:
            self._on_source_combo_changed(self.source_combo.currentIndex())

    def collect_state(self, proj: DetectKitProject) -> None:
        """Write panel state back into the project."""
        proj.last_source_index = max(self.source_combo.currentIndex(), 0)

    def navigate_prev(self) -> None:
        """Navigate to the previous image in the current source."""
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)

    def navigate_next(self) -> None:
        """Navigate to the next image in the current source."""
        row = self.image_list.currentRow()
        if row < self.image_list.count() - 1:
            self.image_list.setCurrentRow(row + 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _selected_source_path(self) -> str | None:
        """Return the path of the currently selected source, or None."""
        idx = self.source_combo.currentIndex()
        if idx < 0:
            return None
        return self.source_combo.itemData(idx)

    def _get_multiple_dirs(self, title: str) -> list[str]:
        """Open a non-native file dialog that allows multi-directory selection."""
        dlg = QFileDialog(self, title)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        for view in dlg.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for view in dlg.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if dlg.exec() != QFileDialog.Accepted:
            return []
        return dlg.selectedFiles()

    def _selected_xal_env(self) -> str | None:
        """Return the selected X-AnyLabeling conda env name, or None."""
        env = self.combo_xal_env.currentText().strip()
        if (
            not env
            or env.startswith("No ")
            or env.startswith("Conda ")
            or env.startswith("Error")
        ):
            return None
        return env

    def _refresh_xal_envs(self) -> None:
        """Scan for conda environments starting with 'x-anylabeling-'."""
        self.combo_xal_env.clear()
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                envs = []
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if parts and parts[0].startswith("x-anylabeling"):
                        envs.append(parts[0])
                if envs:
                    self.combo_xal_env.addItems(envs)
                    self.btn_xanylabeling.setEnabled(True)
                    self._xal_hint.setText(
                        "Open the selected source in X-AnyLabeling, then refresh labels when you return to DetectKit."
                    )
                    logger.info("Found %d X-AnyLabeling conda env(s)", len(envs))
                else:
                    self.combo_xal_env.addItem("No X-AnyLabeling envs found")
                    self.btn_xanylabeling.setEnabled(False)
                    self._xal_hint.setText(
                        "No X-AnyLabeling environment was detected. Create one to enable external annotation round-trips."
                    )
                    logger.warning(
                        "No conda envs starting with 'x-anylabeling-' found. "
                        "Create one: conda create -n x-anylabeling-cpu python=3.10 "
                        "&& conda activate x-anylabeling-cpu && pip install x-anylabeling"
                    )
            else:
                self.combo_xal_env.addItem("Conda not available")
                self.btn_xanylabeling.setEnabled(False)
                self._xal_hint.setText(
                    "Conda is not available in this environment, so X-AnyLabeling launch is disabled."
                )
        except FileNotFoundError:
            self.combo_xal_env.addItem("Conda not installed")
            self.btn_xanylabeling.setEnabled(False)
            self._xal_hint.setText(
                "Conda is not installed, so X-AnyLabeling launch is disabled."
            )
        except Exception as exc:
            self.combo_xal_env.addItem("Error detecting envs")
            self.btn_xanylabeling.setEnabled(False)
            self._xal_hint.setText(
                "DetectKit could not scan conda environments for X-AnyLabeling."
            )
            logger.warning("Failed to scan conda envs: %s", exc)

    def _validate_source(self, path: str) -> None:
        """Validate a candidate DetectKit source against the current project scheme."""
        source_dir = ensure_detectkit_source_structure(path)
        try:
            from hydra_suite.training.dataset_inspector import (
                inspect_obb_or_detect_dataset,
            )

            inspect_obb_or_detect_dataset(source_dir)
        except Exception:
            self._try_xlabel_convert(path)
            inspect_obb_or_detect_dataset(source_dir)

        if self._project is not None:
            source_class_id_map(source_dir, self._project.class_names)

    def _try_xlabel_convert(self, path: str) -> None:
        """Attempt to convert xlabel JSON labels to YOLO format via conda env."""
        env = self._selected_xal_env()
        try:
            from hydra_suite.integrations.xanylabeling.cli import convert_project

            ok, msg = convert_project(path, path, conda_env=env)
            if ok:
                logger.info("Auto-converted xlabel labels in %s: %s", path, msg)
            else:
                logger.debug("xlabel conversion not applicable for %s: %s", path, msg)
        except Exception:
            logger.debug("xlabel conversion failed for %s", path, exc_info=True)

    def _run_dataset_analysis(self) -> None:
        """Analyze all configured sources and display the merged report."""
        if self._project is None:
            self._analysis_view.setPlainText("No dataset sources configured.")
            return

        report, warnings = build_dataset_analysis_report(self._project)
        self._analysis_view.setPlainText(report)
        if warnings:
            QMessageBox.warning(self, "Dataset Analysis Warnings", "\n".join(warnings))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_source_combo_changed(self, index: int) -> None:
        """Populate image list when source combo selection changes."""
        self.image_list.clear()
        if index < 0:
            self._image_summary.setText("Select a source to browse its images.")
            return
        source_path = self.source_combo.itemData(index)
        if not source_path:
            self._image_summary.setText("Select a source to browse its images.")
            return
        images = list_images_in_source(source_path)
        self.image_list.blockSignals(True)
        for img in images:
            img_item = QListWidgetItem(img.name)
            img_item.setData(Qt.UserRole, str(img))
            self.image_list.addItem(img_item)
        self.image_list.blockSignals(False)
        source_name = self.source_combo.currentText().strip() or "Selected source"
        image_count = len(images)
        self._image_summary.setText(
            f"{source_name}: {image_count} image(s) available for review."
        )
        if self._main_window is not None and hasattr(self._main_window, "_tools_panel"):
            self._main_window._tools_panel.set_image_counter(
                1 if image_count else 0,
                image_count,
            )
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)

    def _on_image_changed(self, row: int) -> None:
        """Show selected image in canvas."""
        if row < 0 or self._main_window is None:
            if self._main_window is not None and hasattr(
                self._main_window, "_tools_panel"
            ):
                self._main_window._tools_panel.set_image_counter(
                    0, self.image_list.count()
                )
            return
        img_item = self.image_list.item(row)
        if img_item is None:
            return
        source_path = self._selected_source_path()
        if source_path is None:
            return
        image_path = img_item.data(Qt.UserRole)
        if hasattr(self._main_window, "_tools_panel"):
            self._main_window._tools_panel.set_image_counter(
                row + 1,
                self.image_list.count(),
            )
        self._main_window.show_image(source_path, str(image_path))

    def _open_xanylabeling(self) -> None:
        """Launch X-AnyLabeling for the selected source in the selected conda env."""
        source_path = self._selected_source_path()
        if source_path is None:
            QMessageBox.information(self, "No Source", "Select a source first.")
            return

        env = self._selected_xal_env()
        if env is None:
            QMessageBox.warning(
                self,
                "No Environment",
                "Select a valid conda environment with X-AnyLabeling installed.\n\n"
                "Create one with:\n"
                "  conda create -n x-anylabeling-cpu python=3.10\n"
                "  conda activate x-anylabeling-cpu\n"
                "  pip install x-anylabeling",
            )
            return

        source_dir = Path(source_path)
        try:
            self._validate_source(str(source_dir))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Source", str(exc))
            return

        # Build the shell command: activate conda env, convert yolo->xlabel, open GUI
        convert_cmd = (
            "xanylabeling convert --task yolo2xlabel --mode obb "
            "--images ./images --labels ./labels --output ./images "
            "--classes classes.txt"
        )
        open_cmd = "xanylabeling --filename ./images"
        full_cmd = f"{convert_cmd} && {open_cmd}"

        system = platform.system()
        try:
            if system == "Darwin":
                # macOS: open Terminal with conda activation via AppleScript
                script = (
                    'tell application "Terminal"\n'
                    "    activate\n"
                    '    do script "source $(conda info --base)/etc/profile.d/conda.sh '
                    f"&& conda activate {env} "
                    f"&& cd '{source_dir}' "
                    f'&& {full_cmd}"\n'
                    "end tell"
                )
                subprocess.Popen(["osascript", "-e", script])
            elif system == "Windows":
                cmd = (
                    f'start cmd /k "conda activate {env} '
                    f"&& cd /d {source_dir} "
                    f'&& {full_cmd}"'
                )
                subprocess.Popen(cmd, shell=True)  # noqa: S602
            else:
                # Linux: try common terminal emulators
                shell_cmd = (
                    f"source $(conda info --base)/etc/profile.d/conda.sh "
                    f"&& conda activate {env} "
                    f"&& cd '{source_dir}' "
                    f"&& {full_cmd}"
                )
                for term_cmd in [
                    ["gnome-terminal", "--", "bash", "-c", shell_cmd],
                    ["konsole", "-e", "bash", "-c", shell_cmd],
                    ["xterm", "-e", "bash", "-c", shell_cmd],
                ]:
                    try:
                        subprocess.Popen(term_cmd)
                        break
                    except FileNotFoundError:
                        continue
                else:
                    QMessageBox.warning(
                        self,
                        "No Terminal",
                        "Could not find a terminal emulator "
                        "(gnome-terminal, konsole, or xterm).",
                    )
        except Exception as exc:
            QMessageBox.warning(
                self, "Launch Error", f"Failed to open X-AnyLabeling:\n{exc}"
            )

    def _refresh_labels(self) -> None:
        """Convert xlabel JSONs to YOLO labels, then refresh image list.

        Always attempts xlabel→YOLO conversion first (in case the user
        edited annotations in X-AnyLabeling), then re-validates.
        """
        source_path = self._selected_source_path()
        if source_path is None:
            return
        self._try_xlabel_convert(source_path)
        self._validate_source(source_path)
        # Refresh image list
        self._on_source_combo_changed(self.source_combo.currentIndex())
