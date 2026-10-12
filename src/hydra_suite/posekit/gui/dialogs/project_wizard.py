"""New-project and project-settings wizard dialogs for PoseKit."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.pose_flip import normalize_flip_pairs
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.widgets.dialogs import BaseDialog

from ..constants import DEFAULT_DATASET_IMAGES_DIR, DEFAULT_POSEKIT_PROJECT_DIR
from ..models import Project
from ..project import default_project_parent_dir
from ..utils import get_default_skeleton_dir
from .skeleton import SkeletonEditorDialog

# ─────────────────────────────────────────────────────────────────────────────
# New-project wizard  (standalone: project location + sources chosen freely)
# ─────────────────────────────────────────────────────────────────────────────


class NewProjectDialog(BaseDialog):
    """
    Guides the user through creating a brand-new PoseKit project:
      1. Choose where to save the project (any folder — independent of datasets)
      2. Configure pose classes, keypoints and skeleton
    3. Create the project and manage dataset sources from the workspace
    """

    def __init__(self, parent=None) -> None:
        super().__init__("Create New Project", parent=parent)
        self.setMinimumSize(QSize(820, 600))

        self._parent_dir: Optional[Path] = default_project_parent_dir()
        self._edges: List[Tuple[int, int]] = []
        self._flip_pairs: List[Tuple[int, int]] = []
        self._kpt_names: List[str] = ["kp1", "kp2"]

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(16)

        header = QLabel("<h2 style='margin:0;'>Create New Project</h2>")
        intro = QLabel(
            "Set the project folder first, then define annotation defaults. Dataset sources are managed from the workspace after creation."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(header)
        layout.addWidget(intro)

        # ── 1. Project location ─────────────────────────────────────────
        loc_group = QGroupBox("Project Details")
        loc_layout = QVBoxLayout(loc_group)
        loc_layout.setSpacing(10)
        loc_form = QFormLayout()

        parent_row = QHBoxLayout()
        self._le_parent = QLineEdit()
        self._le_parent.setReadOnly(True)
        self._le_parent.setPlaceholderText("Choose a parent folder…")
        self._le_parent.setText(str(self._parent_dir))
        btn_loc = QPushButton("Browse…")
        btn_loc.clicked.connect(self._pick_location)
        parent_row.addWidget(self._le_parent, 1)
        parent_row.addWidget(btn_loc)
        loc_form.addRow("Parent folder", parent_row)

        self._le_name = QLineEdit()
        self._le_name.setPlaceholderText("e.g. my_ants_project")
        self._le_name.textChanged.connect(self._on_location_changed)
        loc_form.addRow("Project name", self._le_name)
        loc_layout.addLayout(loc_form)

        preview_label = QLabel("Project Folder")
        preview_label.setStyleSheet("color: #cfcfcf;")
        loc_layout.addWidget(preview_label)

        self._lbl_preview = QLabel()
        self._lbl_preview.setWordWrap(True)
        self._lbl_preview.setStyleSheet(
            "padding: 10px; background-color: #252526; border-radius: 6px; "
            "border-left: 3px solid #0e639c;"
        )
        loc_layout.addWidget(self._lbl_preview)

        layout.addWidget(loc_group)

        # ── 2. Classes + Skeleton ───────────────────────────────────────
        setup_group = QGroupBox("Annotation Setup")
        setup_layout = QVBoxLayout(setup_group)
        setup_layout.setSpacing(12)

        classes_help = QLabel(
            "Add one pose class per line, then confirm the keypoint names and skeleton used by the project."
        )
        classes_help.setWordWrap(True)
        classes_help.setStyleSheet("color: #aaaaaa;")
        setup_layout.addWidget(classes_help)

        self._classes_edit = QPlainTextEdit("object")
        self._classes_edit.setMaximumHeight(72)
        setup_layout.addWidget(self._classes_edit)

        skel_group = QGroupBox("Keypoints & Skeleton")
        skel_layout = QHBoxLayout(skel_group)
        self._skel_lbl = QLabel(self._get_skeleton_summary())
        self._skel_lbl.setWordWrap(True)
        btn_skel = QPushButton("Edit Keypoints & Skeleton…")
        btn_skel.clicked.connect(self._edit_skeleton)
        skel_layout.addWidget(self._skel_lbl, 1)
        skel_layout.addWidget(btn_skel)
        setup_layout.addWidget(skel_group)
        layout.addWidget(setup_group)

        # ── 4. Options ──────────────────────────────────────────────────
        opt_group = QGroupBox("Options")
        opt_form = QFormLayout(opt_group)
        self._autosave_cb = QCheckBox("Autosave when changing frames")
        self._autosave_cb.setChecked(True)
        opt_form.addRow("", self._autosave_cb)
        self._pad_spin = QDoubleSpinBox()
        self._pad_spin.setRange(0.0, 0.25)
        self._pad_spin.setSingleStep(0.01)
        self._pad_spin.setValue(0.03)
        opt_form.addRow("Keypoint → bbox padding", self._pad_spin)
        layout.addWidget(opt_group)

        info = QLabel(
            "PoseKit creates a standalone project folder with labels and annotation defaults at the selected location. Use Source Manager after creation to add dataset sources."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding: 10px; background-color: #252526; border-radius: 6px; "
            "border-left: 3px solid #0e639c; color: #aaaaaa;"
        )
        layout.addWidget(info)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.add_content(scroll)

        self._buttons.accepted.disconnect()
        self._buttons.accepted.connect(self._do_accept)
        create_button = self._buttons.button(QDialogButtonBox.Ok)
        create_button.setText("Create Project")
        create_button.setDefault(True)

        self._on_location_changed()

    # ── internal helpers ────────────────────────────────────────────────

    def _refresh_create_btn(self):
        name = self._le_name.text().strip()
        create_button = self._buttons.button(QDialogButtonBox.Ok)
        create_button.setEnabled(self._parent_dir is not None and bool(name))

    def _get_skeleton_summary(self) -> str:
        k = len(self._kpt_names)
        e = len(self._edges)
        return (
            f"{k} keypoint{'s' if k != 1 else ''}, "
            f"{e} edge{'s' if e != 1 else ''} — click Edit to change"
        )

    def _pick_location(self):
        from hydra_suite.paths import get_projects_dir

        start = str(self._parent_dir or get_projects_dir())
        d = QFileDialog.getExistingDirectory(self, "Choose parent folder", start)
        if not d:
            return
        self._parent_dir = Path(d).expanduser().resolve()
        self._le_parent.setText(str(self._parent_dir))
        self._on_location_changed()

    def _on_location_changed(self):
        name = self._le_name.text().strip()
        if self._parent_dir and name:
            self._lbl_preview.setText(f"Project folder: {self._parent_dir / name}")
        elif self._parent_dir:
            self._lbl_preview.setText("Project folder: enter a project name.")
        else:
            self._lbl_preview.setText(
                "Project folder: choose a parent folder and enter a project name."
            )
        self._refresh_create_btn()

    def _do_accept(self):
        loc = self.get_project_location()
        if loc is None:
            return
        if (loc / "pose_project.json").exists():
            r = QMessageBox.question(
                self,
                "Project already exists",
                f"A project already exists at:\n{loc}\n\nOverwrite it?",
            )
            if r != QMessageBox.Yes:
                return
        self.accept()

    def _edit_skeleton(self):
        dlg = SkeletonEditorDialog(
            self._kpt_names,
            self._edges,
            self,
            default_dir=get_default_skeleton_dir(),
            flip_pairs=self._flip_pairs,
        )
        if dlg.exec() == QDialog.Accepted:
            names, edges = dlg.get_result()
            self._flip_pairs = dlg.get_flip_pairs()
            self._kpt_names = list(names) if names else ["kp1", "kp2"]
            k = len(self._kpt_names)
            self._edges = [
                (a, b) for (a, b) in edges if 0 <= a < k and 0 <= b < k and a != b
            ]
            self._skel_lbl.setText(self._get_skeleton_summary())

    # ── accessors ───────────────────────────────────────────────────────

    def get_project_location(self) -> Optional[Path]:
        """Return the full path for the new project directory, or ``None`` if not yet configured."""
        if self._parent_dir is None:
            return None
        name = self._le_name.text().strip()
        if not name:
            return None
        return self._parent_dir / name

    def get_classes(self) -> List[str]:
        """Return the list of class names entered by the user, defaulting to ``['object']`` if empty."""
        lines = [s.strip() for s in self._classes_edit.toPlainText().splitlines()]
        out = [s for s in lines if s]
        return out if out else ["object"]

    def get_keypoints(self) -> List[str]:
        """Return the ordered list of keypoint names defined in the wizard."""
        return list(self._kpt_names) if self._kpt_names else ["kp1", "kp2"]

    def get_edges(self) -> List[Tuple[int, int]]:
        """Return the skeleton edge list as pairs of 0-based keypoint indices."""
        return list(self._edges)

    def get_flip_pairs(self) -> List[Tuple[int, int]]:
        """Left/right keypoint pairs set in the Skeleton Editor."""
        return normalize_flip_pairs(self._flip_pairs, len(self.get_keypoints()))

    def get_options(self) -> Tuple[bool, float]:
        """Return ``(autosave_enabled, bbox_pad_fraction)`` from the wizard options page."""
        return bool(self._autosave_cb.isChecked()), float(self._pad_spin.value())


# ─────────────────────────────────────────────────────────────────────────────
# Project-settings wizard  (edit existing project)
# ─────────────────────────────────────────────────────────────────────────────


class ProjectWizard(QDialog):
    """
    Used on first run (no project file) OR to edit project settings.
    """

    def __init__(
        self, dataset_dir: Path, existing: Optional[Project] = None, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("PoseKit Setup" if existing is None else "Project Settings")
        self.setMinimumSize(QSize(820, 560))

        self.dataset_dir = dataset_dir.resolve()
        self.images_dir = self.dataset_dir / DEFAULT_DATASET_IMAGES_DIR
        self.existing = existing

        # Defaults
        default_root = self.dataset_dir / DEFAULT_POSEKIT_PROJECT_DIR
        default_labels = default_root / "labels"
        if existing is None:
            default_classes = ["object"]
            default_kpts = ["kp1", "kp2"]
            default_pad = 0.03
            default_autosave = True
            default_edges: List[Tuple[int, int]] = []
        else:
            default_root = existing.out_root
            default_labels = existing.labels_dir
            default_classes = existing.class_names
            default_kpts = existing.keypoint_names
            default_pad = existing.bbox_pad_frac
            default_autosave = existing.autosave
            default_edges = existing.skeleton_edges

        self._edges = list(default_edges)
        self._flip_pairs = list(getattr(existing, "flip_pairs", []) or [])
        self._kpt_names = list(default_kpts)

        layout = QVBoxLayout(self)

        # Paths
        form = QFormLayout()
        self.dataset_dir_line = QLineEdit(str(self.dataset_dir))
        self.dataset_dir_line.setReadOnly(True)
        form.addRow("Dataset folder:", self.dataset_dir_line)

        self.images_dir_line = QLineEdit(str(self.images_dir))
        self.images_dir_line.setReadOnly(True)
        form.addRow("Images folder:", self.images_dir_line)

        # Hide legacy single-source path rows for multi-source projects —
        # sources are listed in the Dataset Sources panel below instead.
        if existing is not None and existing.sources:
            for w in (self.dataset_dir_line, self.images_dir_line):
                lbl = form.labelForField(w)
                if lbl:
                    lbl.setVisible(False)
                w.setVisible(False)

        self.out_root = QLineEdit(str(default_root))
        self.out_root.setReadOnly(True)
        form.addRow("PoseKit project folder:", self.out_root)

        self.labels_dir = QLineEdit(str(default_labels))
        self.labels_dir.setReadOnly(True)
        form.addRow("Annotation labels folder:", self.labels_dir)

        self.autosave_cb = QCheckBox("Autosave when changing frames")
        self.autosave_cb.setChecked(default_autosave)
        form.addRow("", self.autosave_cb)

        self.pad_spin = QDoubleSpinBox()
        self.pad_spin.setRange(0.0, 0.25)
        self.pad_spin.setSingleStep(0.01)
        self.pad_spin.setValue(default_pad)
        form.addRow(
            "How much padding should be added around keypoints for bbox export?",
            self.pad_spin,
        )

        layout.addLayout(form)

        # Classes
        cls_box = QWidget()
        cls_layout = QVBoxLayout(cls_box)
        cls_layout.addWidget(QLabel("Classes (one per line)"))
        self.classes_edit = QPlainTextEdit("\n".join(default_classes))
        cls_layout.addWidget(self.classes_edit, 1)
        layout.addWidget(cls_box, 1)

        # Skeleton Editor button
        skel_box = QWidget()
        skel_layout = QVBoxLayout(skel_box)
        skel_layout.addWidget(QLabel("Keypoints and skeleton"))
        self.btn_skel = QPushButton("Edit Keypoints & Skeleton…")
        skel_layout.addWidget(self.btn_skel)

        # Summary label
        self.skel_summary = QLabel(self._get_skeleton_summary())
        self.skel_summary.setWordWrap(True)
        skel_layout.addWidget(self.skel_summary)
        skel_layout.addStretch(1)

        # Migration (only if editing existing project)
        self.mig_box = QWidget()
        mig_layout = QVBoxLayout(self.mig_box)
        self.migrate_cb = QCheckBox(
            "Migrate existing label files to new keypoint layout"
        )
        self.migrate_cb.setChecked(True)
        mig_layout.addWidget(self.migrate_cb)

        self.mig_mode_group = QButtonGroup(self)
        self.rb_by_name = QRadioButton(
            "Map by keypoint NAME (recommended if names stable)"
        )
        self.rb_by_index = QRadioButton(
            "Map by INDEX (recommended if you append/remove at end)"
        )
        self.rb_by_name.setChecked(True)
        self.mig_mode_group.addButton(self.rb_by_name)
        self.mig_mode_group.addButton(self.rb_by_index)
        mig_layout.addWidget(self.rb_by_name)
        mig_layout.addWidget(self.rb_by_index)

        if existing is None:
            self.mig_box.setVisible(False)

        skel_layout.addWidget(self.mig_box)
        layout.addWidget(skel_box)

        self.sources_modified = False

        # OK / Cancel
        bottom = QHBoxLayout()
        self.ok_btn = QPushButton("Create Project" if existing is None else "Apply")
        self.cancel_btn = QPushButton("Cancel")
        bottom.addStretch(1)
        bottom.addWidget(self.ok_btn)
        bottom.addWidget(self.cancel_btn)
        layout.addLayout(bottom)

        # wiring
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.btn_skel.clicked.connect(self._edit_skeleton)

    def _get_skeleton_summary(self) -> str:
        k = len(self._kpt_names)
        e = len(self._edges)
        return f"{k} keypoint{'s' if k != 1 else ''}, {e} edge{'s' if e != 1 else ''}"

    def _edit_skeleton(self):
        dlg = SkeletonEditorDialog(
            self._kpt_names,
            self._edges,
            self,
            default_dir=get_default_skeleton_dir(),
            flip_pairs=self._flip_pairs,
        )
        if dlg.exec() == QDialog.Accepted:
            names, edges = dlg.get_result()
            self._flip_pairs = dlg.get_flip_pairs()
            self._kpt_names = list(names) if names else ["kp1", "kp2"]
            k = len(self._kpt_names)
            self._edges = [
                (a, b) for (a, b) in edges if 0 <= a < k and 0 <= b < k and a != b
            ]
            self.skel_summary.setText(self._get_skeleton_summary())

    def get_classes(self) -> List[str]:
        """get_classes method documentation."""
        lines = [s.strip() for s in self.classes_edit.toPlainText().splitlines()]
        out = [s for s in lines if s]
        return out if out else ["object"]

    def get_keypoints(self) -> List[str]:
        """get_keypoints method documentation."""
        return list(self._kpt_names) if self._kpt_names else ["kp1", "kp2"]

    def get_edges(self) -> List[Tuple[int, int]]:
        """get_edges method documentation."""
        return list(self._edges)

    def get_flip_pairs(self) -> List[Tuple[int, int]]:
        """Left/right keypoint pairs set in the Skeleton Editor."""
        return normalize_flip_pairs(self._flip_pairs, len(self.get_keypoints()))

    def get_paths(self) -> Tuple[Path, Path]:
        """get_paths method documentation."""
        root = (self.dataset_dir / DEFAULT_POSEKIT_PROJECT_DIR).resolve()
        labels = (root / "labels").resolve()
        return root, labels

    def get_options(self) -> Tuple[bool, float]:
        """get_options method documentation."""
        return bool(self.autosave_cb.isChecked()), float(self.pad_spin.value())

    def get_migration(self) -> Tuple[bool, str]:
        """get_migration method documentation."""
        if self.existing is None:
            return False, "name"
        do = bool(self.migrate_cb.isChecked())
        mode = "name" if self.rb_by_name.isChecked() else "index"
        return do, mode
