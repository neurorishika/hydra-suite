"""Project discovery, creation, path repair, and multi-source management."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtWidgets import QDialog, QMessageBox

from hydra_suite.data.project_bundle import (
    DEFAULT_BUNDLE_HISTORY_DIRNAME,
    DEFAULT_BUNDLE_MANIFEST_FILENAME,
    DEFAULT_BUNDLE_STATE_DIRNAME,
    ProjectBundleManifest,
    bundle_paths,
    ensure_bundle_subdirectories,
    ensure_project_bundle_layout,
    load_project_bundle_manifest,
    save_project_bundle_manifest,
    write_json_atomic,
)

from .constants import (
    DEFAULT_DATASET_IMAGES_DIR,
    DEFAULT_POSEKIT_PROJECT_DIR,
    DEFAULT_PROJECT_NAME,
    LARGE_DATASET_SIEVE_THRESHOLD,
)
from .models import DataSource, Project
from .utils import _resolve_project_path, list_images

_KIT_NAME = "posekit"
_POSEKIT_ARTIFACT_DIRS = {
    "cache": "artifacts/cache",
    "exports": "artifacts/exports",
}

# -----------------------------
# Bootstrap / project discovery
# -----------------------------


def default_project_parent_dir() -> Path:
    """Return the default parent directory for new standalone PoseKit projects."""
    from hydra_suite.paths import get_projects_dir

    parent = get_projects_dir() / "PoseKit"
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass
    return parent


def _resolve_images_dir(folder: Path) -> Path:
    """Auto-detect where images live within *folder*.

    Returns ``folder/images/`` when that subdirectory exists **and** contains
    image files (legacy layout), otherwise returns ``folder`` itself so that
    collections organised with images directly in the root are accepted
    without any additional structure requirements.
    """
    candidate = folder / DEFAULT_DATASET_IMAGES_DIR
    if candidate.is_dir() and list_images(candidate):
        return candidate
    return folder


def _default_project_images_dir(project_root: Path) -> Path:
    """Return the placeholder image directory used by empty standalone projects."""
    return project_root / DEFAULT_DATASET_IMAGES_DIR


def resolve_dataset_paths(dataset_dir: Path) -> Tuple[Path, Path, Path, Path, Path]:
    """Resolve canonical dataset structure paths.

    Expected structure:
      dataset_dir/
        images/
        posekit_project/
          state/
            pose_project.json
          labels/
    """
    ds = dataset_dir.expanduser().resolve()
    images_dir = ds / DEFAULT_DATASET_IMAGES_DIR
    out_root = ds / DEFAULT_POSEKIT_PROJECT_DIR
    labels_dir = out_root / "labels"
    project_path = canonical_project_path(out_root)
    return ds, images_dir, out_root, labels_dir, project_path


def canonical_project_path(project_root: Path) -> Path:
    """Return the canonical bundle state file path for a PoseKit project."""
    return bundle_paths(project_root).state_dir / DEFAULT_PROJECT_NAME


def posekit_artifact_paths(project_root: Path) -> dict[str, Path]:
    """Return typed artifact directories for a PoseKit bundle."""
    created = ensure_bundle_subdirectories(
        project_root,
        tuple(_POSEKIT_ARTIFACT_DIRS.values()),
    )
    return {
        name: created[relative] for name, relative in _POSEKIT_ARTIFACT_DIRS.items()
    }


def _manifest_for_project(
    project_root: Path, labels_dir: Path
) -> ProjectBundleManifest:
    """Build the shared bundle manifest for a PoseKit project."""
    return ProjectBundleManifest(
        kit=_KIT_NAME,
        display_name=Path(project_root).name,
        state_path=str(Path(DEFAULT_BUNDLE_STATE_DIRNAME) / DEFAULT_PROJECT_NAME),
        artifacts_dir="artifacts",
        history_dir=DEFAULT_BUNDLE_HISTORY_DIRNAME,
        meta={
            "labels_dir": str(labels_dir.relative_to(project_root)),
            "artifact_dirs": dict(_POSEKIT_ARTIFACT_DIRS),
        },
    )


def _project_root_from_selection(path: Path) -> Path:
    """Infer the PoseKit project root from a selected file or directory."""
    path = path.expanduser().resolve()
    if path.is_dir():
        return path
    if path.name == DEFAULT_BUNDLE_MANIFEST_FILENAME:
        return path.parent
    if (
        path.name == DEFAULT_PROJECT_NAME
        and path.parent.name == DEFAULT_BUNDLE_STATE_DIRNAME
    ):
        return path.parent.parent
    if path.name == DEFAULT_PROJECT_NAME and path.parent.name == "labels":
        return path.parent.parent
    return path.parent


def _archive_legacy_project_file(project_root: Path, legacy_path: Path) -> None:
    """Move a legacy PoseKit project JSON into history after migration."""
    if not legacy_path.exists() or legacy_path == canonical_project_path(project_root):
        return
    history_dir = ensure_project_bundle_layout(project_root).history_dir
    archive_path = history_dir / f"legacy_{legacy_path.name}"
    if archive_path.exists():
        legacy_path.unlink()
        return
    shutil.move(str(legacy_path), str(archive_path))


def _find_legacy_project_file(project_root: Path) -> Path | None:
    """Return the first legacy PoseKit project JSON inside *project_root*."""
    candidates = [
        project_root / DEFAULT_PROJECT_NAME,
        project_root / "labels" / DEFAULT_PROJECT_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def save_project_state(
    project: Project, *, legacy_source_path: Path | None = None
) -> Path:
    """Persist a PoseKit project into the canonical bundle layout."""
    project_root = project.out_root.expanduser().resolve()
    ensure_project_bundle_layout(project_root)
    posekit_artifact_paths(project_root)
    project.project_path = canonical_project_path(project_root)
    write_json_atomic(project.project_path, project.to_json())
    save_project_bundle_manifest(
        project_root,
        _manifest_for_project(project_root, project.labels_dir),
    )
    if legacy_source_path is not None:
        _archive_legacy_project_file(project_root, legacy_source_path)
    return project.project_path


def _load_bundle_project(project_root: Path) -> Optional[Project]:
    """Load a PoseKit project from the shared bundle layout if present."""
    manifest = load_project_bundle_manifest(project_root)
    if manifest is None:
        return None
    if manifest.kit and manifest.kit != _KIT_NAME:
        return None
    state_path = project_root / Path(manifest.state_path)
    if not state_path.exists():
        return None
    return Project.from_json(state_path)


def find_project(dataset_dir: Path) -> Optional[Path]:
    """Find an existing project for a dataset root.

        Primary location:
            dataset_dir/posekit_project/state/pose_project.json

    The primary location is trusted by virtue of its path — if the file
    exists there it belongs to this dataset even if the stored ``images_dir``
    is stale (e.g. after moving between machines).  ``load_project_with_repairs``
    will fix the paths afterwards.

    Legacy fallback paths are only accepted when the stored ``images_dir``
    matches, to avoid accidentally loading an unrelated project.
    """
    dataset_dir, images_dir, out_root, _labels_dir, project_path = (
        resolve_dataset_paths(dataset_dir)
    )

    # ── Primary location: trust by filesystem position ─────────────────────
    if project_path.exists():
        try:
            json.loads(project_path.read_text(encoding="utf-8"))  # validate JSON
            return project_path
        except (json.JSONDecodeError, OSError):
            pass  # corrupt file — fall through to legacy search

    # ── Legacy fallbacks: require images_dir match to avoid false positives ─
    legacy_candidates: List[Path] = [
        out_root / "labels" / DEFAULT_PROJECT_NAME,
        dataset_dir / DEFAULT_PROJECT_NAME,
        dataset_dir / "labels" / DEFAULT_PROJECT_NAME,
    ]
    for p in legacy_candidates:
        if not p.exists():
            continue
        try:
            proj_data = json.loads(p.read_text(encoding="utf-8"))
            base = p.parent
            proj_images_dir = _resolve_project_path(
                Path(proj_data["images_dir"]), base, proj_data
            )
            if proj_images_dir.resolve() == images_dir.resolve():
                return p
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return None


def _has_labels(labels_dir: Path) -> bool:
    try:
        return labels_dir.is_dir() and any(labels_dir.glob("*.txt"))
    except OSError:
        return False


def _repair_project_paths(proj: Project, dataset_dir: Path) -> bool:
    """Repair project paths when moved across machines. Returns True if changed."""
    (
        _dataset_dir,
        expected_images_dir,
        expected_out_root,
        expected_labels_dir,
        expected_project_path,
    ) = resolve_dataset_paths(dataset_dir)
    changed = False
    if proj.images_dir.resolve() != expected_images_dir.resolve():
        proj.images_dir = expected_images_dir
        changed = True
    # Legacy layouts kept labels in <dataset>/labels: keep using them rather
    # than pointing the project at an empty posekit_project/labels.
    legacy_labels = expected_images_dir.parent / "labels"
    if not _has_labels(expected_labels_dir) and _has_labels(legacy_labels):
        expected_labels_dir = legacy_labels
    if proj.labels_dir.resolve() != expected_labels_dir.resolve():
        proj.labels_dir = expected_labels_dir
        changed = True
    if proj.out_root.resolve() != expected_out_root.resolve():
        proj.out_root = expected_out_root
        changed = True
    if proj.project_path.resolve() != expected_project_path.resolve():
        proj.project_path = expected_project_path
        changed = True
    expected_out_root.mkdir(parents=True, exist_ok=True)
    expected_labels_dir.mkdir(parents=True, exist_ok=True)
    return changed


def load_project_with_repairs(project_path: Path, dataset_dir: Path) -> Project:
    """Load a project and repair any stale paths for the current machine."""
    proj = Project.from_json(project_path)
    changed = _repair_project_paths(proj, dataset_dir)
    if changed or project_path != canonical_project_path(proj.out_root):
        save_project_state(proj, legacy_source_path=project_path)
    return proj


def create_project_via_wizard(dataset_dir: Path) -> Optional[Project]:
    """Run the ProjectWizard dialog and create a new project from user input."""
    # Import here to avoid a circular dependency between project.py and dialogs.
    from .dialogs.project_wizard import ProjectWizard

    (
        dataset_dir,
        images_dir,
        _default_out_root,
        _default_labels_dir,
        default_project_path,
    ) = resolve_dataset_paths(dataset_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        QMessageBox.warning(
            None,
            "Invalid Dataset",
            f"Dataset is missing `{DEFAULT_DATASET_IMAGES_DIR}/`:\n{dataset_dir}",
        )
        return None

    # Suggest FilterKit preprocessing for very large datasets.
    image_count = len(list_images(images_dir))
    if image_count > LARGE_DATASET_SIEVE_THRESHOLD:
        msg = QMessageBox(None)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Large Dataset Detected")
        msg.setText(
            f"This dataset has {image_count:,} images.\n\n"
            "For large datasets, FilterKit can reduce near-duplicates and create "
            "a smaller representative subset before labeling."
        )
        msg.setInformativeText("Open this dataset in FilterKit now?")
        btn_open = msg.addButton("Open in FilterKit", QMessageBox.AcceptRole)
        btn_continue = msg.addButton("Continue in PoseKit", QMessageBox.DestructiveRole)
        msg.addButton(QMessageBox.Cancel)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_open:
            try:
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "hydra_suite.filterkit.gui",
                        str(dataset_dir),
                    ],
                    start_new_session=True,
                )
            except Exception as exc:
                QMessageBox.warning(
                    None,
                    "Launch Failed",
                    f"Could not launch FilterKit:\n{exc}",
                )
            return None
        if clicked != btn_continue:
            return None

    wiz = ProjectWizard(dataset_dir, existing=None)
    if wiz.exec() != QDialog.Accepted:
        return None

    out_root, labels_dir = wiz.get_paths()
    class_names = wiz.get_classes()
    kpt_names = wiz.get_keypoints()
    edges = wiz.get_edges()
    flip_pairs = wiz.get_flip_pairs()
    autosave, pad = wiz.get_options()

    out_root.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    project_path = default_project_path
    proj = Project(
        images_dir=images_dir,
        out_root=out_root,
        labels_dir=labels_dir,
        project_path=project_path,
        class_names=class_names,
        keypoint_names=kpt_names,
        skeleton_edges=edges,
        flip_pairs=flip_pairs,
        autosave=autosave,
        bbox_pad_frac=pad,
    )
    save_project_state(proj)
    return proj


def create_standalone_project_via_wizard(parent_widget=None) -> Optional[Project]:
    """Run NewProjectDialog and build a standalone multi-source project.

    Unlike ``create_project_via_wizard()``, the project file lives at a
    user-chosen location independent of any dataset folder. Sources are managed
    after project creation via the standalone Source Manager.
    """
    from .dialogs.project_wizard import NewProjectDialog

    dlg = NewProjectDialog(parent=parent_widget)
    if dlg.exec() != QDialog.Accepted:
        return None

    project_location = dlg.get_project_location()
    if project_location is None:
        return None

    class_names = dlg.get_classes()
    kpt_names = dlg.get_keypoints()
    edges = dlg.get_edges()
    flip_pairs = dlg.get_flip_pairs()
    autosave, pad = dlg.get_options()

    out_root = project_location
    out_root.mkdir(parents=True, exist_ok=True)
    images_dir = _default_project_images_dir(out_root)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    proj = Project(
        images_dir=images_dir,
        out_root=out_root,
        labels_dir=labels_dir,
        project_path=canonical_project_path(out_root),
        class_names=class_names,
        keypoint_names=kpt_names,
        skeleton_edges=edges,
        flip_pairs=flip_pairs,
        autosave=autosave,
        bbox_pad_frac=pad,
    )
    save_project_state(proj)

    return proj


def open_project_from_path(project_path: Path) -> Optional[Project]:
    """Load a project directly from its .json path (no dataset_dir required).

    Used when the user browses to a standalone project file.  Path repairs
    are skipped because the project location is trusted as-is.
    """
    try:
        project_root = _project_root_from_selection(project_path)
        proj = _load_bundle_project(project_root)
        if proj is None:
            selected = project_path.expanduser().resolve()
            if selected.name == DEFAULT_BUNDLE_MANIFEST_FILENAME:
                legacy_path = _find_legacy_project_file(project_root)
                if legacy_path is None:
                    raise FileNotFoundError(
                        f"PoseKit bundle manifest does not reference a valid {DEFAULT_PROJECT_NAME}"
                    )
                selected = legacy_path
            proj = Project.from_json(selected)
            if selected != canonical_project_path(proj.out_root):
                save_project_state(proj, legacy_source_path=selected)
        return proj
    except Exception as exc:
        QMessageBox.critical(
            None, "Failed to open project", f"Could not load:\n{project_path}\n\n{exc}"
        )
        return None


def build_image_list(project: Project) -> list:
    """Return all image paths for *project*, spanning every registered source.

    Multi-source projects aggregate images from all ``project.sources`` in
    registration order.  Legacy single-source projects fall back to the
    top-level ``project.images_dir``.
    """
    if project.sources:
        imgs = []
        for src in project.sources:
            imgs.extend(list_images(src.images_dir))
        return imgs
    return list_images(project.images_dir)


def add_source_to_project(
    project: Project,
    dataset_dir: Path,
    description: str = "",
) -> DataSource:
    """Ingest a new source dataset into *project* and persist the change.

    If the project is currently in legacy single-source mode (``sources=[]``)
    the existing ``images_dir`` / ``labels_dir`` is first promoted to
    ``source_0`` so that routing remains consistent.

    Returns the newly created :class:`DataSource`.
    """
    dataset_dir = dataset_dir.expanduser().resolve()
    images_dir = _resolve_images_dir(dataset_dir)

    # --- promote legacy project to multi-source on first add ---
    if not project.sources and list_images(project.images_dir):
        src0 = DataSource(
            source_id="source_0",
            dataset_root=project.images_dir.parent,
            images_dir=project.images_dir,
            labels_dir=project.labels_dir,
            description=project.images_dir.parent.name,
        )
        project.sources.append(src0)

    # --- generate a unique source_id ---
    existing_ids = {s.source_id for s in project.sources}
    i = len(project.sources)
    while f"source_{i}" in existing_ids:
        i += 1
    source_id = f"source_{i}"

    # --- per-source labels dir ---
    labels_dir = project.out_root / "labels" / source_id
    labels_dir.mkdir(parents=True, exist_ok=True)

    src = DataSource(
        source_id=source_id,
        dataset_root=dataset_dir,
        images_dir=images_dir,
        labels_dir=labels_dir,
        description=description or dataset_dir.name,
    )
    project.sources.append(src)
    if not list_images(project.images_dir):
        project.images_dir = images_dir
    if not project.last_source_id:
        project.last_source_id = source_id
    save_project_state(project)
    return src


def remove_source_from_project(project: Project, source_id: str) -> bool:
    """Remove a source from *project* and delete its project-local label cache."""
    source = next((src for src in project.sources if src.source_id == source_id), None)
    if source is None:
        return False

    project.sources = [src for src in project.sources if src.source_id != source_id]

    try:
        if source.labels_dir.exists() and source.labels_dir != project.labels_dir:
            shutil.rmtree(source.labels_dir)
    except OSError:
        pass

    if project.sources:
        if project.last_source_id == source_id or not any(
            src.source_id == project.last_source_id for src in project.sources
        ):
            project.last_source_id = project.sources[0].source_id
        project.images_dir = project.sources[0].images_dir
    else:
        project.last_source_id = ""
        placeholder_dir = _default_project_images_dir(project.out_root)
        placeholder_dir.mkdir(parents=True, exist_ok=True)
        project.images_dir = placeholder_dir

    save_project_state(project)
    return True


def create_empty_startup_project() -> Project:
    """Create a minimal placeholder project for startup-without-dataset mode."""
    dataset_root = (Path.home() / ".posekit" / "startup").resolve()
    images_dir = dataset_root / DEFAULT_DATASET_IMAGES_DIR
    out_root = dataset_root / DEFAULT_POSEKIT_PROJECT_DIR
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    ensure_project_bundle_layout(out_root)
    posekit_artifact_paths(out_root)
    return Project(
        images_dir=images_dir,
        out_root=out_root,
        labels_dir=labels_dir,
        project_path=canonical_project_path(out_root),
        class_names=["object"],
        keypoint_names=["kp1", "kp2"],
        skeleton_edges=[(0, 1)],
        autosave=True,
    )
