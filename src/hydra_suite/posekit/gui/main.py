"""PoseKit application entry point: CLI argument parsing and startup."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication

from .constants import (
    DEFAULT_DATASET_IMAGES_DIR,
    DEFAULT_POSEKIT_PROJECT_DIR,
    DEFAULT_PROJECT_NAME,
)
from .main_window import MainWindow
from .models import Project
from .project import (
    _project_root_from_selection,
    build_image_list,
    create_empty_startup_project,
    create_project_via_wizard,
    find_project,
    load_project_with_repairs,
    open_project_from_path,
    resolve_dataset_paths,
)
from .utils import _resolve_project_path  # noqa: F401  kept for external callers


def parse_args() -> argparse.Namespace:
    """parse_args function documentation."""
    ap = argparse.ArgumentParser(description="PoseKit labeler")
    ap.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset root folder (contains images/) OR path to a pose_project.json file",
    )
    ap.add_argument("--project", default=None, help="Explicit project json path")
    ap.add_argument(
        "--new", action="store_true", help="Force setup wizard even if project exists"
    )
    return ap.parse_args()


def dataset_dir_for_cli_project(
    project_path: Path, dataset_dir: Optional[Path]
) -> Optional[Path]:
    """Return the dataset root whose layout a ``--project`` file should follow.

    Only a dataset-anchored project (``<dataset>/posekit_project/...``, or the
    legacy ``<dataset>/pose_project.json`` / ``<dataset>/labels/pose_project.json``,
    next to ``<dataset>/images/``) or an explicit dataset argument gets its
    paths repaired.  A standalone project bundle is opened where it is, like the
    GUI's Open Project; inferring a dataset from its images folder would move
    the project under the images' parent.
    """
    if dataset_dir is not None:
        return dataset_dir

    def _dataset_of(root: Path) -> Optional[Path]:
        if (root / DEFAULT_DATASET_IMAGES_DIR).is_dir() and (
            root.name != DEFAULT_POSEKIT_PROJECT_DIR
        ):
            return root
        return None

    selected = Path(project_path).expanduser().absolute()
    for root in (_project_root_from_selection(project_path), selected.parent):
        # <dataset>/posekit_project/... (also when posekit_project is a symlink:
        # the unresolved parent keeps its name).
        for candidate in (root, *root.parents[:2]):
            if (
                candidate.name == DEFAULT_POSEKIT_PROJECT_DIR
                and (candidate.parent / DEFAULT_DATASET_IMAGES_DIR).is_dir()
            ):
                return candidate.parent
    # Legacy files: <dataset>/pose_project.json, <dataset>/labels/pose_project.json
    if selected.name == DEFAULT_PROJECT_NAME:
        parent = selected.parent
        if parent.name == "labels":
            found = _dataset_of(parent.parent)
            if found is not None:
                return found
        found = _dataset_of(parent)
        if found is not None:
            return found
    return None


def main() -> None:
    """main function documentation."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("PoseKitLabeler")
    app.setApplicationDisplayName("PoseKit Labeler")
    app.setOrganizationName("NeuroRishika")
    app.setDesktopFileName("posekit-labeler")
    try:
        from hydra_suite.paths import get_brand_qicon

        icon = get_brand_qicon("posekit.svg")
        if icon and not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass

    # Startup-without-input mode: open full UI first, then show chooser overlay.
    if not args.dataset and not args.project:
        proj = create_empty_startup_project()
        win = MainWindow(proj, [])
        try:
            win.setWindowIcon(app.windowIcon())
        except Exception:
            pass
        win.resize(1500, 860)
        win.showMaximized()
        sys.exit(app.exec())

    dataset_dir: Optional[Path] = None
    if args.dataset:
        raw = Path(args.dataset).expanduser().resolve()
        if not raw.exists():
            print(f"Path not found: {raw}", file=sys.stderr)
            sys.exit(2)

        # Accept a project JSON passed as a positional argument
        if raw.is_file() and raw.suffix == ".json":
            args.project = args.project or str(raw)
        else:
            dataset_dir = raw
            # Backward compatibility: allow passing .../images directly.
            if dataset_dir.name == DEFAULT_DATASET_IMAGES_DIR:
                dataset_dir = dataset_dir.parent

            _, images_dir, _out_root, _labels_dir, _project_path = (
                resolve_dataset_paths(dataset_dir)
            )
            if not images_dir.exists() or not images_dir.is_dir():
                print(
                    f"Invalid dataset root (missing `{DEFAULT_DATASET_IMAGES_DIR}/`): {dataset_dir}",
                    file=sys.stderr,
                )
                sys.exit(2)

    proj: Optional[Project] = None

    if args.project:
        project_path = Path(args.project).expanduser().resolve()
        if project_path.exists():
            if not args.new:
                # Try loading directly (works for both standalone and legacy)
                proj = open_project_from_path(project_path)
                if proj is None:
                    sys.exit(2)
                # For legacy (dataset-anchored) projects, run path repairs
                # The unresolved path keeps a symlinked posekit_project's name.
                dataset_dir = dataset_dir_for_cli_project(
                    Path(args.project).expanduser().absolute(), dataset_dir
                )
                if dataset_dir is not None:
                    from .project import _repair_project_paths as _rpaths

                    if _rpaths(proj, dataset_dir):
                        proj.project_path.write_text(
                            json.dumps(proj.to_json(), indent=2),
                            encoding="utf-8",
                        )
        else:
            print(f"Project file not found: {project_path}", file=sys.stderr)
            sys.exit(2)

    if proj is None and not args.new:
        found = find_project(dataset_dir) if dataset_dir is not None else None
        if found:
            proj = load_project_with_repairs(found, dataset_dir)

    if proj is None:
        if dataset_dir is None:
            print("No dataset directory provided.", file=sys.stderr)
            sys.exit(2)
        proj = create_project_via_wizard(dataset_dir)
        if proj is None:
            sys.exit(0)

    imgs = build_image_list(proj)
    win = MainWindow(proj, imgs, show_welcome_when_empty=False)
    try:
        win.setWindowIcon(app.windowIcon())
    except Exception:
        pass
    win.resize(1500, 860)
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
