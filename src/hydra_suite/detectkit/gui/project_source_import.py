"""Plan a portable import of the datasets listed by another DetectKit project."""

from __future__ import annotations

from pathlib import Path

from .models import DetectKitProject
from .project import open_project, project_exists
from .source_import import inspect_detectkit_source


def project_sources_to_import(
    destination: DetectKitProject, source_project_dir: Path
) -> list[Path]:
    """Validate a project and return its distinct, compatible source roots."""
    source_project_dir = source_project_dir.expanduser().resolve()
    if source_project_dir == destination.project_dir.expanduser().resolve():
        raise ValueError("Select a different DetectKit project.")
    if not project_exists(source_project_dir):
        raise ValueError("The selected folder is not a DetectKit project.")
    source_project = open_project(source_project_dir)
    if source_project is None:
        raise ValueError("The selected DetectKit project could not be opened.")
    if source_project.class_names != destination.class_names:
        raise ValueError(
            "Project classes must match in name and order before importing sources. "
            f"Destination: {destination.class_names}; selected: {source_project.class_names}"
        )

    existing = {
        Path(candidate).expanduser().resolve()
        for source in destination.sources
        for candidate in (source.path, source.original_path)
        if candidate
    }
    pending: list[Path] = []
    for source in source_project.sources:
        if not source.reviewed or source.staged_review is not None:
            raise ValueError(f"Source {source.name!r} has unreviewed labels.")
        root = Path(source.path).expanduser().resolve()
        if root in existing or root in pending:
            continue
        if not root.is_dir():
            raise ValueError(f"Source directory is missing: {root}")
        inspection = inspect_detectkit_source(root)
        if inspection.discovered_labels != destination.class_names:
            raise ValueError(
                f"Source {source.name or root.name!r} has incompatible classes: "
                f"{inspection.discovered_labels}"
            )
        pending.append(root)
    return pending
