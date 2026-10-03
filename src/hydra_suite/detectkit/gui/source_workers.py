"""Background source inspection and import workers for DetectKit."""

from __future__ import annotations

from pathlib import Path

from hydra_suite.training.geometry_levels import GeometryLevel, scan_source_levels
from hydra_suite.widgets.workers import BaseWorker

from .project_source_import import project_sources_to_import
from .source_import import (
    IMPORT_MODE_PORTABLE,
    compute_positional_class_remap,
    inspect_detectkit_source,
    materialize_detectkit_source,
    remap_materialized_source_classes,
)


class SourceInspectionWorker(BaseWorker):
    """Inspect and scan source labels before opening the review dialog."""

    def __init__(self, selected_path: str, parent=None) -> None:
        super().__init__(parent)
        self.selected_path = selected_path
        self.result = None

    def execute(self) -> None:
        self.status.emit("Inspecting source…")
        inspection = inspect_detectkit_source(self.selected_path)
        self.status.emit("Scanning label geometry…")
        intended = (
            GeometryLevel.AABB
            if inspection.source_kind == "yolo_detect"
            else GeometryLevel.OBB
        )
        level_scan = scan_source_levels(
            inspection.dataset_root / "labels", intended_level=intended
        )
        self.result = inspection, level_scan


class SourceImportWorker(BaseWorker):
    """Materialize and optionally remap a source outside the GUI thread."""

    def __init__(
        self,
        selected_path: str,
        project_dir: Path,
        import_mode: str,
        source_classes: list[str],
        project_classes: list[str],
        force_remap: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.selected_path = selected_path
        self.project_dir = project_dir
        self.import_mode = import_mode
        self.source_classes = source_classes
        self.project_classes = project_classes
        self.force_remap = force_remap
        self.result = None

    def execute(self) -> None:
        self.status.emit("Importing images and labels…")
        last_percent = -1

        def report(done: int, total: int) -> None:
            nonlocal last_percent
            percent = min(95, round(95 * done / max(1, total)))
            if percent != last_percent:
                self.progress.emit(percent)
                last_percent = percent
            if done == 0 or done == total or done % max(1, total // 10) == 0:
                self.status.emit(f"Importing images and labels… {done} / {total}")

        materialized = materialize_detectkit_source(
            self.selected_path,
            self.project_dir,
            import_mode=self.import_mode,
            force_import=self.force_remap,
            progress=report,
        )
        if self.force_remap:
            self.status.emit("Remapping label classes…")
            remap = compute_positional_class_remap(
                self.source_classes, self.project_classes
            )

            def report_remap(done: int, total: int) -> None:
                nonlocal last_percent
                percent = 95 + round(5 * done / max(1, total))
                if percent != last_percent:
                    self.progress.emit(percent)
                    last_percent = percent

            remap_materialized_source_classes(
                Path(materialized.canonical_path),
                self.project_classes,
                remap,
                progress=report_remap,
            )
        self.result = materialized
        self.progress.emit(100)


class ProjectSourcesImportWorker(BaseWorker):
    """Validate and copy every compatible source from another project."""

    def __init__(self, destination, source_project_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self.destination = destination
        self.source_project_dir = source_project_dir
        self.result = None

    def execute(self) -> None:
        self.status.emit("Checking project sources…")
        roots = project_sources_to_import(self.destination, self.source_project_dir)
        results = []
        for index, root in enumerate(roots, start=1):
            self.status.emit(f"Importing source {index} of {len(roots)}: {root.name}")
            results.append(
                materialize_detectkit_source(
                    root, self.destination.project_dir, import_mode=IMPORT_MODE_PORTABLE
                )
            )
            self.progress.emit(round(100 * index / len(roots)))
        self.result = results
