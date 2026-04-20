"""DetectKit project lifecycle: create, open, save, recent projects."""

from __future__ import annotations

import copy
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hydra_suite.data.project_bundle import (
    DEFAULT_BUNDLE_HISTORY_DIRNAME,
    DEFAULT_BUNDLE_STATE_DIRNAME,
    ProjectBundleManifest,
    bundle_paths,
    ensure_bundle_subdirectories,
    ensure_project_bundle_layout,
    load_project_bundle_manifest,
    save_project_bundle_manifest,
)

from .constants import DEFAULT_PROJECT_FILENAME, DEFAULT_PROJECTS_ROOT_NAME
from .models import DetectKitProject, normalize_class_names

logger = logging.getLogger(__name__)

_MAX_RECENT = 20
_KIT_NAME = "detectkit"
_LEGACY_ARCHIVE_PREFIX = "legacy_"
_PROJECT_MODELS_DIRNAME = "models"
_PREVIEWABLE_HISTORY_ROLES = {"", "obb_direct"}
_DETECTKIT_ARTIFACT_DIRS = {
    "training_runs": "artifacts/training_runs",
    "evaluation": "artifacts/evaluation",
    "exports": "artifacts/exports",
}


# ---------------------------------------------------------------------------
# Recent-projects persistence
# ---------------------------------------------------------------------------


def get_recent_projects_path() -> Path:
    """Return the path to the recent-projects JSON file."""
    try:
        from hydra_suite.paths import _user_data_dir

        return _user_data_dir() / "detectkit" / "recent_projects.json"
    except Exception:
        return Path.home() / ".detectkit" / "recent_projects.json"


def load_recent_projects() -> list[str]:
    """Load the list of recent project directory paths."""
    rp = get_recent_projects_path()
    if not rp.exists():
        return []
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(p) for p in data]
    except Exception:
        logger.debug("Failed to read recent projects file", exc_info=True)
    return []


def save_recent_projects(paths: list[str]) -> None:
    """Persist at most *_MAX_RECENT* recent project paths."""
    rp = get_recent_projects_path()
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(paths[:_MAX_RECENT], indent=2), encoding="utf-8")


def add_to_recent(project_dir: str) -> None:
    """Add *project_dir* to the top of the recent list, de-duplicating."""
    paths = load_recent_projects()
    # Remove any existing occurrence
    paths = [p for p in paths if p != project_dir]
    paths.insert(0, project_dir)
    save_recent_projects(paths)


# ---------------------------------------------------------------------------
# Project file helpers
# ---------------------------------------------------------------------------


def project_file_path(project_dir: Path) -> Path:
    """Return the canonical project-file path inside *project_dir*."""
    return bundle_paths(project_dir).state_dir / DEFAULT_PROJECT_FILENAME


def legacy_project_file_path(project_dir: Path) -> Path:
    """Return the legacy DetectKit project-file path at the project root."""
    return project_dir / DEFAULT_PROJECT_FILENAME


def project_exists(project_dir: Path) -> bool:
    """Return True when *project_dir* contains either a bundle or legacy project."""
    manifest_path = bundle_paths(project_dir).manifest_path
    return (
        manifest_path.exists()
        or project_file_path(project_dir).exists()
        or legacy_project_file_path(project_dir).exists()
    )


def _manifest_for_project(project_dir: Path) -> ProjectBundleManifest:
    """Build the shared bundle manifest for a DetectKit project."""
    return ProjectBundleManifest(
        kit=_KIT_NAME,
        display_name=project_dir.name,
        state_path=str(Path(DEFAULT_BUNDLE_STATE_DIRNAME) / DEFAULT_PROJECT_FILENAME),
        artifacts_dir="artifacts",
        history_dir=DEFAULT_BUNDLE_HISTORY_DIRNAME,
        meta={
            "state_format": DEFAULT_PROJECT_FILENAME,
            "artifact_dirs": dict(_DETECTKIT_ARTIFACT_DIRS),
            "models_dir": _PROJECT_MODELS_DIRNAME,
        },
    )


def detectkit_models_dir(project_dir: Path) -> Path:
    """Return the project-local models directory for DetectKit."""
    models_dir = project_dir / _PROJECT_MODELS_DIRNAME
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def detectkit_artifact_paths(project_dir: Path) -> dict[str, Path]:
    """Return typed artifact directories for DetectKit bundle projects."""
    created = ensure_bundle_subdirectories(
        project_dir,
        tuple(_DETECTKIT_ARTIFACT_DIRS.values()),
    )
    return {
        name: created[relative] for name, relative in _DETECTKIT_ARTIFACT_DIRS.items()
    }


def _normalize_path_list(values: Any) -> list[str]:
    if isinstance(values, list):
        return [str(value).strip() for value in values if str(value).strip()]
    if isinstance(values, tuple):
        return [str(value).strip() for value in values if str(value).strip()]
    if isinstance(values, str) and values.strip():
        return [values.strip()]
    return []


def _slug(value: Any, *, fallback: str, max_len: int = 48) -> str:
    raw = str(value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = fallback
    return cleaned[:max_len]


def _dedupe_path(dest_dir: Path, desired_name: str) -> Path:
    candidate = dest_dir / desired_name
    counter = 1
    while candidate.exists():
        candidate = dest_dir / f"{candidate.stem}_{counter}{candidate.suffix}"
        counter += 1
    return candidate


def _path_within_project(project_dir: Path, candidate: str | Path) -> Path | None:
    try:
        resolved = Path(candidate).expanduser().resolve()
        resolved.relative_to(project_dir.resolve())
    except Exception:
        return None
    return resolved


def _history_index(entries: list[dict[str, Any]], run_id: str) -> int:
    for index, entry in enumerate(entries):
        if str(entry.get("run_id", "")).strip() == run_id:
            return index
    return -1


def _upsert_history_entry(entries: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    run_id = str(entry.get("run_id", "")).strip()
    if not run_id:
        entries.append(entry)
        return
    index = _history_index(entries, run_id)
    if index >= 0:
        entries[index] = entry
    else:
        entries.append(entry)


def _load_run_record(run_id: str) -> dict[str, Any] | None:
    if not run_id:
        return None
    try:
        from hydra_suite.training.registry import load_registry

        for entry in load_registry().get("runs", []):
            if str(entry.get("run_id", "")).strip() == run_id:
                return copy.deepcopy(entry)
    except Exception:
        logger.debug("Could not load run record %s", run_id, exc_info=True)
    return None


def _build_export_stem(entry: dict[str, Any]) -> str:
    spec = entry.get("spec") or {}
    base_model = Path(str(spec.get("base_model", "") or "model")).stem
    run_id = _slug(entry.get("run_id", ""), fallback="run", max_len=64)
    role = _slug(entry.get("role", "model"), fallback="model", max_len=32)
    base_slug = _slug(base_model, fallback="weights", max_len=32)
    return f"detectkit_{run_id}_{role}_{base_slug}"


def _build_export_name(
    src_path: Path,
    entry: dict[str, Any],
    index: int,
    total_artifacts: int,
) -> str:
    stem = _build_export_stem(entry)
    ext = src_path.suffix or ".pt"
    if total_artifacts == 1:
        return f"{stem}{ext}"
    return f"{stem}_f{index + 1}{ext}"


def _entry_model_paths(entry: dict[str, Any]) -> list[str]:
    project_paths = _normalize_path_list(entry.get("project_model_paths"))
    if project_paths:
        return project_paths
    single = str(entry.get("project_model_path", "") or "").strip()
    if single:
        return [single]
    published = str(entry.get("published_model_path", "") or "").strip()
    if published:
        return [published]
    return _normalize_path_list(entry.get("artifact_paths"))


def _export_entry_artifacts(
    project_dir: Path,
    entry: dict[str, Any],
) -> dict[str, Any]:
    updated = copy.deepcopy(entry)
    existing_exports = [
        path
        for path in _normalize_path_list(updated.get("project_model_paths"))
        if Path(path).exists()
    ]
    if existing_exports:
        updated["project_model_paths"] = existing_exports
        updated["project_model_path"] = existing_exports[0]
        return updated

    artifact_paths = _normalize_path_list(updated.get("artifact_paths"))
    if not artifact_paths:
        updated["project_model_paths"] = []
        updated["project_model_path"] = ""
        return updated

    dest_dir = detectkit_models_dir(project_dir)
    copied: list[str] = []
    failed: list[str] = []
    total_artifacts = len(artifact_paths)
    for index, src in enumerate(artifact_paths):
        src_path = Path(src).expanduser()
        if not src_path.exists():
            failed.append(f"{src_path.name}: file not found")
            continue
        desired_name = _build_export_name(src_path, updated, index, total_artifacts)
        dst = _dedupe_path(dest_dir, desired_name)
        shutil.copy2(str(src_path), str(dst))
        copied.append(str(dst))

    updated["project_model_paths"] = copied
    updated["project_model_path"] = copied[0] if copied else ""
    if failed:
        updated["export_errors"] = failed
    else:
        updated.pop("export_errors", None)
    return updated


def detectkit_project_model_paths(project: DetectKitProject) -> list[str]:
    """Return available project-scoped model paths, newest-first with active first."""
    seen: set[str] = set()
    paths: list[str] = []

    preferred = str(project.active_model_path or "").strip()
    if preferred and Path(preferred).exists():
        seen.add(preferred)
        paths.append(preferred)

    for entry in reversed(project.training_history or []):
        for model_path in _entry_model_paths(entry):
            if model_path in seen or not Path(model_path).exists():
                continue
            seen.add(model_path)
            paths.append(model_path)
    return paths


def detectkit_model_path_is_previewable(
    project: DetectKitProject,
    model_path: str,
) -> bool:
    """Return whether a model path supports full-image preview overlays."""
    candidate = str(model_path or "").strip()
    if not candidate or not Path(candidate).exists():
        return False

    matched_history = False
    for entry in project.training_history or []:
        entry_paths = {str(path).strip() for path in _entry_model_paths(entry)}
        if candidate not in entry_paths:
            continue
        matched_history = True
        role = str(entry.get("role", "") or "").strip().lower()
        if role in _PREVIEWABLE_HISTORY_ROLES:
            return True

    return not matched_history


def detectkit_project_preview_model_paths(project: DetectKitProject) -> list[str]:
    """Return project-scoped model paths that can render preview overlays."""
    return [
        path
        for path in detectkit_project_model_paths(project)
        if detectkit_model_path_is_previewable(project, path)
    ]


def detectkit_training_history_entry_for_model_path(
    project: DetectKitProject,
    model_path: str,
) -> dict[str, Any] | None:
    """Return the newest training-history entry that owns *model_path*."""
    candidate = str(model_path or "").strip()
    if not candidate:
        return None

    for entry in reversed(project.training_history or []):
        if candidate in {str(path).strip() for path in _entry_model_paths(entry)}:
            return copy.deepcopy(entry)
    return None


def detectkit_latest_model_path_for_role(
    project: DetectKitProject,
    role: str,
) -> str:
    """Return the newest existing exported model path for *role*, if any."""
    expected_role = str(role or "").strip().lower()
    if not expected_role:
        return ""

    for entry in reversed(project.training_history or []):
        entry_role = str(entry.get("role", "") or "").strip().lower()
        if entry_role != expected_role:
            continue
        for path in _entry_model_paths(entry):
            resolved = str(path or "").strip()
            if resolved and Path(resolved).exists():
                return resolved
    return ""


def record_training_results(
    project: DetectKitProject,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Persist DetectKit training results into project-local history and models."""
    history = list(project.training_history or [])
    training_runs_dir = detectkit_artifact_paths(project.project_dir)["training_runs"]
    persisted: list[dict[str, Any]] = []
    latest_project_model = ""
    latest_previewable_project_model = ""

    for raw_result in results:
        result = copy.deepcopy(raw_result)
        run_id = str(result.get("run_id", "")).strip() or datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        entry = _load_run_record(run_id) or {}
        entry["run_id"] = run_id
        entry["role"] = str(result.get("role") or entry.get("role") or "")
        entry["status"] = str(
            entry.get("status")
            or (
                "canceled"
                if result.get("canceled")
                else "completed" if result.get("success") else "failed"
            )
        )
        entry["success"] = bool(result.get("success", False))
        if result.get("error"):
            entry["error_message"] = str(result.get("error"))
        if result.get("dataset_fingerprint"):
            entry["dataset_fingerprint"] = str(result.get("dataset_fingerprint"))
        if result.get("command"):
            entry["command"] = list(result.get("command") or [])

        artifact_paths = _normalize_path_list(entry.get("artifact_paths"))
        if not artifact_paths:
            artifact_paths = _normalize_path_list(result.get("artifact_paths"))
        if not artifact_paths:
            artifact_path = str(result.get("artifact_path", "") or "").strip()
            artifact_paths = [artifact_path] if artifact_path else []
        entry["artifact_paths"] = artifact_paths

        metrics_paths = _normalize_path_list(entry.get("metrics_paths"))
        if not metrics_paths:
            metrics_paths = _normalize_path_list(result.get("metrics_paths"))
        if not metrics_paths:
            metrics_path = str(result.get("metrics_path", "") or "").strip()
            metrics_paths = [metrics_path] if metrics_path else []
        entry["metrics_paths"] = metrics_paths

        published_model_path = str(
            result.get("published_model_path")
            or entry.get("published_model_path")
            or ""
        ).strip()
        if published_model_path:
            entry["published_model_path"] = published_model_path
        published_registry_key = str(
            result.get("published_registry_key")
            or entry.get("published_registry_entry")
            or ""
        ).strip()
        if published_registry_key:
            entry["published_registry_entry"] = published_registry_key

        run_dir = str(entry.get("run_dir") or result.get("_run_dir") or "").strip()
        if run_dir:
            entry["run_dir"] = run_dir

        project_run_dir = training_runs_dir / _slug(run_id, fallback="run", max_len=96)
        project_run_dir.mkdir(parents=True, exist_ok=True)
        entry["project_run_dir"] = str(project_run_dir)

        copied_metrics: list[str] = []
        for metrics_path in metrics_paths:
            src_path = Path(metrics_path).expanduser()
            if not src_path.exists():
                continue
            dst = _dedupe_path(project_run_dir, src_path.name)
            shutil.copy2(str(src_path), str(dst))
            copied_metrics.append(str(dst))
        entry["project_metrics_paths"] = copied_metrics

        training_log = str(result.get("training_log", "") or "").rstrip()
        if training_log:
            log_path = project_run_dir / "training.log"
            log_path.write_text(training_log + "\n", encoding="utf-8")
            entry["project_log_path"] = str(log_path)

        if entry["success"]:
            entry = _export_entry_artifacts(project.project_dir, entry)
            if not latest_project_model and entry.get("project_model_path"):
                latest_project_model = str(entry["project_model_path"])
            role = str(entry.get("role", "") or "").strip().lower()
            if (
                not latest_previewable_project_model
                and role in _PREVIEWABLE_HISTORY_ROLES
                and entry.get("project_model_path")
            ):
                latest_previewable_project_model = str(entry["project_model_path"])

        _upsert_history_entry(history, entry)
        persisted.append(copy.deepcopy(entry))

    project.training_history = history
    selected_model = latest_previewable_project_model or latest_project_model
    if selected_model and (project.auto_select or not project.active_model_path):
        project.active_model_path = selected_model
    save_project(project)
    return persisted


def export_training_history_entry(
    project: DetectKitProject,
    run_id: str,
) -> dict[str, Any] | None:
    """Export one stored DetectKit history entry into the project's models folder."""
    history = list(project.training_history or [])
    index = _history_index(history, run_id)
    if index < 0:
        return None
    updated = _export_entry_artifacts(project.project_dir, history[index])
    history[index] = updated
    project.training_history = history
    if updated.get("project_model_path") and not project.active_model_path:
        project.active_model_path = str(updated["project_model_path"])
    save_project(project)
    return copy.deepcopy(updated)


def delete_training_history_entry(project: DetectKitProject, run_id: str) -> bool:
    """Delete one project-local training history entry and its owned artifacts."""
    history = list(project.training_history or [])
    index = _history_index(history, run_id)
    if index < 0:
        return False

    entry = history.pop(index)
    project_dir = project.project_dir.resolve()

    for model_path in _normalize_path_list(entry.get("project_model_paths")):
        owned = _path_within_project(project_dir, model_path)
        if owned is not None and owned.exists():
            owned.unlink()

    project_run_dir = _path_within_project(
        project_dir, entry.get("project_run_dir", "")
    )
    if project_run_dir is not None and project_run_dir.exists():
        shutil.rmtree(project_run_dir, ignore_errors=True)

    active_model = str(project.active_model_path or "").strip()
    project.training_history = history
    if active_model and active_model in _normalize_path_list(
        entry.get("project_model_paths")
    ):
        project.active_model_path = ""
        remaining_paths = detectkit_project_model_paths(project)
        project.active_model_path = remaining_paths[0] if remaining_paths else ""
    save_project(project)
    return True


def _state_path_from_manifest(
    project_dir: Path, manifest: ProjectBundleManifest
) -> Path:
    """Resolve the DetectKit state file from the shared bundle manifest."""
    return project_dir / Path(manifest.state_path)


def _archive_legacy_project_file(project_dir: Path) -> None:
    """Move the legacy root project file into the bundle history directory."""
    legacy_path = legacy_project_file_path(project_dir)
    canonical_path = project_file_path(project_dir)
    if not legacy_path.exists() or legacy_path == canonical_path:
        return

    history_dir = ensure_project_bundle_layout(project_dir).history_dir
    archive_path = history_dir / f"{_LEGACY_ARCHIVE_PREFIX}{DEFAULT_PROJECT_FILENAME}"
    if archive_path.exists():
        legacy_path.unlink()
        return
    shutil.move(str(legacy_path), str(archive_path))


def _load_bundle_project(project_dir: Path) -> Optional[DetectKitProject]:
    """Load a DetectKit project from the shared bundle layout if present."""
    manifest = load_project_bundle_manifest(project_dir)
    if manifest is None:
        return None
    if manifest.kit and manifest.kit != _KIT_NAME:
        logger.warning("Bundle manifest kit mismatch for %s", project_dir)
        return None

    state_path = _state_path_from_manifest(project_dir, manifest)
    if not state_path.exists():
        logger.warning("DetectKit state file not found: %s", state_path)
        return None

    proj = DetectKitProject.load(state_path)
    proj.project_dir = project_dir
    return proj


def _load_legacy_project(project_dir: Path) -> Optional[DetectKitProject]:
    """Load and migrate a legacy DetectKit root project file if present."""
    legacy_path = legacy_project_file_path(project_dir)
    if not legacy_path.exists():
        return None

    proj = DetectKitProject.load(legacy_path)
    proj.project_dir = project_dir
    save_project(proj)
    return proj


def _ensure_bundle_manifest(project_dir: Path) -> None:
    """Create or refresh the shared bundle manifest for *project_dir*."""
    detectkit_artifact_paths(project_dir)
    save_project_bundle_manifest(project_dir, _manifest_for_project(project_dir))


def default_project_parent_dir() -> Path:
    """Return the default parent directory for new DetectKit projects."""
    from hydra_suite.paths import get_projects_dir

    parent = get_projects_dir() / DEFAULT_PROJECTS_ROOT_NAME
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass
    return parent


def open_project(project_dir: Path) -> Optional[DetectKitProject]:
    """Open an existing project from *project_dir*."""
    project_dir = project_dir.expanduser().resolve()
    proj = _load_bundle_project(project_dir)
    if proj is None:
        proj = _load_legacy_project(project_dir)
    if proj is None and project_file_path(project_dir).exists():
        proj = DetectKitProject.load(project_file_path(project_dir))
        proj.project_dir = project_dir
        _ensure_bundle_manifest(project_dir)
    if proj is None:
        logger.warning("Project file not found in: %s", project_dir)
        return None

    add_to_recent(str(project_dir))
    return proj


def create_project(
    project_dir: Path,
    class_name: str = "object",
    *,
    class_names: list[str] | None = None,
) -> DetectKitProject:
    """Create a new project in *project_dir* and persist defaults."""
    project_dir = project_dir.expanduser().resolve()
    ensure_project_bundle_layout(project_dir)
    resolved_class_names = normalize_class_names(
        class_names if class_names is not None else [class_name]
    )
    proj = DetectKitProject(project_dir=project_dir, class_names=resolved_class_names)
    save_project(proj)
    add_to_recent(str(project_dir))
    return proj


def save_project(proj: DetectKitProject) -> None:
    """Save *proj* to its canonical project file."""
    ensure_project_bundle_layout(proj.project_dir)
    detectkit_models_dir(proj.project_dir)
    proj.save(project_file_path(proj.project_dir))
    _ensure_bundle_manifest(proj.project_dir)
    _archive_legacy_project_file(proj.project_dir)
