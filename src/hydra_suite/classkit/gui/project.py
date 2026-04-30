"""ClassKit project bundle helpers and migration utilities."""

from __future__ import annotations

import shutil
from pathlib import Path

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

DEFAULT_CLASSKIT_DB_FILENAME = "classkit.db"
DEFAULT_CLASSKIT_CONFIG_FILENAME = "project.json"
DEFAULT_CLASSKIT_SCHEME_FILENAME = "scheme.json"

_KIT_NAME = "classkit"
_CLASSKIT_ARTIFACT_DIRS = {
    "models": "artifacts/models",
    "exports": "artifacts/exports",
}
_CLASSKIT_STATE_CACHE_DIRS = {
    "embeddings": "state/embeddings",
    "clusters": "state/clusters",
    "umap": "state/umap",
    "predictions": "state/predictions",
}
_LEGACY_FILENAMES = (
    DEFAULT_CLASSKIT_DB_FILENAME,
    DEFAULT_CLASSKIT_CONFIG_FILENAME,
    DEFAULT_CLASSKIT_SCHEME_FILENAME,
)


def classkit_db_path(project_dir: Path) -> Path:
    """Return the canonical database path for a ClassKit project."""
    return bundle_paths(project_dir).state_dir / DEFAULT_CLASSKIT_DB_FILENAME


def classkit_config_path(project_dir: Path) -> Path:
    """Return the canonical config JSON path for a ClassKit project."""
    return bundle_paths(project_dir).state_dir / DEFAULT_CLASSKIT_CONFIG_FILENAME


def classkit_scheme_path(project_dir: Path) -> Path:
    """Return the canonical scheme JSON path for a ClassKit project."""
    return bundle_paths(project_dir).state_dir / DEFAULT_CLASSKIT_SCHEME_FILENAME


def legacy_classkit_db_path(project_dir: Path) -> Path:
    """Return the legacy root database path."""
    return Path(project_dir) / DEFAULT_CLASSKIT_DB_FILENAME


def legacy_classkit_config_path(project_dir: Path) -> Path:
    """Return the legacy root config path."""
    return Path(project_dir) / DEFAULT_CLASSKIT_CONFIG_FILENAME


def legacy_classkit_scheme_path(project_dir: Path) -> Path:
    """Return the legacy root scheme path."""
    return Path(project_dir) / DEFAULT_CLASSKIT_SCHEME_FILENAME


def classkit_artifact_paths(project_dir: Path) -> dict[str, Path]:
    """Return typed artifact directories for a ClassKit project bundle."""
    created = ensure_bundle_subdirectories(
        project_dir,
        tuple(_CLASSKIT_ARTIFACT_DIRS.values()),
    )
    return {
        name: created[relative] for name, relative in _CLASSKIT_ARTIFACT_DIRS.items()
    }


def classkit_model_dir(project_dir: Path) -> Path:
    """Return the canonical models artifact directory for a ClassKit bundle."""
    return classkit_artifact_paths(project_dir)["models"]


def classkit_export_dir(project_dir: Path) -> Path:
    """Return the canonical exports artifact directory for a ClassKit bundle."""
    return classkit_artifact_paths(project_dir)["exports"]


def classkit_state_cache_dir(project_dir: Path, cache_name: str) -> Path:
    """Return a canonical state-scoped cache directory for a ClassKit bundle."""
    if cache_name not in _CLASSKIT_STATE_CACHE_DIRS:
        raise KeyError(f"Unknown ClassKit cache dir: {cache_name}")
    created = ensure_bundle_subdirectories(
        project_dir,
        (_CLASSKIT_STATE_CACHE_DIRS[cache_name],),
    )
    return created[_CLASSKIT_STATE_CACHE_DIRS[cache_name]]


def _manifest_for_project(project_dir: Path) -> ProjectBundleManifest:
    """Build the shared bundle manifest for a ClassKit project."""
    return ProjectBundleManifest(
        kit=_KIT_NAME,
        display_name=Path(project_dir).name,
        state_path=str(
            Path(DEFAULT_BUNDLE_STATE_DIRNAME) / DEFAULT_CLASSKIT_CONFIG_FILENAME
        ),
        database_path=str(
            Path(DEFAULT_BUNDLE_STATE_DIRNAME) / DEFAULT_CLASSKIT_DB_FILENAME
        ),
        artifacts_dir="artifacts",
        history_dir=DEFAULT_BUNDLE_HISTORY_DIRNAME,
        meta={
            "scheme_path": str(
                Path(DEFAULT_BUNDLE_STATE_DIRNAME) / DEFAULT_CLASSKIT_SCHEME_FILENAME
            ),
            "artifact_dirs": dict(_CLASSKIT_ARTIFACT_DIRS),
        },
    )


def ensure_classkit_project_layout(project_dir: Path) -> Path:
    """Create the canonical ClassKit bundle layout and return the DB path."""
    project_dir = Path(project_dir).expanduser().resolve()
    ensure_project_bundle_layout(project_dir)
    classkit_artifact_paths(project_dir)
    for cache_name in _CLASSKIT_STATE_CACHE_DIRS:
        classkit_state_cache_dir(project_dir, cache_name)
    save_project_bundle_manifest(project_dir, _manifest_for_project(project_dir))
    return classkit_db_path(project_dir)


def project_exists(project_dir: Path) -> bool:
    """Return True when a ClassKit bundle or legacy root project exists."""
    project_dir = Path(project_dir).expanduser().resolve()
    manifest_path = bundle_paths(project_dir).manifest_path
    return (
        manifest_path.exists()
        or classkit_db_path(project_dir).exists()
        or legacy_classkit_db_path(project_dir).exists()
    )


def classkit_project_linked_image_count(
    project_dir: Path,
    image_paths: list[Path | str],
) -> int:
    """Return how many ingested images still point outside the project bundle."""
    project_root = Path(project_dir).expanduser().resolve()
    linked_count = 0
    for image_path in image_paths:
        try:
            Path(image_path).expanduser().resolve().relative_to(project_root)
        except Exception:
            linked_count += 1
    return linked_count


def classkit_project_is_portable(
    project_dir: Path,
    image_paths: list[Path | str],
) -> bool:
    """Return whether all ingested images are owned by the project bundle."""
    return classkit_project_linked_image_count(project_dir, image_paths) == 0


def prepare_project_directory(project_dir: Path) -> Path:
    """Migrate any legacy ClassKit root files into the canonical bundle layout."""
    project_dir = Path(project_dir).expanduser().resolve()
    ensure_classkit_project_layout(project_dir)
    history_dir = bundle_paths(project_dir).history_dir

    legacy_mapping = {
        legacy_classkit_db_path(project_dir): classkit_db_path(project_dir),
        legacy_classkit_config_path(project_dir): classkit_config_path(project_dir),
        legacy_classkit_scheme_path(project_dir): classkit_scheme_path(project_dir),
    }
    for legacy_path, canonical_path in legacy_mapping.items():
        if legacy_path == canonical_path or not legacy_path.exists():
            continue

        if not canonical_path.exists():
            shutil.move(str(legacy_path), str(canonical_path))
            continue

        archive_path = history_dir / f"legacy_{legacy_path.name}"
        if archive_path.exists():
            legacy_path.unlink()
        else:
            shutil.move(str(legacy_path), str(archive_path))

    save_project_bundle_manifest(project_dir, _manifest_for_project(project_dir))
    return classkit_db_path(project_dir)


def load_project_manifest(project_dir: Path) -> ProjectBundleManifest | None:
    """Load the shared bundle manifest for a ClassKit project, if present."""
    return load_project_bundle_manifest(Path(project_dir).expanduser().resolve())


def default_project_parent_dir() -> Path:
    """Return the default parent directory for new ClassKit projects."""
    from hydra_suite.paths import get_projects_dir

    parent = get_projects_dir() / "ClassKit"
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass
    return parent
