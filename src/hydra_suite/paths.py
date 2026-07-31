"""Central path resolution for hydra-suite.

All modules in the package import from here to locate user data,
configuration, and bundled assets. Never use ``Path(__file__).parents[N]``
to navigate to the repo root — use this module instead.

User-writable directories are managed via *platformdirs* with optional
environment variable overrides:

    HYDRA_CONFIG_DIR  — override config directory (presets, skeletons, advanced config)
    HYDRA_DATA_DIR    — override data directory (models, training runs)
    HYDRA_PROJECTS_DIR — override default projects directory (browse/open default)

Bundled read-only assets are accessed via *importlib.resources*.
Qt helpers provide lazy-loaded QIcon construction.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from platformdirs import user_config_dir, user_data_dir, user_documents_dir

logger = logging.getLogger(__name__)

APP_NAME = "hydra-suite"
APP_AUTHOR = "Rishika Mohanta"

# ---------------------------------------------------------------------------
# Internal helpers — user-writable directories
# ---------------------------------------------------------------------------


def _user_config_dir() -> Path:
    """Return (and create) the user configuration directory.

    Override with the ``HYDRA_CONFIG_DIR`` environment variable.
    """
    override = os.environ.get("HYDRA_CONFIG_DIR")
    if override:
        p = Path(override)
    else:
        p = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _user_data_dir() -> Path:
    """Return (and create) the user data directory.

    Override with the ``HYDRA_DATA_DIR`` environment variable.
    """
    override = os.environ.get("HYDRA_DATA_DIR")
    if override:
        p = Path(override)
    else:
        p = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _user_projects_dir() -> Path:
    """Return the default projects directory, creating it if possible.

    Override with the ``HYDRA_PROJECTS_DIR`` environment variable.
    Defaults to ``~/Documents/hydra-projects``.

    On macOS the ``~/Documents`` folder may be gated by Full Disk Access.
    If directory creation fails with a permission error the path is returned
    as-is — callers that only need a starting directory for a file dialog can
    still use it even if it does not exist yet.
    """
    override = os.environ.get("HYDRA_PROJECTS_DIR")
    if override:
        p = Path(override)
    else:
        p = Path(user_documents_dir()) / "hydra-projects"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass  # macOS Full Disk Access not granted — return path without creating
    return p


# ---------------------------------------------------------------------------
# Public — user-writable directories
# ---------------------------------------------------------------------------


def get_projects_dir() -> Path:
    """Return (and create) the default projects directory.

    Respects the ``HYDRA_PROJECTS_DIR`` environment variable override.
    """
    return _user_projects_dir()


def get_data_dir() -> Path:
    """Return (and create) the root data directory.

    Respects the ``HYDRA_DATA_DIR`` environment variable override.
    """
    return _user_data_dir()


def get_config_dir() -> Path:
    """Return (and create) the root config directory.

    Respects the ``HYDRA_CONFIG_DIR`` environment variable override.
    """
    return _user_config_dir()


def get_app_data_dir(app_name: str) -> Path:
    """Return (and create) a per-app subdirectory under the data directory.

    Respects the ``HYDRA_DATA_DIR`` environment variable override.
    """
    p = _user_data_dir() / app_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_models_dir() -> Path:
    """Return (and create) the models directory."""
    p = _user_data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_vitpose_cache_dir() -> Path:
    """Return (and create) the cache dir for downloaded ViTPose checkpoints."""
    p = _user_data_dir() / "vitpose-assets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_training_runs_dir() -> Path:
    """Return (and create) the training runs directory."""
    p = _user_data_dir() / "training" / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_training_workspace_dir(subdir: str = "YOLO") -> Path:
    """Return (and create) the training workspace directory."""
    p = _user_data_dir() / "training" / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_advanced_config_path() -> Path:
    """Return the path to the advanced configuration file."""
    return _user_config_dir() / "advanced_config.json"


def get_presets_dir() -> Path:
    """Return (and create) the presets directory, seeding bundled defaults on first use."""
    p = _user_config_dir() / "presets"
    p.mkdir(parents=True, exist_ok=True)
    marker = p / ".seeded"
    if not marker.exists():
        _seed_bundled_configs(p)
        marker.touch()
    return p


def get_skeleton_dir() -> Path:
    """Return (and create) the skeletons directory, seeding bundled defaults on first use."""
    p = _user_config_dir() / "skeletons"
    p.mkdir(parents=True, exist_ok=True)
    marker = p / ".seeded"
    if not marker.exists():
        _seed_bundled_skeletons(p)
        marker.touch()
    return p


def get_classkit_config_dir() -> Path:
    """Return (and create) the ClassKit-specific config directory."""
    p = _user_config_dir() / "classkit"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_classkit_scheme_presets_dir() -> Path:
    """Return (and create) the ClassKit scheme preset directory."""
    p = get_classkit_config_dir() / "scheme_presets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_classkit_timm_backbones_path() -> Path:
    """Return the persistent ClassKit custom-backbone registry path."""
    return get_classkit_config_dir() / "custom_timm_backbones.json"


def print_paths() -> None:
    """Print all resolved paths. Useful for debugging and user support.

    Usage::

        python -c "from hydra_suite.paths import print_paths; print_paths()"
    """
    print(f"Config dir:        {_user_config_dir()}")
    print(f"Data dir:          {_user_data_dir()}")
    print(f"Projects dir:      {_user_projects_dir()}")
    print(f"Models:            {get_models_dir()}")
    print(f"ViTPose cache:     {get_vitpose_cache_dir()}")
    print(f"Training runs:     {get_training_runs_dir()}")
    print(f"Presets:           {get_presets_dir()}")
    print(f"Skeletons:         {get_skeleton_dir()}")
    print(f"ClassKit config:   {get_classkit_config_dir()}")
    print(f"ClassKit schemes:  {get_classkit_scheme_presets_dir()}")
    print(f"Advanced config:   {get_advanced_config_path()}")
    override_cfg = os.environ.get("HYDRA_CONFIG_DIR")
    override_data = os.environ.get("HYDRA_DATA_DIR")
    if override_cfg:
        print(f"  (HYDRA_CONFIG_DIR override active: {override_cfg})")
    if override_data:
        print(f"  (HYDRA_DATA_DIR override active: {override_data})")
    override_proj = os.environ.get("HYDRA_PROJECTS_DIR")
    if override_proj:
        print(f"  (HYDRA_PROJECTS_DIR override active: {override_proj})")


# ---------------------------------------------------------------------------
# Internal — seed helpers
# ---------------------------------------------------------------------------


def _seed_bundled_configs(dest: Path) -> None:
    """Copy bundled config presets into *dest*."""
    for name in get_bundled_config_names():
        data = get_default_config(name)
        if data is not None:
            (dest / name).write_text(json.dumps(data, indent=4), encoding="utf-8")


def _seed_bundled_skeletons(dest: Path) -> None:
    """Copy bundled skeleton configs into *dest*."""
    for name in get_bundled_skeleton_names():
        data = get_skeleton_config(name)
        if data is not None:
            (dest / name).write_text(json.dumps(data, indent=4), encoding="utf-8")


# ---------------------------------------------------------------------------
# Bundled read-only assets (importlib.resources)
# ---------------------------------------------------------------------------


def _resources_files(package: str):
    """Get importlib.resources files handle for a package."""
    from importlib.resources import files

    return files(package)


def get_brand_icon_bytes(name: str) -> Optional[bytes]:
    """Read a brand icon asset by filename. Returns None if not found."""
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    try:
        ref = _resources_files("hydra_suite.resources.brand").joinpath(name)
        return ref.read_bytes()
    except (FileNotFoundError, TypeError, ModuleNotFoundError):
        return None


def get_default_config(name: str) -> Optional[dict]:
    """Read a bundled config preset as a dict. Returns None if not found."""
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    try:
        ref = _resources_files("hydra_suite.resources.configs").joinpath(name)
        return json.loads(ref.read_text(encoding="utf-8"))
    except (FileNotFoundError, TypeError, ModuleNotFoundError, json.JSONDecodeError):
        return None


def get_skeleton_config(name: str) -> Optional[dict]:
    """Read a bundled skeleton config as a dict. Returns None if not found."""
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    try:
        ref = _resources_files("hydra_suite.resources.configs.skeletons").joinpath(name)
        return json.loads(ref.read_text(encoding="utf-8"))
    except (FileNotFoundError, TypeError, ModuleNotFoundError, json.JSONDecodeError):
        return None


def get_bundled_config_names() -> list[str]:
    """List .json files bundled in hydra_suite.resources.configs."""
    try:
        root = _resources_files("hydra_suite.resources.configs")
        return sorted(
            item.name for item in root.iterdir() if item.name.endswith(".json")
        )
    except (TypeError, ModuleNotFoundError):
        return []


def get_bundled_skeleton_names() -> list[str]:
    """List .json files bundled in hydra_suite.resources.configs.skeletons."""
    try:
        root = _resources_files("hydra_suite.resources.configs.skeletons")
        return sorted(
            item.name for item in root.iterdir() if item.name.endswith(".json")
        )
    except (TypeError, ModuleNotFoundError):
        return []


# ---------------------------------------------------------------------------
# Qt helpers
# ---------------------------------------------------------------------------


def get_brand_qicon(name: str):
    """Load a brand asset as a QIcon. Returns an empty QIcon on failure.

    All Qt imports are lazy to avoid import-time failures when Qt is not
    installed or when running headless.
    """
    data = get_brand_icon_bytes(name)
    if data is None:
        try:
            from PySide6.QtGui import QIcon

            return QIcon()
        except ImportError:
            return None

    try:
        from PySide6.QtCore import QByteArray
        from PySide6.QtGui import QIcon, QPixmap

        if name.lower().endswith(".svg"):
            from PySide6.QtCore import QSize
            from PySide6.QtGui import QImage, QPainter
            from PySide6.QtSvg import QSvgRenderer

            renderer = QSvgRenderer(QByteArray(data))
            image = QImage(QSize(256, 256), QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(0)
            painter = QPainter(image)
            renderer.render(painter)
            painter.end()
            pixmap = QPixmap.fromImage(image)
        else:
            pixmap = QPixmap()
            pixmap.loadFromData(QByteArray(data))

        return QIcon(pixmap)
    except (ImportError, Exception):
        try:
            from PySide6.QtGui import QIcon

            return QIcon()
        except ImportError:
            return None
