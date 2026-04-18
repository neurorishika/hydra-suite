"""Model repository utility functions for TrackerKit.

These functions manage YOLO and pose model paths, metadata registry,
and file-system layout for the per-user model repository.
"""

from __future__ import annotations

import json
import logging
import os
import shutil

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def get_models_root_directory() -> str:
    """Return user-local models/ root and create it when missing."""
    from hydra_suite.paths import get_models_dir

    return str(get_models_dir())


def get_models_directory() -> object:
    """
    Get the path to the default YOLO OBB model repository.

    Returns models/obb (direct OBB models).
    Creates the directory if it doesn't exist.
    """
    return get_yolo_model_repository_directory(
        task_family="obb", usage_role="obb_direct"
    )


def get_yolo_model_repository_directory(
    task_family: str | None = None, usage_role: str | None = None
) -> object:
    """Return repository directory for a YOLO model role."""
    tf = str(task_family or "").strip().lower()
    ur = _normalize_usage_role(usage_role)
    models_root = get_models_root_directory()

    if ur == "seq_detect" or tf == "detect":
        repo_dir = os.path.join(models_root, "detection")
    elif ur == "seq_crop_obb":
        repo_dir = os.path.join(models_root, "obb", "cropped")
    elif ur == "headtail":
        repo_dir = os.path.join(models_root, "classification", "orientation")
    elif ur == "colortag" or (tf == "classify" and ur not in ("headtail",)):
        repo_dir = os.path.join(models_root, "classification", "colortag")
    else:
        repo_dir = os.path.join(models_root, "obb")

    os.makedirs(repo_dir, exist_ok=True)
    return repo_dir


def get_pose_models_directory(backend: str | None = None) -> object:
    """
    Get the local pose-model repository directory.

    Layout:
      models/pose/YOLO/
      models/pose/SLEAP/
      models/pose/ViTPose/
    """
    models_root = get_models_root_directory()
    pose_root = os.path.join(models_root, "pose")
    os.makedirs(pose_root, exist_ok=True)
    if not backend:
        return pose_root
    key = str(backend or "").strip().lower()
    if key == "sleap":
        backend_dirname = "SLEAP"
    elif key == "vitpose":
        backend_dirname = "ViTPose"
    else:
        backend_dirname = "YOLO"
    backend_dir = os.path.join(pose_root, backend_dirname)
    os.makedirs(backend_dir, exist_ok=True)
    return backend_dir


def resolve_pose_model_path(model_path: object, backend: str | None = None) -> object:
    """Resolve a pose model path (relative or absolute) to an absolute path when possible."""
    if not model_path:
        return model_path

    path_str = str(model_path).strip()
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str

    models_root = get_models_root_directory()
    candidates = [os.path.join(models_root, path_str)]
    if backend:
        candidates.append(os.path.join(get_pose_models_directory(backend), path_str))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    return path_str


def make_pose_model_path_relative(model_path: object) -> object:
    """Convert absolute pose-model paths under models/ into relative paths."""
    if not model_path or not os.path.isabs(str(model_path)):
        return model_path
    models_root = get_models_root_directory()
    try:
        rel_path = os.path.relpath(str(model_path), models_root)
        if not rel_path.startswith(".."):
            return rel_path
    except (ValueError, TypeError):
        pass
    return model_path


def resolve_model_path(model_path: object) -> object:
    """Resolve a model path to an absolute path."""
    if not model_path:
        return model_path

    path_str = str(model_path).strip()
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str

    models_root = get_models_root_directory()
    candidate = os.path.join(models_root, path_str)
    if os.path.exists(candidate):
        return candidate

    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    return model_path


def make_model_path_relative(model_path: object) -> object:
    """Convert an absolute model path to relative if it's in the models directory."""
    if not model_path or not os.path.isabs(model_path):
        return model_path

    models_root = get_models_root_directory()
    try:
        rel_path = os.path.relpath(model_path, models_root)
        if not rel_path.startswith(".."):
            return rel_path
    except (ValueError, TypeError):
        pass
    return model_path


# ---------------------------------------------------------------------------
# YOLO model registry
# ---------------------------------------------------------------------------


def get_yolo_model_registry_path() -> object:
    """Return path to the local YOLO model metadata registry JSON."""
    return os.path.join(get_models_root_directory(), "model_registry.json")


def _sanitize_model_token(text: object) -> object:
    """Sanitize a species/info token for filenames and metadata."""
    raw = str(text or "").strip()
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw)
    return cleaned.strip("_")


def _normalize_usage_role(value: object) -> str:
    """Normalize legacy and current usage-role spellings to a single token."""
    normalized = _sanitize_model_token(value).lower()
    if normalized == "head_tail":
        return "headtail"
    return normalized


def _normalize_yolo_model_metadata(metadata: object) -> object:
    """Normalize legacy model metadata to species + model_info schema."""
    if not isinstance(metadata, dict):
        return {}

    normalized = dict(metadata)
    species = _sanitize_model_token(normalized.get("species", ""))
    model_info = _sanitize_model_token(normalized.get("model_info", ""))

    if species:
        normalized["species"] = species
    if model_info:
        normalized["model_info"] = model_info

    task_family = _sanitize_model_token(normalized.get("task_family", "")).lower()
    usage_role = _normalize_usage_role(normalized.get("usage_role", ""))
    if task_family:
        normalized["task_family"] = task_family
    else:
        normalized.pop("task_family", None)
    if usage_role:
        normalized["usage_role"] = usage_role
    else:
        normalized.pop("usage_role", None)
    return normalized


def _extract_registry_entries(data: object) -> dict[str, dict]:
    if (
        isinstance(data, dict)
        and data.get("schema_version") == 2
        and isinstance(data.get("entries"), dict)
    ):
        source = data["entries"]
    elif isinstance(data, dict):
        source = data
    else:
        return {}
    return {
        str(key): _normalize_yolo_model_metadata(value)
        for key, value in source.items()
        if isinstance(value, dict)
    }


def _infer_task_family_for_model(rel_path: str, metadata: dict) -> str:
    task_family = _sanitize_model_token(metadata.get("task_family", "")).lower()
    if task_family:
        return str(task_family)
    usage_role = _normalize_usage_role(metadata.get("usage_role", ""))
    if usage_role in {"obb_direct", "seq_crop_obb"}:
        return "obb"
    if usage_role == "seq_detect":
        return "detect"
    if usage_role in {"headtail", "cnn_identity", "colortag"}:
        return "classify"
    rel_lower = str(rel_path or "").replace("\\", "/").lower()
    if rel_lower.startswith("pose/"):
        return "pose"
    if rel_lower.startswith("detection/"):
        return "detect"
    if rel_lower.startswith("classification/"):
        return "classify"
    if rel_lower.startswith("obb/"):
        return "obb"
    return ""


def load_yolo_model_registry() -> object:
    """Load YOLO model metadata registry (path -> metadata)."""
    registry_path = get_yolo_model_registry_path()
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r") as f:
            data = json.load(f)
        return _extract_registry_entries(data)
    except Exception as e:
        logger.warning(f"Failed to load YOLO model registry: {e}")
        return {}


def save_yolo_model_registry(registry: object) -> object:
    """Persist YOLO model metadata registry JSON."""
    registry_path = get_yolo_model_registry_path()
    try:
        entries = _extract_registry_entries(registry)
        with open(registry_path, "w") as f:
            json.dump({"schema_version": 2, "entries": entries}, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save YOLO model registry: {e}")


def get_yolo_model_metadata(model_path: object) -> object:
    """Get metadata for a model path if registered."""
    rel_path = make_model_path_relative(model_path)
    registry = load_yolo_model_registry()
    return _normalize_yolo_model_metadata(registry.get(rel_path, {}))


def register_yolo_model(model_path: object, metadata: object) -> object:
    """Register/overwrite metadata entry for a model path."""
    rel_path = make_model_path_relative(model_path)
    registry = load_yolo_model_registry()
    normalized = _normalize_yolo_model_metadata(metadata)
    inferred_task_family = _infer_task_family_for_model(str(rel_path), normalized)
    if inferred_task_family and not normalized.get("task_family"):
        normalized["task_family"] = inferred_task_family
    registry[rel_path] = normalized
    save_yolo_model_registry(registry)


def unregister_yolo_model(model_path: object) -> bool:
    """Remove a model metadata entry from the registry when present."""
    rel_path = make_model_path_relative(model_path)
    registry = load_yolo_model_registry()
    if rel_path not in registry:
        return False
    registry.pop(rel_path, None)
    save_yolo_model_registry(registry)
    return True


def remove_model_from_repository(model_path: object) -> bool:
    """Delete a stored model file or directory and clear any matching registry entry."""
    path_str = str(model_path or "").strip()
    if not path_str:
        return False

    models_root = os.path.abspath(get_models_root_directory())
    if os.path.isabs(path_str):
        abs_path = os.path.abspath(path_str)
    else:
        abs_path = os.path.abspath(os.path.join(models_root, path_str))

    try:
        if os.path.commonpath([models_root, abs_path]) != models_root:
            logger.warning("Refusing to remove model outside repository: %s", abs_path)
            return False
    except ValueError:
        logger.warning("Refusing to remove model with incompatible path: %s", abs_path)
        return False

    removed_registry_entry = False
    try:
        rel_path = os.path.relpath(abs_path, models_root)
    except ValueError:
        rel_path = path_str
    registry = load_yolo_model_registry()
    if rel_path in registry:
        registry.pop(rel_path, None)
        save_yolo_model_registry(registry)
        removed_registry_entry = True

    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path)
        return True
    if os.path.isfile(abs_path):
        os.remove(abs_path)
        return True
    return removed_registry_entry
