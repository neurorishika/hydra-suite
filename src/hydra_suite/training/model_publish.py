"""Model publishing into MAT model repositories with metadata lineage."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .contracts import TrainingRole


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _use_project_root_override() -> bool:
    return getattr(_project_root, "__module__", __name__) != __name__


def get_models_root() -> Path:
    """Return the platform-appropriate directory where published trained models are stored."""
    if _use_project_root_override():
        root = _project_root() / "models"
        root.mkdir(parents=True, exist_ok=True)
        return root

    from hydra_suite.paths import get_models_dir

    return get_models_dir()


def _repo_dir_for_role(role: TrainingRole, scheme_name: str = "classkit") -> Path:
    root = get_models_root()
    if role == TrainingRole.OBB_DIRECT:
        out = root / "YOLO-obb"
    elif role == TrainingRole.SEQ_DETECT:
        out = root / "YOLO-detect"
    elif role == TrainingRole.SEQ_CROP_OBB:
        out = root / "YOLO-obb" / "cropped"
    elif role == TrainingRole.CLASSIFY_FLAT_YOLO:
        out = root / "YOLO-classify" / scheme_name
    elif role == TrainingRole.CLASSIFY_FLAT_TINY:
        out = root / "tiny-classify" / scheme_name
    elif role == TrainingRole.CLASSIFY_MULTIHEAD_YOLO:
        out = root / "YOLO-classify" / "multihead" / scheme_name
    elif role == TrainingRole.CLASSIFY_MULTIHEAD_TINY:
        out = root / "tiny-classify" / "multihead" / scheme_name
    elif role == TrainingRole.CLASSIFY_FLAT_CUSTOM:
        out = root / "custom-classify" / scheme_name
    elif role == TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM:
        out = root / "custom-classify" / "multihead" / scheme_name
    else:
        raise RuntimeError(f"Unsupported publish role: {role.value}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _task_usage_for_role(role: TrainingRole) -> tuple[str, str]:
    if role == TrainingRole.OBB_DIRECT:
        return "obb", "obb_direct"
    if role == TrainingRole.SEQ_DETECT:
        return "detect", "seq_detect"
    if role == TrainingRole.SEQ_CROP_OBB:
        return "obb", "seq_crop_obb"
    if role in (TrainingRole.CLASSIFY_FLAT_YOLO, TrainingRole.CLASSIFY_MULTIHEAD_YOLO):
        return "classify", "classify_yolo"
    if role in (TrainingRole.CLASSIFY_FLAT_TINY, TrainingRole.CLASSIFY_MULTIHEAD_TINY):
        return "classify", "classify_tiny"
    if role in (
        TrainingRole.CLASSIFY_FLAT_CUSTOM,
        TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM,
    ):
        return "classify", "classify_custom"
    raise RuntimeError(f"Unsupported publish role: {role.value}")


def _registry_path() -> Path:
    return get_models_root() / "model_registry.json"


def load_model_registry() -> dict[str, Any]:
    """Load the published-model registry JSON, returning an empty dict on missing or corrupt files."""
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


_CNN_IDENTITY_ROOTS = (
    "classification/identity",
    "tiny-classify",
    "custom-classify",
    "YOLO-classify",
)

_HEAD_TAIL_ROOTS = (
    "classification/orientation",
    "tiny-classify",
    "custom-classify",
    "YOLO-classify",
)

_CLASSIFIER_EXTS = (".pth", ".pt", ".multihead.json")


def enumerate_classifier_artifacts(
    *,
    roles: tuple[str, ...],
) -> "Iterable[dict[str, Any]]":
    """Yield descriptors for every classifier artifact under any of the
    discovery roots associated with ``roles``.

    Each descriptor has keys:
        ``path`` (str, relative to models root, posix-style)
        ``abs_path`` (str)
        ``root`` (str, which top-level root it was discovered under)
        ``is_managed`` (bool — True when the file lives under a ClassKit
            publish root rather than the TrackerKit canonical
            ``classification/{identity,orientation}``)

    Callers validate per-role suitability via ``ClassifierBackend.metadata``.
    """
    models_root = get_models_root()
    seen: set[str] = set()
    role_roots: list[str] = []
    for role in roles:
        if role == "cnn_identity":
            role_roots.extend(_CNN_IDENTITY_ROOTS)
        elif role == "head_tail":
            role_roots.extend(_HEAD_TAIL_ROOTS)
    for root_rel in role_roots:
        root_abs = models_root / root_rel
        if not root_abs.exists():
            continue
        for path in root_abs.rglob("*"):
            if not path.is_file():
                continue
            name_lower = path.name.lower()
            if not any(name_lower.endswith(ext) for ext in _CLASSIFIER_EXTS):
                continue
            # For multi-head bundles, only yield the manifest — not the per-factor .pt files.
            if path.suffix.lower() == ".pt" and any(
                p.name.lower().endswith(".multihead.json")
                for p in path.parent.parent.glob("*.multihead.json")
            ):
                continue
            try:
                rel = path.relative_to(models_root).as_posix()
            except ValueError:
                continue
            if rel in seen:
                continue
            seen.add(rel)
            is_managed = root_rel not in (
                "classification/identity",
                "classification/orientation",
            )
            yield {
                "path": rel,
                "abs_path": str(path),
                "root": root_rel,
                "is_managed": is_managed,
            }


def iter_registry_entries() -> "Iterable[tuple[str, dict[str, Any]]]":
    """Yield ``(key, metadata)`` pairs from the registry, accepting both the
    current flat-root format and the future v2 root format.

    During phases 1–6 the registry is still written in flat-root format;
    this helper lets UI code read either layout transparently. Phase 7
    removes flat-root support.
    """
    data = load_model_registry()
    if not data:
        return
    if data.get("schema_version") == 2 and isinstance(data.get("entries"), dict):
        for key, meta in data["entries"].items():
            if isinstance(meta, dict):
                yield str(key), meta
        return
    for key, meta in data.items():
        if isinstance(meta, dict):
            yield str(key), meta


def save_model_registry(registry: dict[str, Any]) -> None:
    """Persist the published-model registry dict to disk as pretty-printed JSON."""
    _registry_path().write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _registry_key_for_model(model_path: Path) -> str:
    """Generate registry key compatible with existing main_window behavior."""

    model_path = model_path.resolve()
    models_root = get_models_root().resolve()
    yolo_obb_root = (models_root / "YOLO-obb").resolve()

    try:
        rel_obb = model_path.relative_to(yolo_obb_root)
        return rel_obb.as_posix()
    except Exception:
        pass

    try:
        rel_root = model_path.relative_to(models_root)
        return rel_root.as_posix()
    except Exception:
        return str(model_path)


def publish_trained_model(
    *,
    role: TrainingRole,
    artifact_path: str,
    size: str,
    species: str,
    model_info: str,
    trained_from_run_id: str,
    dataset_fingerprint: str,
    base_model: str,
    scheme_name: str = "",
    factor_index: int | None = None,
    factor_name: str | None = None,
    training_params: dict[str, Any] | None = None,
    classifier_v2_meta: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Copy trained artifact into repository and register metadata.

    Returns:
        (registry_key, absolute_model_path)
    """

    src = Path(artifact_path).expanduser().resolve()
    if not src.exists():
        raise RuntimeError(f"Trained artifact not found: {src}")

    repo_dir = _repo_dir_for_role(role, scheme_name=scheme_name or "classkit")
    task_family, usage_role = _task_usage_for_role(role)

    now = datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    added_at = now.isoformat(timespec="seconds")

    safe_species = (
        "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(species or "species")
        ).strip("_")
        or "species"
    )
    safe_info = (
        "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(model_info or "model")
        ).strip("_")
        or "model"
    )
    safe_size = (
        "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(size or "unknown")
        ).strip("_")
        or "unknown"
    )

    ext = src.suffix.lower() or ".pt"
    base_name = f"{stamp}_{safe_size}_{safe_species}_{safe_info}"
    dst = repo_dir / f"{base_name}{ext}"
    counter = 1
    while dst.exists():
        dst = repo_dir / f"{base_name}_{counter}{ext}"
        counter += 1

    shutil.copy2(src, dst)

    key = _registry_key_for_model(dst)
    metadata = {
        "size": safe_size,
        "species": safe_species,
        "model_info": safe_info,
        "added_at": added_at,
        "source_path": str(src),
        "stored_filename": dst.name,
        "task_family": task_family,
        "usage_role": usage_role,
        "trained_from_run_id": str(trained_from_run_id or ""),
        "dataset_fingerprint": str(dataset_fingerprint or ""),
        "base_model": str(base_model or ""),
        "scheme_name": str(scheme_name or ""),
        "factor_index": factor_index,
        "factor_name": str(factor_name) if factor_name is not None else None,
    }
    if training_params:
        metadata["training_params"] = dict(training_params)

    # v2 sidecar for YOLO-style artifacts whose weight file cannot embed our schema.
    if classifier_v2_meta:
        import json as _json

        sidecar = dst.with_suffix(".v2meta.json")
        payload = {"schema_version": 2, **dict(classifier_v2_meta)}
        sidecar.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        metadata["v2_sidecar"] = sidecar.name

    reg = load_model_registry()
    reg[key] = metadata
    save_model_registry(reg)

    return key, str(dst)
