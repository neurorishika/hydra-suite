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

_CLASSIFIER_ROLE_SET = {
    TrainingRole.CLASSIFY_FLAT_YOLO,
    TrainingRole.CLASSIFY_FLAT_TINY,
    TrainingRole.CLASSIFY_MULTIHEAD_YOLO,
    TrainingRole.CLASSIFY_MULTIHEAD_TINY,
    TrainingRole.CLASSIFY_FLAT_CUSTOM,
    TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM,
}

_TRACKERKIT_MULTIHEAD_KIND = "classifier_multihead_bundle"


def _normalize_bundle_factor_entries(
    factor_entries: list[dict[str, Any]],
    *,
    manifest_dir: Path,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in factor_entries:
        factor_name = str(entry.get("factor") or "").strip()
        factor_path = Path(str(entry.get("path") or "")).expanduser().resolve()
        try:
            rel_path = factor_path.relative_to(manifest_dir).as_posix()
        except ValueError:
            rel_path = factor_path.name
        normalized.append(
            {
                "factor": factor_name,
                "path": rel_path,
                "class_names": [str(name) for name in entry.get("class_names") or []],
            }
        )
    return normalized


def write_classifier_multihead_manifest(
    manifest_path: str | Path,
    *,
    factor_entries: list[dict[str, Any]],
    input_size: tuple[int, int],
    monochrome: bool,
    recommended_confidence_threshold: float | None = None,
    kind: str = _TRACKERKIT_MULTIHEAD_KIND,
) -> Path:
    """Write a TrackerKit-readable multi-head classifier manifest.

    Each factor entry must provide ``factor``, ``path``, and ``class_names``.
    The stored paths are relative to the manifest location when possible.
    """
    manifest_abs = Path(manifest_path).expanduser().resolve()
    manifest_abs.parent.mkdir(parents=True, exist_ok=True)
    normalized_entries = _normalize_bundle_factor_entries(
        factor_entries,
        manifest_dir=manifest_abs.parent,
    )
    payload = {
        "schema_version": 2,
        "kind": str(kind or _TRACKERKIT_MULTIHEAD_KIND),
        "factor_names": [entry["factor"] for entry in normalized_entries],
        "factor_models": normalized_entries,
        "input_size": [int(input_size[0]), int(input_size[1])],
        "monochrome": bool(monochrome),
    }
    if recommended_confidence_threshold is not None:
        payload["recommended_confidence_threshold"] = float(
            min(1.0, max(0.0, recommended_confidence_threshold))
        )
    manifest_abs.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_abs


def _load_classifier_v2_sidecar(artifact_path: str | Path) -> dict[str, Any] | None:
    sidecar_path = Path(artifact_path).with_suffix(".v2meta.json")
    if not sidecar_path.exists():
        return None
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("schema_version") != 2:
        return None
    return data


def _normalize_classifier_meta(meta: dict[str, Any]) -> dict[str, Any]:
    raw_input_size = meta.get("input_size", [224, 224])
    if isinstance(raw_input_size, int):
        input_size = [int(raw_input_size), int(raw_input_size)]
    else:
        input_size = [int(raw_input_size[0]), int(raw_input_size[1])]

    factor_names = [str(name) for name in meta.get("factor_names") or ["flat"]]
    class_names_per_factor = [
        [str(name) for name in class_names]
        for class_names in (meta.get("class_names_per_factor") or [])
    ]
    if not class_names_per_factor:
        class_names_per_factor = [[str(name) for name in meta.get("class_names") or []]]

    normalized = {
        "schema_version": 2,
        "arch": str(meta.get("arch") or ""),
        "factor_names": factor_names,
        "class_names_per_factor": class_names_per_factor,
        "input_size": input_size,
        "monochrome": bool(meta.get("monochrome", False)),
        "num_classes": sum(len(names) for names in class_names_per_factor),
    }
    recommended_confidence_threshold = meta.get("recommended_confidence_threshold")
    if recommended_confidence_threshold is not None:
        try:
            normalized["recommended_confidence_threshold"] = float(
                min(1.0, max(0.0, float(recommended_confidence_threshold)))
            )
        except (TypeError, ValueError):
            pass
    if len(class_names_per_factor) == 1:
        normalized["class_names"] = list(class_names_per_factor[0])
    return normalized


def classifier_metadata_for_artifact(
    artifact_path: str | Path,
    *,
    fallback_input_size: tuple[int, int] | None = None,
    fallback_monochrome: bool | None = None,
) -> dict[str, Any]:
    """Return TrackerKit classifier metadata for an artifact.

    For ``.pth`` and ``.multihead.json`` artifacts this reads the metadata via
    ``ClassifierBackend``. For YOLO ``.pt`` artifacts it prefers a sibling
    ``.v2meta.json`` sidecar and otherwise falls back to the model's class names
    plus caller-provided size/monochrome hints.
    """
    path = Path(artifact_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if path.name.lower().endswith(".multihead.json") or suffix == ".pth":
        from hydra_suite.core.identity.classification.backend import ClassifierBackend
        from hydra_suite.runtime.resolver import ResolvedBackend

        backend = ClassifierBackend(str(path), ResolvedBackend("torch", "cpu", False))
        try:
            meta = backend.metadata
        finally:
            backend.close()
        return _normalize_classifier_meta(
            {
                "arch": meta.arch,
                "factor_names": meta.factor_names,
                "class_names_per_factor": meta.class_names_per_factor,
                "input_size": list(meta.input_size),
                "monochrome": meta.monochrome,
                "recommended_confidence_threshold": meta.recommended_confidence_threshold,
            }
        )

    if suffix == ".pt":
        sidecar = _load_classifier_v2_sidecar(path) or {}
        if sidecar.get("class_names_per_factor"):
            return _normalize_classifier_meta(sidecar)

        from ultralytics import YOLO

        yolo = YOLO(str(path))
        names = getattr(yolo, "names", None) or {}
        inferred_class_names = [str(names[i]) for i in sorted(names.keys())]
        del yolo

        input_size = sidecar.get("input_size")
        if not input_size:
            if fallback_input_size is not None:
                input_size = [int(fallback_input_size[0]), int(fallback_input_size[1])]
            else:
                input_size = [224, 224]

        return _normalize_classifier_meta(
            {
                "arch": sidecar.get("arch") or "yolo",
                "factor_names": sidecar.get("factor_names") or ["flat"],
                "class_names_per_factor": sidecar.get("class_names_per_factor")
                or [inferred_class_names],
                "input_size": input_size,
                "monochrome": sidecar.get(
                    "monochrome",
                    (
                        bool(fallback_monochrome)
                        if fallback_monochrome is not None
                        else False
                    ),
                ),
            }
        )

    raise RuntimeError(f"Unsupported classifier artifact: {path}")


def _copy_classifier_sidecar(src: Path, dst: Path) -> bool:
    src_sidecar = src.with_suffix(".v2meta.json")
    if not src_sidecar.exists():
        return False
    dst_sidecar = dst.with_suffix(".v2meta.json")
    shutil.copy2(src_sidecar, dst_sidecar)
    return True


def _sanitize_classifier_filename_token(text: object) -> str:
    raw = str(text or "").strip()
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw)
    return cleaned.strip("_")


def _classifier_artifact_suffix(path: Path) -> str:
    name_lower = path.name.lower()
    if name_lower.endswith(".multihead.json"):
        return ".multihead.json"
    return path.suffix.lower() or ".pth"


def _dedupe_classifier_artifact_path(
    dest_dir: Path, base_name: str, suffix: str
) -> Path:
    candidate = dest_dir / f"{base_name}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = dest_dir / f"{base_name}_{counter}{suffix}"
        counter += 1
    return candidate


def _copy_single_classifier_artifact(
    source_path: Path,
    *,
    dest_dir: Path,
    base_name: str,
) -> Path:
    dest_path = _dedupe_classifier_artifact_path(
        dest_dir,
        base_name,
        _classifier_artifact_suffix(source_path),
    )
    shutil.copy2(source_path, dest_path)
    if source_path.suffix.lower() == ".pt":
        _copy_classifier_sidecar(source_path, dest_path)
    return dest_path


def _copy_classifier_artifact_to_repository(
    source_path: Path,
    *,
    dest_dir: Path,
    base_name: str,
) -> Path:
    source_name_lower = source_path.name.lower()
    if not source_name_lower.endswith(".multihead.json"):
        return _copy_single_classifier_artifact(
            source_path,
            dest_dir=dest_dir,
            base_name=base_name,
        )

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 2:
        raise RuntimeError(
            f"unsupported classifier bundle manifest: {source_path.name}"
        )
    factor_models = data.get("factor_models")
    if not isinstance(factor_models, list) or not factor_models:
        raise RuntimeError(f"bundle manifest missing factor_models: {source_path.name}")

    bundle_meta = classifier_metadata_for_artifact(source_path)
    factor_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(factor_models):
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"bundle manifest factor entry {index} is not a dict: {source_path.name}"
            )
        rel_factor_path = str(entry.get("path") or "").strip()
        if not rel_factor_path:
            raise RuntimeError(
                f"bundle manifest factor entry {index} missing path: {source_path.name}"
            )
        factor_source = (source_path.parent / rel_factor_path).expanduser().resolve()
        if not factor_source.exists():
            raise RuntimeError(
                f"bundle manifest factor missing on disk: {factor_source.name}"
            )
        if factor_source.name.lower().endswith(".multihead.json"):
            raise RuntimeError(
                f"nested multi-head manifests are unsupported: {factor_source.name}"
            )

        factor_name = str(bundle_meta["factor_names"][index])
        factor_dest = _copy_single_classifier_artifact(
            factor_source,
            dest_dir=dest_dir,
            base_name=(
                f"{base_name}_{_sanitize_classifier_filename_token(factor_name) or f'factor_{index + 1}'}"
            ),
        )
        factor_entries.append(
            {
                "factor": factor_name,
                "path": factor_dest,
                "class_names": list(bundle_meta["class_names_per_factor"][index]),
            }
        )

    manifest_dest = _dedupe_classifier_artifact_path(
        dest_dir,
        base_name,
        ".multihead.json",
    )
    return write_classifier_multihead_manifest(
        manifest_dest,
        factor_entries=factor_entries,
        input_size=(
            int(bundle_meta["input_size"][0]),
            int(bundle_meta["input_size"][1]),
        ),
        monochrome=bool(bundle_meta.get("monochrome", False)),
        kind=str(data.get("kind") or _TRACKERKIT_MULTIHEAD_KIND),
    )


def import_classifier_artifact(
    *,
    source_path: str | Path,
    usage_role: str,
    species: str,
    classification_label: str = "",
    description: str = "",
    scoring_mode: str = "atomic",
) -> str:
    """Register a classifier artifact, copying it into TrackerKit storage when needed."""
    from hydra_suite.core.identity.classification.errors import ClassifierFormatError
    from hydra_suite.core.identity.classification.headtail import (
        validate_headtail_labels,
    )

    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"classifier artifact not found: {source}")

    role = str(usage_role or "").strip().lower()
    if role not in {"cnn_identity", "head_tail"}:
        raise RuntimeError(f"unsupported classifier usage role: {usage_role!r}")

    preview_meta = classifier_metadata_for_artifact(source)
    if role == "head_tail" and len(preview_meta.get("factor_names") or []) != 1:
        raise ClassifierFormatError("head-tail requires a flat classifier artifact")

    models_root = get_models_root().resolve()
    try:
        stored_abs = models_root / source.relative_to(models_root)
    except ValueError:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        descriptor = description if role == "head_tail" else classification_label
        name_parts = [
            timestamp,
            _sanitize_classifier_filename_token(
                preview_meta.get("arch") or "classifier"
            )
            or "classifier",
            _sanitize_classifier_filename_token(species or "unknown") or "unknown",
        ]
        descriptor_token = _sanitize_classifier_filename_token(descriptor)
        if descriptor_token:
            name_parts.append(descriptor_token)
        dest_dir = models_root / (
            "classification/orientation"
            if role == "head_tail"
            else "classification/identity"
        )
        dest_dir.mkdir(parents=True, exist_ok=True)
        stored_abs = _copy_classifier_artifact_to_repository(
            source,
            dest_dir=dest_dir,
            base_name="_".join(name_parts),
        )

    rel_path = stored_abs.relative_to(models_root).as_posix()
    metadata = (
        preview_meta
        if stored_abs == source
        else classifier_metadata_for_artifact(stored_abs)
    )
    if role == "head_tail":
        normalized = validate_headtail_labels(metadata["class_names_per_factor"][0])
        metadata["factor_names"] = ["flat"]
        metadata["class_names_per_factor"] = [normalized]
        metadata["class_names"] = list(normalized)
        metadata["num_classes"] = len(normalized)

    entry = dict(metadata)
    entry.update(
        {
            "species": str(species or "unknown").strip() or "unknown",
            "added_at": datetime.now().isoformat(timespec="seconds"),
            "task_family": "classify",
            "usage_role": role,
        }
    )
    if role == "cnn_identity":
        entry["classification_label"] = str(classification_label or "").strip()
        entry["scoring_mode"] = (
            "atomic"
            if len(entry.get("factor_names") or []) <= 1
            else str(scoring_mode or "atomic")
        )
    else:
        entry["description"] = str(description or "").strip()

    registry = load_model_registry()
    if registry.get("schema_version") == 2 and isinstance(
        registry.get("entries"), dict
    ):
        entries = dict(registry["entries"])
    else:
        entries = {
            str(key): value
            for key, value in registry.items()
            if isinstance(value, dict)
        }
    entries[rel_path] = entry
    save_model_registry(entries)
    return rel_path


def _is_classifier_role(role: TrainingRole) -> bool:
    return role in _CLASSIFIER_ROLE_SET


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

    def _covered_by_multihead_manifest(path: Path) -> bool:
        if path.suffix.lower() not in {".pth", ".pt"}:
            return False
        candidate_manifests = list(path.parent.glob("*.multihead.json"))
        if path.parent.parent.exists():
            candidate_manifests.extend(path.parent.parent.glob("*.multihead.json"))
        abs_path = path.resolve()
        for manifest_path in candidate_manifests:
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            factor_models = data.get("factor_models")
            if not isinstance(factor_models, list):
                continue
            for entry in factor_models:
                rel = entry.get("path") if isinstance(entry, dict) else None
                if not rel:
                    continue
                factor_path = (manifest_path.parent / str(rel)).expanduser().resolve()
                if factor_path == abs_path:
                    return True
        return False

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
            if _covered_by_multihead_manifest(path):
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


def count_legacy_registry_entries() -> int:
    """Return the number of registry entries missing v2-required fields.

    These are skipped silently by ``iter_registry_entries`` but reported via
    the GUI banner so users know why a model disappeared.
    """
    data = load_model_registry()
    if not data:
        return 0
    if data.get("schema_version") != 2 or not isinstance(data.get("entries"), dict):
        # Whole file is pre-v2 — every entry is legacy.
        count = sum(1 for v in data.values() if isinstance(v, dict))
        return count
    count = 0
    for meta in data["entries"].values():
        if not isinstance(meta, dict):
            continue
        task_family = str(meta.get("task_family") or "").strip().lower()
        if task_family and task_family != "classify":
            continue
        required = (
            "schema_version",
            "factor_names",
            "class_names_per_factor",
            "input_size",
        )
        if any(k not in meta for k in required) or meta.get("schema_version") != 2:
            count += 1
    return count


def iter_registry_entries() -> "Iterable[tuple[str, dict[str, Any]]]":
    """Yield ``(key, metadata)`` pairs from the v2-rooted registry.

    Flat-root registries from the Phase 1-6 rollout are no longer accepted.
    The legacy entries appear in the GUI banner and must be re-imported.
    """
    data = load_model_registry()
    if not data:
        return
    if data.get("schema_version") != 2 or not isinstance(data.get("entries"), dict):
        return
    for key, meta in data["entries"].items():
        if isinstance(meta, dict):
            yield str(key), meta


def save_model_registry(registry: dict[str, Any]) -> None:
    """Persist the published-model registry dict to disk in the v2 root shape."""
    # Callers may pass either the bare entries dict or a v2-shaped dict.
    if "schema_version" in registry and "entries" in registry:
        payload = dict(registry)
        payload["schema_version"] = 2
    else:
        payload = {"schema_version": 2, "entries": dict(registry)}
    _registry_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    dst_sidecar = dst.with_suffix(".v2meta.json")

    shutil.copy2(src, dst)

    classifier_meta: dict[str, Any] | None = None
    if _is_classifier_role(role):
        try:
            classifier_meta = _normalize_classifier_meta(
                dict(classifier_v2_meta)
                if classifier_v2_meta
                else classifier_metadata_for_artifact(str(src))
            )
        except Exception:
            classifier_meta = None
        if dst.suffix.lower() == ".pt":
            if classifier_meta is not None:
                dst_sidecar.write_text(
                    json.dumps(classifier_meta, indent=2),
                    encoding="utf-8",
                )
            else:
                _copy_classifier_sidecar(src, dst)

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
    if classifier_meta:
        metadata.update(classifier_meta)

    # v2 sidecar for YOLO-style artifacts whose weight file cannot embed our schema.
    if dst.suffix.lower() == ".pt" and dst_sidecar.exists():
        metadata["v2_sidecar"] = dst_sidecar.name

    reg = load_model_registry()
    if reg.get("schema_version") == 2 and isinstance(reg.get("entries"), dict):
        reg["entries"][key] = metadata
    else:
        reg = {"schema_version": 2, "entries": {key: metadata}}
    save_model_registry(reg)

    return key, str(dst)
