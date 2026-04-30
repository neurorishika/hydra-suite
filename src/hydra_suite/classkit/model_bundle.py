"""Helpers for portable ClassKit model bundle discovery."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

MODEL_BUNDLE_TYPE = "classkit_model_bundle"
MODEL_BUNDLE_VERSION = 1
MODEL_BUNDLE_MANIFEST_SUFFIX = ".bundle.json"


def write_model_bundle_manifest(
    manifest_path: str | Path,
    *,
    mode: str,
    artifact_paths: list[str | Path],
    class_names: list[str] | None = None,
) -> Path:
    """Persist a portable model bundle manifest next to exported artifacts."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = manifest_path.parent.resolve()
    payload = {
        "bundle_type": MODEL_BUNDLE_TYPE,
        "bundle_version": MODEL_BUNDLE_VERSION,
        "mode": str(mode or ""),
        "class_names": list(class_names or []),
        "artifacts": [
            {
                "path": _relative_artifact_path(Path(path), base_dir),
            }
            for path in artifact_paths
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def discover_multihead_model_bundle(selected_path: str | Path) -> dict[str, Any] | None:
    """Discover a portable multi-head model bundle for *selected_path*.

    Resolution order:
    1. Any sibling ``*.bundle.json`` manifest that explicitly lists the artifact.
    2. Best-effort sibling discovery for older exports without manifests.
    """
    selected = Path(selected_path).expanduser().resolve()
    if not selected.exists() or not selected.is_file():
        return None

    manifest_bundle = _bundle_from_manifests(selected)
    if manifest_bundle is not None:
        return manifest_bundle

    fallback_paths = _discover_bundle_siblings(selected)
    if len(fallback_paths) < 2:
        return None

    mode = ""
    suffix = selected.suffix.lower()
    if suffix == ".pth":
        mode = "multihead_custom"
    elif suffix == ".pt":
        mode = "multihead_yolo"
    if not mode:
        return None
    return {
        "mode": mode,
        "artifact_paths": [str(path) for path in fallback_paths],
        "class_names": [],
    }


def _relative_artifact_path(path: Path, base_dir: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(base_dir).as_posix()
    except Exception:
        return resolved.name


def _bundle_from_manifests(selected: Path) -> dict[str, Any] | None:
    for manifest_path in sorted(
        selected.parent.glob(f"*{MODEL_BUNDLE_MANIFEST_SUFFIX}")
    ):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        if raw.get("bundle_type") != MODEL_BUNDLE_TYPE:
            continue
        mode = str(raw.get("mode") or "")
        if not mode.startswith("multihead"):
            continue
        artifact_paths = _resolve_manifest_artifacts(manifest_path, raw)
        resolved_paths = [path.resolve() for path in artifact_paths if path.exists()]
        if selected.resolve() not in resolved_paths or len(resolved_paths) < 2:
            continue
        return {
            "mode": mode,
            "artifact_paths": [str(path) for path in resolved_paths],
            "class_names": list(raw.get("class_names") or []),
        }
    return None


def _resolve_manifest_artifacts(manifest_path: Path, raw: dict[str, Any]) -> list[Path]:
    artifacts = raw.get("artifacts") or []
    resolved: list[Path] = []
    for item in artifacts:
        if isinstance(item, dict):
            rel_path = item.get("path")
        else:
            rel_path = item
        if not rel_path:
            continue
        candidate = (manifest_path.parent / Path(str(rel_path))).expanduser().resolve()
        resolved.append(candidate)
    return resolved


def _discover_bundle_siblings(selected: Path) -> list[Path]:
    suffix = selected.suffix.lower()
    if suffix not in {".pth", ".pt"}:
        return []

    prefixes = _candidate_prefixes(selected.stem)
    for prefix in prefixes:
        siblings = sorted(
            path.resolve()
            for path in selected.parent.iterdir()
            if path.is_file()
            and path.suffix.lower() == suffix
            and _matches_bundle_prefix(path.stem, prefix)
        )
        if selected.resolve() not in siblings or len(siblings) < 2:
            continue
        if suffix == ".pth" and not all(
            _looks_like_classkit_checkpoint(path) for path in siblings
        ):
            continue
        return siblings
    return []


def _candidate_prefixes(stem: str) -> list[str]:
    candidates = []
    for value in (
        re.sub(r"_\d+$", "", stem),
        stem.rsplit("_", 1)[0] if "_" in stem else "",
        (
            re.sub(r"_\d+$", "", stem).rsplit("_", 1)[0]
            if "_" in re.sub(r"_\d+$", "", stem)
            else ""
        ),
    ):
        value = str(value or "").strip("_")
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def _matches_bundle_prefix(stem: str, prefix: str) -> bool:
    return stem == prefix or stem.startswith(prefix + "_")


def _looks_like_classkit_checkpoint(path: Path) -> bool:
    try:
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return False
    return isinstance(ckpt, dict) and (
        "model_state_dict" in ckpt or "arch" in ckpt or "class_names" in ckpt
    )
