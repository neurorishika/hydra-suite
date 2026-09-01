"""Merge SAM3 LoRA adapters into the base checkpoint and publish the result.

Deliberately free of any import of the ``sam3``/ultralytics package: merging
is pure tensor arithmetic on state dicts, and this module must work on a
machine where the licence-gated SAM3 package is not installed (training
runs elsewhere; this seam only needs to run at publish time).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from hydra_suite.core.inference.semantic.sam3 import PREDICTOR_IMGSZ

from .lora import lora_config_from_params, merge_adapters

logger = logging.getLogger(__name__)


def stripped_keys(state_dict: dict[str, Any]) -> list[str]:
    """Reproduce ultralytics' ``build_sam3.py:357`` key transform exactly.

    That code is a SUBSTRING test, not a prefix strip:
    ``{k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}``.
    A prefix-only reimplementation would disagree on keys containing
    "detector" anywhere else in the dotted path.
    """
    return [k.replace("detector.", "") for k in state_dict if "detector" in k]


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def publish_sam3_model(
    *,
    run_id: str,
    adapters_path: str | Path,
    base_checkpoint: str | Path,
    build_manifest: dict[str, Any],
    params: Any,
    source_fingerprint: str,
    models_root: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> tuple[str, str]:
    """Merge adapters into base, write the merged checkpoint + sidecar, register.

    Returns:
        (registry_key, artifact_path) -- the same shape as
        ``model_publish.publish_trained_model``, because ``run_role_training``
        unpacks it as ``(published_key, published_path)``.
    """
    if models_root is None:
        from hydra_suite.paths import get_models_dir

        models_root = get_models_dir()
    models_root = Path(models_root)

    base = torch.load(Path(base_checkpoint), map_location="cpu", weights_only=True)
    if isinstance(base, dict) and "model" in base and isinstance(base["model"], dict):
        base = base["model"]
    adapters = torch.load(Path(adapters_path), map_location="cpu", weights_only=True)

    cfg = lora_config_from_params(params)
    merged = merge_adapters(base, adapters, cfg)

    # Every key merge_adapters could touch is already present in `merged`
    # because it starts from a clone of `base` -- so stock-only keys (e.g.
    # the spike's 22 vision_backbone.sam2_convs.* tensors that Meta's builder
    # never instantiates) are carried across untouched automatically. Just
    # count and log them so a future regression is visible.
    adapter_paths = sorted({k.rsplit(".", 1)[0] for k in adapters})
    touched_keys = {f"detector.{path}.weight" for path in adapter_paths}
    untouched_count = len(set(merged) - touched_keys)
    logger.info(
        "sam3 publish: merged %d adapter tensors into %d base keys "
        "(%d keys carried across untouched)",
        len(adapter_paths),
        len(merged),
        untouched_count,
    )

    fingerprint_keys = sorted(touched_keys)[:3]
    tuned_fingerprints = {k: _tensor_sha256(merged[k]) for k in fingerprint_keys}

    out_dir = models_root / "sam3_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / f"{run_id}.pt"
    torch.save(merged, artifact_path)

    sidecar = {
        "base_variant": "sam3",
        "prompt": getattr(params, "prompt", ""),
        "train_tile_px": build_manifest.get("tile_px"),
        "reference_body_px": build_manifest.get("reference_body_px"),
        "object_tile_fraction": build_manifest.get("object_tile_fraction"),
        "imgsz": PREDICTOR_IMGSZ,
        "stripped_keys": stripped_keys(merged),
        "tuned_fingerprints": tuned_fingerprints,
        "source_fingerprint": source_fingerprint,
        "label_quality_acknowledged": getattr(
            params, "label_quality_acknowledged", False
        ),
    }
    sidecar_path = artifact_path.with_name(artifact_path.name + ".sam3_meta.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    registry_path = (
        Path(registry_path)
        if registry_path is not None
        else models_root / "model_registry.json"
    )
    _register(
        registry_path=registry_path,
        artifact_path=artifact_path,
        sidecar_path=sidecar_path,
        run_id=run_id,
        prompt=sidecar["prompt"],
        source_fingerprint=source_fingerprint,
    )

    key = f"sam3_finetuned/{artifact_path.name}"
    return key, str(artifact_path)


def _register(
    *,
    registry_path: Path,
    artifact_path: Path,
    sidecar_path: Path,
    run_id: str,
    prompt: str,
    source_fingerprint: str,
) -> None:
    data: dict[str, Any] = {"schema_version": 2, "entries": {}}
    if registry_path.exists():
        try:
            loaded = json.loads(registry_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("entries"), dict):
                data = loaded
        except Exception:
            pass

    key = f"sam3_finetuned/{artifact_path.name}"
    data.setdefault("entries", {})[key] = {
        "task_family": "semantic",
        "usage_role": "semantic_sam3",
        "stored_filename": artifact_path.name,
        "stored_path": str(artifact_path),
        "sidecar_path": str(sidecar_path),
        "trained_from_run_id": run_id,
        "prompt": prompt,
        "dataset_fingerprint": source_fingerprint,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
