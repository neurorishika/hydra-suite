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

from hydra_suite.utils.sam3_constants import PREDICTOR_IMGSZ

from .lora import adapter_touched_keys, lora_config_from_params, merge_adapters

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
    # Normalised to float32 BEFORE hashing, on both this side (publish) and
    # the consumer side (core.inference.semantic.sam3._tensor_sha256): the
    # merged checkpoint may be saved in bf16 (this repo's mixed_precision
    # default), but ultralytics reconstructs the live model and may cast
    # dtype during that build. Hashing raw bytes at two different dtypes
    # would make the guard raise on every CORRECTLY loaded checkpoint --
    # turning a safety net into an outage. `.float()` also sidesteps
    # `.numpy()` raising TypeError on bf16 (numpy has no bf16 dtype).
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().float().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _load_base_checkpoint(base_checkpoint: Path) -> dict[str, torch.Tensor]:
    try:
        base = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc_weights_only:
        # The stock SAM3 checkpoint is known (per the spike's reprefix.py)
        # to be a wrapper dict carrying non-tensor metadata that
        # weights_only=True's restricted unpickler rejects. Fall back to
        # weights_only=False -- safe here because base_checkpoint is a
        # trusted, first-party (Meta-distributed) artifact, not user input.
        # If it *still* fails, that's a genuine corruption/missing-file
        # error and must propagate, not be silenced.
        logger.warning(
            "SAM3 base checkpoint %s failed to load with weights_only=True "
            "(%s); retrying with weights_only=False (trusted first-party "
            "checkpoint).",
            base_checkpoint,
            exc_weights_only,
        )
        try:
            base = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        except Exception as exc_fallback:
            raise RuntimeError(
                f"Failed to load SAM3 base checkpoint {base_checkpoint} with "
                "both weights_only=True and weights_only=False."
            ) from exc_fallback
    if isinstance(base, dict) and "model" in base and isinstance(base["model"], dict):
        base = base["model"]
    return base


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
        # Same helper `model_publish._registry_path()` uses (get_models_root,
        # not paths.get_models_dir) so a SAM3 entry can never land in a
        # second registry file under a monkeypatched project root.
        from .model_publish import get_models_root

        models_root = get_models_root()
    models_root = Path(models_root)

    base = _load_base_checkpoint(Path(base_checkpoint))
    adapters = torch.load(Path(adapters_path), map_location="cpu", weights_only=True)
    if not adapters:
        raise RuntimeError(
            "adapters.pt is empty -- there is nothing to merge. Publishing it "
            "would silently produce a checkpoint byte-for-byte identical to "
            "stock SAM3 while reporting success."
        )

    cfg = lora_config_from_params(params)
    merged = merge_adapters(base, adapters, cfg)

    # Every key merge_adapters could touch is already present in `merged`
    # because it starts from a clone of `base` -- so stock-only keys (e.g.
    # the spike's 22 vision_backbone.sam2_convs.* tensors that Meta's builder
    # never instantiates) are carried across untouched automatically. Just
    # count and log them so a future regression is visible. Shared formula
    # with merge_adapters via adapter_touched_keys -- no re-derived prefix.
    touched_keys = adapter_touched_keys(adapters)
    untouched_count = len(set(merged) - touched_keys)
    discarded_on_load = len(merged) - len(stripped_keys(merged))
    logger.info(
        "sam3 publish: merged %d adapter tensors into %d base keys "
        "(%d keys carried across untouched, %d keys ultralytics' substring "
        "filter will discard on load)",
        len(touched_keys),
        len(merged),
        untouched_count,
        discarded_on_load,
    )

    fingerprint_keys = sorted(touched_keys)[:3]
    tuned_fingerprints = {k: _tensor_sha256(merged[k]) for k in fingerprint_keys}
    base_fingerprints = {
        k: _tensor_sha256(base[k]) for k in fingerprint_keys if k in base
    }
    unchanged = [
        k
        for k in fingerprint_keys
        if k in base_fingerprints and base_fingerprints[k] == tuned_fingerprints[k]
    ]
    if unchanged and set(unchanged) == set(fingerprint_keys):
        raise RuntimeError(
            "Merge produced a checkpoint indistinguishable from stock SAM3: "
            f"every sampled tuned key ({sorted(unchanged)}) hashes identically "
            "to base. This usually means lora_B is still at zero init (an "
            "aborted or zero-step training run) or the adapters are otherwise "
            "all-zero. Refusing to publish a model that would silently be a "
            "no-op."
        )

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
        # Stored under the STRIPPED namespace, matching `stripped_keys`
        # above and the live (post-load) model's state dict --
        # `tuned_fingerprints` here is built pre-strip (it indexes `merged`,
        # which still carries the `detector.` prefix ultralytics' load
        # transform later removes). A consumer guarding a live, already-
        # stripped state dict must never see the pre-strip namespace, or
        # every lookup KeyErrors instead of raising the intended refusal.
        "tuned_fingerprints": {
            k.replace("detector.", ""): v for k, v in tuned_fingerprints.items()
        },
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
