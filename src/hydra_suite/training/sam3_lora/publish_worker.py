"""Torch-owning SAM3 publish worker.

This module is imported only inside the protected publish sidecar (or focused
unit tests). The parent launcher lives in :mod:`.publish` and remains free of
Torch/SAM3 imports.

PyTorch's monolithic ``torch.save`` checkpoint format has no supported
state-dict streaming writer. This implementation therefore keeps one base
state plus one active merge tensor and relies on Torch's incremental zip-file
serialization; a future true streaming design requires a versioned sharded or
safetensors checkpoint contract and a coordinated consumer migration.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import torch

from hydra_suite.runtime.sam3_checkpoint_guard import (
    assert_sam3_checkpoint_loaded,
    tensor_sha256,
)
from hydra_suite.utils.sam3_constants import PREDICTOR_IMGSZ

from .lora import (
    _validated_adapter_pairs,
    adapter_touched_keys,
    lora_config_from_params,
    merge_adapters,
)

logger = logging.getLogger(__name__)

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ATTEMPT_ID = re.compile(r"[0-9a-f]{32}\Z")


def stripped_keys(state_dict: dict[str, Any]) -> list[str]:
    """Reproduce ultralytics' substring filter and replacement exactly."""

    return [key.replace("detector.", "") for key in state_dict if "detector" in key]


def _load_base_checkpoint(base_checkpoint: Path) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc_weights_only:
        logger.warning(
            "SAM3 base checkpoint %s failed weights-only loading (%s); "
            "retrying the trusted first-party checkpoint with pickle enabled",
            base_checkpoint,
            exc_weights_only,
        )
        try:
            loaded = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        except Exception as exc_fallback:
            raise RuntimeError(
                f"Failed to load SAM3 base checkpoint {base_checkpoint} with "
                "both weights_only=True and weights_only=False."
            ) from exc_fallback
    if (
        isinstance(loaded, dict)
        and "model" in loaded
        and isinstance(loaded["model"], dict)
    ):
        loaded = loaded["model"]
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError("SAM3 base checkpoint did not contain a non-empty state dict")
    if any(
        not isinstance(key, str) or not torch.is_tensor(value)
        for key, value in loaded.items()
    ):
        raise ValueError("SAM3 base checkpoint state contains a non-tensor entry")
    return loaded


def _save_checkpoint(state_dict: dict[str, torch.Tensor], staged_path: Path) -> None:
    with staged_path.open("xb") as stream:
        torch.save(state_dict, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _write_sidecar(metadata: dict[str, Any], staged_path: Path) -> None:
    with staged_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_staged_artifact(
    staged_artifact: Path,
    metadata: dict[str, Any],
    *,
    expected_keys: tuple[str, ...],
    expected_dtypes: dict[str, torch.dtype],
) -> None:
    """Reload through the serialized file and exercise the consumer guard."""

    reloaded = torch.load(
        staged_artifact, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(reloaded, dict):
        raise RuntimeError("staged SAM3 checkpoint did not reload as a state dict")
    if tuple(reloaded) != expected_keys:
        raise RuntimeError("staged SAM3 checkpoint changed its state-dict key layout")
    for key, expected_dtype in expected_dtypes.items():
        value = reloaded.get(key)
        if not torch.is_tensor(value) or value.dtype != expected_dtype:
            raise RuntimeError(f"staged SAM3 checkpoint changed dtype for {key!r}")
    live_state = {
        key.replace("detector.", ""): value
        for key, value in reloaded.items()
        if "detector" in key
    }
    assert_sam3_checkpoint_loaded(live_state, metadata, imgsz=PREDICTOR_IMGSZ)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _promote_without_overwrite(source: Path, target: Path) -> None:
    """Atomically expose *source* while refusing a raced existing target."""

    os.link(source, target)
    source.unlink()


_atomic_replace = _promote_without_overwrite


def _promote_staged_pair(
    staged_artifact: Path,
    staged_sidecar: Path,
    artifact_path: Path,
    sidecar_path: Path,
) -> None:
    """Promote metadata first and checkpoint last, rolling back on failure."""

    promoted_sidecar = False
    promoted_artifact = False
    try:
        _atomic_replace(staged_sidecar, sidecar_path)
        promoted_sidecar = True
        _atomic_replace(staged_artifact, artifact_path)
        promoted_artifact = True
        _fsync_directory(artifact_path.parent)
    except BaseException:
        # Final-name collisions are rejected before work starts, so anything
        # promoted here belongs to this failed attempt and is safe to remove.
        if promoted_sidecar:
            sidecar_path.unlink(missing_ok=True)
        if promoted_artifact:
            artifact_path.unlink(missing_ok=True)
        _fsync_directory(sidecar_path.parent)
        raise


def _artifact_paths(models_root: Path, run_id: str) -> tuple[Path, Path]:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("SAM3 publish run_id is not a safe artifact name")
    artifact = models_root / "sam3_finetuned" / f"{run_id}.pt"
    return artifact, artifact.with_name(artifact.name + ".sam3_meta.json")


def publish_sam3_artifact(
    *,
    run_id: str,
    adapters_path: str | Path,
    base_checkpoint: str | Path,
    build_manifest: dict[str, Any],
    params: Any,
    source_fingerprint: str,
    models_root: str | Path,
    publish_attempt_id: str | None = None,
) -> tuple[Path, Path]:
    """Build, validate, fsync, and atomically expose one merged checkpoint."""

    artifact_path, sidecar_path = _artifact_paths(Path(models_root), run_id)
    if artifact_path.exists() or sidecar_path.exists():
        raise FileExistsError(
            f"SAM3 publish target already exists for run {run_id!r}; refusing "
            "to overwrite a previously published artifact"
        )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    publish_attempt_id = publish_attempt_id or uuid.uuid4().hex
    if not _ATTEMPT_ID.fullmatch(publish_attempt_id):
        raise ValueError("SAM3 publish attempt identity is invalid")
    staged_artifact = artifact_path.with_name(
        f".{artifact_path.name}.{publish_attempt_id}.tmp"
    )
    staged_sidecar = sidecar_path.with_name(
        f".{sidecar_path.name}.{publish_attempt_id}.tmp"
    )

    try:
        base = _load_base_checkpoint(Path(base_checkpoint))
        adapters = torch.load(
            Path(adapters_path), map_location="cpu", weights_only=True
        )
        if not isinstance(adapters, dict) or not adapters:
            raise RuntimeError(
                "adapters.pt is empty or invalid; refusing to publish a stock-equivalent model"
            )
        cfg = lora_config_from_params(params)
        # Validate every mapping, pair, rank, shape, and dtype before the first
        # in-place write. This is intentionally separate from merge_adapters'
        # own defensive validation because the original fingerprints below
        # must describe a wholly unmodified checkpoint.
        _validated_adapter_pairs(base, adapters, cfg, prefix="detector.")
        touched_keys = adapter_touched_keys(adapters, base)
        fingerprint_keys = sorted(touched_keys)[:3]
        original_fingerprints = {
            key: tensor_sha256(base[key]) for key in fingerprint_keys
        }
        expected_keys = tuple(base)
        expected_dtypes = {key: value.dtype for key, value in base.items()}

        merged = merge_adapters(base, adapters, cfg)
        tuned_fingerprints = {
            key.replace("detector.", ""): tensor_sha256(merged[key])
            for key in fingerprint_keys
        }
        if not any(
            tuned_fingerprints[key.replace("detector.", "")]
            != original_fingerprints[key]
            for key in fingerprint_keys
        ):
            raise RuntimeError(
                "Merge produced a checkpoint indistinguishable from stock SAM3: "
                "no adapter-touched tensor changed after consumer-normalized hashing."
            )

        metadata = {
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
            # Lets the parent clean up after a hard-killed child without ever
            # deleting an artifact that raced into the same final pathname.
            "publish_attempt_id": publish_attempt_id,
        }
        logger.info(
            "sam3 publish: merged %d adapter targets into %d base keys "
            "(%d keys carried through untouched)",
            len(touched_keys),
            len(merged),
            len(merged) - len(touched_keys),
        )
        _save_checkpoint(merged, staged_artifact)
        _write_sidecar(metadata, staged_sidecar)

        # Serialization is complete. Drop the in-memory checkpoint before the
        # mmap reload so validation never retains two checkpoint-sized states.
        del merged, base, adapters
        gc.collect()
        _validate_staged_artifact(
            staged_artifact,
            metadata,
            expected_keys=expected_keys,
            expected_dtypes=expected_dtypes,
        )
        _promote_staged_pair(
            staged_artifact,
            staged_sidecar,
            artifact_path,
            sidecar_path,
        )
        return artifact_path, sidecar_path
    finally:
        staged_artifact.unlink(missing_ok=True)
        staged_sidecar.unlink(missing_ok=True)
