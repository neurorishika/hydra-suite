"""Dependency-light SAM3 checkpoint fingerprint and load guard."""

from __future__ import annotations

import hashlib
from typing import Any

_HASH_CHUNK_ELEMENTS = 1_048_576


def tensor_sha256(tensor: Any) -> str:
    """Hash consumer-normalized float32 bytes with bounded temporaries."""

    source = tensor.detach().cpu()
    if not source.is_contiguous():
        source = source.contiguous()
    flat = source.view(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), _HASH_CHUNK_ELEMENTS):
        chunk = flat[start : start + _HASH_CHUNK_ELEMENTS].float().contiguous()
        digest.update(chunk.view(-1).numpy().tobytes())
        del chunk
    return digest.hexdigest()


def assert_sam3_checkpoint_loaded(
    live_state_dict: dict,
    meta: dict | None,
    *,
    imgsz: int,
) -> None:
    """Refuse a finetuned checkpoint that did not populate the live model."""

    if meta is None:
        return
    stripped = set(meta.get("stripped_keys", []))
    live_keys = set(live_state_dict)
    if not live_keys or not stripped:
        raise RuntimeError(
            "SAM3 finetuned checkpoint cannot be guarded: "
            f"{len(live_keys)} live keys against {len(stripped)} recorded "
            "checkpoint keys. An empty side makes the coverage check pass "
            "vacuously -- refusing rather than serving unchecked."
        )
    uncovered = sorted(live_keys - stripped)
    if uncovered:
        raise RuntimeError(
            "SAM3 finetuned checkpoint failed to load: "
            f"{len(uncovered)} of {len(live_keys)} live model keys are not "
            f"present in the published checkpoint (e.g. {uncovered[:3]}). "
            "ultralytics' load transform likely changed, so those weights "
            "stayed stock SAM3 rather than coming from the checkpoint."
        )
    for key, expected_fp in meta.get("tuned_fingerprints", {}).items():
        if key not in live_state_dict:
            raise RuntimeError(
                f"SAM3 finetuned checkpoint failed to load: tuned key "
                f"{key!r} recorded at publish time is absent from the live "
                "model's state dict."
            )
        actual_fp = tensor_sha256(live_state_dict[key])
        if actual_fp != expected_fp:
            raise RuntimeError(
                f"SAM3 finetuned checkpoint failed to load: tensor {key!r} "
                f"has fingerprint {actual_fp}, expected {expected_fp}. The "
                "key is present but holds different weights than the "
                "published checkpoint -- the load likely fell back to base "
                "weights while reporting success."
            )
    if "imgsz" in meta and meta["imgsz"] is None:
        raise RuntimeError(
            "SAM3 finetuned checkpoint sidecar has imgsz=null. A published "
            "artifact must record the training imgsz to be guarded against "
            "a silent train/serve scale mismatch -- refusing rather than "
            "skipping the check."
        )
    meta_imgsz = meta.get("imgsz")
    if meta_imgsz is not None and meta_imgsz != imgsz:
        raise RuntimeError(
            f"SAM3 finetuned checkpoint was trained at imgsz={meta_imgsz} "
            f"but is being served at imgsz={imgsz}. This loads perfectly "
            "cleanly -- keys and tensors all match -- so nothing else in "
            "the system would ever notice the train/serve scale mismatch. "
            "Refusing rather than silently rescaling."
        )
