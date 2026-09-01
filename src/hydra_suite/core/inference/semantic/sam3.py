"""SAM3 promptable-concept-segmentation backend for the SemanticLabeler seam.

Wraps ultralytics' ``SAM3SemanticPredictor``. Construction is guarded by
``probe_availability`` so a missing dependency raises with an actionable
message instead of letting ultralytics AutoUpdate pip-install packages.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch

from hydra_suite.core.inference.masks import mask_to_contour
from hydra_suite.core.inference.torch_device import resolve_torch_device

from .base import SemanticInstance
from .checkpoints import DEFAULT_VARIANT, ensure_checkpoint, probe_availability

logger = logging.getLogger(__name__)

# Pinned rather than inherited. Ultralytics' BasePredictor.__init__ sets
# ``args.conf = 0.25`` when it is None, and ``postprocess`` filters
# ``pred_scores > args.conf`` BEFORE anything of ours runs -- so a cache
# advertised as "collected at floor 0.05" would in fact hold nothing below
# 0.25, and every calibration cell from 0.05 to 0.25 would be identical.
# The predictor floor must therefore be the floor WE asked for.
DEFAULT_CONFIDENCE_FLOOR = 0.05
# The predictor's own class-agnostic NMS IoU. Ultralytics defaults this to
# 0.7; pinned here so an upstream default change cannot silently alter what
# reaches our cross-tile merge (which applies its own, separate merge_iou).
PREDICTOR_NMS_IOU = 0.7
# Pinned for the same reason as PREDICTOR_NMS_IOU. ultralytics' default cfg
# imgsz is 640 -- rounded up to 644 for the stride-14 backbone -- but
# build_sam3.py builds the SAM3 architecture at img_size=1008 and
# BasePredictor calls model.set_imgsz(self.imgsz). Inheriting the default
# therefore runs a 1008-native model at 644 with no warning. It also makes
# train/serve scale disagree for any finetuned checkpoint.
PREDICTOR_IMGSZ = 1008


def predictor_overrides(
    checkpoint: Path | str,
    device: str,
    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
) -> dict:
    """The override dict handed to ``SAM3SemanticPredictor``.

    Split out from ``from_variant`` so it can be asserted on without a GPU,
    a checkpoint, or ultralytics installed.
    """
    return {
        "model": str(checkpoint),
        "device": device,
        "save": False,
        "verbose": False,
        # See DEFAULT_CONFIDENCE_FLOOR / PREDICTOR_NMS_IOU above.
        "conf": float(max(0.0, min(1.0, confidence_floor))),
        "iou": PREDICTOR_NMS_IOU,
        "imgsz": PREDICTOR_IMGSZ,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    # MUST match training.sam3_lora.publish._tensor_sha256 exactly: both
    # normalise to float32 before hashing, because the live model's dtype
    # after ultralytics reconstructs it may differ from what was saved to
    # disk. Hashing at two different dtypes would make the guard raise on
    # every correctly-loaded checkpoint. `.float()` also sidesteps
    # `.numpy()` raising TypeError on bf16.
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().float().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def assert_checkpoint_loaded(
    live_state_dict: dict,
    meta: dict | None,
    *,
    imgsz: int = PREDICTOR_IMGSZ,
) -> None:
    """Refuse to proceed on a load that "succeeded" but changed nothing.

    ultralytics' ``build_sam3.py`` loads a checkpoint with
    ``load_state_dict(strict=False)`` and discards the return value, so a
    checkpoint whose keys don't line up with the live model loads NOTHING
    and reports success -- the resident weights stay stock. This guard
    catches that (and the silent train/serve imgsz mismatch) by comparing
    the live model's state dict against the sidecar's recorded expectations.

    *meta* is ``None`` for a stock variant: it ships no sidecar and makes no
    claim about what should be resident, so there is nothing to guard.

    The key check runs LIVE-TO-CHECKPOINT, not the reverse: every key the
    live model exposes must be covered by the published checkpoint. The
    opposite direction is not an invariant -- a checkpoint is legitimately a
    SUPERSET of the semantic predictor's live state dict, carrying
    non-persistent RoPE buffers (``*.attn.freqs_cis``) and the point-prompt
    ``geometry_encoder.points_*`` projections that the semantic build never
    instantiates. Requiring those to be resident refused every correctly
    published model. Coverage still catches the failure this guard exists
    for: if ultralytics' key transform drifts, the namespaces stop
    overlapping and nearly every live key goes uncovered.
    """
    if meta is None:
        return
    stripped = set(meta.get("stripped_keys", []))
    live_keys = set(live_state_dict)
    if not live_keys or not stripped:
        # A coverage test over an empty set passes trivially -- exactly the
        # silent no-op this guard exists to prevent.
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
        # `tuned_fingerprints` keys are recorded in the STRIPPED namespace
        # (training.sam3_lora.publish.publish_sam3_model normalises them to
        # match `stripped_keys` / the live, post-load state dict) -- a bare
        # subscript here would KeyError instead of raising the intended
        # refusal if that namespace convention is ever violated again.
        if key not in live_state_dict:
            raise RuntimeError(
                f"SAM3 finetuned checkpoint failed to load: tuned key "
                f"{key!r} recorded at publish time is absent from the live "
                "model's state dict."
            )
        actual_fp = _tensor_sha256(live_state_dict[key])
        if actual_fp != expected_fp:
            raise RuntimeError(
                f"SAM3 finetuned checkpoint failed to load: tensor {key!r} "
                f"has fingerprint {actual_fp}, expected {expected_fp}. The "
                "key is present but holds different weights than the "
                "published checkpoint -- the load likely fell back to base "
                "weights while reporting success."
            )
    if "imgsz" in meta and meta["imgsz"] is None:
        # An explicit ``null`` is not "absent" -- publish.py always writes a
        # concrete imgsz, so a null here means the sidecar was corrupted or
        # hand-edited. Silently skipping the comparison would defeat the
        # very guard this field exists for.
        raise RuntimeError(
            "SAM3 finetuned checkpoint sidecar has imgsz=null. A published "
            "artifact must record the training imgsz to be guarded against "
            "a silent train/serve scale mismatch -- refusing rather than "
            "skipping the check."
        )
    meta_imgsz = meta.get("imgsz")
    if meta_imgsz is not None and meta_imgsz != imgsz:
        # This can only fire if PREDICTOR_IMGSZ changes between publish time
        # and serve time: publish.py stamps the sidecar with the SAME
        # PREDICTOR_IMGSZ constant this module defines, so today the two
        # always agree. It is not dead code -- it is the guard against this
        # constant ever drifting (or a future multi-imgsz world) without
        # anyone noticing the train/serve scale disagreement.
        raise RuntimeError(
            f"SAM3 finetuned checkpoint was trained at imgsz={meta_imgsz} "
            f"but is being served at imgsz={imgsz}. This loads perfectly "
            "cleanly -- keys and tensors all match -- so nothing else in "
            "the system would ever notice the train/serve scale mismatch. "
            "Refusing rather than silently rescaling."
        )


def _sidecar_for_checkpoint(checkpoint: Path) -> dict:
    """Read the ``<artifact>.sam3_meta.json`` sidecar next to *checkpoint*.

    Mirrors the naming convention ``training.sam3_lora.publish`` writes
    (``artifact_path.with_name(artifact_path.name + ".sam3_meta.json")``)
    without importing that module -- this package must stay training-free.

    Only called when the caller has explicitly named a *checkpoint* (a
    published artifact), never for a stock variant -- so unlike
    ``checkpoints.sidecar_for`` (which legitimately returns ``None`` for a
    stock key that ships no sidecar), a missing or corrupt sidecar HERE
    means the sidecar this module itself wrote is gone or broken, which is
    exactly the state to refuse rather than silently skip the guard for.
    """
    import json

    sidecar_path = checkpoint.with_name(checkpoint.name + ".sam3_meta.json")
    if not sidecar_path.exists():
        raise RuntimeError(
            f"SAM3 finetuned checkpoint {checkpoint} has no sidecar at "
            f"{sidecar_path}. A published artifact without its sidecar "
            "cannot be guarded -- refusing rather than serving it unchecked."
        )
    try:
        loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"SAM3 finetuned checkpoint sidecar {sidecar_path} is not valid "
            f"JSON ({exc}). Refusing to serve an unguardable checkpoint."
        ) from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"SAM3 finetuned checkpoint sidecar {sidecar_path} did not "
            f"parse to a JSON object (got {type(loaded).__name__}). "
            "Refusing to serve an unguardable checkpoint."
        )
    if "imgsz" not in loaded:
        # publish.py always writes `imgsz`; its absence means the sidecar
        # predates that field or was hand-edited -- either way, exactly the
        # kind of drift the guard exists to catch, not skip silently.
        raise RuntimeError(
            f"SAM3 finetuned checkpoint sidecar {sidecar_path} has no "
            "'imgsz' field. Refusing to serve an unguardable checkpoint."
        )
    return loaded


class Sam3SemanticLabeler:
    """Text-prompted instance segmentation via SAM3."""

    def __init__(self, predictor, device: str) -> None:
        self._predictor = predictor
        self._device = device

    @property
    def name(self) -> str:
        return "sam3"

    @classmethod
    def from_variant(
        cls,
        variant: str = DEFAULT_VARIANT,
        device: str | None = None,
        *,
        allow_download: bool = True,
        cache_dir: Path | None = None,
        confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR,
        checkpoint: Path | str | None = None,
    ) -> "Sam3SemanticLabeler":
        """Build a labeler whose predictor keeps everything at or above
        *confidence_floor*.

        ``confidence_floor`` is not cosmetic: it is the hard lower bound of
        what the candidate cache can ever contain, so it must be set to the
        lowest threshold any later offline re-threshold or calibration sweep
        will ask for.

        *checkpoint*, when given, names a specific artifact to load -- a
        published finetuned model's path, resolved by the caller via
        ``resolve_checkpoint``. A ``None`` checkpoint reproduces today's
        stock-variant behaviour exactly.
        """
        if checkpoint is None:
            avail = probe_availability(variant, cache_dir)
            # A merely-undownloaded checkpoint is tolerated (ensure_checkpoint
            # below fetches it); anything else is fatal. Keyed on the
            # structured flag, never on a substring of the human-readable
            # reason.
            if not avail.usable and not avail.checkpoint_missing:
                raise RuntimeError(f"SAM3 is unavailable: {avail.reason}")
            ckpt = ensure_checkpoint(
                variant, allow_download=allow_download, cache_dir=cache_dir
            )
        else:
            ckpt = Path(checkpoint)

        # Lazy import: only paid when semantic escalation actually runs.
        from ultralytics.models.sam import SAM3SemanticPredictor

        dev = device or resolve_torch_device()
        predictor = SAM3SemanticPredictor(
            overrides=predictor_overrides(ckpt, dev, confidence_floor)
        )
        if checkpoint is not None:
            # Force eager model construction so there is a live state dict
            # to guard BEFORE the first inference call, not after.
            # No `model=` argument: ultralytics' `setup_model` treats that
            # parameter as an already-constructed nn.Module and calls `.to()`
            # on it (predict.py:458), so passing the checkpoint PATH raises
            # AttributeError. Passing None makes it build from
            # `self.args.model` -- which `predictor_overrides` already set to
            # this same checkpoint.
            predictor.setup_model()
            meta = _sidecar_for_checkpoint(ckpt)
            live_state_dict = predictor.model.state_dict()
            assert_checkpoint_loaded(live_state_dict, meta, imgsz=PREDICTOR_IMGSZ)
        return cls(predictor, dev)

    def label_image(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        *,
        confidence_threshold: float = 0.0,
        max_instances: int = 0,
    ) -> list[SemanticInstance]:
        """Segment every instance of *prompt*, sorted by descending score."""
        # NOTE: ultralytics' predictor.__call__ forwards unmatched kwargs into
        # SAM3SemanticPredictor.inference()'s **kwargs sink and silently drops
        # them -- the text prompt keyword there is `text` (a list[str]), not
        # `prompt`. Passing `prompt=` would make every call behave as if no
        # prompt were given (falls back to `self.model.names`).
        results = self._predictor(source=image_bgr, text=[prompt])
        out: list[SemanticInstance] = []
        for res in results:
            masks = getattr(res, "masks", None)
            boxes = getattr(res, "boxes", None)
            if masks is None or masks.data is None:
                continue
            confs = (
                boxes.conf.detach().cpu().numpy()
                if boxes is not None and boxes.conf is not None
                else np.ones(len(masks.data), dtype=np.float32)
            )
            for mask_t, conf in zip(masks.data, confs):
                score = float(conf)
                if score < confidence_threshold:
                    continue
                contour = mask_to_contour(mask_t.detach().cpu().numpy().astype(bool))
                if contour is None or contour.shape[0] < 3:
                    continue
                out.append(SemanticInstance(contour, score))
        out.sort(key=lambda i: -i.confidence)
        if max_instances > 0:
            out = out[:max_instances]
        return out
