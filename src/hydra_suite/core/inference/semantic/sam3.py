"""SAM3 promptable-concept-segmentation backend for the SemanticLabeler seam.

Wraps ultralytics' ``SAM3SemanticPredictor``. Construction is guarded by
``probe_availability`` so a missing dependency raises with an actionable
message instead of letting ultralytics AutoUpdate pip-install packages.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

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
    }


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
    ) -> "Sam3SemanticLabeler":
        """Build a labeler whose predictor keeps everything at or above
        *confidence_floor*.

        ``confidence_floor`` is not cosmetic: it is the hard lower bound of
        what the candidate cache can ever contain, so it must be set to the
        lowest threshold any later offline re-threshold or calibration sweep
        will ask for.
        """
        avail = probe_availability(variant, cache_dir)
        # A merely-undownloaded checkpoint is tolerated (ensure_checkpoint
        # below fetches it); anything else is fatal. Keyed on the structured
        # flag, never on a substring of the human-readable reason.
        if not avail.usable and not avail.checkpoint_missing:
            raise RuntimeError(f"SAM3 is unavailable: {avail.reason}")
        ckpt = ensure_checkpoint(
            variant, allow_download=allow_download, cache_dir=cache_dir
        )
        # Lazy import: only paid when semantic escalation actually runs.
        from ultralytics.models.sam import SAM3SemanticPredictor

        dev = device or resolve_torch_device()
        predictor = SAM3SemanticPredictor(
            overrides=predictor_overrides(ckpt, dev, confidence_floor)
        )
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
