"""Dataset inference as the third producer of the staging contract.

Inference and escalation are the same kind of thing -- a machine proposal a
human must accept or reject -- and used to be completely different objects
in the code: predictions lived only in an in-memory dict that was cleared on
source switch and never written anywhere. Staging them makes them
reviewable by exactly the code that reviews SAM2 and SAM3 output.

The in-memory preview path is untouched. Staging is a separate, explicit
action, so running inference merely to LOOK at it never creates reviewable
state.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.data.project_bundle import ensure_bundle_subdirectory
from hydra_suite.detectkit.gui.models import OBBSource, StagedReview
from hydra_suite.detectkit.inference_geometry import native_prediction_level

from .sam2_escalation import PENDING_ESCALATIONS_RELDIR

logger = logging.getLogger(__name__)


def stage_predictions(
    source: OBBSource,
    project_dir: str | Path,
    per_image: Mapping[str, list[dict]] | Iterable[tuple[str, list[dict]]],
    *,
    model_path: str,
    inference_kind: str,
    confidence: float,
    device: str,
    class_names: Sequence[str] = (),
) -> StagedReview:
    """Write dataset-inference predictions into the staging contract.

    `per_image` is exactly `_DetectKitDatasetInferenceWorker`'s payload:
    image path -> list of ``{class_id, polygon_px, confidence}`` dicts in
    PIXEL space.

    ``class_id`` in those dicts indexes the PROJECT's class list (that is
    what `PredictionProvider` renders against), NOT the source's own
    ``classes.txt``. ``class_names`` must therefore be the project's class
    names -- callers should pass ``project.class_names`` -- so the staged
    ``classes.txt`` matches the ids actually written, and
    `resolve_staged_class_ids` can do real name-based staged->source
    mapping instead of degenerating to identity.

    The staged label's relative path mirrors the image's under ``images/``
    -- that mirroring IS the review's per-frame key, relied on by
    `find_staged_label_for_image` and by `staged_review.accept_frame`.

    Staging lands under the project's ``artifacts/pending_escalations/`` so
    that `_is_safe_to_delete` keeps bounding the recursive delete that
    finishing or rejecting a review performs.

    Frames with no detections are not staged at all: an empty staged label
    would mean "accept this to delete the frame's labels", which is not what
    running inference asks for. If that leaves NO frames staged, the whole
    call is refused rather than creating a review that cannot be finished.
    """
    if source.staged_review is not None:
        raise RuntimeError(
            f"Source '{source.name}' already has a staged review. Finish or "
            "revert it before staging predictions."
        )

    level = native_prediction_level(inference_kind)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    staged_root = Path(
        ensure_bundle_subdirectory(
            Path(project_dir),
            str(PENDING_ESCALATIONS_RELDIR / f"{source.name}-inference-{stamp}"),
        )
    )
    (staged_root / "labels").mkdir(parents=True, exist_ok=True)

    images_dir = Path(source.path) / "images"
    staged_frames = 0
    items = sorted(per_image.items()) if isinstance(per_image, Mapping) else per_image
    for image_path, detections in items:
        if not detections:
            continue
        image = Path(image_path)
        try:
            rel = image.relative_to(images_dir)
        except ValueError:
            logger.warning(
                "Skipping image outside the source's images/ directory "
                "while staging: %s",
                image,
            )
            continue

        frame = cv2.imread(str(image))
        if frame is None:
            logger.warning("Skipping unreadable image while staging: %s", image)
            continue
        height, width = int(frame.shape[0]), int(frame.shape[1])

        records = []
        for det in detections:
            pts = np.asarray(det.get("polygon_px") or [], dtype=np.float32).reshape(
                -1, 2
            )
            if pts.shape[0] < 3:
                continue
            records.append(
                LabelRecord(
                    class_id=int(det.get("class_id", 0)),
                    confidence=float(det.get("confidence", 0.0)),
                    points=pts,
                    level=level,
                )
            )
        if not records:
            continue

        out = staged_root / "labels" / rel.with_suffix(".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_label_file(out, records, (height, width), level)
        staged_frames += 1

    if staged_frames == 0:
        # A zero-frame review is unfinishable: is_complete needs total > 0,
        # reject-all rejects nothing, and revert has no snapshot. Refuse
        # before the source's field is set, and clean the empty dir up.
        shutil.rmtree(staged_root, ignore_errors=True)
        raise RuntimeError(
            "There were no detections to stage. Lower the confidence "
            "threshold or re-run inference before staging."
        )

    (staged_root / "classes.txt").write_text(
        "".join(f"{name}\n" for name in class_names) if class_names else "object\n"
    )

    params = {
        "model_path": str(model_path),
        "inference_kind": str(inference_kind),
        "confidence": float(confidence),
        "device": str(device),
    }
    (staged_root / "run.json").write_text(
        json.dumps(
            {
                "producer": "inference",
                "params": params,
                "staged_frames": staged_frames,
                "created_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    review = StagedReview(
        staged_path=str(staged_root),
        target_level=level.label,
        producer="inference",
        producer_variant=Path(model_path).name,
        prompt="",
        params=params,
        created_at=datetime.now().isoformat(),
    )
    source.staged_review = review
    return review
