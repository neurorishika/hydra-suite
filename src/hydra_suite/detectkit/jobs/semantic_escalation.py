"""SAM3 semantic escalation: a prompt -> staged polygon labels for review.

Mirrors ``sam2_escalation.run_escalation``'s staging mechanics and departs
from them in four deliberate places, each marked below: no polygon-level
filter, the prompt enters the staging hash, the pre-write wipe is
conditional on a run fingerprint (so a multi-hour run can resume), and
cancellation is honoured between tiles.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.core.inference.semantic.tiling import (
    DEFAULT_MERGE_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_SEAM_MARGIN_PX,
    SEMANTIC_TILE_FRACTION_SEED,
    TileCandidate,
    collect_candidates,
    full_frame_plan,
    merge_candidates,
    plan_for_frame,
    resolve_tile_px,
)
from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.data.project_bundle import ensure_bundle_subdirectory
from hydra_suite.detectkit.gui.constants import IMG_EXTS
from hydra_suite.detectkit.gui.models import OBBSource, PendingEscalation
from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.widgets.workers import BaseWorker

from .sam2_escalation import remove_staged_escalation_dir

logger = logging.getLogger(__name__)

CANDIDATES_FILENAME = "candidates.json"
RUN_FILENAME = "run.json"
# A run staging nothing on a majority of frames is a prompt failure, not a
# quiet success -- the dominant failure mode is a noun phrase the model does
# not match, and it looks exactly like a clean run otherwise.
PROMPT_FAILURE_FRACTION = 0.5


@dataclass
class SemanticEscalationRequest:
    project: object  # has .project_dir and .sources (list[OBBSource])
    source_names: list[str]
    variant: str
    prompt: str
    confidence: float = 0.35
    max_instances: int = 0
    reference_body_px: float = 0.0
    overlap: float = DEFAULT_OVERLAP
    seam_margin_px: float = DEFAULT_SEAM_MARGIN_PX
    merge_iou: float = DEFAULT_MERGE_IOU
    # Calibrated by the dialog (Task 12); SEMANTIC_TILE_FRACTION_SEED is only
    # the prefill when the user skips calibration. None = full frame.
    tile_fraction: float | None = SEMANTIC_TILE_FRACTION_SEED
    tile_px: int | None = None  # explicit override; wins over tile_fraction
    overwrite: bool = False


@dataclass
class SemanticEscalationResult:
    staged: list[str] = field(default_factory=list)
    labelled: int = 0  # instances staged
    empty_images: int = 0  # frames where the model returned nothing
    degenerate: int = 0  # contours with P < 3, dropped not fatal
    tile_px: int | None = None  # resolved tile size, None = full frame
    skipped: list[tuple[str, str]] = field(default_factory=list)


def is_prompt_failure(result: SemanticEscalationResult, frames_processed: int) -> bool:
    """True when the run should be reported as a PROMPT failure, not success."""
    if frames_processed <= 0:
        return False
    return result.empty_images >= PROMPT_FAILURE_FRACTION * frames_processed


def prompt_slug(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.strip().lower()).strip("-")
    return (slug or "prompt")[:24]


def _fingerprint(
    req: SemanticEscalationRequest, src_root: Path, tile_px: int | None
) -> dict:
    return {
        "prompt": req.prompt,
        "variant": req.variant,
        "tile_px": tile_px,
        "tile_fraction": req.tile_fraction,
        "overlap": float(req.overlap),
        "seam_margin_px": float(req.seam_margin_px),
        "max_instances": int(req.max_instances),
        "confidence_floor": float(req.confidence),
        "source_root": str(src_root.resolve()),
    }


def _load_cache(staged_root: Path) -> dict:
    path = staged_root / CANDIDATES_FILENAME
    if not path.exists():
        return {"version": 1, "images": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.warning("Unreadable candidate cache at %s; starting over", path)
        return {"version": 1, "images": {}}


def _save_cache(staged_root: Path, cache: dict) -> None:
    (staged_root / CANDIDATES_FILENAME).write_text(json.dumps(cache))


def _candidates_to_json(cands: list[TileCandidate]) -> list[dict]:
    return [
        {
            "p": np.asarray(c.polygon_px, dtype=float).round(2).tolist(),
            "c": round(float(c.confidence), 4),
            "t": int(c.tile_index),
        }
        for c in cands
    ]


def _candidates_from_json(entries: list[dict]) -> list[TileCandidate]:
    return [
        TileCandidate(
            np.asarray(e["p"], dtype=np.float32).reshape(-1, 2),
            float(e["c"]),
            int(e.get("t", 0)),
        )
        for e in entries
    ]


class _DegenerateCountingLabeler:
    """Wraps a SemanticLabeler to count instances ``collect_candidates``
    silently drops with no signal back to the caller.

    ``tiling.collect_candidates`` (Task 4) discards any polygon with fewer
    than 3 points *before* it is ever offset into a ``TileCandidate`` -- it
    just ``continue``s, with no count returned. That means a degenerate
    contour never survives to reach the candidate cache, so counting
    ``pts.shape[0] < 3`` on cached/merged candidates downstream (as this
    module's ``_write_labels_from_candidates`` also defensively does) can
    never observe one. This wrapper intercepts the labeler's raw output --
    before collect_candidates' silent drop -- so a degenerate contour is
    still reflected in ``SemanticEscalationResult.degenerate``.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.count = 0

    @property
    def name(self) -> str:
        return self._inner.name

    def label_image(
        self, image_bgr, prompt, *, confidence_threshold=0.0, max_instances=0
    ):
        instances = self._inner.label_image(
            image_bgr,
            prompt,
            confidence_threshold=confidence_threshold,
            max_instances=max_instances,
        )
        for inst in instances:
            pts = np.asarray(inst.polygon_px, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 3:
                self.count += 1
        return instances


def _write_labels_from_candidates(
    staged_root: Path, cache: dict, *, confidence: float, merge_iou: float
) -> tuple[int, int]:
    """(instances written, degenerate dropped) across every cached image."""
    written = degenerate = 0
    for rel, entry in cache["images"].items():
        h, w = entry["hw"]
        merged = merge_candidates(
            _candidates_from_json(entry["candidates"]),
            confidence_threshold=confidence,
            iou_threshold=merge_iou,
        )
        records: list[LabelRecord] = []
        for inst in merged:
            pts = np.asarray(inst.polygon_px, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 3:
                # Defensive: collect_candidates already drops these before
                # they reach the cache (see _DegenerateCountingLabeler), so
                # this branch should not fire in practice. Kept because
                # write_label_file's _polygon_points RAISES on <3 points
                # (data/al/labels.py) and this cache is untrusted on-disk
                # state a resumed run reads back.
                degenerate += 1
                continue
            records.append(
                LabelRecord(
                    class_id=0,
                    confidence=float(inst.confidence),
                    points=pts,
                    level=GeometryLevel.POLYGON,
                )
            )
        label_path = staged_root / "labels" / Path(rel).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        write_label_file(
            label_path, records, frame_size=(h, w), level=GeometryLevel.POLYGON
        )
        written += len(records)
    return written, degenerate


def run_semantic_escalation(
    req: SemanticEscalationRequest,
    labeler,
    *,
    overwrite: bool = False,
    progress: Callable[[int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> SemanticEscalationResult:
    """Stage prompt-driven polygon labels for each named source.

    Never touches a source's own labels. Promotion is
    ``accept_pending_semantic_escalation``, which writes a NEW SIBLING
    SOURCE -- SAM3's output is a different instance set at a different
    geometry convention, so overwriting in place (as SAM2 does) would
    delete the user's curated labels.
    """
    result = SemanticEscalationResult()
    by_name = {s.name: s for s in req.project.sources}
    # DEPARTURE 1: no `level != "polygon"` filter. Finding animals the
    # existing polygons missed is a primary use case for this feature.
    todo = [by_name[n] for n in req.source_names if n in by_name]
    project_root = Path(req.project.project_dir)
    tile_px = req.tile_px or resolve_tile_px(req.reference_body_px, req.tile_fraction)
    result.tile_px = tile_px
    slug = prompt_slug(req.prompt)
    counting_labeler = _DegenerateCountingLabeler(labeler)

    for si, src in enumerate(todo):
        if src.pending_escalation is not None and not (overwrite or req.overwrite):
            result.skipped.append(
                (
                    src.name,
                    f"'{src.name}' already has a pending escalation; review it, or "
                    "re-run with overwrite to replace it.",
                )
            )
            continue

        src_root = Path(src.path)
        images_dir = src_root / "images"
        # DEPARTURE 2: the PROMPT enters the hash. Without it two prompts on
        # one source collide and the replaced-pending cleanup no-ops.
        content_hash = sha1(
            (str(src_root.resolve()) + req.variant + req.prompt).encode("utf-8")
        ).hexdigest()[:10]
        staged_dirname = f"{src.name}-sam3-{slug}-{content_hash}"
        staged_root = ensure_bundle_subdirectory(
            project_root, f"artifacts/pending_escalations/{staged_dirname}"
        )

        old_pending = src.pending_escalation
        if old_pending is not None and old_pending.staged_path != str(staged_root):
            remove_staged_escalation_dir(old_pending.staged_path, project_root)

        # DEPARTURE 3: the wipe is CONDITIONAL on the fingerprint, so a
        # cancelled multi-hour run resumes instead of restarting.
        fingerprint = _fingerprint(req, src_root, tile_px)
        run_path = staged_root / RUN_FILENAME
        stale = True
        if run_path.exists():
            try:
                stale = json.loads(run_path.read_text()) != fingerprint
            except Exception:
                stale = True
        if stale:
            remove_staged_escalation_dir(staged_root, project_root)
            staged_root = ensure_bundle_subdirectory(
                project_root, f"artifacts/pending_escalations/{staged_dirname}"
            )
        (staged_root / "labels").mkdir(parents=True, exist_ok=True)
        (staged_root / RUN_FILENAME).write_text(json.dumps(fingerprint))

        cache = {"version": 1, "images": {}} if stale else _load_cache(staged_root)
        images = sorted(
            p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
        )
        for ii, img_path in enumerate(images):
            # DEPARTURE 4: cancellation, honoured between images and (inside
            # collect_candidates) between tiles.
            if should_stop is not None and should_stop():
                break
            rel = str(img_path.relative_to(images_dir))
            if rel in cache["images"]:
                continue  # already inferred by an earlier, cancelled run
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]
            plan = (
                plan_for_frame((h, w), tile_px, req.overlap)
                if tile_px
                else full_frame_plan((h, w))
            )
            cands = collect_candidates(
                counting_labeler,
                image,
                plan,
                req.prompt,
                confidence_threshold=req.confidence,
                max_instances=req.max_instances,
                seam_margin_px=req.seam_margin_px,
                should_stop=should_stop,
            )
            cache["images"][rel] = {
                "hw": [h, w],
                "candidates": _candidates_to_json(cands),
            }
            _save_cache(staged_root, cache)
            if progress:
                progress(
                    int(
                        100 * (si + (ii + 1) / max(len(images), 1)) / max(len(todo), 1)
                    ),
                    f"{src.name}: {ii + 1}/{len(images)}",
                )

        written, degenerate = _write_labels_from_candidates(
            staged_root, cache, confidence=req.confidence, merge_iou=req.merge_iou
        )
        result.labelled += written
        result.degenerate += degenerate
        result.empty_images += sum(
            1 for e in cache["images"].values() if not e["candidates"]
        )
        (staged_root / "classes.txt").write_text(f"{req.prompt.strip() or 'object'}\n")
        src.pending_escalation = PendingEscalation(
            staged_path=str(staged_root),
            target_level=GeometryLevel.POLYGON.label,
            created_at=datetime.now().isoformat(),
            primer_kind="sam3",
            primer_variant=req.variant,
            primer_prompt=req.prompt,
            primer_params={
                "confidence": float(req.confidence),
                "merge_iou": float(req.merge_iou),
                "tile_px": tile_px,
                "overlap": float(req.overlap),
                "seam_margin_px": float(req.seam_margin_px),
                "max_instances": int(req.max_instances),
            },
        )
        result.staged.append(src.name)

    result.degenerate += counting_labeler.count
    return result


def rethreshold_staged(
    source: OBBSource, *, confidence: float, merge_iou: float
) -> int:
    """Rewrite a staged result at a new confidence. No inference.

    This is why the candidate cache exists: a 30-hour run must not be a
    one-shot commitment to one threshold. NMS is redone here rather than
    post-filtering the previous labels, because suppression is
    survivor-dependent.
    """
    pending = source.pending_escalation
    if pending is None or pending.primer_kind != "sam3":
        raise ValueError(f"Source '{source.name}' has no staged SAM3 escalation.")
    staged_root = Path(pending.staged_path)
    cache = _load_cache(staged_root)
    if not cache["images"]:
        raise RuntimeError(
            f"The candidate cache for '{source.name}' is missing or empty; "
            "re-run the escalation."
        )
    written, _degenerate = _write_labels_from_candidates(
        staged_root, cache, confidence=confidence, merge_iou=merge_iou
    )
    pending.primer_params = {
        **pending.primer_params,
        "confidence": float(confidence),
        "merge_iou": float(merge_iou),
    }
    return written


class SemanticEscalationWorker(BaseWorker):
    """QThread wrapper around run_semantic_escalation, with cancellation."""

    result_ready = Signal(object)  # SemanticEscalationResult

    def __init__(
        self, request: SemanticEscalationRequest, labeler=None, parent=None
    ) -> None:
        super().__init__(parent)
        self._request = request
        self._labeler = labeler
        self._cancel = False

    def cancel(self) -> None:
        """Ask the run to stop at the next tile boundary."""
        self._cancel = True

    def execute(self) -> None:
        labeler = self._labeler
        if labeler is None:
            from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

            labeler = Sam3SemanticLabeler.from_variant(self._request.variant)
        self.status.emit(
            f"Segmenting '{self._request.prompt}' across "
            f"{len(self._request.source_names)} source(s)..."
        )
        result = run_semantic_escalation(
            self._request,
            labeler,
            overwrite=self._request.overwrite,
            progress=lambda pct, msg: (
                self.progress.emit(pct),
                self.status.emit(msg),
            ),
            should_stop=lambda: self._cancel,
        )
        self.result_ready.emit(result)


def _unique_source_name(project, base: str) -> str:
    existing = {s.name for s in project.sources}
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


def accept_pending_semantic_escalation(
    source: OBBSource,
    project,
    project_dir: str | Path | None = None,
) -> OBBSource:
    """Promote a staged SAM3 result to a NEW SIBLING SOURCE.

    Deliberately NOT ``sam2_escalation.accept_pending_escalation``, which
    rmtree's the origin's labels and copies the staged ones over them. That
    is correct for SAM2 (a lossless upgrade of the SAME instances) and
    destructive here: SAM3's output is a different instance set at a
    different geometry convention (masks trace legs and antennae; tracking
    labels bound the body core), all class 0. Merging the two conventions
    into one source would degrade YOLO training -- and deleting the user's
    curated labels to do it would be worse.

    The staged labels and hardlinked images become a new source the user can
    keep, merge, or delete with the tools they already have. The candidate
    cache and run fingerprint stay behind in staging: they are consumed
    here, never shipped, so they cannot go stale against later user edits.
    """
    from hydra_suite.data.al.export import _link_or_copy

    pending = source.pending_escalation
    if pending is None:
        raise ValueError(f"Source '{source.name}' has no pending escalation.")
    if pending.primer_kind != "sam3":
        raise ValueError(
            f"Source '{source.name}' has a {pending.primer_kind!r} pending "
            "escalation, not a SAM3 one; use the SAM2 accept path."
        )

    staged_root = Path(pending.staged_path)
    staged_labels = staged_root / "labels"
    if not staged_labels.is_dir():
        raise RuntimeError(
            f"Staged escalation for '{source.name}' is missing on disk "
            f"({staged_labels}); nothing was changed. Reject it and re-run."
        )

    project_root = Path(project.project_dir)
    sibling_name = _unique_source_name(
        project, f"{source.name}-sam3-{prompt_slug(pending.primer_prompt)}"
    )
    sibling_root = Path(
        ensure_bundle_subdirectory(project_root, f"sources/{sibling_name}")
    )
    (sibling_root / "images").mkdir(parents=True, exist_ok=True)
    (sibling_root / "labels").mkdir(parents=True, exist_ok=True)

    origin_images = Path(source.path) / "images"
    for label_path in sorted(staged_labels.rglob("*.txt")):
        rel = label_path.relative_to(staged_labels)
        dst_label = sibling_root / "labels" / rel
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_bytes(label_path.read_bytes())
        for img in origin_images.rglob(f"{rel.stem}.*"):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            dst_img = sibling_root / "images" / img.relative_to(origin_images)
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                _link_or_copy(img, dst_img)
            break

    classes_src = staged_root / "classes.txt"
    (sibling_root / "classes.txt").write_text(
        classes_src.read_text() if classes_src.exists() else "object\n"
    )

    sibling = OBBSource(
        path=str(sibling_root),
        name=sibling_name,
        validated=False,
        original_path=source.path,
        source_kind="detectkit_sam3",
        imported=False,
        level=GeometryLevel.POLYGON.label,
        # Machine-derived and not yet human-confirmed: excluded from training
        # until the user runs "Mark reviewed...".
        reviewed=False,
        derived_from=source.name,
    )
    project.sources.append(sibling)

    remove_staged_escalation_dir(staged_root, project_dir or project_root)
    source.pending_escalation = None
    return sibling
