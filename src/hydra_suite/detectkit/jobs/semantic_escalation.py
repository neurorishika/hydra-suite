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
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.core.inference.semantic.base import SemanticInstance
from hydra_suite.core.inference.semantic.calibration import CONFIDENCE_GRID
from hydra_suite.core.inference.semantic.shape_prior import AreaBand
from hydra_suite.core.inference.semantic.tiling import (
    DEFAULT_MERGE_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_SEAM_MARGIN_PX,
    SEMANTIC_TILE_FRACTION_SEED,
    TileCandidate,
    TileCollectionCancelled,
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
from hydra_suite.detectkit.gui.models import OBBSource, StagedReview
from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.widgets.workers import BaseWorker

from .sam2_escalation import remove_staged_escalation_dir

logger = logging.getLogger(__name__)

CANDIDATES_FILENAME = "candidates.json"
RUN_FILENAME = "run.json"
# I4: candidates are CACHED at the bottom of the sweep grid, not at the run's
# own confidence. Inference cost is identical either way -- the model runs on
# every tile regardless and the threshold only filters what is KEPT -- but
# caching at req.confidence would silently truncate the cache, so
# rethreshold_staged DOWNWARD (the entire reason the cache exists) would
# quietly return fewer instances than a fresh run at that threshold.
CACHE_CONFIDENCE_FLOOR = CONFIDENCE_GRID[0]
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
    # The label-derived area gate from calibration (shape_prior.AreaBand),
    # flattened to two floats so it survives the request/params/JSON round
    # trip. 0/0 disables it. It gates at MERGE time, not collection time, so
    # it deliberately stays OUT of the cache fingerprint: changing it is a
    # re-threshold, not a re-run.
    area_min_px2: float = 0.0
    area_max_px2: float = 0.0
    overwrite: bool = False


@dataclass
class SemanticEscalationResult:
    staged: list[str] = field(default_factory=list)
    labelled: int = 0  # instances staged
    empty_images: int = 0  # frames where the model returned nothing
    degenerate: int = 0  # contours with P < 3, dropped not fatal
    tile_px: int | None = None  # resolved tile size, None = full frame
    skipped: list[tuple[str, str]] = field(default_factory=list)
    # I1: FRAMES, not instances. `labelled` counts instances staged, so it is
    # the wrong denominator for the prompt-failure rule -- 40 frames x 10
    # instances made `labelled + empty_images` 460, and the rule never fired.
    frames_processed: int = 0
    # I9: a cancelled run is a PARTIAL result. Without this flag the caller
    # reports it as an unqualified success and the user may accept it
    # believing the whole source was covered.
    cancelled: bool = False
    # I7: staged labels whose origin image could not be located; skipped
    # rather than written as an image-less orphan label.
    orphaned: int = 0


def is_prompt_failure(
    result: SemanticEscalationResult, frames_processed: int | None = None
) -> bool:
    """True when the run should be reported as a PROMPT failure, not success.

    The denominator defaults to the run's own ``frames_processed``. Callers
    should not compute one: ``labelled`` counts INSTANCES, so deriving frames
    from it inflates the denominator and disarms the rule entirely (I1).
    """
    frames = result.frames_processed if frames_processed is None else frames_processed
    if frames <= 0:
        return False
    return result.empty_images >= PROMPT_FAILURE_FRACTION * frames


def _resolved(path: str | Path) -> Path:
    """The canonical form of *path*, matching ``bundle_paths``' own.

    Every staging path the job writes comes back from
    ``ensure_bundle_subdirectory`` -> ``bundle_paths``, which
    ``.expanduser().resolve()``s the project root (data/project_bundle.py).
    ``project_dir`` itself arrives UNRESOLVED (from QFileDialog, or from the
    project JSON), so any project reached through a symlink -- macOS /tmp, a
    symlinked home, a symlinked lab share -- compares unequal to its own
    staging directory unless both sides are canonicalised here. That
    mismatch made a resume look like a replacement of a different run.
    """
    return Path(path).expanduser().resolve()


def cache_confidence_floor(confidence: float) -> float:
    """The floor a run at *confidence* must collect its candidates at.

    Normally the bottom of the sweep grid (see CACHE_CONFIDENCE_FLOOR), but
    NEVER above the run's own threshold: the dialog allows confidences down
    to 0.01, and collecting at 0.05 for a run at 0.02 would drop candidates
    in [0.02, 0.05) that the run itself asked to keep -- a silently
    truncated staged set, not just a truncated cache.
    """
    return min(float(CACHE_CONFIDENCE_FLOOR), float(confidence))


def prompt_slug(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.strip().lower()).strip("-")
    return (slug or "prompt")[:24]


def staged_dirname_for(src: OBBSource, variant: str, prompt: str) -> str:
    """The staging directory NAME a (source, variant, prompt) run targets.

    Shared with the GUI so it can tell a RESUME of the same run (same
    directory) from a REPLACE of a different one (different directory)
    without duplicating the hashing rule.
    """
    # DEPARTURE 2: the PROMPT enters the hash. Without it two prompts on
    # one source collide and the replaced-pending cleanup no-ops.
    content_hash = sha1(
        (str(Path(src.path).resolve()) + variant + prompt).encode("utf-8")
    ).hexdigest()[:10]
    return f"{src.name}-sam3-{prompt_slug(prompt)}-{content_hash}"


def sources_pending_replacement(req: "SemanticEscalationRequest") -> list[str]:
    """Selected sources whose staged escalation this run would DESTROY.

    A source whose pending escalation is the very run being re-issued is a
    RESUME and is not listed -- only a different prompt, a different variant,
    or an unreviewed SAM2 result would be wiped, and the GUI must confirm
    those before they are.
    """
    by_name = {s.name: s for s in req.project.sources}
    project_root = _resolved(req.project.project_dir)
    out: list[str] = []
    for name in req.source_names:
        src = by_name.get(name)
        if src is None or src.staged_review is None:
            continue
        target = (
            project_root
            / "artifacts"
            / "pending_escalations"
            / staged_dirname_for(src, req.variant, req.prompt)
        )
        if _resolved(src.staged_review.staged_path) != target:
            out.append(name)
    return out


def band_from_bounds(min_px2: float, max_px2: float) -> AreaBand | None:
    """Rebuild the area gate from two persisted floats. 0/0 = no gate.

    The median and label count are not persisted -- nothing downstream of
    calibration uses them -- so they are reported as the bounds' midpoint
    and 0 rather than invented.
    """
    lo, hi = float(min_px2 or 0.0), float(max_px2 or 0.0)
    if lo <= 0.0 or hi <= lo:
        return None
    return AreaBand(min_px2=lo, max_px2=hi, median_px2=(lo + hi) / 2.0, n_labels=0)


def band_from_params(params: dict | None) -> AreaBand | None:
    """The area gate recorded on a staged run's ``params``.

    Absent on runs staged before the gate existed, which is exactly the
    ungated behaviour those runs were produced under.
    """
    if not params:
        return None
    return band_from_bounds(
        params.get("area_min_px2", 0.0), params.get("area_max_px2", 0.0)
    )


def _fingerprint(
    req: SemanticEscalationRequest,
    src_root: Path,
    tile_px: int | None,
    floor: float,
) -> dict:
    return {
        "prompt": req.prompt,
        "variant": req.variant,
        "tile_px": tile_px,
        "tile_fraction": req.tile_fraction,
        "overlap": float(req.overlap),
        "seam_margin_px": float(req.seam_margin_px),
        "max_instances": int(req.max_instances),
        # The floor the CACHE was collected at (not the run's display
        # threshold): a cache collected at a higher floor is not
        # interchangeable with one collected at the grid bottom.
        "confidence_floor": float(floor),
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
    staged_root: Path,
    cache: dict,
    *,
    confidence: float,
    merge_iou: float,
    area_band: AreaBand | None = None,
    origin_images: Path | None = None,
) -> tuple[int, int, int]:
    """(instances written, degenerate dropped, orphans skipped) per cache.

    I7: a cached image whose ORIGIN file has since disappeared (a resume of a
    run whose source was edited in between) would otherwise be written as a
    staged label with no image behind it -- an orphan promotion has to throw
    away silently. Counted here so the run can report it.
    """
    written = degenerate = orphaned = 0
    check_origins = origin_images is not None and origin_images.is_dir()
    for rel, entry in cache["images"].items():
        if check_origins and _origin_image_for(origin_images, Path(rel)) is None:
            logger.warning("Cached frame %s has no origin image; skipping it.", rel)
            orphaned += 1
            continue
        h, w = entry["hw"]
        merged = merge_candidates(
            _candidates_from_json(entry["candidates"]),
            confidence_threshold=confidence,
            iou_threshold=merge_iou,
            area_band=area_band,
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
        if not records:
            # Do not stage a label file for a zero-record frame: an empty
            # staged label means "accept this to delete the frame's
            # labels", which is not what running a prompt asks for. Same
            # contract inference_stager.py already documents/enforces.
            continue
        label_path = staged_root / "labels" / Path(rel).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        write_label_file(
            label_path, records, frame_size=(h, w), level=GeometryLevel.POLYGON
        )
        written += len(records)
    return written, degenerate, orphaned


def run_semantic_escalation(
    req: SemanticEscalationRequest,
    labeler,
    *,
    overwrite: bool = False,
    progress: Callable[[int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> SemanticEscalationResult:
    """Stage prompt-driven polygon labels for each named source.

    Never touches a source's own labels. Promotion is frame-granular review
    (``jobs/staged_review.py``): the user accepts or rejects individual
    staged frames into the source the run was made against, rather than
    promoting the whole staged run into a new sibling source.
    """
    result = SemanticEscalationResult()
    by_name = {s.name: s for s in req.project.sources}
    # DEPARTURE 1: no `level != "polygon"` filter. Finding animals the
    # existing polygons missed is a primary use case for this feature.
    todo = [by_name[n] for n in req.source_names if n in by_name]
    # Canonical, so every staged-path comparison below matches the paths
    # ensure_bundle_subdirectory hands back (see _resolved).
    project_root = _resolved(req.project.project_dir)
    floor = cache_confidence_floor(req.confidence)
    tile_px = req.tile_px or resolve_tile_px(req.reference_body_px, req.tile_fraction)
    result.tile_px = tile_px
    counting_labeler = _DegenerateCountingLabeler(labeler)

    for si, src in enumerate(todo):
        staged_dirname = staged_dirname_for(src, req.variant, req.prompt)
        target_root = (
            project_root / "artifacts" / "pending_escalations" / staged_dirname
        )
        # I2: a RESUME of this very run (same target directory) proceeds
        # without overwrite -- the run.json fingerprint decides what is
        # reusable. Only a REPLACE of a DIFFERENT staged result (another
        # prompt, or an unreviewed SAM2 escalation) needs the caller's
        # explicit consent, because it destroys unreviewed work.
        would_replace = (
            src.staged_review is not None
            and _resolved(src.staged_review.staged_path) != target_root
        )
        if would_replace and not (overwrite or req.overwrite):
            result.skipped.append(
                (
                    src.name,
                    f"'{src.name}' already has a different pending escalation; "
                    "review it, or re-run with overwrite to replace it.",
                )
            )
            continue

        src_root = Path(src.path)
        images_dir = src_root / "images"
        staged_root = ensure_bundle_subdirectory(
            project_root, f"artifacts/pending_escalations/{staged_dirname}"
        )

        old_pending = src.staged_review
        if (
            old_pending is not None
            and _resolved(old_pending.staged_path) != staged_root
        ):
            remove_staged_escalation_dir(old_pending.staged_path, project_root)
            # F7: the pointer dies with the directory. src.staged_review
            # is only REPLACED at the end of a successful source, so any
            # failure in between (e.g. plan_for_frame's ValueError above the
            # tile ceiling at overlap 0.9) used to leave the source pointing
            # at a directory that no longer exists -- the review dialog then
            # offers a pending escalation that cannot be opened or dismissed.
            src.staged_review = None

        # DEPARTURE 3: the wipe is CONDITIONAL on the fingerprint, so a
        # cancelled multi-hour run resumes instead of restarting.
        fingerprint = _fingerprint(req, src_root, tile_px, floor)
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
                result.cancelled = True
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
            try:
                cands = collect_candidates(
                    counting_labeler,
                    image,
                    plan,
                    req.prompt,
                    # I4: the CACHE floor, not req.confidence -- see
                    # cache_confidence_floor (never ABOVE req.confidence).
                    confidence_threshold=floor,
                    max_instances=req.max_instances,
                    seam_margin_px=req.seam_margin_px,
                    should_stop=should_stop,
                )
            except TileCollectionCancelled:
                # F1: a half-tiled frame is NOT cached. Caching it would let
                # the `rel in cache["images"]` resume check skip the frame
                # forever, so "re-run to carry on" would do nothing and the
                # staged labels would be silently truncated. Leaving the key
                # absent makes the resume redo this frame from tile zero.
                result.cancelled = True
                break
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

        written, degenerate, orphaned = _write_labels_from_candidates(
            staged_root,
            cache,
            confidence=req.confidence,
            merge_iou=req.merge_iou,
            area_band=band_from_bounds(req.area_min_px2, req.area_max_px2),
            origin_images=images_dir,
        )
        result.labelled += written
        result.degenerate += degenerate
        result.orphaned += orphaned
        result.empty_images += sum(
            1 for e in cache["images"].values() if not e["candidates"]
        )
        # I1: the prompt-failure denominator. Frames actually inferred (the
        # cache's key set), which under cancellation or resume is NOT
        # len(images) either.
        result.frames_processed += len(cache["images"])
        (staged_root / "classes.txt").write_text(f"{req.prompt.strip() or 'object'}\n")
        src.staged_review = StagedReview(
            staged_path=str(staged_root),
            target_level=GeometryLevel.POLYGON.label,
            created_at=datetime.now().isoformat(),
            producer="sam3",
            producer_variant=req.variant,
            prompt=req.prompt,
            params={
                # The threshold the staged LABELS were written at -- distinct
                # from confidence_floor (what the CACHE holds). The review
                # dialog prefills its re-threshold prompt from this, and
                # nothing else records it, so dropping it made the dialog
                # claim a run at 0.60 was staged at its 0.35 default.
                "confidence": float(req.confidence),
                "confidence_floor": float(floor),
                "merge_iou": float(req.merge_iou),
                "tile_px": tile_px,
                "overlap": float(req.overlap),
                "seam_margin_px": float(req.seam_margin_px),
                "max_instances": int(req.max_instances),
                # The calibrated size gate, carried so a re-threshold
                # replays under the SAME rule the run emitted under.
                "area_min_px2": float(req.area_min_px2),
                "area_max_px2": float(req.area_max_px2),
            },
        )
        result.staged.append(src.name)

    result.degenerate += counting_labeler.count
    if should_stop is not None and should_stop():
        result.cancelled = True
    return result


def recorded_confidence_floor(staged_root: str | Path) -> float | None:
    """The floor this staging dir's candidates were collected at, if recorded.

    An OLD staging directory (collected at req.confidence, before I4) records
    its own higher floor here, so it cannot pretend a downward re-threshold
    is complete. Public because the review dialog uses it as the MINIMUM of
    its re-threshold input: offering a value the cache cannot honestly serve
    only lets the user pick a refusal.
    """
    staged_root = Path(staged_root)
    try:
        data = json.loads((staged_root / RUN_FILENAME).read_text())
    except Exception:
        return None
    value = data.get("confidence_floor")
    return float(value) if isinstance(value, (int, float)) else None


def rethreshold_floor_for(sources) -> float:
    """The lowest confidence a re-threshold of ALL *sources* can honour.

    The HIGHEST recorded floor among them: a value below that would be
    refused for at least one source, so offering it in the UI only lets the
    user pick an error. Falls back to CACHE_CONFIDENCE_FLOOR when nothing
    records one.
    """
    floors = [
        recorded_confidence_floor(s.staged_review.staged_path)
        for s in sources
        if getattr(s, "staged_review", None) is not None
    ]
    known = [f for f in floors if f is not None]
    return max(known) if known else float(CACHE_CONFIDENCE_FLOOR)


def rethreshold_staged(
    source: OBBSource, *, confidence: float, merge_iou: float
) -> int:
    """Rewrite a staged result at a new confidence. No inference.

    This is why the candidate cache exists: a 30-hour run must not be a
    one-shot commitment to one threshold. NMS is redone here rather than
    post-filtering the previous labels, because suppression is
    survivor-dependent.
    """
    pending = source.staged_review
    if pending is None or pending.producer != "sam3":
        raise ValueError(f"Source '{source.name}' has no staged SAM3 escalation.")
    staged_root = Path(pending.staged_path)
    floor = recorded_confidence_floor(staged_root)
    if floor is not None and float(confidence) < floor - 1e-9:
        # Two different situations, and the advice differs. Saying "re-run to
        # collect at a lower floor" for the second was impossible advice: a
        # re-run at the SAME confidence collects at the same floor.
        remedy = (
            "Re-run the escalation: it now collects down to "
            f"{CACHE_CONFIDENCE_FLOOR:.2f}."
            if floor > CACHE_CONFIDENCE_FLOOR + 1e-9
            else (
                "This is the collection floor, so a re-run at the same "
                f"confidence would not change it -- re-run with a confidence "
                f"of {confidence:.2f} or lower, which collects candidates "
                "down to that value."
            )
        )
        raise ValueError(
            f"This staged run's candidate cache was collected at confidence "
            f">= {floor:.2f}, so re-thresholding down to {confidence:.2f} "
            f"would silently return a truncated set. {remedy}"
        )
    cache = _load_cache(staged_root)
    if not cache["images"]:
        raise RuntimeError(
            f"The candidate cache for '{source.name}' is missing or empty; "
            "re-run the escalation."
        )
    written, _degenerate, _orphaned = _write_labels_from_candidates(
        staged_root,
        cache,
        confidence=confidence,
        merge_iou=merge_iou,
        area_band=band_from_params(pending.params),
        origin_images=Path(source.path) / "images",
    )
    pending.params = {
        **pending.params,
        "confidence": float(confidence),
        "merge_iou": float(merge_iou),
    }
    return written


def _label_path_for(images_dir: Path, labels_dir: Path, img_path: Path) -> Path:
    return labels_dir / img_path.relative_to(images_dir).with_suffix(".txt")


def has_labelled_frames(source: OBBSource) -> bool:
    """True if any frame in *source* carries a non-empty label file.

    Deliberately does NOT call ``labelled_frames_for``: answering "are there
    any labels?" by decoding every labelled image cost the GUI thread a full
    image-set decode every time the dialog opened. This is a label-FILE scan
    and touches no pixels.
    """
    root = Path(source.path)
    images_dir, labels_dir = root / "images", root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return False
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = _label_path_for(images_dir, labels_dir, img_path)
        try:
            if label_path.exists() and label_path.read_text().strip():
                return True
        except OSError:  # pragma: no cover - unreadable label file
            continue
    return False


# Enough frames for a stable median without decoding a whole image set on the
# GUI thread while the dialog is opening.
MEDIAN_BODY_SAMPLE_FRAMES = 20
# F4: a PROJECT-WIDE budget, not just a per-source one. The per-source limit
# alone still decoded 20 frames x every source in the project on the GUI
# thread at dialog open -- on 4512^2 frames that is seconds to minutes of a
# frozen window with no feedback. The median only needs a sample, so the
# sample is bounded globally and the truncation is SURFACED (see
# measure_median_body_px), never silently applied.
MEDIAN_BODY_TOTAL_FRAMES = 20


def measure_median_body_px(
    sources,
    *,
    sample_frames: int = MEDIAN_BODY_SAMPLE_FRAMES,
    max_total_frames: int = MEDIAN_BODY_TOTAL_FRAMES,
) -> tuple[float, int, bool]:
    """(median longest side px, frames sampled, whether the budget truncated).

    Link 2 of the ``reference_body_px`` resolution chain (project setting ->
    this -> the user). Returns 0.0 when nothing can be measured. Without it,
    a project with no ``slice_training.reference_body_px`` silently runs with
    tiling OFF -- the measured-worst configuration (F1 0.719 -> 0.075).

    ``max_total_frames`` bounds the decode across ALL sources; the caller is
    expected to report the sample size so the cap is visible rather than a
    silent change of what "median of your labels" means.
    """
    sides: list[float] = []
    used = 0
    truncated = False
    for source in sources:
        if used >= max_total_frames:
            truncated = True
            break
        budget = min(sample_frames, max_total_frames - used)
        # Ask for one MORE than the budget: if it comes back, this source had
        # frames we are declining to read, which is exactly what `truncated`
        # is supposed to tell the caller. Setting the flag only on the next
        # iteration misses the single-source case the cap exists for.
        frames = labelled_frames_for(source, limit=budget + 1)
        if len(frames) > budget:
            truncated = True
            frames = frames[:budget]
        used += len(frames)
        for _path, records in frames:
            for rec in records:
                pts = np.asarray(rec.points, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] < 2:
                    continue
                extent = pts.max(axis=0) - pts.min(axis=0)
                longest = float(max(extent[0], extent[1]))
                if longest > 0:
                    sides.append(longest)
    if not sides:
        return 0.0, used, truncated
    return float(np.median(np.asarray(sides, dtype=np.float64))), used, truncated


def median_body_px_for(
    sources, *, sample_frames: int = MEDIAN_BODY_SAMPLE_FRAMES
) -> float:
    """The median alone; see ``measure_median_body_px`` for the sample size."""
    return measure_median_body_px(sources, sample_frames=sample_frames)[0]


def labelled_frames_for(
    source: OBBSource, *, limit: int = 0
) -> list[tuple[Path, list[LabelRecord]]]:
    """(image path, LabelRecords) for every non-empty labelled frame.

    Uses ``gui/utils.parse_obb_label``, which already handles 5-field AABB,
    9-field quad and odd-count polygon lines. Deliberately NOT
    ``sam2_prompts.read_boxes_from_label``, which accepts only 4- and
    8-value lines and silently drops polygon lines (jobs/sam2_prompts.py:49-60)
    -- calibration must work at ANY geometry level, because choosing an
    operating point needs instance COUNTS, not masks.
    """
    from hydra_suite.detectkit.gui.utils import parse_obb_label

    root = Path(source.path)
    images_dir, labels_dir = root / "images", root / "labels"
    out: list[tuple[Path, list[LabelRecord]]] = []
    for img_path in sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS
    ):
        if limit and len(out) >= limit:
            break
        label_path = _label_path_for(images_dir, labels_dir, img_path)
        if not label_path.exists() or not label_path.read_text().strip():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        parsed = parse_obb_label(label_path, w, h)
        if not parsed:
            continue
        out.append(
            (
                img_path,
                [
                    LabelRecord(
                        class_id=int(d["class_id"]),
                        confidence=1.0,
                        points=np.asarray(d["polygon_px"], dtype=np.float32).reshape(
                            -1, 2
                        ),
                        level=GeometryLevel.POLYGON,
                    )
                    for d in parsed
                ],
            )
        )
    return out


@dataclass
class FramePreviewResult:
    """A complete randomly-selected frame processed with the run settings."""

    image_path: Path
    source_name: str
    predictions: list[SemanticInstance]
    ground_truth: list[LabelRecord]
    seconds: float
    tile_px: int | None
    tiles_per_frame: int


def preview_random_frame(
    labeler,
    sources: list[OBBSource],
    prompt: str,
    *,
    reference_body_px: float,
    tile_fraction: float | None,
    overlap: float = DEFAULT_OVERLAP,
    seam_margin_px: float = DEFAULT_SEAM_MARGIN_PX,
    merge_iou: float = DEFAULT_MERGE_IOU,
    confidence: float = 0.35,
    max_instances: int = 0,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> FramePreviewResult:
    """Process one random complete image exactly as escalation would.

    "Complete image" describes the user-facing unit: configured tiling is
    still applied internally when enabled.  The measured duration therefore
    includes every tile and the merge step, and can be multiplied directly
    by a frame count for a transparent run-time estimate.
    """
    choices: list[tuple[OBBSource, Path]] = []
    for source in sources:
        images_dir = Path(source.path) / "images"
        choices.extend(
            (source, path)
            for path in sorted(images_dir.rglob("*"))
            if path.suffix.lower() in IMG_EXTS
        )
    if not choices:
        raise RuntimeError("The selected source(s) have no images to preview.")
    if should_stop is not None and should_stop():
        raise TileCollectionCancelled(0, 1)
    source, img_path = random.choice(choices)
    image = cv2.imread(str(img_path))
    if image is None:
        raise RuntimeError(f"Could not read {img_path.name}.")
    h, w = image.shape[:2]
    tile_px = resolve_tile_px(reference_body_px, tile_fraction)
    plan = (
        plan_for_frame((h, w), tile_px, overlap) if tile_px else full_frame_plan((h, w))
    )
    started = time.perf_counter()
    candidates = collect_candidates(
        labeler,
        image,
        plan,
        prompt,
        confidence_threshold=cache_confidence_floor(confidence),
        max_instances=max_instances,
        seam_margin_px=seam_margin_px,
        progress=progress,
        should_stop=should_stop,
    )
    if should_stop is not None and should_stop():
        raise RuntimeError("Complete-frame preview cancelled.")
    predictions = merge_candidates(
        candidates,
        confidence_threshold=confidence,
        iou_threshold=merge_iou,
    )
    elapsed = time.perf_counter() - started

    ground_truth: list[LabelRecord] = []
    images_dir = Path(source.path) / "images"
    labels_dir = Path(source.path) / "labels"
    label_path = _label_path_for(images_dir, labels_dir, img_path)
    if label_path.is_file():
        from hydra_suite.detectkit.gui.utils import parse_obb_label

        for item in parse_obb_label(label_path, w, h):
            ground_truth.append(
                LabelRecord(
                    class_id=int(item["class_id"]),
                    confidence=1.0,
                    points=np.asarray(item["polygon_px"], dtype=np.float32).reshape(
                        -1, 2
                    ),
                    level=GeometryLevel.POLYGON,
                )
            )
    return FramePreviewResult(
        image_path=img_path,
        source_name=source.name,
        predictions=predictions,
        ground_truth=ground_truth,
        seconds=elapsed,
        tile_px=tile_px,
        tiles_per_frame=len(plan.tiles),
    )


class FramePreviewWorker(BaseWorker):
    """QThread wrapper around a random, complete-frame preview."""

    result_ready = Signal(object)  # FramePreviewResult

    def __init__(self, sources, prompt, variant, params, labeler=None, parent=None):
        super().__init__(parent)
        self._sources = list(sources)
        self._prompt = prompt
        self._variant = variant
        self._params = dict(params)
        self._labeler = labeler
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def execute(self) -> None:
        labeler = self._labeler
        if labeler is None:
            from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

            # F2: the preview must see what the run will see, so its
            # predictor floor is the same cache floor the run would use.
            labeler = Sam3SemanticLabeler.from_variant(
                self._variant,
                confidence_floor=cache_confidence_floor(
                    self._params.get("confidence", 0.35)
                ),
            )
        self.result_ready.emit(
            preview_random_frame(
                labeler,
                self._sources,
                self._prompt,
                reference_body_px=self._params.get("reference_body_px", 0.0),
                tile_fraction=self._params.get("tile_fraction"),
                overlap=self._params.get("overlap", DEFAULT_OVERLAP),
                seam_margin_px=self._params.get(
                    "seam_margin_px", DEFAULT_SEAM_MARGIN_PX
                ),
                merge_iou=self._params.get("merge_iou", DEFAULT_MERGE_IOU),
                confidence=self._params.get("confidence", 0.35),
                max_instances=self._params.get("max_instances", 0),
                progress=lambda done, total: (
                    self.progress.emit(int(100 * done / max(total, 1))),
                    self.status.emit(f"Running tile {done}/{total}..."),
                ),
                should_stop=lambda: self._cancel,
            )
        )


# Kept as an import-compatible alias for extensions written against the
# earlier internal worker name. The UI now describes the complete-frame work.
TilePreviewWorker = FramePreviewWorker


class CalibrationWorker(BaseWorker):
    """QThread wrapper around calibrate(), cancellable between frames.

    F4: takes SOURCES, not decoded frames. The dialog used to build the
    frame list itself -- ``labelled_frames_for`` cv2.imreads every labelled
    image of every selected source, with no limit -- on the GUI thread,
    before the progress dialog even existed. Two hundred 4512^2 frames is
    minutes of a frozen window with no feedback and no cancel. The decode
    now happens here, behind the progress dialog and under should_stop.
    """

    result_ready = Signal(object)  # list[CalibrationPoint]

    def __init__(
        self, sources, prompt, variant, params, labeler=None, parent=None
    ) -> None:
        super().__init__(parent)
        self._sources = list(sources)
        self._prompt = prompt
        self._variant = variant
        self._params = dict(params)
        self._labeler = labeler
        self._cancel = False
        self.preview_frames: list = []

    def cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        """True if this sweep was cut short, so its frontier is PARTIAL.

        Read by the results dialog: calibrate() drops fractions whose frames
        were not fully inferred, which keeps the surviving rows comparable
        but leaves them resting on fewer frames than the user selected.
        """
        return self._cancel

    def execute(self) -> None:
        from hydra_suite.core.inference.semantic.calibration import calibrate

        frames: list = []
        for i, source in enumerate(self._sources):
            if self._cancel:
                break
            name = getattr(source, "name", "?")
            self.status.emit(
                f"Reading labelled frames from '{name}' "
                f"({i + 1}/{len(self._sources)})..."
            )
            frames.extend(labelled_frames_for(source))
        if self._cancel or not frames:
            self.result_ready.emit([])
            return

        labeler = self._labeler
        if labeler is None:
            from hydra_suite.core.inference.semantic.sam3 import Sam3SemanticLabeler

            # F2: the sweep's own bottom cell, so cells 0.05-0.25 are not
            # all silently identical to 0.25.
            labeler = Sam3SemanticLabeler.from_variant(
                self._variant, confidence_floor=CONFIDENCE_GRID[0]
            )
        points = calibrate(
            labeler,
            frames,
            self._prompt,
            # The GRID, not the dialog's single fraction: calibration exists
            # precisely to choose the fraction, so passing the current one
            # would make the answer its own input.
            reference_body_px=self._params.get("reference_body_px", 0.0),
            overlap=self._params.get("overlap", DEFAULT_OVERLAP),
            seam_margin_px=self._params.get("seam_margin_px", DEFAULT_SEAM_MARGIN_PX),
            merge_iou=self._params.get("merge_iou", DEFAULT_MERGE_IOU),
            max_instances=self._params.get("max_instances", 0),
            progress=lambda pct, msg: (self.progress.emit(pct), self.status.emit(msg)),
            should_stop=lambda: self._cancel,
            preview_sink=self.preview_frames.extend,
        )
        self.result_ready.emit(points)


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

            # F2: the predictor's OWN conf gate must sit at the cache floor,
            # not ultralytics' 0.25 default, or the cache silently holds
            # nothing below 0.25 and every offline re-threshold below it lies.
            labeler = Sam3SemanticLabeler.from_variant(
                self._request.variant,
                confidence_floor=cache_confidence_floor(self._request.confidence),
            )
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


def _origin_image_for(origin_images: Path, rel: Path) -> Path | None:
    """The origin image a staged label at *rel* came from, or None.

    F5: *rel* is the cache key, which was derived from the real filename via
    ``relative_to``, so it already CARRIES the true extension casing. Trying
    it verbatim first is what makes mixed case work. The old code tried only
    ``ext`` and ``ext.upper()``, so ``a.Jpg`` -- which passes the run scan's
    ``suffix.lower() in IMG_EXTS`` and costs real GPU time -- matched
    neither ``a.jpg`` nor ``a.JPG`` and was silently orphaned at promotion.
    Invisible on macOS's case-insensitive filesystem; a data-loss bug on the
    case-sensitive Linux lab shares this is deployed to. The ext loop is
    retained as a fallback for callers that pass a stem-only or
    differently-cased rel.
    """
    direct = origin_images / rel
    if direct.is_file():
        return direct
    stem = origin_images / rel.parent / rel.stem
    for ext in sorted(IMG_EXTS):
        for candidate in (
            stem.with_name(stem.name + ext),
            stem.with_name(stem.name + ext.upper()),
        ):
            if candidate.is_file():
                return candidate
    return None
