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
import math
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from hydra_suite.core.inference.semantic.base import SemanticInstance
from hydra_suite.core.inference.semantic.calibration import CONFIDENCE_GRID
from hydra_suite.core.inference.semantic.checkpoints import (
    SAM3_VARIANTS,
    resolve_checkpoint,
)
from hydra_suite.core.inference.semantic.shape_prior import AreaBand
from hydra_suite.core.inference.semantic.tiling import (
    DEFAULT_MERGE_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_SEAM_MARGIN_PX,
    SEMANTIC_TILE_FRACTION_SEED,
    TileCandidate,
    TileCollectionCancelled,
    TileProgressReporter,
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
from hydra_suite.utils.sam3_constants import PREDICTOR_IMGSZ

from .sam2_escalation import remove_staged_escalation_dir
from .semantic_workers import CalibrationWorker as CalibrationWorker  # noqa: F401
from .semantic_workers import FramePreviewWorker as FramePreviewWorker  # noqa: F401
from .semantic_workers import (  # noqa: F401
    SemanticEscalationWorker as SemanticEscalationWorker,
)
from .semantic_workers import TilePreviewWorker as TilePreviewWorker  # noqa: F401

logger = logging.getLogger(__name__)

CANDIDATES_FILENAME = "candidates.json"
CANDIDATE_STORE_DIRNAME = "candidates.v2"
MAX_CANDIDATE_FRAME_BYTES = 16 * 1024 * 1024
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
    # The PROJECT CLASS the staged instances are, which is not the prompt.
    # The prompt is a noun phrase chosen to make the model find things
    # ("ant with color patch"); writing it into the staging dir's
    # classes.txt made accept append it to the source as a NEW class the
    # project does not have -- and both the overlay and the training
    # dataset builder drop labels whose class is outside the project
    # scheme, so the accepted work rendered blank and would have trained
    # on nothing. Empty falls back to the prompt, which is what pre-fix
    # staging directories on disk contain.
    class_name: str = ""
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
    # Stable source identities for GUI requests. Kept after the pre-existing
    # fields so positional construction remains backward compatible.
    source_paths: list[str] = field(default_factory=list)


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


def staged_dirname_for(
    src: OBBSource,
    variant: str,
    prompt: str,
    *,
    imgsz: int = PREDICTOR_IMGSZ,
) -> str:
    """The staging directory NAME a (source, variant, prompt, imgsz) run targets.

    Shared with the GUI so it can tell a RESUME of the same run (same
    directory) from a REPLACE of a different one (different directory)
    without duplicating the hashing rule.
    """
    # DEPARTURE 2: the PROMPT enters the hash. Without it two prompts on
    # one source collide and the replaced-pending cleanup no-ops.
    # DEPARTURE 3: IMGSZ enters the hash. Candidates collected at one input
    # size are not interchangeable with another's, and nothing else would
    # invalidate them -- a silently stale cache reads as a successful reuse.
    content_hash = sha1(
        (
            str(Path(src.path).resolve()) + variant + prompt + f"|imgsz={int(imgsz)}"
        ).encode("utf-8")
    ).hexdigest()[:10]
    return f"{src.name}-sam3-{prompt_slug(prompt)}-{content_hash}"


def labeler_checkpoint_for(model_key: str):
    """Resolve a UI model key to a checkpoint path.

    Stock variants and published finetuned models are both selectable, so the
    key is no longer necessarily a SAM3_VARIANTS entry.

    DEVIATION from the literal brief: a stock variant resolves to ``None``,
    not its on-disk path. ``Sam3SemanticLabeler.from_variant`` treats a
    non-None ``checkpoint`` as "load exactly this artifact and require its
    sidecar" (``_sidecar_for_checkpoint`` raises if the sidecar is missing),
    which stock checkpoints never have -- and it also skips the
    probe/ensure-download flow entirely. Always resolving to a path here
    would turn every stock run into an immediate RuntimeError and silently
    drop the "offer to download the 3.45 GB checkpoint" UX. Only a
    published, finetuned registry key -- which DOES ship a sidecar -- should
    flow through as an explicit checkpoint.
    """
    if model_key in SAM3_VARIANTS:
        return None
    return resolve_checkpoint(model_key)


def _requested_sources(req: "SemanticEscalationRequest") -> list[OBBSource]:
    """Resolve stable source paths first; names remain a legacy fallback."""
    if req.source_paths:
        by_path = {
            str(Path(source.path).expanduser().resolve()): source
            for source in req.project.sources
        }
        return [
            by_path[resolved]
            for path in req.source_paths
            if (resolved := str(Path(path).expanduser().resolve())) in by_path
        ]
    by_name = {source.name: source for source in req.project.sources}
    return [by_name[name] for name in req.source_names if name in by_name]


def source_paths_pending_replacement(req: "SemanticEscalationRequest") -> list[str]:
    """Selected sources whose staged escalation this run would DESTROY.

    A source whose pending escalation is the very run being re-issued is a
    RESUME and is not listed -- only a different prompt, a different variant,
    or an unreviewed SAM2 result would be wiped, and the GUI must confirm
    those before they are.
    """
    project_root = _resolved(req.project.project_dir)
    out: list[str] = []
    for src in _requested_sources(req):
        if src.staged_review is None:
            continue
        target = (
            project_root
            / "artifacts"
            / "pending_escalations"
            / staged_dirname_for(src, req.variant, req.prompt)
        )
        if _resolved(src.staged_review.staged_path) != target:
            out.append(src.path)
    return out


def sources_pending_replacement(req: "SemanticEscalationRequest") -> list[str]:
    """Display names for selected sources whose staged work would be replaced."""
    pending_paths = set(source_paths_pending_replacement(req))
    return [
        source.name
        for source in _requested_sources(req)
        if source.path in pending_paths
    ]


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


class CandidateFrameStore:
    """Frame-atomic semantic candidates; memory is independent of source length."""

    def __init__(self, staged_root: str | Path) -> None:
        self.root = Path(staged_root) / CANDIDATE_STORE_DIRNAME

    @staticmethod
    def _name(rel: str) -> str:
        from hashlib import sha256

        return sha256(rel.encode("utf-8")).hexdigest() + ".json"

    def _path(self, rel: str) -> Path:
        return self.root / self._name(rel)

    def contains(self, rel: str) -> bool:
        path = self._path(rel)
        if not path.is_file():
            return False
        try:
            entry = self._read(path)
        except (OSError, TypeError, ValueError):
            return False
        return entry[0] == rel

    @staticmethod
    def _read(path: Path) -> tuple[str, dict]:
        with path.open("rb") as stream:
            encoded = stream.read(MAX_CANDIDATE_FRAME_BYTES + 1)
        if len(encoded) > MAX_CANDIDATE_FRAME_BYTES:
            raise ValueError("semantic candidate frame exceeds safe size")
        raw = json.loads(encoded)
        if not isinstance(raw, dict) or set(raw) != {
            "version",
            "relative_path",
            "hw",
            "candidates",
        }:
            raise ValueError("semantic candidate frame has invalid fields")
        rel = raw["relative_path"]
        hw = raw["hw"]
        candidates = raw["candidates"]
        if (
            raw["version"] != 2
            or not isinstance(rel, str)
            or len(rel) > 4096
            or Path(rel).is_absolute()
            or ".." in Path(rel).parts
            or not isinstance(hw, list)
            or len(hw) != 2
            or any(not isinstance(value, int) or value < 1 for value in hw)
            or not isinstance(candidates, list)
            or len(candidates) > 100_000
        ):
            raise ValueError("semantic candidate frame is malformed")
        for candidate in candidates:
            if not isinstance(candidate, dict) or set(candidate) != {"p", "c", "t"}:
                raise ValueError("semantic candidate is malformed")
            points = candidate["p"]
            if (
                not isinstance(points, list)
                or not 3 <= len(points) <= 1_000_000
                or any(
                    not isinstance(point, list)
                    or len(point) != 2
                    or any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                        for value in point
                    )
                    for point in points
                )
                or not isinstance(candidate["c"], (int, float))
                or isinstance(candidate["c"], bool)
                or not math.isfinite(candidate["c"])
                or not isinstance(candidate["t"], int)
                or isinstance(candidate["t"], bool)
                or candidate["t"] < 0
            ):
                raise ValueError("semantic candidate polygon is malformed")
        return rel, {"hw": hw, "candidates": candidates}

    def write(self, rel: str, hw: tuple[int, int], candidates: list[dict]) -> None:
        payload = {
            "version": 2,
            "relative_path": rel,
            "hw": [int(hw[0]), int(hw[1])],
            "candidates": candidates,
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_CANDIDATE_FRAME_BYTES:
            raise ValueError("semantic candidate frame exceeds safe size")
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self._name(rel)}.", suffix=".tmp", dir=self.root
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path(rel))
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def iter_entries(self):
        if not self.root.is_dir():
            return
        for path in sorted(self.root.glob("*.json")):
            rel, entry = self._read(path)
            if path.name != self._name(rel):
                raise ValueError("semantic candidate path identity mismatch")
            yield rel, entry

    def count(self) -> int:
        return sum(1 for _item in self.iter_entries())


def _image_hw(path: Path) -> tuple[int, int] | None:
    """Read image dimensions without decoding a full frame for ETA planning."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.height), int(image.width)
    except Exception:
        logger.warning("Could not read image dimensions for ETA planning: %s", path)
        return None


def _remaining_tile_count(
    req: SemanticEscalationRequest,
    src: OBBSource,
    *,
    project_root: Path,
    tile_px: int | None,
    floor: float,
) -> int:
    """Tile work still needed for one source, including resume-aware caching."""
    src_root = Path(src.path)
    staged_root = (
        project_root
        / "artifacts"
        / "pending_escalations"
        / staged_dirname_for(src, req.variant, req.prompt)
    )
    fingerprint = _fingerprint(req, src_root, tile_px, floor)
    completed: set[str] = set()
    run_path = staged_root / RUN_FILENAME
    if run_path.exists():
        try:
            if json.loads(run_path.read_text()) == fingerprint:
                store = CandidateFrameStore(staged_root)
                if store.root.is_dir():
                    completed = {rel for rel, _entry in store.iter_entries()}
                else:
                    completed = set(_load_cache(staged_root)["images"])
        except Exception:
            pass
    images_dir = src_root / "images"
    total = 0
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        try:
            rel = str(img_path.relative_to(images_dir))
        except ValueError:  # pragma: no cover - defensive filesystem race
            continue
        if rel in completed:
            continue
        hw = _image_hw(img_path)
        if hw is None:
            continue
        plan = (
            plan_for_frame(hw, tile_px, req.overlap) if tile_px else full_frame_plan(hw)
        )
        total += len(plan.tiles)
    return total


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
    cache: dict | CandidateFrameStore,
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
    entries = (
        cache.iter_entries()
        if isinstance(cache, CandidateFrameStore)
        else cache["images"].items()
    )
    for rel, entry in entries:
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
    on_mutated: Callable[[], None] | None = None,
) -> SemanticEscalationResult:
    """Stage prompt-driven polygon labels for each named source.

    Never touches a source's own labels. Promotion is frame-granular review
    (``jobs/staged_review.py``): the user accepts or rejects individual
    staged frames into the source the run was made against, rather than
    promoting the whole staged run into a new sibling source.

    ``on_mutated`` is called immediately after EVERY write to a source's
    ``staged_review``, so the caller can persist the project right then.
    Without it the pointer lived only in memory until the run returned and
    the caller saved on success -- so an exception on source 2, or the app
    being closed before the run returned, left the fully-written staging
    directory on disk with NOTHING pointing at it. The review bar keys off
    the pointer, so those staged frames became unreviewable and invisible
    (recovery was a full re-run). The clear at the top of the loop matters
    just as much as the set at the bottom: by then the old staging
    directory is already deleted, so an unpersisted clear leaves the source
    pointing at a directory that no longer exists.
    """

    def _mutated() -> None:
        if on_mutated is not None:
            on_mutated()

    result = SemanticEscalationResult()
    # DEPARTURE 1: no `level != "polygon"` filter. Finding animals the
    # existing polygons missed is a primary use case for this feature.
    todo = _requested_sources(req)
    # Canonical, so every staged-path comparison below matches the paths
    # ensure_bundle_subdirectory hands back (see _resolved).
    project_root = _resolved(req.project.project_dir)
    floor = cache_confidence_floor(req.confidence)
    tile_px = req.tile_px or resolve_tile_px(req.reference_body_px, req.tile_fraction)
    result.tile_px = tile_px
    counting_labeler = _DegenerateCountingLabeler(labeler)
    if progress is not None:
        progress(0, "Planning SAM3 tile work…")
    remaining_tiles = sum(
        _remaining_tile_count(
            req, src, project_root=project_root, tile_px=tile_px, floor=floor
        )
        for src in todo
        if not (
            src.staged_review is not None
            and _resolved(src.staged_review.staged_path)
            != project_root
            / "artifacts"
            / "pending_escalations"
            / staged_dirname_for(src, req.variant, req.prompt)
            and not (overwrite or req.overwrite)
        )
    )
    tile_progress = TileProgressReporter(remaining_tiles)
    tiles_completed = 0
    if progress is not None:
        progress(0, f"Running SAM3 inference across {remaining_tiles} tile(s)…")

    for src in todo:
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
            _mutated()

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

        store = CandidateFrameStore(staged_root)
        legacy_cache = (
            {"version": 1, "images": {}}
            if stale or store.root.is_dir()
            else _load_cache(staged_root)
        )
        if legacy_cache["images"]:
            # One compatibility read of the old monolith, followed by bounded
            # frame-at-a-time writes. New and resumed writes never rewrite the
            # whole-source JSON document again.
            for rel, entry in legacy_cache["images"].items():
                store.write(
                    rel,
                    (int(entry["hw"][0]), int(entry["hw"][1])),
                    list(entry["candidates"]),
                )
            legacy_cache = {"version": 1, "images": {}}
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
            if store.contains(rel) or rel in legacy_cache["images"]:
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
                tiles_before_frame = tiles_completed
                frame_total = len(images)

                def _report_tile(
                    done: int,
                    total: int,
                    *,
                    tiles_before_frame: int = tiles_before_frame,
                    source_name: str = src.name,
                    frame_number: int = ii + 1,
                    frame_total: int = frame_total,
                ) -> None:
                    if progress is None:
                        return
                    pct, message = tile_progress.report(
                        tiles_before_frame + done,
                        f"{source_name}: frame {frame_number}/{frame_total}, "
                        f"tile {done}/{total}",
                    )
                    progress(pct, message)

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
                    progress=_report_tile,
                )
            except TileCollectionCancelled:
                # F1: a half-tiled frame is NOT cached. Caching it would let
                # the `rel in cache["images"]` resume check skip the frame
                # forever, so "re-run to carry on" would do nothing and the
                # staged labels would be silently truncated. Leaving the key
                # absent makes the resume redo this frame from tile zero.
                result.cancelled = True
                break
            tiles_completed += len(plan.tiles)
            store.write(rel, (h, w), _candidates_to_json(cands))

        written, degenerate, orphaned = _write_labels_from_candidates(
            staged_root,
            store if store.root.is_dir() else legacy_cache,
            confidence=req.confidence,
            merge_iou=req.merge_iou,
            area_band=band_from_bounds(req.area_min_px2, req.area_max_px2),
            origin_images=images_dir,
        )
        result.labelled += written
        result.degenerate += degenerate
        result.orphaned += orphaned
        entries = (
            store.iter_entries()
            if store.root.is_dir()
            else legacy_cache["images"].items()
        )
        processed = empty = 0
        for _rel, entry in entries:
            processed += 1
            empty += int(not entry["candidates"])
        result.empty_images += empty
        # I1: the prompt-failure denominator. Frames actually inferred (the
        # cache's key set), which under cancellation or resume is NOT
        # len(images) either.
        result.frames_processed += processed
        # Deliberately NOT in `_fingerprint`: which class the instances ARE
        # is a labelling decision, not an inference input, so changing it
        # must not wipe a cache full of candidates. Same precedent as the
        # area band -- a re-threshold, not a re-run. classes.txt is
        # rewritten on every run including a resume, so the new class lands
        # without re-inferring a single tile.
        staged_class = req.class_name.strip() or req.prompt.strip() or "object"
        (staged_root / "classes.txt").write_text(f"{staged_class}\n")
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
        _mutated()
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
    store = CandidateFrameStore(staged_root)
    cache: dict | CandidateFrameStore = (
        store if store.root.is_dir() else _load_cache(staged_root)
    )
    if (
        store.count() == 0
        if isinstance(cache, CandidateFrameStore)
        else not cache["images"]
    ):
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
CALIBRATION_SAMPLE_FRAMES = 12


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


def stratified_calibration_frames(
    sources, *, budget: int = CALIBRATION_SAMPLE_FRAMES
) -> list[tuple[Path, list[LabelRecord]]]:
    """Return a deterministic, globally bounded sample spread across sources."""
    sources = list(sources)
    budget = max(1, int(budget))
    if not sources:
        return []
    per_source = max(1, budget // len(sources))
    sampled = [labelled_frames_for(source, limit=per_source) for source in sources]
    output: list[tuple[Path, list[LabelRecord]]] = []
    for row in range(per_source):
        for frames in sampled:
            if row < len(frames):
                output.append(frames[row])
                if len(output) >= budget:
                    return output
    return output


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
    status: Callable[[str], None] | None = None,
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
    tile_progress = TileProgressReporter(len(plan.tiles))

    def _report_tile(done: int, total: int) -> None:
        if progress is not None:
            progress(done, total)
        if status is not None:
            _pct, message = tile_progress.report(
                done, f"Segmenting {img_path.name}: tile {done}/{total}"
            )
            status(message)

    candidates = collect_candidates(
        labeler,
        image,
        plan,
        prompt,
        confidence_threshold=cache_confidence_floor(confidence),
        max_instances=max_instances,
        seam_margin_px=seam_margin_px,
        progress=_report_tile,
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
