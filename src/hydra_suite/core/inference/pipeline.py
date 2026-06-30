"""Inference pipeline orchestrator (depth=1).

The :class:`Pipeline` turns a stream of video frames into per-type cache writes
by running the batch-native stage layer over fixed, frame-indexed windows. Batch
boundaries are a pure function of frame index (window ``k`` = frames
``[k*W, (k+1)*W)``), never of arrival order, so reruns and backward passes see
identical batching.

This module implements **depth=1 only**: window ``k`` is fully processed before
window ``k+1`` begins (synchronous, no threads). Depths >1 (pipelined prefetch /
overlap) and the async ``CacheWriter`` are later tasks; the structure here is
deliberately shaped so they can be slotted in without changing the windowing or
the per-window stage sequence.

Numeric contract (depth=1): the per-type caches produced for a given video MUST
match what the legacy per-frame ``runner._run_batch`` produced — same detections,
same head-tail / CNN / pose / AprilTag per frame. The batch stage functions were
proven equivalent to per-frame calls in Task 3; the pose path additionally uses
``extract_canonical_crops_batch`` (pad-to-window-max, never resize) so pose crops
are bit-identical to the per-frame ``extract_canonical_crops`` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from .result import FrameResult, OBBResult
from .stages.apriltag import run_apriltag
from .stages.cnn import run_cnn_batch
from .stages.crops import extract_aabb_crops, extract_canonical_crops_batch
from .stages.filtering import filter_with_indices
from .stages.headtail import run_headtail_batch
from .stages.obb import _RawOBBTensors, materialize_tensors, run_obb
from .stages.pose import run_pose_batch


@dataclass
class BatchWindow:
    """A window of consecutive frames plus their absolute frame indices.

    ``frames`` are the raw frame payloads (numpy BGR arrays in production, or any
    object in tests). ``frame_indices[i]`` is the absolute video frame index of
    ``frames[i]``. The two lists are always parallel and equal length.
    """

    frames: list
    frame_indices: list[int]

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class InferencePassResult:
    """Outcome of a full pipeline pass.

    ``frames_processed`` counts frames fed through windows. ``frame_results`` is
    the in-memory list of per-frame :class:`FrameResult`s (assembled via
    ``scatter``); cache writes are a separate, raw-result side effect performed
    inside ``_process_window`` to preserve exact legacy cache parity.
    """

    frames_processed: int = 0
    frame_results: list = field(default_factory=list)


@dataclass
class PipelineStages:
    """The loaded models + config the pipeline needs to call the stage layer.

    A thin bundle (no behavior) so :class:`Pipeline` does not reach into the
    runner's private ``_AllModels`` / ``InferenceConfig`` shapes directly.
    """

    config: Any
    obb_models: Any
    headtail_model: Any
    cnn_models: list
    pose_model: Any
    apriltag_model: Any


# --- test-only frame shim -------------------------------------------------


@dataclass
class _TestFrame:
    """Index-carrying frame used by ``Pipeline.for_test`` / ``run_frames``.

    The depth=1 windowing logic is a pure function of frame index; tests need to
    exercise ``_iter_windows`` without real models or numpy frames, so the test
    shim wraps each index in an object exposing ``.index``.
    """

    index: int


class Pipeline:
    """Synchronous (depth=1) inference orchestrator over frame-indexed windows."""

    def __init__(
        self,
        stages: PipelineStages,
        runtime: Any,
        cache_writer: Any,
        *,
        depth: int = 1,
        queue_bound: int | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError(f"pipeline depth must be >= 1, got {depth}")
        # Only the synchronous (depth=1) execution model is implemented. Higher
        # depths request pipelined prefetch/overlap (a later task); until that
        # lands they run synchronously here, which is numerically identical —
        # depth changes *when* windows run, never their per-frame output. The
        # effective depth is therefore pinned to 1 for execution.
        self._requested_depth = depth
        self.stages = stages
        self.runtime = runtime
        self.cache_writer = cache_writer
        self.depth = 1  # execution model: synchronous depth=1
        self.queue_bound = queue_bound
        self._window_size = (
            int(stages.config.detection_batch_size) if stages is not None else 1
        )
        # Test-only per-window stage callable (set by ``for_test``); when present,
        # ``run_frames`` uses it instead of the production ``_process_window``.
        self._test_stage: Callable[[BatchWindow], list] | None = None

    # --- windowing ------------------------------------------------------

    @property
    def window_size(self) -> int:
        return max(1, self._window_size)

    def _iter_windows(
        self, frames: list, frame_indices: list[int]
    ) -> Iterator[BatchWindow]:
        """Yield fixed-size windows over a materialized (frames, indices) pair.

        Windows are sliced by position in the provided lists; callers guarantee
        ``frame_indices`` is contiguous and ascending so positional windows equal
        frame-index windows ``[k*W, (k+1)*W)``.
        """
        w = self.window_size
        for start in range(0, len(frames), w):
            yield BatchWindow(
                frames=frames[start : start + w],
                frame_indices=frame_indices[start : start + w],
            )

    # --- production stage sequence -------------------------------------

    def _process_window(self, window: BatchWindow) -> list[FrameResult]:
        """Run OBB → crops → HT/CNN/pose → AprilTag → scatter for one window.

        Cache writes mirror ``runner._run_batch`` exactly (raw stage results, no
        foreign-keypoint suppression at cache time — that is an assemble-layer
        concern). The returned :class:`FrameResult`s are the in-memory assembled
        view via ``scatter``; the runner discards them (parity is in the caches).
        """
        from .stages.assemble import scatter

        cfg = self.stages.config
        frames = window.frames
        frame_indices = window.frame_indices

        ar = cfg.headtail.canonical_aspect_ratio if cfg.headtail else 2.0
        mg = cfg.headtail.canonical_margin if cfg.headtail else 1.3

        # --- OBB: cross-frame native batch (unchanged from _run_batch) ---
        raw_list = run_obb(frames, self.stages.obb_models, cfg.obb, self.runtime)

        # Materialize + re-stamp detection_ids per frame, write detection cache
        # for EVERY frame (including empty), and collect filtered OBBs.
        filtered_by_frame: dict[int, OBBResult] = {}
        det_indices_by_frame: dict[int, np.ndarray] = {}
        nonempty_frames: list = []
        nonempty_obbs: list[OBBResult] = []

        for frame, frame_idx, raw in zip(frames, frame_indices, raw_list):
            obb_result = (
                materialize_tensors(raw, cfg.obb.raw_detection_cap)
                if isinstance(raw, _RawOBBTensors)
                else raw
            )
            obb_result = OBBResult(
                frame_idx=frame_idx,
                centroids=obb_result.centroids,
                angles=obb_result.angles,
                sizes=obb_result.sizes,
                shapes=obb_result.shapes,
                confidences=obb_result.confidences,
                corners=obb_result.corners,
                detection_ids=OBBResult.make_detection_ids(
                    frame_idx, obb_result.num_detections
                ),
            )
            self.cache_writer.write_detection(frame_idx, obb_result)

            filtered_obb, det_indices = filter_with_indices(obb_result, cfg.obb)
            if filtered_obb.num_detections == 0:
                # _run_batch ``continue``s: no downstream stage / cache writes.
                continue
            filtered_by_frame[frame_idx] = filtered_obb
            det_indices_by_frame[frame_idx] = det_indices
            nonempty_frames.append(frame)
            nonempty_obbs.append(filtered_obb)

        if not nonempty_obbs:
            return []

        # --- downstream batch stages over the non-empty frames -------------
        headtail: dict[int, Any] | None = None
        if self.stages.headtail_model is not None:
            headtail = run_headtail_batch(
                nonempty_frames,
                nonempty_obbs,
                self.stages.headtail_model,
                cfg.headtail,
                self.runtime,
                ar,
                mg,
            )

        cnns_by_frame: dict[int, list] = {idx: [] for idx in filtered_by_frame}
        cnn_per_phase: list[dict[int, Any]] = []
        for cfg_cnn, mdl in zip(cfg.cnn_phases, self.stages.cnn_models):
            phase = run_cnn_batch(
                nonempty_frames,
                nonempty_obbs,
                mdl,
                cfg_cnn,
                self.runtime,
                ar,
                mg,
            )
            cnn_per_phase.append(phase)
            for idx, result in phase.items():
                cnns_by_frame[idx].append(result)

        pose: dict[int, Any] | None = None
        if self.stages.pose_model is not None:
            crop_batch = extract_canonical_crops_batch(
                nonempty_frames, nonempty_obbs, ar, mg, self.runtime
            )
            pose = run_pose_batch(
                crop_batch, self.stages.pose_model, cfg.pose, self.runtime
            )

        apriltag: dict[int, Any] | None = None
        if self.stages.apriltag_model is not None:
            apriltag = {}
            for frame, obb in zip(nonempty_frames, nonempty_obbs):
                aabb_crops = extract_aabb_crops(
                    frame, obb, padding=cfg.apriltag.crop_padding
                )
                apriltag[obb.frame_idx] = run_apriltag(
                    aabb_crops, obb, self.stages.apriltag_model, cfg.apriltag
                )

        # --- write RAW per-frame results to the per-type caches ------------
        # Mirrors _run_batch: writes raw stage outputs (no foreign suppression),
        # only for non-empty frames, keyed by that frame's det_indices.
        for frame_idx in sorted(filtered_by_frame):
            det_indices = det_indices_by_frame[frame_idx]
            ht = headtail.get(frame_idx) if headtail is not None else None
            cnn_results = [
                phase[frame_idx] for phase in cnn_per_phase if frame_idx in phase
            ]
            pose_result = pose.get(frame_idx) if pose is not None else None
            at_result = apriltag.get(frame_idx) if apriltag is not None else None
            self.cache_writer.write_downstream(
                frame_idx,
                det_indices=det_indices,
                headtail=ht,
                cnn_results=cnn_results,
                pose=pose_result,
                apriltag=at_result,
            )

        # In-memory assembled view (foreign suppression applied here, per config).
        return scatter(
            filtered_by_frame,
            headtail,
            cnns_by_frame,
            pose,
            apriltag,
            cfg,
            overrides_headtail=(
                cfg.pose.overrides_headtail if cfg.pose is not None else True
            ),
        )

    # --- production driver ---------------------------------------------

    def run(
        self,
        frame_source: Iterable,
        frame_range: range,
    ) -> InferencePassResult:
        """Drive depth=1 windows over ``frame_source`` for ``frame_range``.

        ``frame_source`` yields ``(frame_idx, frame)`` pairs in ascending index
        order. Windows are emitted as soon as ``W`` frames have arrived, so the
        whole video is never buffered. ``frame_range`` is informational here
        (the runner already clamps the read loop); it lets ``run`` cross-check the
        expected count for the in-memory result.
        """
        result = InferencePassResult()
        frames_buf: list = []
        indices_buf: list[int] = []
        w = self.window_size

        for frame_idx, frame in frame_source:
            frames_buf.append(frame)
            indices_buf.append(int(frame_idx))
            result.frames_processed += 1
            if len(frames_buf) == w:
                window = BatchWindow(
                    frames=list(frames_buf), frame_indices=list(indices_buf)
                )
                result.frame_results.extend(self._process_window(window))
                frames_buf.clear()
                indices_buf.clear()

        if frames_buf:
            window = BatchWindow(
                frames=list(frames_buf), frame_indices=list(indices_buf)
            )
            result.frame_results.extend(self._process_window(window))

        return result

    # --- test harness ---------------------------------------------------

    @classmethod
    def for_test(
        cls,
        window_size: int,
        depth: int,
        stage: Callable[[BatchWindow], list],
    ) -> "Pipeline":
        """Build a model-free Pipeline whose per-window stage is ``stage``.

        Used to exercise the windowing logic (``_iter_windows`` + per-window
        dispatch) without loading real models. ``run_frames`` feeds the same
        frame-indexed windows the production ``run`` would build.
        """
        pipe = cls.__new__(cls)
        if depth != 1:
            raise NotImplementedError("Pipeline.for_test supports depth=1 only")
        pipe.stages = None
        pipe.runtime = None
        pipe.cache_writer = None
        pipe.depth = depth
        pipe.queue_bound = None
        pipe._window_size = int(window_size)
        pipe._test_stage = stage
        return pipe

    def run_frames(self, frame_range: Iterable[int]) -> list:
        """Test shim: window ``frame_range`` and run the fake stage per window.

        Builds ``BatchWindow``s whose ``frames`` are index-carrying ``_TestFrame``
        objects, so windows are frame-indexed (``[0,1],[2,3],[4]`` for range(5),
        W=2) — identical boundaries to production. Returns the flat list of
        per-window stage outputs.
        """
        assert self._test_stage is not None, "run_frames requires for_test()"
        indices = list(frame_range)
        frames = [_TestFrame(index=i) for i in indices]
        results: list = []
        for window in self._iter_windows(frames, indices):
            results.extend(self._test_stage(window))
        return results
