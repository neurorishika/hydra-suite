"""Inference pipeline orchestrator (depth=1 synchronous, depth>=2 deep prefetch).

The :class:`Pipeline` turns a stream of video frames into per-type cache writes
by running the batch-native stage layer over fixed, frame-indexed windows. Batch
boundaries are a pure function of frame index (window ``k`` = frames
``[k*W, (k+1)*W)``), never of arrival order, so reruns and backward passes see
identical batching.

Two execution models are implemented:

* **depth=1** — fully synchronous: window ``k`` is OBB'd and fully processed
  (crops → HT/CNN/pose → AprilTag → scatter → cache write) before window ``k+1``
  begins. No threads.
* **depth>=2** — deep prefetch: a single PRODUCER thread runs decode+OBB and
  pushes ``(window, obb_raw_list)`` onto a bounded
  ``queue.Queue(maxsize=depth-1)``, while a SINGLE in-order consumer (the calling
  thread) pulls windows in strict ascending order and runs crops → individual
  stages → scatter → cache write. ``maxsize=depth-1`` lets the producer run up to
  ``depth-1`` windows ahead (depth=2 is the classic double buffer with one
  in-flight window; depth=4 allows three). The producer blocks when the queue is
  full — natural backpressure. depth scales *only* the prefetch runway; it never
  adds a second consumer and never reorders windows. The OBB output tensors are
  synced across threads via ``RuntimeContext.handoff`` (producer) /
  ``await_handoff`` (consumer) on the SAME tensor objects (a detach/clone would
  create a new key and silently miss the event). Stop is checked only at window
  boundaries so no window is ever half-written. A supervisor re-raises any
  stage/producer error to the caller after joining threads and flushing the
  cache writer.

Numeric contract: the per-type caches produced for a given video MUST match what
the legacy per-frame ``runner._run_batch`` produced — same detections, same
head-tail / CNN / pose / AprilTag per frame — AND output MUST be byte-identical
across ALL depths. Byte parity holds for any depth because (a) batch boundaries
are a pure function of frame index, (b) there is always exactly ONE consumer
processing windows in strict frame order (a deeper queue only buffers more
produced windows, it does not reorder or parallelize consumption), and (c) the
per-window downstream + cache-write code path is literally the same method for
every depth.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from .result import FrameResult, OBBResult
from .stages.apriltag import run_apriltag
from .stages.bgsub import run_bgsub_batch
from .stages.cnn import run_cnn_batch
from .stages.crops import extract_aabb_crops, extract_canonical_crops_batch
from .stages.filtering import filter_for_source
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
    # Set instead of ``obb_models`` when config.detection_source == "bgsub".
    # Last, with a default, so existing keyword constructions are unaffected.
    bgsub_model: Any = None


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
    """Inference orchestrator over frame-indexed windows.

    depth=1 runs fully synchronously; depth>=2 runs a producer (decode+OBB)
    ahead of a single in-order consumer via a bounded queue (``maxsize=depth-1``).
    All depths produce byte-identical caches.
    """

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
        # depth=1 -> synchronous (no threads). depth>=2 -> producer/consumer
        # with a single in-order consumer and a bounded prefetch queue. Increasing
        # depth deepens the prefetch (the OBB producer may run up to ``depth-1``
        # windows ahead) but never changes per-frame output: there is still ONE
        # consumer pulling windows in strict ascending order and writing caches
        # in-order, so output is byte-identical across all depths.
        self.stages = stages
        self.runtime = runtime
        self.cache_writer = cache_writer
        # Effective execution model: 1 (sync) or N>=2 (deep prefetch). depth
        # takes effect directly — no clamping.
        self.depth = depth
        # Bounded hand-off queue size for depth>=2. The default scales with depth:
        # ``maxsize = depth - 1`` lets the producer run up to ``depth-1`` windows
        # ahead of the single consumer (depth=2 -> 1, the classic double buffer;
        # depth=4 -> 3). An explicit ``queue_bound`` overrides this.
        if queue_bound is not None:
            self.queue_bound = queue_bound
        else:
            self.queue_bound = max(1, depth - 1)
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

    def _run_detection_for_window(self, window: BatchWindow) -> list:
        """Producer stage: run the detection source for one window.

        Returns the raw per-frame detection list (``_RawOBBTensors`` or
        ``OBBResult`` per frame). On CUDA the raw OBB tensors are passed through
        ``RuntimeContext.handoff`` so the consumer thread can ``await_handoff``
        the SAME tensor objects before reading them — the tensor objects are
        returned unchanged (no detach/clone), preserving the handoff key. On
        CPU/MPS ``handoff`` is an identity no-op.

        bg-sub is CPU numpy throughout and never yields ``_RawOBBTensors``, so
        the handoff/CUDA-event path does not apply to it. It is also strictly
        sequential (the BackgroundModel carries cross-frame state) — safe here
        because the producer is the single thread that ever touches the model,
        and it walks windows in ascending frame order.
        """
        cfg = self.stages.config
        if cfg.detection_source == "bgsub":
            return run_bgsub_batch(
                window.frames,
                window.frame_indices,
                self.stages.bgsub_model,
                cfg.bgsub,
                self.runtime,
            )
        raw_list = run_obb(window.frames, self.stages.obb_models, cfg.obb, self.runtime)
        for raw in raw_list:
            if isinstance(raw, _RawOBBTensors):
                # Same-object handoff: record a CUDA event keyed by each device
                # tensor. Do NOT detach/clone — that would create a new key.
                self.runtime.handoff(raw.xywhr)
                self.runtime.handoff(raw.corners)
                self.runtime.handoff(raw.conf)
        return raw_list

    def _process_window(self, window: BatchWindow) -> list[FrameResult]:
        """Run OBB → crops → HT/CNN/pose → AprilTag → scatter for one window.

        Synchronous (depth=1) entry point: OBB then downstream in one thread.
        """
        raw_list = self._run_detection_for_window(window)
        return self._process_obb_results(window, raw_list)

    def _process_obb_results(
        self, window: BatchWindow, raw_list: list
    ) -> list[FrameResult]:
        """Consumer stage: crops → HT/CNN/pose → AprilTag → scatter + cache write.

        ``raw_list`` is the detection output (from ``_run_detection_for_window``) for
        ``window``. Cache writes mirror ``runner._run_batch`` exactly (raw stage
        results, no foreign-keypoint suppression at cache time — that is an
        assemble-layer concern). The returned :class:`FrameResult`s are the
        in-memory assembled view via ``scatter``; the runner discards them
        (parity is in the caches).
        """
        from .stages.assemble import scatter

        cfg = self.stages.config
        frames = window.frames
        frame_indices = window.frame_indices

        ar = cfg.headtail.canonical_aspect_ratio if cfg.headtail else 2.0
        mg = cfg.headtail.canonical_margin if cfg.headtail else 1.3

        # Consumer-side stream-sync: wait on the producer's handoff events for the
        # SAME tensor objects before the first device read (no-op on CPU/MPS).
        for raw in raw_list:
            if isinstance(raw, _RawOBBTensors):
                self.runtime.await_handoff(raw.xywhr)
                self.runtime.await_handoff(raw.corners)
                self.runtime.await_handoff(raw.conf)

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

            filtered_obb, det_indices = filter_for_source(cfg, obb_result)
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
            suppress_foreign = (
                cfg.pose.suppress_foreign_regions if cfg.pose is not None else False
            )
            background_color = (
                cfg.pose.background_color if cfg.pose is not None else (0, 0, 0)
            )
            crop_batch = extract_canonical_crops_batch(
                nonempty_frames,
                nonempty_obbs,
                ar,
                mg,
                self.runtime,
                suppress_foreign=suppress_foreign,
                background_color=background_color,
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
        progress_cb: Callable[[int, int], None] | None = None,
        range_total: int = 0,
        should_stop: Callable[[], bool] | None = None,
    ) -> InferencePassResult:
        """Drive the full pass over ``frame_source`` for ``frame_range``.

        ``frame_source`` yields ``(frame_idx, frame)`` pairs in ascending index
        order. Windows are emitted as soon as ``W`` frames have arrived, so the
        whole video is never buffered. ``frame_range`` is informational here
        (the runner already clamps the read loop).

        ``progress_cb(processed, range_total)`` is called with the same cadence
        as the legacy runner read loop (every ``max(1, range_total // 100)``
        frames read, plus a final call). At depth=1 this runs synchronously;
        at depth>=2 a producer thread runs decode+OBB up to ``depth-1`` windows
        ahead (bounded queue) while this single in-order consumer thread runs the
        downstream stages + cache writes.

        ``should_stop``, if given, is polled at window boundaries; when it
        returns ``True`` the pass returns early with whatever frames were
        already processed (a partial ``InferencePassResult``).
        """
        if self.depth == 1:
            return self._run_sync(frame_source, progress_cb, range_total, should_stop)
        return self._run_double_buffer(
            frame_source, progress_cb, range_total, should_stop
        )

    # --- depth=1: synchronous --------------------------------------------

    def _run_sync(
        self,
        frame_source: Iterable,
        progress_cb: Callable[[int, int], None] | None,
        range_total: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> InferencePassResult:
        result = InferencePassResult()
        w = self.window_size
        step = max(1, range_total // 100) if range_total > 0 else 1

        for window in self._stream_windows(frame_source, w):
            if should_stop is not None and should_stop():
                break
            result.frames_processed += len(window)
            result.frame_results.extend(self._process_window(window))
            if progress_cb and range_total > 0:
                # Mirror the legacy per-frame cadence: emit at each multiple of
                # ``step`` crossed within this window.
                self._emit_progress(
                    progress_cb,
                    result.frames_processed,
                    len(window),
                    step,
                    range_total,
                )

        if progress_cb:
            progress_cb(result.frames_processed, range_total)
        return result

    # --- depth>=2: producer/consumer double buffer ------------------------

    def _run_double_buffer(
        self,
        frame_source: Iterable,
        progress_cb: Callable[[int, int], None] | None,
        range_total: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> InferencePassResult:
        result = InferencePassResult()
        w = self.window_size
        step = max(1, range_total // 100) if range_total > 0 else 1

        # Bounded hand-off: (window, raw_obb_list) producer -> consumer. A
        # sentinel ``None`` marks end-of-stream. maxsize bounds in-flight windows
        # so the producer can run at most ``queue_bound`` windows ahead.
        handoff_q: queue.Queue = queue.Queue(maxsize=max(1, int(self.queue_bound)))
        stop = threading.Event()
        producer_error: list[BaseException] = []
        # Frames read so far (written by producer, read for progress). Guarded by
        # being the producer's sole responsibility; the consumer only reads it
        # after the producer has put the corresponding window on the queue.
        read_counter = {"n": 0}

        def producer() -> None:
            try:
                for window in self._stream_windows(frame_source, w):
                    if stop.is_set() or (should_stop is not None and should_stop()):
                        stop.set()
                        break
                    raw_list = self._run_detection_for_window(window)
                    read_counter["n"] += len(window)
                    # Carry the running read count so the consumer can emit
                    # progress with the same cadence as the sync path.
                    handoff_q.put((window, raw_list, read_counter["n"]))
            except BaseException as exc:  # noqa: BLE001,B036 supervisor
                producer_error.append(exc)
                stop.set()
            finally:
                handoff_q.put(None)  # sentinel (always, even on error)

        producer_thread = threading.Thread(
            target=producer, name="pipeline-obb-producer", daemon=True
        )
        producer_thread.start()

        consumer_error: BaseException | None = None
        try:
            while True:
                item = handoff_q.get()
                if item is None:  # producer finished or errored
                    break
                window, raw_list, read_n = item
                result.frames_processed += len(window)
                result.frame_results.extend(self._process_obb_results(window, raw_list))
                if progress_cb and range_total > 0:
                    self._emit_progress(
                        progress_cb,
                        read_n,
                        len(window),
                        step,
                        range_total,
                    )
        except BaseException as exc:  # noqa: BLE001,B036 supervisor
            consumer_error = exc
        finally:
            # Supervisor teardown: stop the producer at the next window boundary,
            # drain the queue so a blocked producer ``put`` unblocks, then join.
            stop.set()
            self._drain_queue(handoff_q)
            producer_thread.join(timeout=30.0)
            # Flush + close the (async) cache writer so no write is left pending,
            # regardless of whether we are unwinding an error or finishing clean.
            # ``cache_writer`` is only ``None`` for Pipeline.for_test() shims used
            # in tests; every real (non-test) Pipeline always has one.
            if self.cache_writer is not None:
                try:
                    self.cache_writer.flush()
                finally:
                    self.cache_writer.close()

        if consumer_error is not None:
            raise consumer_error
        if producer_error:
            raise producer_error[0]

        if progress_cb:
            progress_cb(result.frames_processed, range_total)
        return result

    # --- shared helpers ---------------------------------------------------

    def _stream_windows(self, frame_source: Iterable, w: int) -> Iterator[BatchWindow]:
        """Yield fixed-size windows from a ``(frame_idx, frame)`` stream.

        Windows close at every ``w`` frames; the final partial window (if any) is
        yielded at end-of-stream. Boundaries are a pure function of arrival order
        (which the runner guarantees is ascending frame index), identical to
        ``_iter_windows`` over a fully materialized list.
        """
        frames_buf: list = []
        indices_buf: list[int] = []
        for frame_idx, frame in frame_source:
            frames_buf.append(frame)
            indices_buf.append(int(frame_idx))
            if len(frames_buf) == w:
                yield BatchWindow(
                    frames=list(frames_buf), frame_indices=list(indices_buf)
                )
                frames_buf.clear()
                indices_buf.clear()
        if frames_buf:
            yield BatchWindow(frames=list(frames_buf), frame_indices=list(indices_buf))

    @staticmethod
    def _emit_progress(
        progress_cb: Callable[[int, int], None],
        processed_now: int,
        window_len: int,
        step: int,
        range_total: int,
    ) -> None:
        """Emit progress at each ``step`` multiple crossed by this window.

        Reproduces the legacy per-frame cadence (``processed % step == 0``)
        without iterating frame-by-frame: fire once for every ``step`` boundary
        that falls within ``(processed_now - window_len, processed_now]``.
        """
        prev = processed_now - window_len
        # First multiple of ``step`` strictly greater than ``prev``.
        first = (prev // step + 1) * step
        m = first
        while m <= processed_now:
            progress_cb(m, range_total)
            m += step

    @staticmethod
    def _drain_queue(q: queue.Queue) -> None:
        """Empty a queue without blocking so a full-blocked producer unblocks."""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

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
        if depth < 1:
            raise ValueError(f"Pipeline.for_test depth must be >= 1, got {depth}")
        pipe.stages = None
        pipe.runtime = None
        pipe.cache_writer = None
        pipe.depth = depth
        pipe.queue_bound = max(1, depth - 1)
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
