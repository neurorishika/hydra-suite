"""Region-source abstraction for the OBB stage (phase C)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch

from ..cancellation import InferenceCancelled


@dataclass(frozen=True)
class Affine:
    """Maps region-local pixel coords -> frame coords: p_frame = p_region * scale + offset."""

    offset: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)

    @property
    def is_translate_only(self) -> bool:
        return self.scale == (1.0, 1.0)


Affine.IDENTITY = Affine()


@dataclass
class Region:
    """A sub-image to run OBB prediction on, plus the mapping back to frame coords.

    ``image`` is EXACTLY what today's pipelines feed to ``model.predict`` for
    this region (the raw frame, a tile crop, or a resized stage-2 crop) --
    planners must not alter crop/resize behavior, only describe it.
    """

    image: Any
    affine: Affine
    frame_idx: int


class RegionSource:
    """Base for OBB region planners.

    ``merge_policy`` tells ``merge_per_frame`` whether cross-region
    detections need overlap-band NMS-style dedup (``"overlap_band_nms"``,
    tiled sources whose regions can double-detect near shared borders) or are
    already disjoint (``"plain"``, whole-frame / stage-1-proposal crops).

    ``device_residency`` tells the executor whether this source's regions can
    stay on-device end to end (``"on_device_capable"``) or require a CPU crop
    boundary (``"cpu_crop_boundary"``, e.g. stage-1 proposals, whose crop
    geometry is data-dependent per detection and is built with numpy/cv2
    today).

    ``force_numpy`` is threaded straight into ``extract_with_transform`` --
    the spec S5.2 opt-in knob. Sources whose ``device_residency`` is
    ``"cpu_crop_boundary"`` set it ``True`` so stage-2 extraction never slips
    into the (unproven, for sequential) raw branch on the gpu-native tier,
    independent of the affine's shape.
    """

    merge_policy: str = "plain"
    device_residency: str = "on_device_capable"
    force_numpy: bool = False

    def plan(
        self, frames, models, config, runtime, roi_mask=None
    ) -> list[list[Region]]:
        """Return, per frame, the list of ``Region``s to predict on.

        Planning only: no model inference is performed by ``WholeFrame``/
        ``Grid``; ``Stage1Proposals`` runs the cheap stage-1 detector to
        derive crop geometry (mirroring ``_run_sequential`` exactly) but
        does not run stage-2 OBB prediction. ``roi_mask`` (frame-space) is
        only consumed by ``Grid`` (tile gating); other sources ignore it.
        """
        raise NotImplementedError

    def execute(self, per_frame_regions, models, config, runtime) -> list[list[Any]]:
        """Run this source's model-prediction routine over planned regions.

        Returns, per frame, one raw ultralytics-style result per region (same
        order as ``per_frame_regions[frame_idx]``). This is the mode's exact
        predict routine, moved verbatim from the retired standalone
        orchestrator (``_run_direct`` / ``run_direct_sliced`` /
        ``_run_sequential``'s stage-2 loop) -- the one piece of genuinely
        per-mode logic ``RegionSource`` does not unify.
        """
        raise NotImplementedError

    def iter_region_results(
        self, frames, models, config, runtime, roi_mask=None, should_stop=None
    ) -> Iterator[list[tuple[int, Region, Any]]]:
        """Yield prediction results in bounded chunks.

        Whole-frame sources are already bounded by the admitted frame window,
        so the compatibility implementation may use their existing plan and
        execute methods. Tile and proposal sources override this method to
        avoid materializing every region image or vendor result.
        """

        if should_stop is not None and should_stop():
            raise InferenceCancelled("inference cancelled before region prediction")
        planned = self.plan(frames, models, config, runtime, roi_mask=roi_mask)
        predicted = self.execute(planned, models, config, runtime)
        if should_stop is not None and should_stop():
            raise InferenceCancelled("inference cancelled during region prediction")
        for frame_idx, (regions, results) in enumerate(zip(planned, predicted)):
            if regions:
                yield [
                    (frame_idx, region, result)
                    for region, result in zip(regions, results)
                ]

    def task(self, config) -> str:
        """The OBB task ('obb'/'detect'/'segment') this source's regions predict."""
        raise NotImplementedError

    def seg_source(self, config) -> Any:
        """Config object supplying seg_* params to ``extract_with_transform``.

        ``None`` lets ``extract_with_transform`` default to ``config.direct``
        (correct for ``WholeFrame``/``Grid``). ``Stage1Proposals`` overrides
        to hand back ``config.sequential`` (its stage-2 seg_* fields).
        """
        return None

    def merge_plan(self, frame_idx: int) -> Any:
        """The geometry ``merge_per_frame`` needs for this frame's merge.

        ``None`` for ``"plain"`` merge policy (no geometry needed). ``Grid``
        overrides to return its memoized ``SlicePlan`` (same for every frame
        in a batch -- tile geometry only depends on frame size).
        """
        return None


class WholeFrame(RegionSource):
    """One region per frame: the frame itself, identity affine."""

    merge_policy = "plain"
    device_residency = "on_device_capable"
    force_numpy = False

    def plan(
        self, frames, models, config, runtime, roi_mask=None
    ) -> list[list[Region]]:
        return [
            [Region(image=frame, affine=Affine.IDENTITY, frame_idx=fi)]
            for fi, frame in enumerate(frames)
        ]

    def task(self, config) -> str:
        return config.direct.model_task if config.direct else "obb"

    def execute(self, per_frame_regions, models, config, runtime) -> list[list[Any]]:
        """Verbatim predict section of the retired ``obb._run_direct``."""
        from .obb import (
            DirectExecutorAdapter,
            _frames_are_cuda_tensors,
            _gpu_letterbox_batch,
            _invert_letterbox_on_result,
            _resolve_imgsz,
            effective_raw_detection_cap,
        )

        model = models.direct_model
        conf_floor = config.direct.confidence_floor if config.direct else 1e-3
        candidate_cap = effective_raw_detection_cap(config)
        frames = [regions[0].image for regions in per_frame_regions]

        # See obb._run_direct's original dispatch comment (preserved there
        # verbatim) for why the CUDA-tensor-frames branch is skipped for
        # DirectExecutorAdapter (gpu_fast): it does its own letterbox and a
        # manual pre-batch here would double-preprocess and corrupt the
        # shape fed to TensorRT.
        if _frames_are_cuda_tensors(frames) and not isinstance(
            model, DirectExecutorAdapter
        ):
            imgsz = _resolve_imgsz(model)
            batched, lb_params = _gpu_letterbox_batch(frames, imgsz)
            results = model.predict(
                batched,
                conf=conf_floor,
                iou=1.0,
                classes=config.target_classes or None,
                verbose=False,
                device=runtime.device,
                max_det=candidate_cap,
            )
            for frame, result, (r, pad_left, pad_top) in zip(
                frames, results, lb_params
            ):
                _invert_letterbox_on_result(
                    result,
                    r,
                    pad_left,
                    pad_top,
                    orig_shape=(int(frame.shape[0]), int(frame.shape[1])),
                )
        else:
            results = model.predict(
                frames,
                conf=conf_floor,
                iou=1.0,
                classes=config.target_classes or None,
                verbose=False,
                device=runtime.device,
                max_det=candidate_cap,
            )
        return [[r] for r in results]


class Grid(RegionSource):
    """SAHI-style tile grid: one region per planned tile (+ optional full frame).

    Reuses ``slicing.plan_slices`` for tile geometry and
    ``slicing._build_tile_jobs`` for the actual tile cropping, so region
    images are byte-identical to what ``run_direct_sliced`` predicted on
    before this task. ``roi_mask`` is threaded straight into ``plan_slices``,
    matching ``run_direct_sliced``'s ROI tile-gating exactly.
    """

    merge_policy = "overlap_band_nms"
    device_residency = "on_device_capable"
    force_numpy = False

    def plan(
        self, frames, models, config, runtime, roi_mask=None
    ) -> list[list[Region]]:
        if not frames:
            self._plan = None
            return []

        from .obb import _frames_are_cuda_tensors, _resolve_imgsz
        from .slicing import _build_tile_jobs, plan_slices

        slice_cfg = config.direct.slice
        model = models.direct_model
        imgsz = _resolve_imgsz(model)
        device_frames = _frames_are_cuda_tensors(frames)

        first = frames[0]
        frame_hw = (int(first.shape[0]), int(first.shape[1]))
        plan = plan_slices(
            frame_hw,
            slice_cfg,
            imgsz,
            roi_mask=roi_mask,
            ref_object_px=slice_cfg.reference_body_px,
        )
        # Stashed for `execute` (chunk sizing) and `merge_plan` (tile
        # geometry for overlap-band NMS) -- same plan for every frame in this
        # batch (memoized on the first frame's shape), so one attribute
        # suffices for the whole call.
        self._plan = plan

        jobs, images = _build_tile_jobs(frames, plan, device_frames)
        # `_build_tile_jobs` flattens frames x (tiles [+ optional full frame])
        # in that fixed per-frame order/count -- same plan for every frame
        # (memoized on the first frame's shape) -- so we can regroup by
        # simple contiguous slicing instead of re-deriving tile geometry.
        n_tiles = len(plan.tiles)
        per_frame_count = n_tiles + (1 if plan.full_frame else 0)

        per_frame: list[list[Region]] = []
        for fi in range(len(frames)):
            start = fi * per_frame_count
            frame_jobs = jobs[start : start + per_frame_count]
            frame_images = images[start : start + per_frame_count]
            regions: list[Region] = []
            for idx, ((_, x0, y0), image) in enumerate(zip(frame_jobs, frame_images)):
                if idx < n_tiles:
                    affine = Affine(offset=(float(x0), float(y0)), scale=(1.0, 1.0))
                else:
                    affine = Affine.IDENTITY
                regions.append(Region(image=image, affine=affine, frame_idx=fi))
            per_frame.append(regions)
        return per_frame

    def task(self, config) -> str:
        return config.direct.model_task if config.direct else "obb"

    def execute(self, per_frame_regions, models, config, runtime) -> list[list[Any]]:
        """Verbatim tile-predict section of the retired ``slicing.run_direct_sliced``."""
        from .obb import DirectExecutorAdapter, _frames_are_cuda_tensors, _resolve_imgsz
        from .slicing import MAX_TILE_CHUNK, _predict_tiles

        model = models.direct_model
        imgsz = _resolve_imgsz(model)
        images = [r.image for regions in per_frame_regions for r in regions]
        device_frames = _frames_are_cuda_tensors(images)
        # DirectExecutorAdapter accepts (and internally letterboxes) a raw
        # list of CUDA frames; pre-batching it double-preprocesses. See
        # slicing.run_direct_sliced's original "TWO ORTHOGONAL DISPATCH
        # DECISIONS" comment (finding C1).
        letterbox = device_frames and not isinstance(model, DirectExecutorAdapter)

        plan = self._plan
        chunk_size = max(1, min(plan.jobs_per_frame, MAX_TILE_CHUNK)) if plan else 1
        results = _predict_tiles(
            images,
            model,
            config,
            runtime,
            imgsz,
            letterbox=letterbox,
            chunk_size=chunk_size,
        )

        out: list[list[Any]] = []
        idx = 0
        for regions in per_frame_regions:
            n = len(regions)
            out.append(results[idx : idx + n])
            idx += n
        return out

    def iter_region_results(
        self, frames, models, config, runtime, roi_mask=None, should_stop=None
    ) -> Iterator[list[tuple[int, Region, Any]]]:
        """Predict tiles without retaining a frame-window's tile pixels/results."""

        if not frames:
            self._plan = None
            return

        from .obb import (
            DirectExecutorAdapter,
            _frames_are_cuda_tensors,
            _resolve_imgsz,
            effective_raw_detection_cap,
        )
        from .slicing import (
            MAX_TILE_CHUNK,
            _predict_tiles,
            admitted_tile_chunk_size,
            iter_tile_job_chunks,
            plan_slices,
        )

        slice_cfg = config.direct.slice
        model = models.direct_model
        imgsz = _resolve_imgsz(model)
        device_frames = _frames_are_cuda_tensors(frames)
        first = frames[0]
        plan = plan_slices(
            (int(first.shape[0]), int(first.shape[1])),
            slice_cfg,
            imgsz,
            roi_mask=roi_mask,
            ref_object_px=slice_cfg.reference_body_px,
        )
        self._plan = plan
        requested = min(
            plan.jobs_per_frame,
            int(getattr(slice_cfg, "tile_batch_size", MAX_TILE_CHUNK)),
        )
        byte_budget = int(
            getattr(slice_cfg, "tile_memory_budget_bytes", 256 * 1024 * 1024)
        )
        chunk_size = admitted_tile_chunk_size(
            plan,
            imgsz=imgsz,
            device_tiles=device_frames,
            requested=requested,
            byte_budget=byte_budget,
            task=self.task(config),
            max_detections=effective_raw_detection_cap(config),
        )
        letterbox = device_frames and not isinstance(model, DirectExecutorAdapter)

        for chunk in iter_tile_job_chunks(
            frames,
            plan,
            device_tiles=device_frames,
            chunk_size=chunk_size,
        ):
            if should_stop is not None and should_stop():
                raise InferenceCancelled("inference cancelled before tile prediction")
            images = [image for _, image in chunk]
            results = _predict_tiles(
                images,
                model,
                config,
                runtime,
                imgsz,
                letterbox=letterbox,
                chunk_size=chunk_size,
            )
            if should_stop is not None and should_stop():
                raise InferenceCancelled("inference cancelled during tile prediction")
            entries: list[tuple[int, Region, Any]] = []
            for (job, image), result in zip(chunk, results):
                entries.append(
                    (
                        job.frame_idx,
                        Region(
                            image=image,
                            affine=Affine(offset=job.offset),
                            frame_idx=job.frame_idx,
                        ),
                        result,
                    )
                )
            yield entries
            # Both sides of a generator suspension retain the yielded object.
            # Drop our reference before the next crop/model call so only one
            # chunk is live in streaming consumers. Do not mutate ``entries``:
            # callers are allowed to retain a yielded batch explicitly.
            chunk.clear()
            del entries, results, images, chunk, job, image, result

    def merge_plan(self, frame_idx: int) -> Any:
        return self._plan


class Stage1Proposals(RegionSource):
    """Stage-1 detector proposals: one region per stage-1 crop (sequential mode).

    Mirrors ``_run_sequential``'s stage-1 predict and crop geometry exactly,
    but creates/resizes only the next admitted stage-2 batch. Region images and
    offset/scale affines stay byte-identical to what ``_run_sequential`` fed to
    stage-2. ``force_numpy=True`` matches A.5's
    always-numpy stage-2 extraction (spec S5.2): a pixel-exact crop yields a
    translate-only affine that would otherwise slip into the (unproven for
    sequential) raw branch on the gpu-native tier.
    """

    merge_policy = "plain"
    device_residency = "cpu_crop_boundary"
    force_numpy = True

    DEFAULT_STAGE2_BATCH_SIZE = 16
    MAX_STAGE2_BATCH_SIZE = 128

    def plan(
        self, frames, models, config, runtime, roi_mask=None
    ) -> list[list[Region]]:
        from .obb import (
            build_crops,
            effective_raw_detection_cap,
            resize_crops_for_stage2,
        )

        seq = config.sequential
        stage1_kwargs: dict[str, Any] = {}
        if seq.detect_image_size > 0:
            stage1_kwargs["imgsz"] = seq.detect_image_size
        stage1 = models.detect_model.predict(
            frames,
            conf=seq.detect_confidence_threshold,
            iou=1.0,
            classes=config.target_classes or None,
            verbose=False,
            device=runtime.device,
            max_det=effective_raw_detection_cap(config),
            **stage1_kwargs,
        )

        per_frame: list[list[Region]] = []
        for frame_idx, (frame, s1) in enumerate(zip(frames, stage1)):
            boxes = s1.boxes
            if boxes is None or len(boxes) == 0:
                per_frame.append([])
                continue
            crops, offsets = build_crops(frame, boxes, seq, runtime)
            if not crops:
                per_frame.append([])
                continue
            orig_sizes = [(c.shape[1], c.shape[0]) for c in crops]  # (w, h)
            crops = resize_crops_for_stage2(crops, seq.stage2_image_size)
            regions: list[Region] = []
            for j, resized in enumerate(crops):
                orig_w, orig_h = orig_sizes[j]
                scale = (
                    (orig_w / seq.stage2_image_size, orig_h / seq.stage2_image_size)
                    if seq.stage2_image_size > 0
                    else (1.0, 1.0)
                )
                regions.append(
                    Region(
                        image=resized,
                        affine=Affine(offset=offsets[j], scale=scale),
                        frame_idx=frame_idx,
                    )
                )
            per_frame.append(regions)
        return per_frame

    def task(self, config) -> str:
        return str(getattr(config.sequential, "stage2_task", "obb"))

    def seg_source(self, config) -> Any:
        return config.sequential

    def execute(self, per_frame_regions, models, config, runtime) -> list[list[Any]]:
        """Verbatim stage-2 predict loop of the retired ``obb._run_sequential``."""
        from .obb import effective_raw_detection_cap

        seq = config.sequential
        out: list[list[Any]] = []
        for regions in per_frame_regions:
            if not regions:
                out.append([])
                continue
            crops = [r.image for r in regions]
            batch_size = self._stage2_batch_size(seq)
            frame_results: list[Any] = []
            for i in range(0, len(crops), batch_size):
                batch = crops[i : i + batch_size]
                s2 = models.obb_model.predict(
                    batch,
                    conf=seq.obb_confidence_threshold,
                    iou=1.0,
                    verbose=False,
                    device=runtime.device,
                    imgsz=seq.stage2_image_size,
                    max_det=effective_raw_detection_cap(config),
                )
                frame_results.extend(s2)
            out.append(frame_results)
        return out

    @classmethod
    def _stage2_batch_size(cls, seq: Any) -> int:
        requested = int(getattr(seq, "stage2_batch_size", 0) or 0)
        if requested <= 0:
            requested = cls.DEFAULT_STAGE2_BATCH_SIZE
        return max(1, min(requested, cls.MAX_STAGE2_BATCH_SIZE))

    def iter_region_results(
        self, frames, models, config, runtime, roi_mask=None, should_stop=None
    ) -> Iterator[list[tuple[int, Region, Any]]]:
        """Run stage 2 over crop chunks created just in time."""

        from .obb import MAX_RAW_CANDIDATES_PER_FRAME, effective_raw_detection_cap
        from .slicing import admitted_prediction_chunk_size

        seq = config.sequential
        stage1_kwargs: dict[str, Any] = {}
        if seq.detect_image_size > 0:
            stage1_kwargs["imgsz"] = seq.detect_image_size
        stage1 = models.detect_model.predict(
            frames,
            conf=seq.detect_confidence_threshold,
            iou=1.0,
            classes=config.target_classes or None,
            verbose=False,
            device=runtime.device,
            max_det=effective_raw_detection_cap(config),
            **stage1_kwargs,
        )
        if should_stop is not None and should_stop():
            raise InferenceCancelled("inference cancelled during stage-1 prediction")
        candidate_cap = effective_raw_detection_cap(config)
        for frame_idx, (frame, stage1_result) in enumerate(zip(frames, stage1)):
            boxes = stage1_result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            if len(boxes) > MAX_RAW_CANDIDATES_PER_FRAME:
                raise ValueError(
                    "Sequential stage-1 proposals are not resource-admissible: "
                    f"frame {frame_idx} produced {len(boxes)} candidates, above "
                    f"the hard {MAX_RAW_CANDIDATES_PER_FRAME}-candidate ceiling; "
                    "increase the confidence threshold"
                )
            stage2_side = int(seq.stage2_image_size or max(frame.shape[:2]))
            batch_size = admitted_prediction_chunk_size(
                imgsz=stage2_side,
                task=self.task(config),
                max_detections=candidate_cap,
                requested=self._stage2_batch_size(seq),
                source_bytes=stage2_side * stage2_side * 3,
                description="Sequential stage-2 crop batch",
            )
            yield from self._iter_stage2_for_boxes(
                frame_idx,
                frame,
                boxes,
                models,
                config,
                runtime,
                candidate_cap,
                batch_size,
                should_stop,
            )

    def _iter_stage2_for_boxes(
        self,
        frame_idx: int,
        frame: Any,
        boxes: Any,
        models: Any,
        config: Any,
        runtime: Any,
        candidate_cap: int,
        batch_size: int,
        should_stop: Any = None,
    ) -> Iterator[list[tuple[int, Region, Any]]]:
        from .obb import iter_crops

        seq = config.sequential
        pending: list[Region] = []
        for crop, offset in iter_crops(
            frame,
            boxes,
            seq,
            runtime,
            max_candidates=candidate_cap,
        ):
            orig_h, orig_w = crop.shape[:2]
            if seq.stage2_image_size > 0 and (
                orig_h != seq.stage2_image_size or orig_w != seq.stage2_image_size
            ):
                import cv2

                resized = cv2.resize(
                    crop,
                    (seq.stage2_image_size, seq.stage2_image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                resized = crop
            scale = (
                (
                    orig_w / seq.stage2_image_size,
                    orig_h / seq.stage2_image_size,
                )
                if seq.stage2_image_size > 0
                else (1.0, 1.0)
            )
            pending.append(
                Region(
                    image=resized,
                    affine=Affine(offset=offset, scale=scale),
                    frame_idx=frame_idx,
                )
            )
            if len(pending) == batch_size:
                if should_stop is not None and should_stop():
                    raise InferenceCancelled(
                        "inference cancelled before stage-2 prediction"
                    )
                entries = self._predict_stage2_chunk(
                    frame_idx, pending, models, seq, runtime, candidate_cap
                )
                if should_stop is not None and should_stop():
                    raise InferenceCancelled(
                        "inference cancelled during stage-2 prediction"
                    )
                yield entries
                pending.clear()
                del entries, crop, resized
                pending = []
        if pending:
            if should_stop is not None and should_stop():
                raise InferenceCancelled(
                    "inference cancelled before stage-2 prediction"
                )
            entries = self._predict_stage2_chunk(
                frame_idx, pending, models, seq, runtime, candidate_cap
            )
            if should_stop is not None and should_stop():
                raise InferenceCancelled(
                    "inference cancelled during stage-2 prediction"
                )
            yield entries
            pending.clear()
            del entries

    @staticmethod
    def _predict_stage2_chunk(
        frame_idx: int,
        regions: list[Region],
        models,
        seq,
        runtime,
        candidate_cap: int,
    ) -> list[tuple[int, Region, Any]]:
        results = models.obb_model.predict(
            [region.image for region in regions],
            conf=seq.obb_confidence_threshold,
            iou=1.0,
            verbose=False,
            device=runtime.device,
            imgsz=seq.stage2_image_size,
            max_det=candidate_cap,
        )
        return [(frame_idx, region, result) for region, result in zip(regions, results)]


class _FrameSpaceBoxes:
    """Minimal ``build_crops``-compatible box container.

    ``build_crops`` (obb.py) reads exactly one attribute off its ``boxes``
    argument: ``boxes.xyxy.cpu().numpy()``. Real stage-1 predict results hand
    it an ultralytics ``Boxes`` object; ``SlicedStage1Proposals`` instead has
    already-merged frame-space boxes (a plain ``(N, 4)`` array), so this
    wraps them in the same minimal shape rather than fabricating a full
    ultralytics ``Boxes``.
    """

    def __init__(self, xyxy: torch.Tensor) -> None:
        self.xyxy = xyxy

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


def _box_overlap(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """IoU or IoS of two axis-aligned ``(x1, y1, x2, y2)`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    if area_a <= 1e-9 or area_b <= 1e-9:
        return 0.0
    denom = min(area_a, area_b) if metric == "ios" else area_a + area_b - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def _merge_axis_aligned_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    policy: str,
    metric: str,
    threshold: float,
) -> np.ndarray:
    """Dedup cross-tile stage-1 boxes in frame space (axis-aligned, greedy).

    Same confidence-descending greedy-group structure as
    ``merge._merge_obb_detections`` (oriented boxes, cv2 hull IoU/IoS), applied
    to plain axis-aligned stage-1 detect boxes instead: ``policy="nms"`` keeps
    the highest-confidence member of each overlapping group and drops the
    rest; ``"nmm"``/``"greedy_nmm"`` union each group into its enclosing
    axis-aligned box. A tile-boundary-straddling object detected in two
    overlapping tiles collapses into ONE frame-space box either way, rather
    than being fed twice into ``build_crops``.
    """
    n = boxes.shape[0]
    if n <= 1:
        return boxes
    order = np.argsort(-scores)
    consumed = np.zeros(n, dtype=bool)
    out_rows: list[np.ndarray] = []
    for i in order:
        if consumed[i]:
            continue
        group = [int(i)]
        for j in order:
            if j == i or consumed[j]:
                continue
            if _box_overlap(boxes[i], boxes[j], metric) >= threshold:
                consumed[j] = True
                group.append(int(j))
        consumed[i] = True
        if policy == "nms" or len(group) == 1:
            out_rows.append(boxes[i])
        else:  # nmm / greedy_nmm -> union into the enclosing box
            g = boxes[group]
            out_rows.append(
                np.array([g[:, 0].min(), g[:, 1].min(), g[:, 2].max(), g[:, 3].max()])
            )
    return np.stack(out_rows, axis=0) if out_rows else np.zeros((0, 4))


class SlicedStage1Proposals(Stage1Proposals):
    """Stage1Proposals, but stage-1 detection runs on a tiled grid (Task 11).

    New capability (off by default via ``config.sequential.stage1_slice.
    enabled``): tiles the frame for the STAGE-1 detect pass only (reusing
    ``Grid``'s tiling geometry), remaps each tile's stage-1 boxes into frame
    space, dedups cross-tile duplicates with ``_merge_axis_aligned_boxes``,
    then hands the merged frame-space boxes to the SAME
    ``build_crops``/``resize_crops_for_stage2``/stage-2-region construction
    ``Stage1Proposals.plan`` uses. ``task``/``seg_source``/``execute``/
    ``force_numpy``/``merge_policy``/``device_residency`` are all inherited
    unchanged from ``Stage1Proposals`` -- only stage-1 detection geometry
    differs.
    """

    def plan(
        self, frames, models, config, runtime, roi_mask=None
    ) -> list[list[Region]]:
        if not frames:
            return []

        from .obb import (
            _frames_are_cuda_tensors,
            _resolve_imgsz,
            build_crops,
            effective_raw_detection_cap,
            resize_crops_for_stage2,
        )
        from .slicing import MAX_TILE_CHUNK, _build_tile_jobs, plan_slices

        seq = config.sequential
        slice_cfg = seq.stage1_slice
        model = models.detect_model
        imgsz = _resolve_imgsz(model)
        device_frames = _frames_are_cuda_tensors(frames)

        first = frames[0]
        frame_hw = (int(first.shape[0]), int(first.shape[1]))
        plan = plan_slices(
            frame_hw,
            slice_cfg,
            imgsz,
            roi_mask=roi_mask,
            ref_object_px=slice_cfg.reference_body_px,
        )
        jobs, images = _build_tile_jobs(frames, plan, device_frames)
        n_tiles = len(plan.tiles)
        per_frame_count = n_tiles + (1 if plan.full_frame else 0)

        stage1_kwargs: dict[str, Any] = {}
        if seq.detect_image_size > 0:
            stage1_kwargs["imgsz"] = seq.detect_image_size
        # Same stage-1 predict kwargs as Stage1Proposals.plan, applied to the
        # flattened tile-image list instead of the whole-frame list -- but
        # CHUNKED (mirrors Grid.execute / _predict_tiles, finding I2): issuing
        # `frames x tiles_per_frame` tile images as ONE predict call multiplies
        # the peak activation memory the user configured by the tile count with
        # no cap. `chunk_size` is bounded by one frame's worth of tiles (and
        # `MAX_TILE_CHUNK`), the same bound Grid uses; `predict` is
        # order-preserving and stateless across the list, so chunking changes
        # peak memory only, never the assembled results.
        chunk_size = max(1, min(per_frame_count, MAX_TILE_CHUNK))
        tile_results: list[Any] = []
        for start in range(0, len(images), chunk_size):
            tile_results.extend(
                model.predict(
                    images[start : start + chunk_size],
                    conf=seq.detect_confidence_threshold,
                    iou=1.0,
                    classes=config.target_classes or None,
                    verbose=False,
                    device=runtime.device,
                    max_det=effective_raw_detection_cap(config),
                    **stage1_kwargs,
                )
            )

        per_frame: list[list[Region]] = []
        for frame_idx, frame in enumerate(frames):
            start = frame_idx * per_frame_count
            frame_jobs = jobs[start : start + per_frame_count]
            frame_results = tile_results[start : start + per_frame_count]

            boxes_parts: list[np.ndarray] = []
            scores_parts: list[np.ndarray] = []
            for (_, x0, y0), res in zip(frame_jobs, frame_results):
                b = getattr(res, "boxes", None)
                if b is None or len(b) == 0:
                    continue
                xyxy = np.array(b.xyxy.detach().cpu().numpy(), dtype=np.float64)
                xyxy[:, [0, 2]] += x0
                xyxy[:, [1, 3]] += y0
                conf = getattr(b, "conf", None)
                scores = (
                    np.array(conf.detach().cpu().numpy(), dtype=np.float64)
                    if conf is not None
                    else np.ones(xyxy.shape[0], dtype=np.float64)
                )
                boxes_parts.append(xyxy)
                scores_parts.append(scores)

            if not boxes_parts:
                per_frame.append([])
                continue

            all_boxes = np.concatenate(boxes_parts, axis=0)
            all_scores = np.concatenate(scores_parts, axis=0)
            merged = _merge_axis_aligned_boxes(
                all_boxes,
                all_scores,
                policy=slice_cfg.merge_policy,
                metric=slice_cfg.merge_metric,
                threshold=slice_cfg.merge_threshold,
            )
            if merged.shape[0] == 0:
                per_frame.append([])
                continue

            frame_boxes = _FrameSpaceBoxes(torch.as_tensor(merged, dtype=torch.float32))
            crops, offsets = build_crops(frame, frame_boxes, seq, runtime)
            if not crops:
                per_frame.append([])
                continue
            orig_sizes = [(c.shape[1], c.shape[0]) for c in crops]  # (w, h)
            crops = resize_crops_for_stage2(crops, seq.stage2_image_size)
            regions: list[Region] = []
            for j, resized in enumerate(crops):
                orig_w, orig_h = orig_sizes[j]
                scale = (
                    (orig_w / seq.stage2_image_size, orig_h / seq.stage2_image_size)
                    if seq.stage2_image_size > 0
                    else (1.0, 1.0)
                )
                regions.append(
                    Region(
                        image=resized,
                        affine=Affine(offset=offsets[j], scale=scale),
                        frame_idx=frame_idx,
                    )
                )
            per_frame.append(regions)
        return per_frame

    def iter_region_results(
        self, frames, models, config, runtime, roi_mask=None, should_stop=None
    ) -> Iterator[list[tuple[int, Region, Any]]]:
        """Stream stage-1 tiles, retain bounded boxes, then stream stage 2."""

        if not frames:
            return

        from .obb import (
            MAX_RAW_CANDIDATES_PER_FRAME,
            _frames_are_cuda_tensors,
            _resolve_imgsz,
            effective_raw_detection_cap,
        )
        from .slicing import (
            MAX_TILE_CHUNK,
            admitted_tile_chunk_size,
            iter_tile_job_chunks,
            plan_slices,
        )

        seq = config.sequential
        slice_cfg = seq.stage1_slice
        model = models.detect_model
        imgsz = _resolve_imgsz(model)
        device_frames = _frames_are_cuda_tensors(frames)
        first = frames[0]
        plan = plan_slices(
            (int(first.shape[0]), int(first.shape[1])),
            slice_cfg,
            imgsz,
            roi_mask=roi_mask,
            ref_object_px=slice_cfg.reference_body_px,
        )
        requested = min(
            plan.jobs_per_frame,
            int(getattr(slice_cfg, "tile_batch_size", MAX_TILE_CHUNK)),
        )
        candidate_cap = effective_raw_detection_cap(config)
        chunk_size = admitted_tile_chunk_size(
            plan,
            imgsz=imgsz,
            device_tiles=device_frames,
            requested=requested,
            byte_budget=int(
                getattr(slice_cfg, "tile_memory_budget_bytes", 256 * 1024 * 1024)
            ),
            task="detect",
            max_detections=candidate_cap,
        )
        boxes_by_frame = [np.zeros((0, 4), dtype=np.float64) for _ in frames]
        scores_by_frame = [np.zeros(0, dtype=np.float64) for _ in frames]

        stage1_kwargs: dict[str, Any] = {}
        if seq.detect_image_size > 0:
            stage1_kwargs["imgsz"] = seq.detect_image_size
        for chunk in iter_tile_job_chunks(
            frames,
            plan,
            device_tiles=device_frames,
            chunk_size=chunk_size,
        ):
            if should_stop is not None and should_stop():
                raise InferenceCancelled(
                    "inference cancelled before sliced stage-1 prediction"
                )
            results = model.predict(
                [image for _, image in chunk],
                conf=seq.detect_confidence_threshold,
                iou=1.0,
                classes=config.target_classes or None,
                verbose=False,
                device=runtime.device,
                max_det=candidate_cap,
                **stage1_kwargs,
            )
            if should_stop is not None and should_stop():
                raise InferenceCancelled(
                    "inference cancelled during sliced stage-1 prediction"
                )
            for (job, _image), result in zip(chunk, results):
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                if len(boxes) > MAX_RAW_CANDIDATES_PER_FRAME:
                    raise ValueError(
                        "Sliced sequential stage-1 proposals are not "
                        f"resource-admissible: one tile produced {len(boxes)} "
                        f"candidates, above the hard "
                        f"{MAX_RAW_CANDIDATES_PER_FRAME}-candidate ceiling; "
                        "increase the confidence threshold"
                    )
                xyxy = np.asarray(
                    boxes.xyxy.detach().cpu().numpy(), dtype=np.float64
                ).copy()
                ox, oy = job.offset
                xyxy[:, [0, 2]] += ox
                xyxy[:, [1, 3]] += oy
                conf = getattr(boxes, "conf", None)
                scores = (
                    np.asarray(conf.detach().cpu().numpy(), dtype=np.float64)
                    if conf is not None
                    else np.ones(xyxy.shape[0], dtype=np.float64)
                )
                frame_idx = job.frame_idx
                combined_boxes = np.concatenate(
                    (boxes_by_frame[frame_idx], xyxy), axis=0
                )
                combined_scores = np.concatenate(
                    (scores_by_frame[frame_idx], scores), axis=0
                )
                order = np.argsort(-combined_scores, kind="stable")[:candidate_cap]
                boxes_by_frame[frame_idx] = combined_boxes[order]
                scores_by_frame[frame_idx] = combined_scores[order]

            # Do not retain the final stage-1 tile/result chunk while stage 2
            # begins. The compact box reservoirs above are the only state that
            # crosses this model boundary.
            chunk.clear()
            results.clear()
            del chunk, results, job, _image, result

        for frame_idx, frame in enumerate(frames):
            boxes = boxes_by_frame[frame_idx]
            if boxes.shape[0] == 0:
                continue
            merged = _merge_axis_aligned_boxes(
                boxes,
                scores_by_frame[frame_idx],
                policy=slice_cfg.merge_policy,
                metric=slice_cfg.merge_metric,
                threshold=slice_cfg.merge_threshold,
            )[:candidate_cap]
            if merged.shape[0] == 0:
                continue
            frame_boxes = _FrameSpaceBoxes(torch.as_tensor(merged, dtype=torch.float32))
            from .slicing import admitted_prediction_chunk_size

            stage2_side = int(seq.stage2_image_size or max(frame.shape[:2]))
            batch_size = admitted_prediction_chunk_size(
                imgsz=stage2_side,
                task=self.task(config),
                max_detections=candidate_cap,
                requested=self._stage2_batch_size(seq),
                source_bytes=stage2_side * stage2_side * 3,
                description="Sequential stage-2 crop batch",
            )
            yield from self._iter_stage2_for_boxes(
                frame_idx,
                frame,
                frame_boxes,
                models,
                config,
                runtime,
                candidate_cap,
                batch_size,
                should_stop,
            )


def select_region_source(config) -> RegionSource:
    """Pick the ``RegionSource`` planner implied by an ``OBBConfig``.

    ``direct`` + sliced -> ``Grid``; ``direct`` (non-sliced) -> ``WholeFrame``;
    ``sequential`` + ``stage1_slice.enabled`` -> ``SlicedStage1Proposals``;
    ``sequential`` (otherwise) -> ``Stage1Proposals``.
    """
    if config.mode == "direct":
        slice_cfg = getattr(config.direct, "slice", None) if config.direct else None
        if slice_cfg is not None and slice_cfg.enabled:
            return Grid()
        return WholeFrame()
    sequential = getattr(config, "sequential", None)
    stage1_slice = getattr(sequential, "stage1_slice", None)
    if stage1_slice is not None and stage1_slice.enabled:
        return SlicedStage1Proposals()
    return Stage1Proposals()
