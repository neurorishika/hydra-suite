"""Region-source abstraction for the OBB stage (phase C)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
        )

        model = models.direct_model
        conf_floor = config.direct.confidence_floor if config.direct else 1e-3
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

    def merge_plan(self, frame_idx: int) -> Any:
        return self._plan


class Stage1Proposals(RegionSource):
    """Stage-1 detector proposals: one region per stage-1 crop (sequential mode).

    Mirrors ``_run_sequential``'s stage-1-predict + crop-building exactly
    (same kwargs, same ``build_crops``/``resize_crops_for_stage2`` calls) so
    region images and offset/scale affines are byte-identical to what
    ``_run_sequential`` fed to stage-2. ``force_numpy=True`` matches A.5's
    always-numpy stage-2 extraction (spec S5.2): a pixel-exact crop yields a
    translate-only affine that would otherwise slip into the (unproven for
    sequential) raw branch on the gpu-native tier.
    """

    merge_policy = "plain"
    device_residency = "cpu_crop_boundary"
    force_numpy = True

    def plan(
        self, frames, models, config, runtime, roi_mask=None
    ) -> list[list[Region]]:
        from .obb import build_crops, resize_crops_for_stage2

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
        return config.sequential.stage2_task

    def seg_source(self, config) -> Any:
        return config.sequential

    def execute(self, per_frame_regions, models, config, runtime) -> list[list[Any]]:
        """Verbatim stage-2 predict loop of the retired ``obb._run_sequential``."""
        seq = config.sequential
        out: list[list[Any]] = []
        for regions in per_frame_regions:
            if not regions:
                out.append([])
                continue
            crops = [r.image for r in regions]
            batch_size = seq.stage2_batch_size or len(crops)
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
                )
                frame_results.extend(s2)
            out.append(frame_results)
        return out


def select_region_source(config) -> RegionSource:
    """Pick the ``RegionSource`` planner implied by an ``OBBConfig``.

    ``direct`` + sliced -> ``Grid``; ``direct`` (non-sliced) -> ``WholeFrame``;
    ``sequential`` -> ``Stage1Proposals``. (A ``SlicedStage1Proposals`` variant
    is a later task, not selected here.)
    """
    if config.mode == "direct":
        slice_cfg = getattr(config.direct, "slice", None) if config.direct else None
        if slice_cfg is not None and slice_cfg.enabled:
            return Grid()
        return WholeFrame()
    return Stage1Proposals()
