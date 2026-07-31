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

    ``merge_policy`` tells the (future) executor whether cross-region
    detections need overlap-band NMS-style dedup (``"overlap_band_nms"``,
    tiled sources whose regions can double-detect near shared borders) or are
    already disjoint (``"plain"``, whole-frame / stage-1-proposal crops).

    ``device_residency`` tells the (future) executor whether this source's
    regions can stay on-device end to end (``"on_device_capable"``) or
    require a CPU crop boundary (``"cpu_crop_boundary"``, e.g. stage-1
    proposals, whose crop geometry is data-dependent per detection and is
    built with numpy/cv2 today).
    """

    merge_policy: str = "plain"
    device_residency: str = "on_device_capable"

    def plan(self, frames, models, config, runtime) -> list[list[Region]]:
        """Return, per frame, the list of ``Region``s to predict on.

        Planning only: no model inference is performed by ``WholeFrame``/
        ``Grid``; ``Stage1Proposals`` runs the cheap stage-1 detector to
        derive crop geometry (mirroring ``_run_sequential`` exactly) but
        does not run stage-2 OBB prediction.
        """
        raise NotImplementedError


class WholeFrame(RegionSource):
    """One region per frame: the frame itself, identity affine."""

    merge_policy = "plain"
    device_residency = "on_device_capable"

    def plan(self, frames, models, config, runtime) -> list[list[Region]]:
        return [
            [Region(image=frame, affine=Affine.IDENTITY, frame_idx=fi)]
            for fi, frame in enumerate(frames)
        ]


class Grid(RegionSource):
    """SAHI-style tile grid: one region per planned tile (+ optional full frame).

    Reuses ``slicing.plan_slices`` for tile geometry and
    ``slicing._build_tile_jobs`` for the actual tile cropping, so region
    images are byte-identical to what ``run_direct_sliced`` predicts on
    today. This planner does not thread ``roi_mask`` (not part of the
    ``RegionSource.plan`` signature) -- tile gating stays a
    ``run_direct_sliced``-only optimization for now.
    """

    merge_policy = "overlap_band_nms"
    device_residency = "on_device_capable"

    def plan(self, frames, models, config, runtime) -> list[list[Region]]:
        if not frames:
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
            roi_mask=None,
            ref_object_px=slice_cfg.reference_body_px,
        )

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


class Stage1Proposals(RegionSource):
    """Stage-1 detector proposals: one region per stage-1 crop (sequential mode).

    Mirrors ``_run_sequential``'s stage-1-predict + crop-building exactly
    (same kwargs, same ``build_crops``/``resize_crops_for_stage2`` calls) so
    region images and offset/scale affines are byte-identical to what
    ``_run_sequential`` feeds to stage-2 today. Only plans regions -- does
    not run stage-2 prediction.
    """

    merge_policy = "plain"
    device_residency = "cpu_crop_boundary"

    def plan(self, frames, models, config, runtime) -> list[list[Region]]:
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
