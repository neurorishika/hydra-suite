"""Confidence density map core module for RefineKit.

This module accumulates per-frame confidence density grids from detection
measurements, smooths them temporally, binarizes, and finds 3D connected
components (x, y, t) that represent "low-confidence high-density" regions
where identity swaps are most likely to occur.

A detection with low confidence contributes a strong Gaussian signal;
high-confidence detections contribute near-zero signal. After accumulating
all frames the volume is smoothed along the time axis, globally normalised,
and thresholded to produce a binary mask. scipy.ndimage.label then finds
connected components in that 3D mask.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, label

from hydra_suite.utils.video_encoder import VideoEncoder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hydra_suite.core.tracking.arenas import ArenaLayout

_GAUSSIAN_TRUNCATE = 4.0
_FULL_SMOOTH_MAX_BYTES = 256 * 1024 * 1024
_SMOOTH_CHUNK_FRAMES = 500

# Read-only autotuner replays rebuild candidate-specific density evidence in
# the GUI process.  Unlike ordinary tracking, a replay must not quietly fall
# back to no density regions when that evidence cannot be built: doing so
# would score a different candidate policy.  The worker applies this bound only
# to ``cache_read_only_replay``; normal production density generation remains
# uncapped by this admission guard.
DEFAULT_AUTOTUNE_DENSITY_MAX_BYTES = 512 * 1024 * 1024


class ConfidenceDensityCancelled(RuntimeError):
    """Raised when a density map is cancelled before all evidence is complete.

    Returning a partially accumulated map would be indistinguishable from a
    genuine sparse-confidence result.  A distinct exception makes callers end
    a read-only replay rather than silently score incomplete density evidence.
    """


@dataclass(frozen=True)
class DensityMapMemoryEstimate:
    """Conservative in-process working-set estimate for a density map.

    The estimate covers only arrays allocated by this module after the caller
    has materialized a detection cache.  It deliberately uses the number of
    cached frame *keys*, never their absolute values, so sparse source timelines
    cannot turn a high frame number into a dense allocation.

    ``smoothing_peak_bytes`` includes the retained raw volume, binary output,
    SciPy's float32 smoothing output/workspace, and threshold temporaries.
    ``labeling_peak_bytes`` includes the retained raw/binary volumes, int32
    connected-component labels, and one uint8-per-voxel bookkeeping margin.
    In multi-arena mode those transient phases also include the retained
    aggregate float32/uint8 diagnostic volumes and the per-arena float32 work
    volume.  Chunked smoothing substitutes its largest extended temporal
    chunk for a full float32 volume, but labeling remains full-volume.
    """

    frame_count: int
    grid_h: int
    grid_w: int
    voxel_count: int
    timeline_bytes: int
    accumulation_peak_bytes: int
    smoothing_peak_bytes: int
    labeling_peak_bytes: int
    peak_bytes: int
    uses_chunked_smoothing: bool
    multi_arena: bool


class DensityReplayBudgetExceeded(RuntimeError):
    """A read-only replay cannot safely materialize density evidence."""

    def __init__(
        self, estimate: DensityMapMemoryEstimate, max_working_bytes: int
    ) -> None:
        self.estimate = estimate
        self.max_working_bytes = int(max_working_bytes)
        super().__init__(
            "read-only confidence-density replay was not admitted: estimated "
            f"peak {_format_density_bytes(estimate.peak_bytes)} for "
            f"{estimate.frame_count} cached frame(s) at "
            f"{estimate.grid_w}x{estimate.grid_h} grid resolution exceeds "
            "AUTOTUNE_DENSITY_MAX_BYTES="
            f"{_format_density_bytes(self.max_working_bytes)}"
        )


def _format_density_bytes(value: int) -> str:
    """Return a short deterministic byte count for diagnostics."""

    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} bytes"


def _density_grid_shape(
    frame_h: int, frame_w: int, downsample_factor: int
) -> tuple[int, int, int]:
    """Return the effective factor and ceil-covered density-grid dimensions.

    Grid coordinates are derived by dividing source detection centres by
    ``ds``.  The grid must therefore retain a final partial cell whenever a
    source dimension is not divisible by ``ds``; floor division silently
    clips detections in that source-frame remainder.
    """

    ds = max(1, int(downsample_factor))
    source_h = max(1, int(frame_h))
    source_w = max(1, int(frame_w))
    return ds, (source_h + ds - 1) // ds, (source_w + ds - 1) // ds


def _largest_smoothing_chunk_frames(frame_count: int, temporal_sigma: float) -> int:
    """Return the largest extended temporal slice used by chunked smoothing."""

    if frame_count <= 0:
        return 0
    try:
        sigma = float(temporal_sigma)
    except (TypeError, ValueError):
        # The caller will ultimately surface the invalid smoothing parameter;
        # until then, refuse to under-estimate its potential working set.
        return frame_count
    if not math.isfinite(sigma) or sigma < 0:
        return frame_count
    radius = int(math.ceil(_GAUSSIAN_TRUNCATE * sigma)) + 1
    return min(frame_count, _SMOOTH_CHUNK_FRAMES + 2 * radius)


def estimate_density_map_working_set(
    frame_count: int,
    frame_h: int,
    frame_w: int,
    downsample_factor: int = 8,
    *,
    temporal_sigma: float = 2.0,
    multi_arena: bool = False,
) -> DensityMapMemoryEstimate:
    """Estimate a conservative peak for density-map arrays without allocating.

    ``frame_count`` is the count of cache keys to replay, not ``max(key)``.
    Whole-frame maps retain one float32 raw grid and one uint8 binary grid per
    cache key.  A full smoothing pass additionally needs a float32 SciPy
    output/workspace and threshold temporaries; connected-components need an
    int32 label volume.  Multi-arena maps retain aggregate float32/uint8
    diagnostic volumes plus one float32 work volume while processing each
    arena.  Above :data:`_FULL_SMOOTH_MAX_BYTES`, the implementation smooths
    chunks, and this estimate uses the largest extended chunk; labeling is
    still a full-volume operation.

    The returned bound intentionally excludes the caller-owned detection cache
    and arbitrary Python component metadata.  It is an admission guard, not a
    process-memory containment boundary.
    """

    frames = int(frame_count)
    if frames < 0:
        raise ValueError("frame_count must be non-negative")
    _ds, grid_h, grid_w = _density_grid_shape(frame_h, frame_w, downsample_factor)
    voxels = frames * grid_h * grid_w
    float_bytes = voxels * np.dtype(np.float32).itemsize
    binary_bytes = voxels * np.dtype(np.uint8).itemsize
    label_bytes = voxels * np.dtype(np.int32).itemsize
    # ``sorted(detection_cache.keys())`` holds references and ``frame_indices``
    # owns int64 copies.  Count both even though they are small beside a map.
    timeline_bytes = frames * (np.dtype(np.int64).itemsize * 2)
    uses_chunked_smoothing = float_bytes > _FULL_SMOOTH_MAX_BYTES

    if multi_arena:
        # Aggregate float32 + uint8 diagnostic volumes and one arena's work
        # float32 volume remain resident across every arena iteration.
        retained_bytes = float_bytes * 2 + binary_bytes
        accumulation = timeline_bytes + retained_bytes
        labeling = (
            timeline_bytes + retained_bytes + binary_bytes + label_bytes + binary_bytes
        )
    else:
        retained_bytes = float_bytes
        accumulation = timeline_bytes + retained_bytes
        labeling = (
            timeline_bytes + retained_bytes + binary_bytes + label_bytes + binary_bytes
        )

    if uses_chunked_smoothing:
        chunk_voxels = _largest_smoothing_chunk_frames(frames, temporal_sigma) * (
            grid_h * grid_w
        )
        chunk_float_bytes = chunk_voxels * np.dtype(np.float32).itemsize
        chunk_binary_bytes = chunk_voxels * np.dtype(np.uint8).itemsize
        # Raw + output binary stay resident; each chunk can require float32
        # output plus a same-size SciPy workspace and uint8 comparison/cast
        # temporaries.  In arena mode ``retained_bytes`` already includes the
        # aggregate and work volumes.
        smoothing = (
            timeline_bytes
            + retained_bytes
            + binary_bytes
            + chunk_float_bytes * 2
            + chunk_binary_bytes * 2
        )
    else:
        # Treat SciPy's full output and workspace as co-resident.  The actual
        # phases are shorter-lived, but this avoids a brittle dependency on
        # private scipy allocation details.
        smoothing = (
            timeline_bytes
            + retained_bytes
            + binary_bytes
            + float_bytes * 2
            + binary_bytes * 2
        )

    peak = max(accumulation, smoothing, labeling)
    return DensityMapMemoryEstimate(
        frame_count=frames,
        grid_h=grid_h,
        grid_w=grid_w,
        voxel_count=voxels,
        timeline_bytes=timeline_bytes,
        accumulation_peak_bytes=accumulation,
        smoothing_peak_bytes=smoothing,
        labeling_peak_bytes=labeling,
        peak_bytes=peak,
        uses_chunked_smoothing=uses_chunked_smoothing,
        multi_arena=bool(multi_arena),
    )


def estimate_density_map_working_bytes(*args: Any, **kwargs: Any) -> int:
    """Return only :attr:`DensityMapMemoryEstimate.peak_bytes`.

    This compact wrapper is useful to callers that only need an admission
    comparison; detailed diagnostics should use
    :func:`estimate_density_map_working_set` instead.
    """

    return estimate_density_map_working_set(*args, **kwargs).peak_bytes


def admit_density_map_working_set(
    frame_count: int,
    frame_h: int,
    frame_w: int,
    downsample_factor: int = 8,
    *,
    temporal_sigma: float = 2.0,
    multi_arena: bool = False,
    max_working_bytes: int | None = None,
) -> DensityMapMemoryEstimate:
    """Return an estimate or raise before a requested density allocation.

    ``None`` leaves admission disabled.  A positive bound refuses work only
    when the conservative estimate is strictly larger, so an exact boundary is
    admitted deterministically.
    """

    estimate = estimate_density_map_working_set(
        frame_count,
        frame_h,
        frame_w,
        downsample_factor,
        temporal_sigma=temporal_sigma,
        multi_arena=multi_arena,
    )
    if max_working_bytes is None:
        return estimate
    limit = int(max_working_bytes)
    if limit <= 0:
        raise ValueError("max_working_bytes must be positive when specified")
    if estimate.peak_bytes > limit:
        raise DensityReplayBudgetExceeded(estimate, limit)
    return estimate


def _raise_if_density_cancelled(should_stop: Callable[[], bool] | None) -> None:
    """Raise instead of returning partial density evidence on cancellation."""

    if should_stop is not None and should_stop():
        raise ConfidenceDensityCancelled(
            "confidence-density map cancelled before complete evidence was available"
        )


# ---------------------------------------------------------------------------
# DensityRegion
# ---------------------------------------------------------------------------


@dataclass
class DensityRegion:
    """A 3-D connected component in the confidence density volume.

    Attributes
    ----------
    label:
        Human-readable identifier, e.g. ``"region-1"``.
    frame_start:
        Inclusive first frame index covered by the region.
    frame_end:
        Inclusive last frame index covered by the region.
    pixel_bbox:
        Bounding box in image pixel coordinates ``(x1, y1, x2, y2)``.
    arena:
        Zero-based id of the arena this region was discovered in, or ``None``
        for an untagged (whole-frame) region. ``None`` is what every
        single-arena run and every pre-multi-arena sidecar produces, and it
        means "applies to every detection" -- i.e. exactly today's behaviour.
        A tagged region applies ONLY to detections in that same arena.
    """

    label: str
    frame_start: int
    frame_end: int
    pixel_bbox: Tuple[int, int, int, int]
    arena: Optional[int] = None

    # ------------------------------------------------------------------
    # Spatial / temporal membership
    # ------------------------------------------------------------------

    def contains(self, frame: int, cx: float, cy: float) -> bool:
        """Return True if *frame* and position ``(cx, cy)`` fall inside.

        Parameters
        ----------
        frame:
            Frame index to test.
        cx, cy:
            Detection centre in pixel coordinates.
        """
        if frame < self.frame_start or frame > self.frame_end:
            return False
        x1, y1, x2, y2 = self.pixel_bbox
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def is_boundary_frame(self, frame: int, margin: int = 3) -> bool:
        """Return True if *frame* is within *margin* frames of either edge.

        Parameters
        ----------
        frame:
            Frame index to test.
        margin:
            Number of frames from each boundary to consider as "boundary".
        """
        return (
            frame <= self.frame_start + margin - 1
            or frame >= self.frame_end - margin + 1
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        The ``arena`` key is emitted ONLY for arena-tagged regions, so a
        single-arena run writes a sidecar byte-identical to the pre-arena
        format and no older reader can trip over an unexpected key.
        """
        d: Dict[str, Any] = {
            "label": self.label,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "pixel_bbox": list(self.pixel_bbox),
        }
        if self.arena is not None:
            d["arena"] = int(self.arena)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DensityRegion":
        """Reconstruct a :class:`DensityRegion` from a dictionary.

        A sidecar written before per-arena regions existed has no ``arena``
        key; it reads back as ``arena=None`` (untagged), which reproduces the
        old whole-frame semantics exactly. That is deliberate: a backward pass
        replaying an old forward pass's regions must behave as that forward
        pass did, not silently acquire arena scoping it was never computed
        with.
        """
        _arena = d.get("arena")
        return cls(
            label=d["label"],
            frame_start=int(d["frame_start"]),
            frame_end=int(d["frame_end"]),
            pixel_bbox=tuple(int(v) for v in d["pixel_bbox"]),  # type: ignore[arg-type]
            arena=None if _arena is None else int(_arena),
        )


# ---------------------------------------------------------------------------
# ConfidenceDensityMap
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceDensityMap:
    """Container for the accumulated density volume and derived regions.

    Attributes
    ----------
    frame_grids:
        Raw per-frame float32 grids, shape ``(T, H, W)``.
    regions:
        List of :class:`DensityRegion` found after binarisation.
    frame_h:
        Height of each grid in pixels.
    frame_w:
        Width of each grid in pixels.
    binary_volume:
        Optional uint8 binarised volume, shape ``(T, H, W)``, aligned with
        *frame_grids*.  Populated by :func:`compute_density_map_from_cache` and
        used by :func:`export_diagnostic_video` to draw region contours.
    frame_indices:
        Absolute source-video frame index for each row in *frame_grids* and
        *binary_volume*. Sparse cache keys remain sparse here; callers must not
        treat row number as a source frame number.
    """

    frame_grids: np.ndarray
    regions: List[DensityRegion]
    frame_h: int
    frame_w: int
    binary_volume: Optional[np.ndarray] = None
    frame_indices: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# accumulate_frame
# ---------------------------------------------------------------------------


def accumulate_frame(
    grid: np.ndarray,
    meas: np.ndarray,
    confidences: np.ndarray,
    sizes: np.ndarray,
    sigma_scale: float,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Add ``(1 - confidence)`` weighted Gaussians to *grid* in-place.

    Each detection contributes a Gaussian blob centred at its spatial
    location.  The weight is ``(1 - confidence)`` so that uncertain
    detections produce the strongest signal.

    Sigma is derived from the bounding-box size:
        ``sigma = sigma_scale * sqrt(size) / 2``

    Parameters
    ----------
    grid:
        Float32 array of shape ``(H, W)`` — modified in-place.
    meas:
        Detection measurements, shape ``(N, 3)`` columns ``[x, y, theta]``.
    confidences:
        Detection confidence values, shape ``(N,)``, range ``[0, 1]``.
    sizes:
        Squared bounding-box diagonal (or area proxy), shape ``(N,)``.
    sigma_scale:
        Scalar that controls the spread of each Gaussian relative to size.
    should_stop:
        Optional cooperative cancellation predicate.  A cancellation raises
        :class:`ConfidenceDensityCancelled` rather than returning a partially
        accumulated grid.

    Returns
    -------
    np.ndarray
        The modified *grid* (same object that was passed in).
    """
    _raise_if_density_cancelled(should_stop)
    if meas.shape[0] == 0:
        return grid

    h, w = grid.shape

    # Extract positions and compute weights/sigmas for all detections at once
    cx = meas[:, 0]  # (N,)
    cy = meas[:, 1]  # (N,)
    weights = np.maximum(1.0 - confidences, 0.0)  # (N,)
    sigmas = sigma_scale * np.sqrt(np.maximum(sizes, 1e-6)) / 2.0  # (N,)

    # Skip zero-weight detections
    mask = weights > 0
    if not mask.any():
        return grid
    cx, cy, weights, sigmas = cx[mask], cy[mask], weights[mask], sigmas[mask]

    for det_cx, det_cy, det_weight, det_sigma in zip(
        cx, cy, weights, sigmas, strict=False
    ):
        _raise_if_density_cancelled(should_stop)
        sigma = max(float(det_sigma), 1e-3)
        radius = max(1, int(np.ceil(_GAUSSIAN_TRUNCATE * sigma)))

        x0 = max(0, int(np.floor(float(det_cx) - radius)))
        x1 = min(w, int(np.ceil(float(det_cx) + radius)) + 1)
        y0 = max(0, int(np.floor(float(det_cy) - radius)))
        y1 = min(h, int(np.ceil(float(det_cy) + radius)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        local_x = np.arange(x0, x1, dtype=np.float32) - np.float32(det_cx)
        local_y = np.arange(y0, y1, dtype=np.float32) - np.float32(det_cy)
        inv2s2 = np.float32(1.0 / (2.0 * sigma * sigma))
        gauss_x = np.exp(-(local_x * local_x) * inv2s2)
        gauss_y = np.exp(-(local_y * local_y) * inv2s2)
        grid[y0:y1, x0:x1] += np.float32(det_weight) * np.multiply.outer(
            gauss_y, gauss_x
        )

    return grid


# ---------------------------------------------------------------------------
# smooth_and_binarize
# ---------------------------------------------------------------------------


def _contiguous_frame_slices(
    frame_indices: np.ndarray | None, frame_count: int
) -> tuple[slice, ...]:
    """Return row slices separated at missing source-video frames."""

    if frame_indices is None:
        return (slice(0, frame_count),) if frame_count else ()
    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) != frame_count:
        raise ValueError("frame_indices must contain one source frame per density row")
    if len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError("frame_indices must be strictly increasing")
    starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
    stops = np.r_[starts[1:], frame_count]
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))


def _smoothed_global_max(
    frames: np.ndarray,
    temporal_sigma: float,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> float:
    """Return the temporal Gaussian maximum without retaining a full output."""

    _raise_if_density_cancelled(should_stop)
    T = len(frames)
    if T == 0:
        return 0.0
    if int(frames.nbytes) <= _FULL_SMOOTH_MAX_BYTES:
        smoothed = gaussian_filter(frames, sigma=(temporal_sigma, 0.0, 0.0))
        _raise_if_density_cancelled(should_stop)
        return float(smoothed.max())

    radius = int(np.ceil(4.0 * temporal_sigma)) + 1
    global_max = 0.0
    for chunk_start in range(0, T, _SMOOTH_CHUNK_FRAMES):
        _raise_if_density_cancelled(should_stop)
        chunk_end = min(T, chunk_start + _SMOOTH_CHUNK_FRAMES)
        ext_start = max(0, chunk_start - radius)
        ext_end = min(T, chunk_end + radius)
        smoothed_chunk = gaussian_filter(
            frames[ext_start:ext_end], sigma=(temporal_sigma, 0.0, 0.0)
        )
        _raise_if_density_cancelled(should_stop)
        trim_start = chunk_start - ext_start
        trim_stop = trim_start + (chunk_end - chunk_start)
        global_max = max(global_max, float(smoothed_chunk[trim_start:trim_stop].max()))
    return global_max


def _binarize_smoothed(
    frames: np.ndarray,
    temporal_sigma: float,
    raw_threshold: float,
    output: np.ndarray,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Write thresholded temporal smoothing into a preallocated output volume."""

    _raise_if_density_cancelled(should_stop)
    T = len(frames)
    if T == 0:
        return
    if int(frames.nbytes) <= _FULL_SMOOTH_MAX_BYTES:
        smoothed = gaussian_filter(frames, sigma=(temporal_sigma, 0.0, 0.0))
        _raise_if_density_cancelled(should_stop)
        output[:] = (smoothed >= raw_threshold).astype(np.uint8)
        _raise_if_density_cancelled(should_stop)
        return

    radius = int(np.ceil(4.0 * temporal_sigma)) + 1
    for chunk_start in range(0, T, _SMOOTH_CHUNK_FRAMES):
        _raise_if_density_cancelled(should_stop)
        chunk_end = min(T, chunk_start + _SMOOTH_CHUNK_FRAMES)
        ext_start = max(0, chunk_start - radius)
        ext_end = min(T, chunk_end + radius)
        smoothed_chunk = gaussian_filter(
            frames[ext_start:ext_end], sigma=(temporal_sigma, 0.0, 0.0)
        )
        _raise_if_density_cancelled(should_stop)
        trim_start = chunk_start - ext_start
        trim_stop = trim_start + (chunk_end - chunk_start)
        output[chunk_start:chunk_end] = (
            smoothed_chunk[trim_start:trim_stop] >= raw_threshold
        ).astype(np.uint8)
        _raise_if_density_cancelled(should_stop)


def smooth_and_binarize(
    frames: np.ndarray,
    temporal_sigma: float,
    threshold: float,
    *,
    frame_indices: np.ndarray | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Smooth the density volume temporally, normalise globally, binarize.

    Parameters
    ----------
    frames:
        Float32 array of shape ``(T, H, W)`` — the stacked per-frame grids.
    temporal_sigma:
        Standard deviation (in frames) for the Gaussian smoothing kernel
        applied along the time axis only.
    threshold:
        Value in ``[0, 1]`` above which a voxel is marked as 1 after global
        normalisation.
    frame_indices:
        Optional absolute source-video frame index for each row. Temporal
        Gaussian smoothing never crosses a gap in these keys, while the
        threshold remains normalized against the global maximum across all
        available cache frames.
    should_stop:
        Optional cooperative cancellation predicate.  Cancellation raises
        :class:`ConfidenceDensityCancelled` and never returns a partial mask.

    Returns
    -------
    np.ndarray
        Uint8 binary array of shape ``(T, H, W)`` with values in ``{0, 1}``.
    """
    _raise_if_density_cancelled(should_stop)
    T, H, W = frames.shape
    binary = np.zeros((T, H, W), dtype=np.uint8)
    if T == 0:
        return binary

    runs = _contiguous_frame_slices(frame_indices, T)
    global_max = max(
        _smoothed_global_max(frames[run], temporal_sigma, should_stop=should_stop)
        for run in runs
    )
    _raise_if_density_cancelled(should_stop)
    if global_max <= 0.0:
        return binary
    raw_threshold = threshold * global_max
    for run in runs:
        _raise_if_density_cancelled(should_stop)
        _binarize_smoothed(
            frames[run],
            temporal_sigma,
            raw_threshold,
            binary[run],
            should_stop=should_stop,
        )

    return binary


# ---------------------------------------------------------------------------
# find_regions
# ---------------------------------------------------------------------------


def find_regions(
    binary: np.ndarray,
    frame_h: int,
    frame_w: int,
    min_frame_duration: int = 3,
    min_area_px: int = 100,
    *,
    frame_indices: np.ndarray | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> List[DensityRegion]:
    """Find 3-D connected components in a binary (T, H, W) volume.

    Uses full 3-D connectivity (26-connected in 3-D) via
    ``scipy.ndimage.label``.

    Parameters
    ----------
    binary:
        Uint8 array of shape ``(T, H, W)``.
    frame_h:
        Frame height in pixels (informational; used for future normalisation).
    frame_w:
        Frame width in pixels (informational).
    min_frame_duration:
        Regions spanning fewer frames than this are discarded.  Eliminates
        transient single-animal noise.  Default 3.
    min_area_px:
        Regions whose spatial bounding-box area (width × height, in grid
        pixels) is smaller than this are discarded.  Default 100.
    frame_indices:
        Optional contiguous absolute source-video frame numbers corresponding
        to the time rows in *binary*. This maps region endpoints back to the
        cache's native timeline. Gaps must be split before calling this
        function; :func:`compute_density_map_from_cache` does so automatically.
    should_stop:
        Optional cooperative cancellation predicate.  It is checked before
        allocating connected-component labels and during component conversion.

    Returns
    -------
    List[DensityRegion]
        One :class:`DensityRegion` per connected component, sorted by first
        occurrence frame then by pixel-space centroid x.  Empty list if no
        foreground voxels exist.
    """
    _raise_if_density_cancelled(should_stop)
    if frame_indices is None:
        indices = None
    else:
        indices = np.asarray(frame_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) != len(binary):
            raise ValueError("frame_indices must contain one value per binary row")
        if len(indices) > 1 and np.any(np.diff(indices) != 1):
            raise ValueError("find_regions requires contiguous frame_indices")

    if binary.max() == 0:
        return []

    # Full 26-connectivity structure for 3-D labelling.
    structure = np.ones((3, 3, 3), dtype=np.int32)
    # Force int32 output (scipy default is int64 on 64-bit platforms, which
    # would allocate 8× the binary volume's size).  int32 supports up to
    # ~2 billion distinct regions — far beyond any practical limit here.
    _raise_if_density_cancelled(should_stop)
    labeled = np.empty(binary.shape, dtype=np.int32)
    # When output= is a pre-allocated array, scipy returns only the feature count.
    label(binary, structure=structure, output=labeled)
    _raise_if_density_cancelled(should_stop)

    # find_objects returns bounding slices per component in O(N) — much
    # faster than per-component np.nonzero which allocates a full (T,H,W)
    # boolean mask each time.
    slices = find_objects(labeled)
    # Free the labeled volume immediately — it can be ~4× the binary volume
    # (int32 vs uint8) and is no longer needed once bounding slices are known.
    del labeled

    regions: List[DensityRegion] = []
    for component_id, obj_slices in enumerate(slices, start=1):
        _raise_if_density_cancelled(should_stop)
        if obj_slices is None:
            continue

        t_slice, y_slice, x_slice = obj_slices
        frame_start = (
            int(indices[t_slice.start]) if indices is not None else t_slice.start
        )
        frame_end = (
            int(indices[t_slice.stop - 1]) if indices is not None else t_slice.stop - 1
        )
        x1 = x_slice.start
        x2 = x_slice.stop - 1
        y1 = y_slice.start
        y2 = y_slice.stop - 1

        # Apply minimum size / duration filters.
        duration = frame_end - frame_start + 1
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if duration < min_frame_duration or area < min_area_px:
            continue

        region_label = f"region-{component_id}"
        regions.append(
            DensityRegion(
                label=region_label,
                frame_start=frame_start,
                frame_end=frame_end,
                pixel_bbox=(x1, y1, x2, y2),
            )
        )

    # Sort by frame_start, then by spatial centroid x for determinism.
    regions.sort(key=lambda r: (r.frame_start, r.pixel_bbox[0]))

    # Reassign sequential labels after sorting.
    for idx, region in enumerate(regions, start=1):
        region.label = f"region-{idx}"

    return regions


def _find_regions_by_frame_runs(
    binary: np.ndarray,
    frame_h: int,
    frame_w: int,
    min_frame_duration: int,
    min_area_px: int,
    frame_indices: np.ndarray,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> List[DensityRegion]:
    """Label contiguous source-frame runs without connecting missing keys."""

    indices = np.asarray(frame_indices, dtype=np.int64)
    regions: List[DensityRegion] = []
    for run in _contiguous_frame_slices(indices, len(binary)):
        _raise_if_density_cancelled(should_stop)
        regions.extend(
            find_regions(
                binary[run],
                frame_h=frame_h,
                frame_w=frame_w,
                min_frame_duration=min_frame_duration,
                min_area_px=min_area_px,
                frame_indices=indices[run],
                should_stop=should_stop,
            )
        )
    regions.sort(key=lambda region: (region.frame_start, region.pixel_bbox[0]))
    for index, region in enumerate(regions, start=1):
        region.label = f"region-{index}"
    return regions


# ---------------------------------------------------------------------------
# tag_detections
# ---------------------------------------------------------------------------


def tag_detections(
    detections: List[Dict[str, Any]],
    regions: List[DensityRegion],
) -> List[Dict[str, Any]]:
    """Annotate detection dicts with confidence-density region membership.

    Each input dict must contain ``frame``, ``cx``, and ``cy`` keys. The
    matching region label is written back in-place as ``region_label`` and the
    temporal edge flag as ``region_boundary``.
    """
    for det in detections:
        frame = int(det["frame"])
        cx = float(det["cx"])
        cy = float(det["cy"])

        matched_region: Optional[DensityRegion] = None
        for region in regions:
            if region.contains(frame, cx, cy):
                matched_region = region
                break

        if matched_region is None:
            det["region_label"] = "open_field"
            det["region_boundary"] = False
        else:
            det["region_label"] = matched_region.label
            det["region_boundary"] = matched_region.is_boundary_frame(frame)

    return detections


# ---------------------------------------------------------------------------
# save_regions / load_regions
# ---------------------------------------------------------------------------


def save_regions(regions: List[DensityRegion], path: str | Path) -> None:
    """Persist a list of :class:`DensityRegion` objects to a JSON file.

    Parameters
    ----------
    regions:
        Regions to serialise.
    path:
        Destination file path.  Parent directories must already exist.
    """
    payload = [r.to_dict() for r in regions]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_regions(path: str | Path) -> List[DensityRegion]:
    """Load :class:`DensityRegion` objects from a JSON file.

    Parameters
    ----------
    path:
        Source file path previously written by :func:`save_regions`.

    Returns
    -------
    List[DensityRegion]
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return [DensityRegion.from_dict(d) for d in payload]


# ---------------------------------------------------------------------------
# per-arena density regions
# ---------------------------------------------------------------------------


def _scaled_for_grid(meas, sizes, ds):
    """Scale detection positions/sizes from frame space into grid space."""
    if meas.shape[0] > 0 and ds > 1:
        meas_scaled = meas.copy()
        meas_scaled[:, 0] /= ds
        meas_scaled[:, 1] /= ds
        return meas_scaled, sizes / (ds**2)
    return meas, sizes


def _scale_region_bboxes_to_frame(
    regions: List[DensityRegion],
    *,
    ds: int,
    grid_h: int,
    grid_w: int,
    frame_h: int,
    frame_w: int,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Map inclusive grid-cell bboxes back to inclusive source pixels.

    A grid maximum is an inclusive cell index, not a coordinate to multiply
    directly.  The final floor-division cell also represents a non-divisible
    source-frame remainder, so it expands to the true frame edge.
    """

    if ds <= 1:
        return
    max_x = max(0, int(frame_w) - 1)
    max_y = max(0, int(frame_h) - 1)
    final_grid_x = max(0, int(grid_w) - 1)
    final_grid_y = max(0, int(grid_h) - 1)
    for region in regions:
        _raise_if_density_cancelled(should_stop)
        x1, y1, x2, y2 = region.pixel_bbox
        region.pixel_bbox = (
            min(max_x, max(0, int(x1) * ds)),
            min(max_y, max(0, int(y1) * ds)),
            max_x if int(x2) >= final_grid_x else min(max_x, (int(x2) + 1) * ds - 1),
            max_y if int(y2) >= final_grid_y else min(max_y, (int(y2) + 1) * ds - 1),
        )


def _compute_density_map_per_arena(
    detection_cache,
    arena_layout,
    frame_h: int,
    frame_w: int,
    grid_h: int,
    grid_w: int,
    ds: int,
    sigma_scale: float,
    temporal_sigma: float,
    threshold: float,
    min_frame_duration: int,
    min_area_px: int,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    should_stop: Callable[[], bool] | None = None,
) -> Tuple[ConfidenceDensityMap, np.ndarray]:
    """Compute density regions independently for each arena.

    Why per-arena at all: the whole-frame pipeline couples arenas in two
    distinct ways, both demonstrated by the arena adversarial review.

    1. *Spatial spillover* -- a crowd entirely inside arena A produces a
       connected component whose Gaussian tails, and whose rectangular
       bounding box, reach across the arena wall, so detections in arena B get
       flagged by arena A's crowd.
    2. *Global-max coupling* -- ``smooth_and_binarize`` thresholds at
       ``threshold * max(volume)`` over the WHOLE frame, so a dense arena
       raises the bar for every other arena and can erase a region that arena
       would have had entirely on its own. This needs no adjacency at all:
       arenas on opposite corners of the frame still couple.

    The fix addresses both at the source rather than post-filtering the
    output, because only (2) is invisible to any post-hoc geometric filter:
    a region that never formed cannot be recovered by intersecting bboxes
    with arena labels. So each arena gets

    * its own volume, accumulated from ONLY its own detections (kills the
      cross-arena Gaussian contribution),
    * that volume masked to its own pixels (kills tails that leak across the
      wall),
    * its own ``smooth_and_binarize`` call, hence its own maximum (kills the
      global-max coupling),
    * its own ``find_regions`` call, and every resulting region tagged with
      ``arena`` so that even a bounding box which geometrically overlaps a
      neighbour (unavoidable for non-rectangular arenas) can never flag that
      neighbour's detections -- see :func:`get_density_region_flags`.

    Each arena's region set is therefore a pure function of that arena's own
    detections plus the static layout, which is the property the whole
    multi-arena feature is defined by.

    Detections that fall outside every arena (``arena_of_points`` -> ``-1``)
    contribute to NO arena's volume and match no arena-tagged region, so they
    are never density-flagged. This mirrors how the rest of the arena work
    treats them (they bootstrap no slot and match no track).

    The returned ``frame_grids``/``binary_volume`` are the union over arenas
    (sum of the masked grids, elementwise max of the binaries) purely so the
    diagnostic video keeps working; nothing in the tracking path reads them.
    """
    _raise_if_density_cancelled(should_stop)
    sorted_frames = sorted(detection_cache.keys())
    frame_indices = np.asarray(sorted_frames, dtype=np.int64)
    n_total = len(sorted_frames)
    n_arenas = int(arena_layout.n_arenas)

    # Arena id per detection, resolved once per frame in the SAME coordinate
    # space the grid is built in ((frame_w, frame_h)), so a detection's arena
    # and its grid cell can never disagree.
    det_arena: List[np.ndarray] = []
    for frame_idx in sorted_frames:
        _raise_if_density_cancelled(should_stop)
        meas, _confs, _sizes = detection_cache[frame_idx]
        if meas.shape[0] == 0:
            det_arena.append(np.zeros(0, dtype=np.int32))
        else:
            det_arena.append(
                arena_layout.arena_of_points(meas[:, :2], frame_size=(frame_w, frame_h))
            )

    # Arena label image at grid resolution (nearest-neighbour, cached by the
    # layout). Labels are 1-based with 0 meaning "outside every arena".
    grid_labels = arena_layout.label_image_for_size(grid_w, grid_h)

    total_grids = np.zeros((n_total, grid_h, grid_w), dtype=np.float32)
    binary_total = np.zeros((n_total, grid_h, grid_w), dtype=np.uint8)
    work = np.zeros((n_total, grid_h, grid_w), dtype=np.float32)
    regions: List[DensityRegion] = []

    for arena_id in range(n_arenas):
        _raise_if_density_cancelled(should_stop)
        work[:] = 0.0
        for i, frame_idx in enumerate(sorted_frames):
            _raise_if_density_cancelled(should_stop)
            meas, confs, sizes = detection_cache[frame_idx]
            if meas.shape[0] == 0:
                continue
            sel = det_arena[i] == arena_id
            if not sel.any():
                continue
            meas_scaled, sizes_scaled = _scaled_for_grid(meas[sel], sizes[sel], ds)
            accumulate_frame(
                work[i],
                meas_scaled,
                confs[sel],
                sizes_scaled,
                sigma_scale=sigma_scale,
                should_stop=should_stop,
            )
        # Mask to this arena's own pixels: a Gaussian tail that spills over the
        # wall must not be able to seed a region on the other side. float32
        # multiply by an exact 0.0/1.0 mask leaves in-arena values bit-exact.
        mask = (grid_labels == (arena_id + 1)).astype(np.float32)
        work *= mask[None, :, :]

        binary = smooth_and_binarize(
            work,
            temporal_sigma=temporal_sigma,
            threshold=threshold,
            frame_indices=frame_indices,
            should_stop=should_stop,
        )
        _raise_if_density_cancelled(should_stop)
        arena_regions = _find_regions_by_frame_runs(
            binary,
            frame_h=grid_h,
            frame_w=grid_w,
            min_frame_duration=min_frame_duration,
            min_area_px=min_area_px,
            frame_indices=frame_indices,
            should_stop=should_stop,
        )
        for r in arena_regions:
            r.arena = arena_id
        regions.extend(arena_regions)

        total_grids += work
        np.maximum(binary_total, binary, out=binary_total)
        del binary
        _raise_if_density_cancelled(should_stop)

        if progress_callback is not None:
            pct = int(45 * (arena_id + 1) / max(1, n_arenas))
            progress_callback(pct, f"Density map: arena {arena_id + 1}/{n_arenas}")

    # Scale inclusive grid-cell bboxes back to original pixel coordinates.
    _scale_region_bboxes_to_frame(
        regions,
        ds=ds,
        grid_h=grid_h,
        grid_w=grid_w,
        frame_h=frame_h,
        frame_w=frame_w,
        should_stop=should_stop,
    )

    # Deterministic global ordering/labelling across arenas (arena breaks ties
    # so two arenas producing an identical (frame_start, x1) still sort
    # stably).
    regions.sort(key=lambda r: (r.frame_start, r.pixel_bbox[0], r.arena))
    for idx, region in enumerate(regions, start=1):
        _raise_if_density_cancelled(should_stop)
        region.label = f"region-{idx}"

    if progress_callback is not None:
        progress_callback(
            48,
            f"Density map complete: {len(regions)} regions found "
            f"across {n_arenas} arenas",
        )

    cdm = ConfidenceDensityMap(
        frame_grids=total_grids,
        regions=regions,
        frame_h=grid_h,
        frame_w=grid_w,
        binary_volume=binary_total,
        frame_indices=frame_indices,
    )
    return cdm, total_grids


# ---------------------------------------------------------------------------
# compute_density_map_from_cache
# ---------------------------------------------------------------------------


def compute_density_map_from_cache(
    detection_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    frame_h: int,
    frame_w: int,
    sigma_scale: float,
    temporal_sigma: float,
    threshold: float,
    downsample_factor: int = 8,
    min_frame_duration: int = 3,
    min_area_px: int = 100,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    arena_layout: Optional["ArenaLayout"] = None,
    *,
    max_working_bytes: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Tuple[ConfidenceDensityMap, List[np.ndarray]]:
    """Run the full density-map pipeline from a detection cache.

    Parameters
    ----------
    detection_cache:
        Mapping from ``frame_index`` to a tuple of
        ``(measurements, confidences, sizes)`` as produced by the MAT
        detection engine.  Each array has ``N`` rows (one per detection).
        *measurements* is shape ``(N, 3)`` with columns ``[x, y, theta]``.
        *confidences* is shape ``(N,)`` in ``[0, 1]``.
        *sizes* is shape ``(N,)`` (squared bbox diagonal or equivalent).
    frame_h:
        Height of the video frames in pixels.
    frame_w:
        Width of the video frames in pixels.
    sigma_scale:
        Passed through to :func:`accumulate_frame`.
    temporal_sigma:
        Passed through to :func:`smooth_and_binarize`.
    threshold:
        Passed through to :func:`smooth_and_binarize`.
    downsample_factor:
        Internal grids operate at ceil-divided ``(frame_h / factor,
        frame_w / factor)`` resolution, retaining a final partial cell for
        non-divisible source dimensions. Detection positions are scaled down
        before accumulation and region bounding boxes are scaled back up on
        output. Default 8.
    progress_callback:
        Optional callable ``(percent: int, message: str) -> None`` invoked
        periodically to report progress.
    arena_layout:
        Optional :class:`~hydra_suite.core.tracking.arenas.ArenaLayout`. When
        it describes more than one arena AND carries a label image, regions
        are computed independently per arena (see
        :func:`_compute_density_map_per_arena`). ``None``, a single-arena
        layout, or a layout with no label image all take the ORIGINAL
        whole-frame path below, unmodified.
    max_working_bytes:
        Optional conservative peak-memory admission cap.  The check happens
        before any density-volume allocation and uses the number of cache keys,
        never their absolute source-frame values.  Ordinary tracking leaves
        this as ``None``; read-only autotuner replay supplies its own cap.
    should_stop:
        Optional cooperative cancellation predicate.  It is propagated through
        accumulation, smoothing, arena iteration, and connected-component
        labeling.  Cancellation raises :class:`ConfidenceDensityCancelled`
        rather than returning partial density evidence.

    Returns
    -------
    (ConfidenceDensityMap, raw_grids)
        *raw_grids* is the list of per-frame float32 arrays before smoothing,
        in frame-index order.  Grids are at the downsampled resolution.

    The :attr:`ConfidenceDensityMap.binary_volume` field on the returned CDM
    holds the binarised volume (at downsampled resolution) so it can be passed
    directly to :func:`export_diagnostic_video` for contour rendering.
    """
    _raise_if_density_cancelled(should_stop)
    ds, grid_h, grid_w = _density_grid_shape(frame_h, frame_w, downsample_factor)
    # Validate an explicit replay admission setting even when an empty cache
    # takes the allocation-free early return below. Silently accepting an
    # invalid override based on cache contents makes configuration behavior
    # nondeterministic between otherwise equivalent replays.
    if max_working_bytes is not None and int(max_working_bytes) <= 0:
        raise ValueError("max_working_bytes must be positive when specified")

    if not detection_cache:
        frame_grids = np.zeros((0, grid_h, grid_w), dtype=np.float32)
        cdm = ConfidenceDensityMap(
            frame_grids=frame_grids,
            regions=[],
            frame_h=grid_h,
            frame_w=grid_w,
            frame_indices=np.zeros(0, dtype=np.int64),
        )
        return cdm, []

    multi_arena = bool(
        arena_layout is not None
        and not arena_layout.is_single_arena
        and arena_layout.label_image is not None
    )
    # Admission is deliberately based on cache cardinality, not max(frame key),
    # and happens before either volume path allocates density grids.
    if max_working_bytes is not None:
        admit_density_map_working_set(
            len(detection_cache),
            frame_h,
            frame_w,
            ds,
            temporal_sigma=temporal_sigma,
            multi_arena=multi_arena,
            max_working_bytes=max_working_bytes,
        )

    if multi_arena:
        return _compute_density_map_per_arena(
            detection_cache=detection_cache,
            arena_layout=arena_layout,
            frame_h=frame_h,
            frame_w=frame_w,
            grid_h=grid_h,
            grid_w=grid_w,
            ds=ds,
            sigma_scale=sigma_scale,
            temporal_sigma=temporal_sigma,
            threshold=threshold,
            min_frame_duration=min_frame_duration,
            min_area_px=min_area_px,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )

    sorted_frames = sorted(detection_cache.keys())
    frame_indices = np.asarray(sorted_frames, dtype=np.int64)
    n_total = len(sorted_frames)
    # Pre-allocate the full (T, grid_h, grid_w) array once and accumulate
    # directly into it — avoids building a Python list then calling np.stack,
    # which would briefly hold two copies of the entire volume in RAM.
    frame_grids = np.zeros((n_total, grid_h, grid_w), dtype=np.float32)
    for i, frame_idx in enumerate(sorted_frames):
        _raise_if_density_cancelled(should_stop)
        meas, confs, sizes = detection_cache[frame_idx]
        # Scale detection positions and sizes to the downsampled grid.
        if meas.shape[0] > 0 and ds > 1:
            meas_scaled = meas.copy()
            meas_scaled[:, 0] /= ds
            meas_scaled[:, 1] /= ds
            sizes_scaled = sizes / (ds**2)
        else:
            meas_scaled = meas
            sizes_scaled = sizes
        accumulate_frame(
            frame_grids[i],
            meas_scaled,
            confs,
            sizes_scaled,
            sigma_scale=sigma_scale,
            should_stop=should_stop,
        )
        if progress_callback is not None and (i % 50 == 0 or i == n_total - 1):
            pct = int(40 * (i + 1) / n_total)  # 0–40% for accumulation
            progress_callback(pct, f"Density map: accumulating frame {i + 1}/{n_total}")

    if progress_callback is not None:
        progress_callback(42, "Density map: temporal smoothing...")

    binary = smooth_and_binarize(
        frame_grids,
        temporal_sigma=temporal_sigma,
        threshold=threshold,
        frame_indices=frame_indices,
        should_stop=should_stop,
    )
    _raise_if_density_cancelled(should_stop)

    if progress_callback is not None:
        progress_callback(45, "Density map: finding regions...")

    regions = _find_regions_by_frame_runs(
        binary,
        frame_h=grid_h,
        frame_w=grid_w,
        min_frame_duration=min_frame_duration,
        min_area_px=min_area_px,
        frame_indices=frame_indices,
        should_stop=should_stop,
    )

    # Scale inclusive grid-cell bboxes back to original pixel coordinates.
    _scale_region_bboxes_to_frame(
        regions,
        ds=ds,
        grid_h=grid_h,
        grid_w=grid_w,
        frame_h=frame_h,
        frame_w=frame_w,
        should_stop=should_stop,
    )

    if progress_callback is not None:
        progress_callback(48, f"Density map complete: {len(regions)} regions found")

    cdm = ConfidenceDensityMap(
        frame_grids=frame_grids,
        regions=regions,
        frame_h=grid_h,
        frame_w=grid_w,
        binary_volume=binary,
        frame_indices=frame_indices,
    )
    # Return frame_grids as the second value (a numpy array supports the same
    # [frame_idx] access as the former raw_grids list, so callers are unaffected).
    return cdm, frame_grids


# ---------------------------------------------------------------------------
# export_diagnostic_video
# ---------------------------------------------------------------------------


def _prepare_diag_frame(frame_reader, frame_idx, frame_h, frame_w, cv2):
    """Read and resize a frame for diagnostic video output."""
    frame = frame_reader(frame_idx)
    if frame is None:
        return np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    if frame.shape[0] != frame_h or frame.shape[1] != frame_w:
        return cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
    return frame.copy()


def _overlay_diag_heatmap(
    frame,
    density_row,
    density_grids,
    global_max,
    frame_h,
    frame_w,
    heatmap_alpha,
    cv2,
):
    """Apply red heatmap overlay for a single frame."""
    if density_row is None or density_row >= len(density_grids):
        return frame
    norm = (density_grids[density_row] / global_max).clip(0, 1)
    if norm.shape[0] != frame_h or norm.shape[1] != frame_w:
        norm = cv2.resize(norm, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    red_mask = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    red_mask[:, :, 2] = (norm * 255).astype(np.uint8)
    return cv2.addWeighted(frame, 1 - heatmap_alpha, red_mask, heatmap_alpha, 0)


def _scaled_region_bbox(r, output_scale):
    """Scale a region's pixel bbox by output_scale."""
    x1, y1, x2, y2 = r.pixel_bbox
    if output_scale != 1.0:
        x1, y1 = int(x1 * output_scale), int(y1 * output_scale)
        x2, y2 = int(x2 * output_scale), int(y2 * output_scale)
    return x1, y1, x2, y2


def _draw_diag_regions(
    frame,
    frame_idx,
    density_row,
    regions,
    binary_volume,
    frame_h,
    frame_w,
    output_scale,
    cv2,
):
    """Draw region outlines/contours and labels on a diagnostic frame."""
    active = [r for r in regions if r.frame_start <= frame_idx <= r.frame_end]
    if not active:
        return

    if (
        binary_volume is not None
        and density_row is not None
        and density_row < len(binary_volume)
    ):
        bin_slice = binary_volume[density_row]
        if bin_slice.max() > 0:
            bin_out = cv2.resize(
                bin_slice,
                (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST,
            )
            contours, _ = cv2.findContours(
                bin_out,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(frame, contours, -1, (0, 0, 200), 1)
        for r in active:
            x1, y1, x2, y2 = _scaled_region_bbox(r, output_scale)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(
                frame,
                f"{r.label} [{r.frame_start}-{r.frame_end}]",
                (cx, max(cy - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 200),
                1,
            )
    else:
        for r in active:
            x1, y1, x2, y2 = _scaled_region_bbox(r, output_scale)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 1)
            cv2.putText(
                frame,
                f"{r.label} [{r.frame_start}-{r.frame_end}]",
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 200),
                1,
            )


def export_diagnostic_video(
    frame_reader,  # callable: frame_idx -> np.ndarray (H,W,3) uint8 or None
    n_frames: int,
    frame_h: int,
    frame_w: int,
    density_grids: list,  # list of (grid_h, grid_w) float32 raw grids
    regions: list,  # list of DensityRegion
    output_path,  # Path
    fps: float = 25.0,
    heatmap_alpha: float = 0.35,
    output_scale: float = 1.0,
    binary_volume: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    *,
    density_frame_indices: Optional[np.ndarray] = None,
    start_frame_index: int = 0,
) -> None:
    """Write diagnostic video with red heatmap overlay on low-confidence zones.

    Renders each video frame with a semi-transparent red heatmap overlay
    proportional to the normalised confidence density.  When *binary_volume*
    is provided the actual contour border of each region is drawn instead of
    a bounding box, which eliminates the visual artefact of nested rectangles.

    Uses BGR colour order (OpenCV convention). Red = channel index 2 in BGR.

    Parameters
    ----------
    frame_reader:
        Callable ``frame_idx -> np.ndarray (H, W, 3) uint8`` or ``None`` if
        the frame is unavailable.  A black frame is substituted when ``None``
        is returned.  The returned frame may be at any resolution — it will
        be resized to the output dimensions.
    n_frames:
        Total number of frames to render.
    frame_h:
        Output frame height in pixels.
    frame_w:
        Output frame width in pixels.
    density_grids:
        Per-frame raw float32 density grids, one array of shape ``(H, W)``
        per frame.  May be shorter than *n_frames*; missing frames get no
        overlay.
    regions:
        List of :class:`DensityRegion` whose outlines are drawn while they
        are temporally active.
    output_path:
        Destination ``.mp4`` file path.
    fps:
        Output video frame rate.
    heatmap_alpha:
        Blend weight for the red heatmap layer (0 = invisible, 1 = opaque).
    output_scale:
        Scale factor applied to region bounding-box / label coordinates so
        they match the output resolution.  E.g. if regions are in
        full-resolution pixel coords and the output is 4× downsampled, pass
        ``0.25``.
    binary_volume:
        Optional uint8 array of shape ``(T, H, W)`` — the binarised density
        volume at the downsampled grid resolution.  When supplied, per-frame
        slices are scaled to ``(frame_h, frame_w)`` and contours are drawn
        instead of bounding boxes.  Defaults to ``None`` (fall back to bbox).
    progress_callback:
        Optional callable ``(percent: int, message: str) -> None`` invoked
        periodically to report progress.
    density_frame_indices:
        Optional absolute source-video frame index for each density-grid row.
        When supplied, missing cache rows render without a heatmap rather than
        assigning an adjacent row's evidence to the wrong source frame.
    start_frame_index:
        Absolute source-video frame represented by output row zero. Defaults to
        zero for backward compatibility with whole-video callers.
    """
    import cv2

    output_path = Path(output_path)
    writer = VideoEncoder(output_path, fps=fps, width=frame_w, height=frame_h)

    _dg = (
        np.asarray(density_grids)
        if not isinstance(density_grids, np.ndarray)
        else density_grids
    )
    global_max = float(_dg.max()) if len(_dg) > 0 and _dg.max() > 0 else 1.0
    if density_frame_indices is None:
        density_row_by_frame = None
    else:
        indices = np.asarray(density_frame_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) != len(_dg):
            raise ValueError(
                "density_frame_indices must contain one source frame per density row"
            )
        if len(indices) > 1 and np.any(np.diff(indices) <= 0):
            raise ValueError("density_frame_indices must be strictly increasing")
        density_row_by_frame = {
            int(source_frame): row for row, source_frame in enumerate(indices)
        }

    try:
        for frame_idx in range(n_frames):
            source_frame = start_frame_index + frame_idx
            density_row = (
                frame_idx
                if density_row_by_frame is None
                else density_row_by_frame.get(source_frame)
            )
            frame = _prepare_diag_frame(
                frame_reader, source_frame, frame_h, frame_w, cv2
            )
            frame = _overlay_diag_heatmap(
                frame,
                density_row,
                _dg,
                global_max,
                frame_h,
                frame_w,
                heatmap_alpha,
                cv2,
            )
            _draw_diag_regions(
                frame,
                source_frame,
                density_row,
                regions,
                binary_volume,
                frame_h,
                frame_w,
                output_scale,
                cv2,
            )
            writer.write(frame)
            if progress_callback is not None and (
                frame_idx % 50 == 0 or frame_idx == n_frames - 1
            ):
                pct = 50 + int(45 * (frame_idx + 1) / n_frames)
                progress_callback(
                    pct, f"Diagnostic video: frame {frame_idx + 1}/{n_frames}"
                )
    finally:
        writer.release()
