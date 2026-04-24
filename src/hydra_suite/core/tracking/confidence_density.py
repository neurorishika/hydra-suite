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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter, label

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
    """

    label: str
    frame_start: int
    frame_end: int
    pixel_bbox: Tuple[int, int, int, int]

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
        """Return a JSON-serialisable dictionary representation."""
        return {
            "label": self.label,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "pixel_bbox": list(self.pixel_bbox),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DensityRegion":
        """Reconstruct a :class:`DensityRegion` from a dictionary."""
        return cls(
            label=d["label"],
            frame_start=int(d["frame_start"]),
            frame_end=int(d["frame_end"]),
            pixel_bbox=tuple(int(v) for v in d["pixel_bbox"]),  # type: ignore[arg-type]
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
    """

    frame_grids: np.ndarray
    regions: List[DensityRegion]
    frame_h: int
    frame_w: int
    binary_volume: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# accumulate_frame
# ---------------------------------------------------------------------------


def accumulate_frame(
    grid: np.ndarray,
    meas: np.ndarray,
    confidences: np.ndarray,
    sizes: np.ndarray,
    sigma_scale: float,
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

    Returns
    -------
    np.ndarray
        The modified *grid* (same object that was passed in).
    """
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

    # Pre-compute coordinate vectors once.
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)

    # Vectorised separable Gaussian: process all N detections in one batch.
    # dy: (N, H), dx: (N, W) — broadcast without Python loop.
    dy = ys[None, :] - cy[:, None]           # (N, H)
    dx = xs[None, :] - cx[:, None]           # (N, W)
    inv2s2 = 1.0 / (2.0 * sigmas[:, None] ** 2)  # (N, 1)
    gauss_y = np.exp(-(dy ** 2) * inv2s2)    # (N, H)
    gauss_x = np.exp(-(dx ** 2) * inv2s2)    # (N, W)
    # Weighted outer-product sum: einsum over detections → (H, W) increment.
    grid += np.einsum("n,nh,nw->hw", weights, gauss_y, gauss_x)

    return grid


# ---------------------------------------------------------------------------
# smooth_and_binarize
# ---------------------------------------------------------------------------


def smooth_and_binarize(
    frames: np.ndarray,
    temporal_sigma: float,
    threshold: float,
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

    Returns
    -------
    np.ndarray
        Uint8 binary array of shape ``(T, H, W)`` with values in ``{0, 1}``.
    """
    T, H, W = frames.shape
    binary = np.zeros((T, H, W), dtype=np.uint8)
    if T == 0:
        return binary

    # Effective Gaussian kernel radius (scipy default truncate=4.0).
    radius = int(np.ceil(4.0 * temporal_sigma)) + 1
    chunk_size = 500  # frames per chunk; tunes peak-RAM vs. pass count

    # --- Pass 1: find global max across all smoothed chunks ---
    # Each slice is a VIEW into `frames` (no copy), so only the
    # gaussian_filter output — at most (chunk_size + 2*radius, H, W) —
    # is allocated per iteration instead of the full (T, H, W) volume.
    global_max = 0.0
    for chunk_start in range(0, T, chunk_size):
        chunk_end = min(T, chunk_start + chunk_size)
        ext_start = max(0, chunk_start - radius)
        ext_end = min(T, chunk_end + radius)
        smoothed_chunk = gaussian_filter(
            frames[ext_start:ext_end], sigma=(temporal_sigma, 0.0, 0.0)
        )
        trim_s = chunk_start - ext_start
        trim_e = trim_s + (chunk_end - chunk_start)
        chunk_max = float(smoothed_chunk[trim_s:trim_e].max())
        if chunk_max > global_max:
            global_max = chunk_max

    if global_max <= 0.0:
        return binary

    # Avoid a second `/` allocation: compare raw values against
    # threshold * global_max (mathematically equivalent to normalising first).
    raw_threshold = threshold * global_max

    # --- Pass 2: binarize chunk by chunk without retaining the full
    # smoothed float32 volume in memory ---
    for chunk_start in range(0, T, chunk_size):
        chunk_end = min(T, chunk_start + chunk_size)
        ext_start = max(0, chunk_start - radius)
        ext_end = min(T, chunk_end + radius)
        smoothed_chunk = gaussian_filter(
            frames[ext_start:ext_end], sigma=(temporal_sigma, 0.0, 0.0)
        )
        trim_s = chunk_start - ext_start
        trim_e = trim_s + (chunk_end - chunk_start)
        binary[chunk_start:chunk_end] = (
            smoothed_chunk[trim_s:trim_e] >= raw_threshold
        ).astype(np.uint8)

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

    Returns
    -------
    List[DensityRegion]
        One :class:`DensityRegion` per connected component, sorted by first
        occurrence frame then by pixel-space centroid x.  Empty list if no
        foreground voxels exist.
    """
    if binary.max() == 0:
        return []

    # Full 26-connectivity structure for 3-D labelling.
    structure = np.ones((3, 3, 3), dtype=np.int32)
    # Force int32 output (scipy default is int64 on 64-bit platforms, which
    # would allocate 8× the binary volume's size).  int32 supports up to
    # ~2 billion distinct regions — far beyond any practical limit here.
    labeled = np.empty(binary.shape, dtype=np.int32)
    # When output= is a pre-allocated array, scipy returns only the feature count.
    label(binary, structure=structure, output=labeled)

    # find_objects returns bounding slices per component in O(N) — much
    # faster than per-component np.nonzero which allocates a full (T,H,W)
    # boolean mask each time.
    slices = find_objects(labeled)
    # Free the labeled volume immediately — it can be ~4× the binary volume
    # (int32 vs uint8) and is no longer needed once bounding slices are known.
    del labeled

    regions: List[DensityRegion] = []
    for component_id, obj_slices in enumerate(slices, start=1):
        if obj_slices is None:
            continue

        t_slice, y_slice, x_slice = obj_slices
        frame_start = t_slice.start
        frame_end = t_slice.stop - 1
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
        Internal grids operate at ``(frame_h // factor, frame_w // factor)``
        resolution.  Detection positions are scaled down before accumulation
        and region bounding boxes are scaled back up on output.  Default 8.
    progress_callback:
        Optional callable ``(percent: int, message: str) -> None`` invoked
        periodically to report progress.

    Returns
    -------
    (ConfidenceDensityMap, raw_grids)
        *raw_grids* is the list of per-frame float32 arrays before smoothing,
        in frame-index order.  Grids are at the downsampled resolution.

    The :attr:`ConfidenceDensityMap.binary_volume` field on the returned CDM
    holds the binarised volume (at downsampled resolution) so it can be passed
    directly to :func:`export_diagnostic_video` for contour rendering.
    """
    ds = max(1, int(downsample_factor))
    grid_h = max(1, frame_h // ds)
    grid_w = max(1, frame_w // ds)

    if not detection_cache:
        frame_grids = np.zeros((0, grid_h, grid_w), dtype=np.float32)
        cdm = ConfidenceDensityMap(
            frame_grids=frame_grids,
            regions=[],
            frame_h=grid_h,
            frame_w=grid_w,
        )
        return cdm, []

    sorted_frames = sorted(detection_cache.keys())
    n_total = len(sorted_frames)
    # Pre-allocate the full (T, grid_h, grid_w) array once and accumulate
    # directly into it — avoids building a Python list then calling np.stack,
    # which would briefly hold two copies of the entire volume in RAM.
    frame_grids = np.zeros((n_total, grid_h, grid_w), dtype=np.float32)
    for i, frame_idx in enumerate(sorted_frames):
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
            frame_grids[i], meas_scaled, confs, sizes_scaled, sigma_scale=sigma_scale
        )
        if progress_callback is not None and (i % 50 == 0 or i == n_total - 1):
            pct = int(40 * (i + 1) / n_total)  # 0–40% for accumulation
            progress_callback(pct, f"Density map: accumulating frame {i + 1}/{n_total}")

    if progress_callback is not None:
        progress_callback(42, "Density map: temporal smoothing...")

    binary = smooth_and_binarize(
        frame_grids, temporal_sigma=temporal_sigma, threshold=threshold
    )

    if progress_callback is not None:
        progress_callback(45, "Density map: finding regions...")

    regions = find_regions(
        binary,
        frame_h=grid_h,
        frame_w=grid_w,
        min_frame_duration=min_frame_duration,
        min_area_px=min_area_px,
    )

    # Scale bounding boxes back to original pixel coordinates.
    if ds > 1:
        for r in regions:
            x1, y1, x2, y2 = r.pixel_bbox
            r.pixel_bbox = (
                x1 * ds,
                y1 * ds,
                x2 * ds,
                y2 * ds,
            )

    if progress_callback is not None:
        progress_callback(48, f"Density map complete: {len(regions)} regions found")

    cdm = ConfidenceDensityMap(
        frame_grids=frame_grids,
        regions=regions,
        frame_h=grid_h,
        frame_w=grid_w,
        binary_volume=binary,
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
    frame_idx,
    density_grids,
    global_max,
    frame_h,
    frame_w,
    heatmap_alpha,
    cv2,
):
    """Apply red heatmap overlay for a single frame."""
    if frame_idx >= len(density_grids):
        return frame
    norm = (density_grids[frame_idx] / global_max).clip(0, 1)
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

    if binary_volume is not None and frame_idx < len(binary_volume):
        bin_slice = binary_volume[frame_idx]
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
    """
    import cv2

    output_path = Path(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    _dg = np.asarray(density_grids) if not isinstance(density_grids, np.ndarray) else density_grids
    global_max = float(_dg.max()) if len(_dg) > 0 and _dg.max() > 0 else 1.0

    try:
        for frame_idx in range(n_frames):
            frame = _prepare_diag_frame(frame_reader, frame_idx, frame_h, frame_w, cv2)
            frame = _overlay_diag_heatmap(
                frame,
                frame_idx,
                _dg,
                global_max,
                frame_h,
                frame_w,
                heatmap_alpha,
                cv2,
            )
            _draw_diag_regions(
                frame,
                frame_idx,
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
