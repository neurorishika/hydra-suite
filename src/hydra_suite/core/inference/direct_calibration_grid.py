"""Bounded, transparent SAHI candidate grid for detector calibration.

Deliberately a stated grid, not an optimizer: the UI prints the exact candidate
list and its tile cost before any model runs. Candidates carry ``SLICE_*``
PARAMS, never a hand-built ``SliceConfig`` -- routing through the shared params
mapping is what makes a measured point expressible as TrackerKit settings.
"""

from __future__ import annotations

from dataclasses import dataclass

FRACTION_STEPS: tuple[float, ...] = (0.75, 1.0, 1.5)
OVERLAP_STEPS: tuple[float, ...] = (0.1, 0.2, 0.3)
DEFAULT_MAX_TOTAL_TILES = 20000


@dataclass(frozen=True)
class CandidateGeometry:
    """One fixed tile plan to measure. ``enabled=False`` is the full-frame baseline."""

    enabled: bool
    geometry_mode: str
    slice_width: int
    slice_height: int
    overlap: float
    object_tile_fraction: float
    trained_body_px: float
    label: str

    def slice_params(self) -> dict:
        """The ``SLICE_*`` params consumed by ``config._slice_config_from_params``."""
        params: dict = {
            "SLICE_ENABLED": bool(self.enabled),
            "SLICE_GEOMETRY_MODE": self.geometry_mode,
            "SLICE_OVERLAP": float(self.overlap),
            "SLICE_OBJECT_TILE_FRACTION": float(self.object_tile_fraction),
            "SLICE_TRAINED_BODY_PX": float(self.trained_body_px),
        }
        if self.geometry_mode == "custom":
            params["SLICE_WIDTH"] = int(self.slice_width)
            params["SLICE_HEIGHT"] = int(self.slice_height)
        return params


@dataclass(frozen=True)
class GridWorkEstimate:
    candidate: CandidateGeometry
    tiles_per_frame: int
    total_tiles: int
    failed_reason: str = ""


def build_candidate_grid(
    training_geometry: dict,
    *,
    custom: tuple[int, int] | None = None,
    fraction_steps: tuple[float, ...] = FRACTION_STEPS,
    overlaps: tuple[float, ...] = OVERLAP_STEPS,
) -> list[CandidateGeometry]:
    """Full frame + training geometry + nearby fractions x overlaps (+ custom)."""
    base_fraction = float(training_geometry.get("object_tile_fraction") or 0.15)
    base_overlap = float(training_geometry.get("overlap") or 0.2)
    mode = str(training_geometry.get("geometry_mode") or "auto_object")
    if mode not in {"auto_model", "auto_object", "custom"}:
        mode = "auto_object"
    body_px = float(training_geometry.get("reference_body_px") or 0.0)
    out = [
        CandidateGeometry(
            enabled=False,
            geometry_mode=mode,
            slice_width=0,
            slice_height=0,
            overlap=base_overlap,
            object_tile_fraction=base_fraction,
            trained_body_px=body_px,
            label="Full frame (no SAHI)",
        )
    ]
    seen: set[tuple] = set()

    def _add(fraction: float, overlap: float, label: str) -> None:
        fraction = max(0.01, min(0.9, round(float(fraction), 4)))
        overlap = max(0.0, min(0.9, round(float(overlap), 4)))
        key = (mode, fraction, overlap)
        if key in seen:
            return
        seen.add(key)
        out.append(
            CandidateGeometry(
                enabled=True,
                geometry_mode=mode,
                slice_width=0,
                slice_height=0,
                overlap=overlap,
                object_tile_fraction=fraction,
                trained_body_px=body_px,
                label=label,
            )
        )

    _add(base_fraction, base_overlap, "Training geometry")
    for step in fraction_steps:
        for overlap in overlaps:
            _add(
                base_fraction * float(step),
                overlap,
                f"fraction x{step:g}, overlap {overlap:g}",
            )
    if custom is not None:
        out.append(
            CandidateGeometry(
                enabled=True,
                geometry_mode="custom",
                slice_width=int(custom[0]),
                slice_height=int(custom[1]),
                overlap=base_overlap,
                object_tile_fraction=base_fraction,
                trained_body_px=body_px,
                label=f"Custom {int(custom[0])}x{int(custom[1])}",
            )
        )
    return out


def estimate_grid_work(
    candidates: list[CandidateGeometry],
    *,
    frame_hw: tuple[int, int],
    imgsz: int,
    frames: int,
    max_total_tiles: int = DEFAULT_MAX_TOTAL_TILES,
) -> list[GridWorkEstimate]:
    """Tiles/frame per candidate; over-budget or unplannable candidates are FLAGGED.

    A silently omitted candidate looks to the user like a measured,
    unremarkable one -- so failures get a row with a reason instead.
    """
    from hydra_suite.core.inference.config import _slice_config_from_params
    from hydra_suite.core.inference.stages.slicing import plan_slices

    frames = max(0, int(frames))
    estimates: list[GridWorkEstimate] = []
    running = 0
    for candidate in candidates:
        if not candidate.enabled:
            tiles, reason = 1, ""
        else:
            try:
                params = candidate.slice_params()
                slice_cfg = _slice_config_from_params(
                    params, "SLICE_", reference_body_px=candidate.trained_body_px
                )
                plan = plan_slices(
                    frame_hw,
                    slice_cfg,
                    int(imgsz),
                    None,
                    float(candidate.trained_body_px),
                )
                tiles, reason = int(plan.jobs_per_frame), ""
            except Exception as exc:  # ValueError from MAX_TILES_PER_FRAME, etc.
                tiles, reason = 0, str(exc)
        total = tiles * frames
        running += total
        if not reason and running > max_total_tiles:
            reason = (
                f"exceeds the {max_total_tiles} tile budget "
                "(confirm a broader sweep to include it)"
            )
        estimates.append(GridWorkEstimate(candidate, tiles, total, reason))
    return estimates
