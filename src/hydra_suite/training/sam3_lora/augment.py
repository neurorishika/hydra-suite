"""Train-time tile augmentation for SAM3 LoRA finetuning.

Pure and Qt-free; numpy only at module scope (cv2/torch lazily) because the
`hydra-sam3` sidecar imports this. Uses the YOLO-side `AugmentationProfile`
vocabulary -- see docs/superpowers/specs/2026-10-05-sam3-lora-augmentation-design.md
for the SAM3 semantics of each field.

Image and polygons are transformed TOGETHER in tile-pixel, edge-convention
coordinates (the convention `datapoints._scale_polygons_to_res` assumes),
before the RES resize and Meta's NormalizeAPI. Only the training loop builds
an augmenter; validation, the autobatch probe and detection-quality never do.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hydra_suite.training.contracts import AugmentationProfile

logger = logging.getLogger(__name__)

Instance = tuple[np.ndarray, bool]
TileAugmenter = Callable[[np.ndarray, list, int], tuple[np.ndarray, list]]

AUGMENTATION_STAMP_FILENAME = "hydra_sam3_augmentation.json"
# Decorrelates per-tile draws from the epoch-shuffle RNG seeded by the same
# epoch seed.
_RNG_SALT = 0x5A3A06
# Ultralytics' letterbox gray. A constant fill, never BORDER_REFLECT: a
# reflected border pastes UNLABELED mirrored animals into an exhaustive query.
_FILL_BGR = (114, 114, 114)

_PROBABILITY_FIELDS = ("fliplr", "flipud", "rot90", "decode_color_sim", "resample_sim")
_UNIT_FIELDS = ("brightness", "contrast", "saturation")
_NUMERIC_OPS = _PROBABILITY_FIELDS + _UNIT_FIELDS + ("rotate", "hue")


def recommended_sam3_augmentation() -> AugmentationProfile:
    """The GUI's fresh-session default: label-exact geometry + mild photometrics.

    `hue` stays 0 because colour can BE the concept (painted tags); `rotate`
    stays 0 because it needs border fill and fragment re-flagging.
    """
    return AugmentationProfile(
        enabled=True,
        fliplr=0.5,
        flipud=0.5,
        rot90=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    )


def validate_sam3_augmentation(profile: AugmentationProfile) -> list[str]:
    """Range + supported-field errors; empty when valid."""
    errors: list[str] = []

    def _check(name: str, lo: float, hi: float) -> None:
        try:
            value = float(getattr(profile, name))
        except (TypeError, ValueError):
            errors.append(f"augmentation.{name} must be a number")
            return
        if not (lo <= value <= hi):  # also rejects NaN
            errors.append(
                f"augmentation.{name} must be in [{lo:g}, {hi:g}], got {value!r}"
            )

    for name in _PROBABILITY_FIELDS + _UNIT_FIELDS:
        _check(name, 0.0, 1.0)
    _check("rotate", 0.0, 180.0)
    _check("hue", 0.0, 0.5)
    if profile.canonical_aug:
        errors.append("augmentation.canonical_aug is not supported for SAM3 training")
    if profile.label_expansion:
        errors.append("augmentation.label_expansion is not supported for SAM3 training")
    if profile.args:
        errors.append(
            "augmentation.args (Ultralytics passthrough) is not supported for SAM3 training"
        )
    return errors


def active_ops(profile: AugmentationProfile | None) -> dict[str, Any]:
    """The ops a profile actually applies (empty when disabled)."""
    if profile is None or not profile.enabled:
        return {}
    ops: dict[str, Any] = {
        name: float(getattr(profile, name))
        for name in _NUMERIC_OPS
        if float(getattr(profile, name)) > 0.0
    }
    if profile.monochrome:
        ops["monochrome"] = True
    return ops


def is_active(profile: AugmentationProfile | None) -> bool:
    return bool(active_ops(profile))


def tile_rng(epoch_seed: int, image_id: int) -> np.random.Generator:
    """Per-(epoch, tile) generator: independent of shuffle/grouping order."""
    return np.random.default_rng(
        [int(epoch_seed) % 2**32, int(image_id) % 2**32, _RNG_SALT]
    )


def _polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0].astype(np.float64)
    y = poly[:, 1].astype(np.float64)
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _map(instances: list[Instance], fn) -> list[Instance]:
    return [(fn(poly).astype(np.float32), crowd) for poly, crowd in instances]


def _rotate(
    img: np.ndarray, instances: list[Instance], angle: float, min_area_ratio: float
) -> tuple[np.ndarray, list[Instance]]:
    import cv2

    from hydra_suite.utils.slice_geometry import clip_polygon_to_tile

    h, w = img.shape[:2]
    # cv2 maps pixel INDICES; rotating about the index centre equals rotating
    # about (w/2, h/2) in edge coordinates. Polygons: edge -> index -> edge.
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
    img = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_FILL_BGR,
    )
    out: list[Instance] = []
    for poly, crowd in instances:
        idx = poly.astype(np.float64) - 0.5
        rotated = idx @ matrix[:, :2].T + matrix[:, 2] + 0.5
        before = _polygon_area(rotated)
        clipped = clip_polygon_to_tile(rotated, (0, 0, w, h))
        if clipped is None or before <= 0.0:
            continue  # rotated out of the image: genuinely absent now
        retained = _polygon_area(clipped) / before
        if retained <= 0.0:
            continue
        # Reference area is the TILE polygon (dataset_build's full-frame area
        # is unavailable here). Sub-floor -> is_crowd, which
        # `datapoints.select_output_objects` excludes AND uses to mark the
        # positive query non-exhaustive. Already-crowd stays crowd.
        out.append(
            (
                np.asarray(clipped, dtype=np.float32),
                bool(crowd or retained < float(min_area_ratio)),
            )
        )
    return img, out


def _photometric(rgb: np.ndarray, p: AugmentationProfile, rng) -> np.ndarray:
    """Same semantics as `runner._apply_tiny_augmentation`, explicit RNG."""
    import cv2

    if p.brightness > 0:
        factor = rng.uniform(max(0.0, 1.0 - p.brightness), 1.0 + p.brightness)
        rgb = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if p.contrast > 0:
        factor = rng.uniform(max(0.0, 1.0 - p.contrast), 1.0 + p.contrast)
        mean = rgb.mean(axis=(0, 1), keepdims=True)
        rgb = np.clip((rgb.astype(np.float32) - mean) * factor + mean, 0, 255).astype(
            np.uint8
        )
    if p.saturation > 0 or p.hue > 0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        if p.saturation > 0:
            factor = rng.uniform(max(0.0, 1.0 - p.saturation), 1.0 + p.saturation)
            hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        if p.hue > 0:
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-p.hue, p.hue) * 179.0) % 180.0
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if p.monochrome:
        rgb = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if p.decode_color_sim > 0 or p.resample_sim > 0:
        from hydra_suite.training.augmentation import (
            simulate_decode_color,
            simulate_resample,
        )

        if p.decode_color_sim > 0:
            rgb = simulate_decode_color(rgb, float(p.decode_color_sim), rng)
        if p.resample_sim > 0:
            rgb = simulate_resample(rgb, float(p.resample_sim), rng)
    return rgb


_PHOTOMETRIC = (
    "brightness",
    "contrast",
    "saturation",
    "hue",
    "decode_color_sim",
    "resample_sim",
)


def augment_tile(
    tile_bgr: np.ndarray,
    instances: list[Instance],
    profile: AugmentationProfile,
    rng: np.random.Generator,
    *,
    min_area_ratio: float,
) -> tuple[np.ndarray, list[Instance]]:
    """Apply `profile` to one BGR tile and its edge-convention polygons."""
    import cv2

    img = tile_bgr
    polys: list[Instance] = [
        (np.asarray(poly, dtype=np.float32).reshape(-1, 2), bool(crowd))
        for poly, crowd in instances
    ]
    h, w = img.shape[:2]
    if profile.flipud > 0 and rng.random() < profile.flipud:
        img = cv2.flip(img, 0)
        polys = _map(polys, lambda p: np.column_stack([p[:, 0], h - p[:, 1]]))
    if profile.fliplr > 0 and rng.random() < profile.fliplr:
        img = cv2.flip(img, 1)
        polys = _map(polys, lambda p: np.column_stack([w - p[:, 0], p[:, 1]]))
    if profile.rot90 > 0 and rng.random() < profile.rot90:
        h, w = img.shape[:2]
        if rng.random() < 0.5:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            polys = _map(polys, lambda p: np.column_stack([h - p[:, 1], p[:, 0]]))
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            polys = _map(polys, lambda p: np.column_stack([p[:, 1], w - p[:, 0]]))
    if profile.rotate > 0:
        angle = float(rng.uniform(-profile.rotate, profile.rotate))
        if angle != 0.0 and math.isfinite(angle):
            img, polys = _rotate(img, polys, angle, min_area_ratio)
    if profile.monochrome or any(getattr(profile, n) > 0 for n in _PHOTOMETRIC):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = _photometric(rgb, profile, rng)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(img), polys


def make_tile_augmenter(
    profile: AugmentationProfile | None, *, epoch_seed: int, min_area_ratio: float
) -> TileAugmenter | None:
    """`None` for a disabled/no-op profile, so that arm is literally today's path."""
    if not is_active(profile):
        return None

    def _augment(tile_bgr, instances, image_id):
        return augment_tile(
            tile_bgr,
            instances,
            profile,
            tile_rng(epoch_seed, image_id),
            min_area_ratio=min_area_ratio,
        )

    return _augment


def write_sam3_augmentation_stamp(
    run_dir: str | Path, profile: AugmentationProfile | None
) -> Path | None:
    """Requested-vs-applied record, written on BOTH arms (scale-grouping shape)."""
    profile = profile or AugmentationProfile(enabled=False)
    ops = active_ops(profile)
    reason = "" if ops else ("disabled" if not profile.enabled else "no active ops")
    payload = {
        "requested": asdict(profile),
        "applied": {"augmentation": bool(ops), "ops": ops, "reason": reason},
    }
    try:
        directory = Path(run_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / AUGMENTATION_STAMP_FILENAME
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
        return target
    except OSError:
        logger.warning(
            "Failed to write SAM3 augmentation stamp to %s; the run continues, "
            "but its published sidecar will not record what augmentation ran.",
            Path(run_dir) / AUGMENTATION_STAMP_FILENAME,
        )
        return None


def read_sam3_augmentation_stamp(run_dir: str | Path) -> dict[str, Any] | None:
    try:
        data = json.loads(
            (Path(run_dir) / AUGMENTATION_STAMP_FILENAME).read_text(encoding="utf-8")
        )
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None
