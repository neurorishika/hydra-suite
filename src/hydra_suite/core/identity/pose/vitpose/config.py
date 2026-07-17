"""Variant table and constants that must not drift.

Values transcribed from upstream ViTPose configs
(configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/).
"""

from __future__ import annotations

from dataclasses import dataclass

IMAGE_SIZE_WH: tuple[int, int] = (192, 256)
HEATMAP_SIZE_WH: tuple[int, int] = (48, 64)
PIXEL_STD: float = 200.0
PADDING_FACTOR: float = 1.25
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

UDP_BLUR_KERNEL: int = 11
TARGET_SIGMA: float = 2.0

NUM_EXPERTS: int = 6
EXPERT_DATASETS: tuple[str, ...] = (
    "COCO",
    "AiC",
    "MPII",
    "AP-10K",
    "APT-36K",
    "COCO-WholeBody",
)

PATCH_SIZE: int = 16
PATCH_PADDING: int = 2


@dataclass(frozen=True)
class ViTPoseVariant:
    embed_dim: int
    depth: int
    num_heads: int
    part_features: int
    drop_path_rate: float
    layer_decay: float


VARIANTS: dict[str, ViTPoseVariant] = {
    "S": ViTPoseVariant(384, 12, 12, 96, 0.10, 0.80),
    "B": ViTPoseVariant(768, 12, 12, 192, 0.30, 0.75),
    "L": ViTPoseVariant(1024, 24, 16, 256, 0.50, 0.80),
    "H": ViTPoseVariant(1280, 32, 16, 320, 0.55, 0.85),
}
