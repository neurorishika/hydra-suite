from hydra_suite.core.identity.pose.vitpose.config import (
    EXPERT_DATASETS,
    HEATMAP_SIZE_WH,
    IMAGE_SIZE_WH,
    NUM_EXPERTS,
    PIXEL_STD,
    TARGET_SIGMA,
    UDP_BLUR_KERNEL,
    VARIANTS,
)


def test_variant_dims():
    assert VARIANTS["S"].embed_dim == 384
    assert VARIANTS["B"].embed_dim == 768
    assert VARIANTS["L"].embed_dim == 1024
    assert VARIANTS["H"].embed_dim == 1280
    assert VARIANTS["B"].depth == 12
    assert VARIANTS["L"].depth == 24
    assert VARIANTS["H"].depth == 32


def test_small_uses_twelve_heads_not_six():
    """ViTPose-S is 12 heads at dim 384 (head_dim 32), NOT the usual 6.
    Getting this from ViT habit rather than the config is a real trap."""
    assert VARIANTS["S"].num_heads == 12
    assert VARIANTS["S"].embed_dim // VARIANTS["S"].num_heads == 32


def test_part_features():
    assert [VARIANTS[k].part_features for k in "SBLH"] == [96, 192, 256, 320]


def test_frozen_constants():
    assert IMAGE_SIZE_WH == (192, 256)  # (w, h) - configs write [192, 256]
    assert HEATMAP_SIZE_WH == (48, 64)  # (w, h)
    assert PIXEL_STD == 200.0
    assert UDP_BLUR_KERNEL == 11
    assert TARGET_SIGMA == 2.0


def test_blur_kernel_matches_training_sigma():
    """OpenCV sigma=0 derives sigma from kernel: 0.3*((k-1)*0.5 - 1) + 0.8.
    For k=11 that is exactly 2.0 == TARGET_SIGMA. HF hardcodes 0.8 instead,
    which does not track kernel size; we deliberately do not follow HF."""
    k = UDP_BLUR_KERNEL
    derived = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    assert abs(derived - TARGET_SIGMA) < 1e-9


def test_expert_dataset_order():
    assert NUM_EXPERTS == 6
    assert EXPERT_DATASETS == (
        "COCO",
        "AiC",
        "MPII",
        "AP-10K",
        "APT-36K",
        "COCO-WholeBody",
    )
