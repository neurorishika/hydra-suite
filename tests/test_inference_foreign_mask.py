import numpy as np
import torch

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.crops import (
    apply_foreign_mask_to_crop_batch,
    extract_canonical_crops_batch,
)

_GEOM = CanonicalGeometry(canvas_wh=(64, 64), margin=1.5, aspect_ratio=1.0)


def _two_adjacent_obbs():
    # two boxes; box B overlaps box A's canonical crop region
    corners = np.array(
        [
            [[10, 10], [30, 10], [30, 30], [10, 30]],
            [[28, 10], [48, 10], [48, 30], [28, 30]],
        ],
        np.float32,
    )
    return OBBResult(
        frame_idx=0,
        centroids=np.array([[20, 20], [38, 20]], np.float32),
        angles=np.zeros(2, np.float32),
        sizes=np.full(2, 400, np.float32),
        shapes=np.ones((2, 2), np.float32),
        confidences=np.ones(2, np.float32),
        corners=corners,
        detection_ids=np.array([0, 1], np.int64),
    )


def test_foreign_mask_blacks_out_neighbor_pixels():
    frame = np.full((64, 64, 3), 200, np.uint8)
    obb = _two_adjacent_obbs()
    rt = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )
    masked = extract_canonical_crops_batch(
        [frame],
        [obb],
        _GEOM,
        rt,
        suppress_foreign=True,
        background_color=(0, 0, 0),
    )
    plain = extract_canonical_crops_batch(
        [frame],
        [obb],
        _GEOM,
        rt,
        suppress_foreign=False,
    )
    # masking must zero strictly more pixels than the unmasked crop
    assert (masked.crops[0] == 0).sum() > (plain.crops[0] == 0).sum()


def test_masking_shared_batch_is_bit_identical_and_keeps_input_unmodified():
    frame = np.full((64, 64, 3), 200, np.uint8)
    obb = _two_adjacent_obbs()
    rt = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )
    shared = extract_canonical_crops_batch([frame], [obb], _GEOM, rt)
    before = shared.crops.clone()
    reference = extract_canonical_crops_batch(
        [frame],
        [obb],
        _GEOM,
        rt,
        suppress_foreign=True,
        background_color=(0, 0, 0),
    )

    actual = apply_foreign_mask_to_crop_batch(shared, _GEOM, (0, 0, 0))

    assert torch.equal(actual.crops, reference.crops)
    assert torch.equal(shared.crops, before)
    assert np.array_equal(actual.detection_ids, reference.detection_ids)
    assert np.array_equal(actual.frame_index, reference.frame_index)
    assert np.array_equal(actual.native_sizes, reference.native_sizes)


def _three_overlapping_obbs():
    xs = np.array([20.0, 30.0, 40.0], np.float32)
    c = np.stack([xs, np.full(3, 20.0, np.float32)], 1)
    corners = np.stack([c + d for d in ([-8, -8], [8, -8], [8, 8], [-8, 8])], 1)
    return OBBResult(
        frame_idx=0,
        centroids=c,
        angles=np.zeros(3, np.float32),
        sizes=np.full(3, 256, np.float32),
        shapes=np.ones((3, 2), np.float32),
        confidences=np.ones(3, np.float32),
        corners=corners.astype(np.float32),
        detection_ids=np.array([0, 1, 2], np.int64),
    )


def test_foreign_set_masks_a_subset_crop_like_the_full_frame():
    """Cropping a subset with the full set as foreign == the full crop's rows."""
    from hydra_suite.core.inference.stages.crops import ForeignSet
    from hydra_suite.core.inference.stages.filtering import _select

    rng = np.random.default_rng(3)
    frame = rng.integers(1, 255, (64, 64, 3), dtype=np.uint8)
    rt = RuntimeContext(
        cuda_mode=False, device="cpu", use_nvdec=False, tensor_on_cuda=False
    )
    full = _three_overlapping_obbs()
    reference = extract_canonical_crops_batch(
        [frame], [full], _GEOM, rt, suppress_foreign=True
    )
    rows = np.array([1])  # a single-row subset still needs masking
    sub = _select(full, rows)
    foreign = {0: ForeignSet(corners=full.corners, self_rows=rows)}

    direct = extract_canonical_crops_batch(
        [frame], [sub], _GEOM, rt, suppress_foreign=True, foreign_by_frame=foreign
    )
    assert torch.equal(direct.crops, reference.crops[1:2])

    shared = extract_canonical_crops_batch([frame], [sub], _GEOM, rt)
    masked = apply_foreign_mask_to_crop_batch(
        shared, _GEOM, (0, 0, 0), foreign_by_frame=foreign
    )
    assert torch.equal(masked.crops, reference.crops[1:2])

    # Without the foreign set the subset crop is NOT masked like the full frame.
    alone = extract_canonical_crops_batch(
        [frame], [sub], _GEOM, rt, suppress_foreign=True
    )
    assert not torch.equal(alone.crops, reference.crops[1:2])
