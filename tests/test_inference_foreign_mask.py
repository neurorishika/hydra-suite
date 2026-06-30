import numpy as np
import torch

from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.crops import extract_canonical_crops_batch


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
        default_runtime="cpu",
    )
    masked = extract_canonical_crops_batch(
        [frame],
        [obb],
        1.0,
        1.5,
        rt,
        suppress_foreign=True,
        background_color=(0, 0, 0),
    )
    plain = extract_canonical_crops_batch(
        [frame],
        [obb],
        1.0,
        1.5,
        rt,
        suppress_foreign=False,
    )
    # masking must zero strictly more pixels than the unmasked crop
    assert (masked.crops[0] == 0).sum() > (plain.crops[0] == 0).sum()
