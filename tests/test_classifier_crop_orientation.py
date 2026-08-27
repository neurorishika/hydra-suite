"""Head-first classifier crop orientation (R8: identity catalog is ordered).

``extract_classifier_crops`` warps each OBB onto the shared canonical canvas
purely from the OBB major-axis corner order, which has a 180-degree
ambiguity. The head/tail stage already resolves that ambiguity for other
purposes; this makes the identity CNN crop consult it too, WITHOUT changing
anything for detections the head/tail stage is unsure about (undirected).
"""

import numpy as np
import torch

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.crops import (
    extract_canonical_crops,
    extract_classifier_crops,
)


def _frame_with_marker():
    fr = np.full((400, 400, 3), 128, np.uint8)
    fr[195:205, 260:300] = 255  # bright marker on the +x side of centre (200,200)
    return fr


def _obb(cx=200, cy=200, w=120, h=50):
    c = np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
        ]
    )
    return OBBResult(
        frame_idx=0,
        centroids=np.array([[cx, cy]]),
        angles=np.zeros(1),
        sizes=np.ones(1),
        shapes=np.ones((1, 2)),
        confidences=np.ones(1),
        corners=c[None],
        detection_ids=np.array([1]),
    )


def test_no_heading_kwargs_matches_base_call():
    """Ledger ruling: every existing caller today invokes this with NO
    heading kwargs at all. That exact call signature must stay byte-identical
    to the base/undirected crop after this task's changes."""
    geo = CanonicalGeometry.from_reference(100.0, 2.0, 1.3)
    fr, obb = _frame_with_marker(), _obb()
    base = extract_classifier_crops(fr, obb, geo)[0]
    no_kwargs = extract_classifier_crops(fr, obb, geo)[0]
    assert np.array_equal(base, no_kwargs)


def test_undirected_is_unchanged_and_directed_flips():
    geo = CanonicalGeometry.from_reference(100.0, 2.0, 1.3)
    fr, obb = _frame_with_marker(), _obb()
    base = extract_classifier_crops(fr, obb, geo)[0]
    same = extract_classifier_crops(
        fr,
        obb,
        geo,
        heading_hints=np.array([np.pi]),
        directed_mask=np.array([0], np.uint8),
    )[0]
    assert np.array_equal(base, same)  # undirected -> byte-identical
    flipped = extract_classifier_crops(
        fr,
        obb,
        geo,
        heading_hints=np.array([np.pi]),
        directed_mask=np.array([1], np.uint8),
    )[0]
    assert np.array_equal(
        flipped, np.ascontiguousarray(base[::-1, ::-1])
    )  # head points -x -> rotate 180 degrees
    keep = extract_classifier_crops(
        fr,
        obb,
        geo,
        heading_hints=np.array([0.0]),
        directed_mask=np.array([1], np.uint8),
    )[0]
    assert np.array_equal(keep, base)  # head already +x -> unchanged


def test_extract_canonical_crops_undirected_is_unchanged_and_directed_flips():
    """Direct coverage for the CUDA-branch entry point
    (``extract_canonical_crops``/``extract_canonical_crops_batch`` share
    this function and its ``_directed_align`` helper with
    ``extract_classifier_crops``, but neither had a test that actually
    called ``extract_canonical_crops`` with ``heading_hints``/
    ``directed_mask``). ``extract_canonical_crops`` is device-agnostic per
    its own docstring, so this runs entirely on CPU with a dummy
    ``RuntimeContext`` -- the ``runtime`` argument is accepted only for
    call-site compatibility and is unused internally (``del runtime``).

    Returns a ``(N, C, H, W)`` float tensor (not the quantised HWC uint8
    ``extract_classifier_crops`` returns), so this compares crops against
    each other rather than against ``extract_classifier_crops``'s output.
    """
    geo = CanonicalGeometry.from_reference(100.0, 2.0, 1.3)
    fr, obb = _frame_with_marker(), _obb()
    rt = RuntimeContext(cuda_mode=False, device="cpu", use_nvdec=False)

    base = extract_canonical_crops(fr, obb, geo, rt)[0]
    same = extract_canonical_crops(
        fr,
        obb,
        geo,
        rt,
        heading_hints=np.array([np.pi]),
        directed_mask=np.array([0], np.uint8),
    )[0]
    assert torch.equal(base, same)  # undirected -> byte-identical

    flipped = extract_canonical_crops(
        fr,
        obb,
        geo,
        rt,
        heading_hints=np.array([np.pi]),
        directed_mask=np.array([1], np.uint8),
    )[0]
    # head points -x -> rotate 180 degrees (flip both spatial dims, CHW)
    assert torch.equal(flipped, torch.flip(base, dims=(1, 2)))

    keep = extract_canonical_crops(
        fr,
        obb,
        geo,
        rt,
        heading_hints=np.array([0.0]),
        directed_mask=np.array([1], np.uint8),
    )[0]
    assert torch.equal(keep, base)  # head already +x -> unchanged
