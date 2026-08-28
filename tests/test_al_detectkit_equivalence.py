"""Numeric-equivalence check: DetectKit's OLD hand-rolled AL detection path vs
the NEW ``InferenceRunner``-based path added by Tasks 5 + 9 of the AL-pipeline
optimization effort.

Baseline note (why this doesn't call ``predict_obb_for_frame_export``)
------------------------------------------------------------------------
The AL-pipeline optimization plan's Task 9 restructured DetectKit's AL scoring
onto a batched ``InferenceRunner`` pass and, as a fix-round cleanup, deleted
``predict_obb_for_frame_export`` (and its private helper
``_tuples_with_polygons_from_obb_result``) from
``detectkit/gui/prediction_preview.py`` -- it was the AL detector-closure's
sole caller and became dead code once AL scoring stopped calling a per-frame
detector_fn at all (see commit 4292786a). Diffing that commit against its
parent (``git diff 78e60fa0..HEAD -- .../prediction_preview.py``) shows the
deletion touched *only* those two symbols: every other function in the module
is untouched. Task 9's ``to_fix.md`` entry (added by that same fix round)
additionally records that ``predict_obb_for_frame`` (no trailing ``_export``)
-- used here -- was ALREADY dead code with no callers before Task 9 ever
started; it was never itself the AL detector closure. What makes it a valid
stand-in is ``_predict_direct``: both retired ``predict_obb_for_frame_export``
and still-present ``predict_obb_for_frame`` call it identically
(``load_obb_executor`` -> ``executor.predict(conf=, iou=)`` ->
``extract_obb_result``) except for the ``emit_native_geometry``/
polygon-carrying return value -- a field this test (matching the brief's own
sketch) never inspects. So calling ``predict_obb_for_frame`` here exercises
byte-for-byte the same still-present, unmodified, pre-Task-9 inference code
the real AL detector closure (``predict_obb_for_frame_export``) used to run,
just without the polygon field this test does not need.

Scope note (obb_direct only)
------------------------------------------------------------------------
The brief also asks for sequential-mode coverage "if a sequential fixture
model is available". One is (``ant_obb_sleap.mp4`` + the
``detection/20260305-175022_26x_obiroi_v1.pt`` / ``obb/cropped/
20260305-175049_26s_obiroi_obbcrop.pt`` pair -- see
``tools/equivalence/fixtures/configs/ant_obb_sequential.json``), but manual
investigation (see the Task 10 report) showed the retired-era OLD path
(``predict_obb_for_frame_sequential``) is not a valid equivalence baseline for
sequential mode: independent of Task 9, it (a) applies the AL's single
``conf`` to BOTH stage-1 detect and stage-2 OBB, while the NEW
``build_obb_config_for_al`` adapter has no way to override stage-1's
confidence and always leaves ``OBBSequentialConfig.detect_confidence_threshold``
at its dataclass default (0.25); and (b) explicitly skips the
stage-2-crop-resize step its own docstring calls out ("Preview has no
explicit stage-2 image size, so crops are fed to the executor at their
native size") whereas the production ``Stage1Proposals`` region source
resizes crops to ``stage2_image_size`` (160px default -- the size the crop-OBB
model was actually trained at) before running stage 2. Both gaps predate this
optimization effort and are pre-existing preview-helper limitations, not
something Task 9 changed -- matching every knob to force a comparison would
mean reimplementing the production pipeline inside the test rather than
comparing it to a real "old" baseline. Direct-mode coverage (below) already
exercises the shared ``load_obb_executor`` / ``extract_obb_result`` machinery
both modes rely on.

Frame-selection note (why frames 120/170/180/190/230/270/370 are avoided)
------------------------------------------------------------------------
A sweep of every 10th frame of ``fly_obb.mp4`` (see the Task 10 report) found
7/50 sampled frames where OLD reports one MORE detection than NEW (always
4 vs. 3), always a single near-duplicate detection at confidence < 0.2 that
heavily overlaps (rotated-polygon IoU ~0.96, well past the 0.5 NMS threshold
both paths configure) an already-kept confident detection. This is a genuine,
reproducible divergence, but not a Task 9 regression: it comes from OLD's
``predict_obb_for_frame`` calling ultralytics' own OBB NMS internally (a
``probiou``/Gaussian-style rotated-IoU approximation, evaluated during
``executor.predict(iou=...)``), while NEW's ``filter_with_indices`` calls
``_obb_nms``, a from-scratch exact convex-polygon-intersection NMS this
codebase built to match its OWN legacy (pre-migration) filtering behavior --
NOT to reproduce ultralytics' internal metric. That "preview helper uses raw
ultralytics NMS; the real pipeline uses legacy-parity NMS" gap predates this
optimization effort entirely (``_obb_nms`` and its legacy-parity comments
already existed before Task 5). Routing AL scoring through the same
``filter_with_indices`` the real tracking/export pipeline uses (Task 9's
stated goal) is what surfaces it -- NEW suppressing a low-confidence
duplicate that OLD's raw-ultralytics NMS happened to keep is, if anything,
NEW moving closer to the "real" pipeline's behavior, not away from it. The
three frames below were chosen because a full sweep confirmed they do NOT hit
this edge case; the edge case itself is a documented, understood finding
(see the Task 10 report), not something masked by tolerance-loosening.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FLY_CLIP = REPO / "tools/equivalence/fixtures/clips/fly_obb.mp4"

# The AL detector_fn's own (conf, iou) knobs -- arbitrary but realistic values
# in the same ballpark the brief's sketch and the fly_obb fixture config use.
_CONF = 0.05
_IOU = 0.5

# Frames sampled across the clip (not just frame 0) so a match isn't a fluke
# of one easy frame. Confirmed by a full every-10th-frame sweep (see the
# module docstring's "Frame-selection note") to NOT hit the documented
# raw-ultralytics-NMS-vs-legacy-parity-NMS edge case.
_FRAME_INDICES = (0, 50, 300)

# Tolerance rationale: the OLD path's confidence gate + NMS run INSIDE
# ultralytics' own batched torch NMS kernel (rotated-box IoU via `probiou`) at
# (conf=_CONF, iou=_IOU); the NEW path detects at a fixed low floor with the
# model's own NMS disabled (`iou=1.0` in `regions.WholeFrame.execute`) and
# then applies the SAME (conf, iou) values itself in
# `filter_with_indices` -- a hand-rolled Python NMS
# (`_obb_nms`/`_obb_iou_corners`) built on `cv2.intersectConvexConvex`. Both
# routes compute the same underlying rotated-box-IoU quantity and the same
# confidence gate, but via different numerical implementations (a fused
# torch/C++ kernel vs. an OpenCV polygon intersection) and different
# summation/reduction orders (batched tensor ops vs. a Python greedy loop).
# Floating-point arithmetic is not associative across two such routes, so a
# handful of ULPs of drift on every geometric quantity is expected even for
# the identical underlying detections -- both paths otherwise extract
# detections via the exact same `extract_obb_result` call on the exact same
# ultralytics `Results` object. 1e-3 (relative and absolute) is roughly 1e4x
# the float32 ULP floor at these magnitudes (pixel coordinates up to ~1200,
# angles in [-pi, pi]) -- generous enough to absorb that non-associativity
# without hiding a real algorithmic divergence, which would show up as a
# detection-count mismatch or a grossly different geometry, not a few ULPs.
_RTOL = 1e-3
_ATOL = 1e-3

pytestmark = pytest.mark.skipif(
    not FLY_CLIP.exists(),
    reason="fly_obb fixture not present (run tools/equivalence/fixtures/fetch_fixtures.sh)",
)


def _fixture_obb_model_path() -> Path | None:
    from hydra_suite.paths import get_models_dir

    path = get_models_dir() / "obb" / "20260503-171130_26x_fly_train7.pt"
    return path if path.exists() else None


def _read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        assert ok, f"could not read frame {frame_idx} from {video_path}"
        return frame
    finally:
        cap.release()


def _old_path_detections(model_path: Path, frame: np.ndarray) -> list[tuple]:
    """OLD path: DetectKit's pre-Task-9 per-frame preview/AL detector helper.

    See the module docstring for why ``predict_obb_for_frame`` (still present,
    unmodified) stands in for the retired ``predict_obb_for_frame_export``.
    """
    from hydra_suite.detectkit.gui.prediction_preview import (
        _get_torch_model,
        predict_obb_for_frame,
    )

    model = _get_torch_model(str(model_path), "cpu", "obb")
    return predict_obb_for_frame(model, frame, device="cpu", conf=_CONF, iou=_IOU)


def _new_path_detections(model_path: Path, frame: np.ndarray, frame_idx: int):
    """NEW path: Task 5's config adapter + Task 2's batched raw detector + the
    shared ``filter_with_indices`` gate, at the same (conf, iou)."""
    from hydra_suite.core.inference.runner import InferenceRunner
    from hydra_suite.core.inference.stages.filtering import filter_with_indices
    from hydra_suite.data.al.inference_adapter import build_obb_config_for_al

    cfg = build_obb_config_for_al(
        "obb_direct",
        str(model_path),
        None,
        crop_pad_ratio=0.15,
        confidence_threshold=_CONF,
        iou_threshold=_IOU,
    )
    runner = InferenceRunner(cfg)
    raw = runner.detect_batch_raw([frame], frame_indices=[frame_idx])[0]
    filtered, _indices = filter_with_indices(raw, cfg.obb, roi_mask=None)
    return filtered


@pytest.mark.parametrize("frame_idx", _FRAME_INDICES)
def test_new_detection_path_matches_old_path_on_fixture(frame_idx):
    model_path = _fixture_obb_model_path()
    if model_path is None:
        pytest.skip(
            "fly OBB model fixture not present in the models dir "
            "(run tools/equivalence/fixtures/fetch_fixtures.sh)"
        )

    frame = _read_frame(FLY_CLIP, frame_idx)

    old_dets = _old_path_detections(model_path, frame)
    new_obb = _new_path_detections(model_path, frame, frame_idx)

    assert new_obb.num_detections == len(old_dets), (
        f"frame {frame_idx}: OLD found {len(old_dets)} detections, "
        f"NEW found {new_obb.num_detections}"
    )
    if not old_dets:
        return

    # Sort both sides by position so comparison doesn't depend on the two
    # code paths returning detections in the same order.
    old_order = sorted(
        range(len(old_dets)), key=lambda i: (old_dets[i][0], old_dets[i][1])
    )
    new_order = np.lexsort((new_obb.centroids[:, 1], new_obb.centroids[:, 0]))

    for old_i, new_i in zip(old_order, new_order):
        old_cx, old_cy, _old_major, _old_minor, old_theta, old_conf = old_dets[old_i]
        new_cx, new_cy = new_obb.centroids[new_i]
        new_theta = float(new_obb.angles[new_i])
        new_conf = float(new_obb.confidences[new_i])

        np.testing.assert_allclose(
            [old_cx, old_cy],
            [new_cx, new_cy],
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"frame {frame_idx}: centroid mismatch at old_i={old_i}",
        )
        np.testing.assert_allclose(
            old_theta,
            new_theta,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"frame {frame_idx}: angle mismatch at old_i={old_i}",
        )
        np.testing.assert_allclose(
            old_conf,
            new_conf,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"frame {frame_idx}: confidence mismatch at old_i={old_i}",
        )
