"""Regression test for the sequential-OBB CUDA over-detection bug.

Root cause: ``YOLOOBBDetector`` built its ``DirectPyTorchCUDAOBBExecutor``
(used only on native CUDA to force ``auto=False`` square letterboxing) at the
*checkpoint's own* embedded/export imgsz, even in ``sequential`` OBB mode
where ``self.model`` is actually the stage-2 *crop* classifier invoked on
``YOLO_SEQ_STAGE2_IMGSZ``-sized crops. That imgsz mismatch fed
already-resized crops into an executor configured for a different square
size, systematically inflating confidence scores on CUDA relative to
CPU/MPS and producing ~10x more detections.

This test exercises ``YOLOOBBDetector._resolve_direct_cuda_obb_imgsz`` in
isolation (no GPU, no model loading) to confirm it resolves the *stage-2
crop* imgsz in sequential mode, and falls back to the checkpoint's own
imgsz in direct mode (unaffected code path).
"""

from types import SimpleNamespace

from hydra_suite.core.detectors.yolo_detector import YOLOOBBDetector


def _make_bare_detector(params, obb_mode):
    """Construct a YOLOOBBDetector instance without running __init__.

    __init__ loads real model files, which we don't want in a unit test.
    We only need ``params`` and ``obb_mode`` for ``_resolve_direct_cuda_obb_imgsz``
    and its dependency ``_current_obb_mode``.
    """
    detector = YOLOOBBDetector.__new__(YOLOOBBDetector)
    detector.params = params
    detector.obb_mode = obb_mode
    return detector


def test_sequential_mode_uses_stage2_crop_imgsz_not_checkpoint_imgsz():
    """Sequential mode must resolve the configured stage-2 crop imgsz,
    ignoring whatever imgsz the crop-classifier checkpoint was exported at.
    """
    params = {"YOLO_OBB_MODE": "sequential", "YOLO_SEQ_STAGE2_IMGSZ": 128}
    detector = _make_bare_detector(params, "sequential")

    # Checkpoint metadata claims a different imgsz (160) than the configured
    # stage-2 crop size (128) — this is the exact real-world mismatch that
    # caused the bug (crop model exported/trained-metadata at 160, sequential
    # pipeline actually resizing crops to 128 before stage-2 inference).
    fake_model = SimpleNamespace(overrides={"imgsz": 160})

    resolved = detector._resolve_direct_cuda_obb_imgsz(fake_model)

    assert resolved == 128, (
        "Sequential mode must build the direct CUDA OBB executor at the "
        "actual stage-2 crop imgsz, not the checkpoint's own imgsz — using "
        "the wrong imgsz here reproduces the CUDA over-detection bug."
    )


def test_sequential_mode_falls_back_to_checkpoint_imgsz_if_stage2_unset():
    """If YOLO_SEQ_STAGE2_IMGSZ isn't configured (falsy/absent), fall back
    to the checkpoint's own imgsz rather than crashing or resolving to 0."""
    params = {"YOLO_OBB_MODE": "sequential"}
    detector = _make_bare_detector(params, "sequential")
    fake_model = SimpleNamespace(overrides={"imgsz": 640})

    resolved = detector._resolve_direct_cuda_obb_imgsz(fake_model)

    assert resolved == 640


def test_direct_mode_still_uses_checkpoint_imgsz():
    """Direct (full-frame) OBB mode is unaffected by this fix: it must keep
    resolving imgsz from the checkpoint's own metadata, as before."""
    params = {"YOLO_OBB_MODE": "direct", "YOLO_SEQ_STAGE2_IMGSZ": 128}
    detector = _make_bare_detector(params, "direct")
    fake_model = SimpleNamespace(overrides={"imgsz": 1024})

    resolved = detector._resolve_direct_cuda_obb_imgsz(fake_model)

    assert resolved == 1024, (
        "Direct mode's OBB model runs on full frames at the checkpoint's "
        "own imgsz and must not be redirected to the stage-2 crop size."
    )
