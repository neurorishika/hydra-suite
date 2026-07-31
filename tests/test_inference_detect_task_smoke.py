"""Real detect-task YOLO run through the production InferenceRunner path on each
available device. NOT mocked. Synthetic high-contrast image so detections fire."""

import numpy as np
import pytest

pytest.importorskip("ultralytics")
import torch


def _available_devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def _synthetic_frame():
    # yolo11n.pt is COCO-pretrained: solid dark rectangles on a white field
    # score essentially zero confidence (verified: <0.002 with a real model,
    # even at conf=0.001) because they don't resemble any COCO class. Random
    # RGB texture in the same blob regions is scored far higher by the
    # backbone (verified: >0.02 reliably) while still being a synthetic,
    # non-photographic image with no external asset dependency. Seeded for a
    # deterministic, reproducible test.
    rng = np.random.default_rng(3)
    img = np.full((640, 640, 3), 255, np.uint8)
    for x, y in [(120, 120), (400, 300), (250, 480)]:
        img[y : y + 90, x : x + 60] = rng.integers(
            0, 255, size=(90, 60, 3), dtype=np.uint8
        )
    return img


@pytest.mark.parametrize("device", _available_devices())
def test_detect_task_direct_runs_end_to_end(device):
    from ultralytics import YOLO

    from hydra_suite.core.inference.config import build_obb_only_config
    from hydra_suite.core.inference.runner import InferenceRunner

    try:
        YOLO("yolo11n.pt")  # ensure the real detect checkpoint is cached
    except Exception as exc:  # network/download unavailable
        pytest.skip(f"yolo11n.pt unavailable: {exc}")

    cfg = build_obb_only_config("yolo11n.pt", compute_runtime=device, mode="direct")
    cfg.obb.direct.model_task = "detect"
    # 0.02: empirically verified to reliably clear the max/second-highest
    # confidence (~0.11 / ~0.026) the seeded synthetic frame produces on
    # yolo11n.pt across cpu and mps; a fresh non-COCO image scores far lower
    # than a typical production confidence_threshold.
    cfg.obb.direct.confidence_threshold = 0.02
    cfg.obb.confidence_threshold = 0.02

    runner = InferenceRunner(cfg)
    try:
        results = runner.detect_batch([_synthetic_frame()], frame_indices=[0])
    finally:
        runner.close()

    assert len(results) == 1
    res = results[0]
    assert res.centroids.shape[1] == 2
    assert res.corners.shape[1:] == (4, 2)
    assert np.isfinite(res.centroids).all()
    assert np.isfinite(res.angles).all()
    if device == "cpu":
        assert (
            res.num_detections >= 1
        ), "expected >=1 detection on synthetic blobs (cpu)"
