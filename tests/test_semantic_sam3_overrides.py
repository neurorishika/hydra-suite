"""imgsz must be PINNED, not inherited from ultralytics' default cfg."""

from hydra_suite.core.inference.semantic.sam3 import (
    PREDICTOR_IMGSZ,
    predictor_overrides,
)


def test_overrides_pin_imgsz_to_the_architecture_size():
    # build_sam3.py builds SAM3 at img_size=1008; ultralytics' default cfg is
    # 640 -> 644. Inheriting it silently runs the model off-size.
    assert PREDICTOR_IMGSZ == 1008
    ov = predictor_overrides("/tmp/fake.pt", "cpu")
    assert ov["imgsz"] == 1008


def test_overrides_still_pin_conf_and_iou():
    ov = predictor_overrides("/tmp/fake.pt", "cpu", confidence_floor=0.05)
    assert ov["conf"] == 0.05
    assert ov["iou"] == 0.7
    assert ov["model"] == "/tmp/fake.pt"
