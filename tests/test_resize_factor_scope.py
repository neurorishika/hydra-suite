"""`RESIZE_FACTOR` is a background-subtraction knob and must be clamped elsewhere.

The frame downscale in `core/tracking/worker.py::_resize_tracking_frame` is
method-agnostic, but only the bg-sub path is coherent under it:

* bg-sub detects on the already-resized frame (`stages/bgsub.py`'s RESIZE
  CONTRACT), and `RESIZE_FACTOR` is in `_BGSUB_KEY_PARAMS` so the cache agrees.
* YOLO's batch pass decodes at NATIVE resolution
  (`InferenceRunner.run_batch_pass` -> `make_frame_source`, which never sees the
  factor) and `RESIZE_FACTOR` is absent from the OBB cache key. So cached
  detections come back in native coordinates while `rescale_coordinates` still
  divides X/Y by the factor -- coordinates inflated by 1/scale -- and realtime
  YOLO (which *does* detect on the downscaled frame) silently disagrees with
  batch YOLO for the same config.

Rather than thread the factor through the OBB pipeline for a saving that YOLO's
own letterbox-to-`imgsz` mostly erases, the builder clamps it to 1.0 for
non-bg-sub methods. Every shipped config already uses 1.0, so this is
byte-identical for them.
"""

from __future__ import annotations

import pytest

from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params


def _runtime() -> RuntimeContext:
    return RuntimeContext(fps=30.0, total_frames=100, frame_width=640, frame_height=480)


def _params(**cfg):
    base = {"file_path": "/tmp/x.mp4", "fps": 30.0}
    base.update(cfg)
    return build_engine_params(base, runtime=_runtime())


@pytest.mark.parametrize("resize", [0.25, 0.5, 0.75, 1.0])
def test_background_subtraction_keeps_the_configured_scale(resize):
    params = _params(detection_method="background_subtraction", resize_factor=resize)
    assert params["RESIZE_FACTOR"] == pytest.approx(resize)


def test_background_subtraction_is_the_default_method_and_keeps_scale():
    """No explicit method means bg-sub, which honors the knob."""
    assert _params(resize_factor=0.5)["RESIZE_FACTOR"] == pytest.approx(0.5)


@pytest.mark.parametrize("resize", [0.25, 0.5, 0.75])
def test_yolo_obb_clamps_the_scale_to_one(resize):
    params = _params(detection_method="yolo_obb", resize_factor=resize)
    assert params["RESIZE_FACTOR"] == 1.0


def test_yolo_obb_at_full_scale_is_untouched():
    params = _params(detection_method="yolo_obb", resize_factor=1.0)
    assert params["RESIZE_FACTOR"] == 1.0


def test_clamping_also_rescales_the_derived_body_size_gates():
    """The clamp must reach every `body * RESIZE_FACTOR` derivative, not just the key.

    `scaled_body_size` feeds MIN/MAX_OBJECT_SIZE, the Kalman velocity gate and
    the assignment distance gates; a clamp applied only to the emitted
    `RESIZE_FACTOR` would leave those computed against the shrunk body.
    """
    scaled = _params(
        detection_method="yolo_obb", resize_factor=0.5, reference_body_size=20.0
    )
    full = _params(
        detection_method="yolo_obb", resize_factor=1.0, reference_body_size=20.0
    )
    assert scaled["MIN_OBJECT_SIZE"] == full["MIN_OBJECT_SIZE"]
    assert scaled["MAX_OBJECT_SIZE"] == full["MAX_OBJECT_SIZE"]


def test_the_clamp_warns_so_an_operator_learns_why(caplog):
    with caplog.at_level("WARNING"):
        _params(detection_method="yolo_obb", resize_factor=0.5)
    assert any("RESIZE_FACTOR" in record.message for record in caplog.records)


def test_the_scale_control_follows_the_detection_method():
    """The GUI must not offer a knob the engine will discard.

    Switching to YOLO OBB disables Scale and resets it to 1.0; switching back to
    background subtraction re-enables it.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    mw = MainWindow()
    try:
        setup = mw._setup_panel
        combo = mw._detection_panel.combo_detection_method

        combo.setCurrentIndex(0)  # background subtraction
        assert setup.spin_resize.isEnabled()
        setup.spin_resize.setValue(0.5)

        combo.setCurrentIndex(1)  # YOLO OBB
        assert not setup.spin_resize.isEnabled()
        assert setup.spin_resize.value() == pytest.approx(1.0)

        combo.setCurrentIndex(0)
        assert setup.spin_resize.isEnabled()
    finally:
        mw.close()
