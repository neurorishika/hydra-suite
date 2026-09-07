"""Regression guard: the exported-SLEAP GPU path must scale float [0,1] canonical
crops to uint8 [0,255] — NOT floor them to a black image.

Bug: `SleapExportedBackend.predict_batch_cuda`'s ONNX fallback did
`c.clamp(0,255).byte()` on float32 [0,1] crops, flooring every pixel to 0. The
downstream `_prepare_export_crop` also casts to uint8, so the model saw a black
image and SLEAP returned zero-confidence keypoints (valid_mask=0) in the full
pipeline. Fix: scale [0,1] → [0,255] before the uint8 cast (pass [0,255] through).
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydra_suite.core.individual.pose.backends.sleap import (
    SleapExportedBackend,
    SleapServiceBackend,
    _prepare_export_crop,
)


def test_prepare_export_crop_scales_float_unit_crops_not_black():
    # THE real fix: the onnx_cuda/tensorrt CPU path hands float32 [0,1] crops to
    # predict_batch -> _prepare_export_crop. A plain uint8 cast floored them to
    # black (valid_mask=0). Must scale to [0,255].
    crop = np.full((8, 8, 3), 0.5, dtype=np.float32)
    arr, _ = _prepare_export_crop(crop, (8, 8), 3)
    assert arr.dtype == np.uint8
    assert int(arr.max()) == 127  # 0.5*255, NOT 0 (the bug)


def test_prepare_export_crop_passes_uint8_through():
    crop = np.full((8, 8, 3), 200, dtype=np.uint8)
    arr, _ = _prepare_export_crop(crop, (8, 8), 3)
    assert arr.dtype == np.uint8
    assert int(arr.max()) == 200  # unchanged


def _capture_cpu_crops(crops):
    be = SleapExportedBackend.__new__(SleapExportedBackend)
    be._runner = object()  # not a _DirectTensorRTEngine → ONNX fallback path
    captured = {}
    be.predict_batch = lambda c: captured.setdefault("crops", c) or []  # type: ignore
    be.predict_batch_cuda(crops)
    return captured["crops"]


def test_float_unit_crops_scaled_to_uint8_not_black():
    crops = [torch.full((3, 8, 8), 0.5, dtype=torch.float32) for _ in range(4)]
    got = _capture_cpu_crops(crops)
    assert len(got) == 4
    arr = np.asarray(got[0])
    assert arr.shape == (8, 8, 3)
    assert arr.dtype == np.uint8
    assert int(arr.max()) == 127  # 0.5 * 255 -> 127, NOT 0 (the bug)
    assert int(arr.min()) == 127


def test_already_255_range_passed_through():
    # A crop already in [0,255] must not be re-scaled to all-white.
    crops = [torch.full((3, 8, 8), 200.0, dtype=torch.float32) for _ in range(2)]
    got = _capture_cpu_crops(crops)
    arr = np.asarray(got[0])
    assert arr.dtype == np.uint8
    assert int(arr.max()) == 200  # unchanged, not clamped to 255


def test_service_to_uint8_image_scales_unit_floats():
    # The native path's shared helper must also scale [0,1] floats (guards the
    # convention the fix relies on).
    out = SleapServiceBackend._to_uint8_image(np.full((8, 8, 3), 0.5, dtype=np.float32))
    assert out.dtype == np.uint8
    assert int(out.max()) == 127


# --- Atomic re-export (fan-out siblings share one export dir) ----------------


def _sleap_config(model_dir, **overrides):
    from types import SimpleNamespace

    base = dict(
        model_path=str(model_dir),
        sleap_export_input_hw=None,
        sleap_batch=1,
        sleap_max_instances=1,
        sleap_env="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_export_into(target_bytes: bytes, observed: dict, watch: Path):
    """Stand-in exporter that records what a CONCURRENT reader would see."""

    def _export(*, model_dir, export_dir, runtime_flavor, sleap_env=None, **kwargs):
        observed["mid_export"] = watch.read_bytes() if watch.exists() else b"<GONE>"
        observed["exported_into"] = str(export_dir)
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        (Path(export_dir) / "model.onnx").write_bytes(target_bytes)
        return True, ""

    return _export


def test_reexport_never_deletes_the_export_a_sibling_is_reading(tmp_path, monkeypatch):
    """The old code rmtree'd ``export_dir`` and exported into it IN PLACE, so a
    concurrent fan-out child with a different batch/input_hw deleted the export
    another child was loading and left a half-built directory visible."""
    from pathlib import Path as _Path

    import hydra_suite.core.individual.pose.backends.sleap as sleap_mod

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "training_config.json").write_text("{}")
    export_dir = tmp_path / "model.onnx"
    export_dir.mkdir()
    (export_dir / "model.onnx").write_bytes(b"OLD-EXPORT")
    (export_dir / ".runtime_meta.json").write_text('{"signature": "stale"}')

    observed: dict = {}
    monkeypatch.setattr(
        sleap_mod,
        "_attempt_sleap_cli_export",
        _fake_export_into(b"NEW-EXPORT", observed, export_dir / "model.onnx"),
    )

    out = sleap_mod.auto_export_sleap_model(_sleap_config(model_dir), "onnx")

    assert _Path(out) == export_dir.resolve()
    # The published export was COMPLETE for the whole build: at export time the
    # old artifact was still there untouched, and the build happened elsewhere.
    assert observed["mid_export"] == b"OLD-EXPORT", observed
    assert observed["exported_into"] != str(export_dir)
    assert (export_dir / "model.onnx").read_bytes() == b"NEW-EXPORT"
    assert (export_dir / ".runtime_meta.json").exists()
    # No staging leftovers beside it (the build lock file is expected).
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "model",
        "model.onnx",
        "model.onnx.lock",
    ]


def test_failed_reexport_leaves_the_previous_export_intact(tmp_path, monkeypatch):
    import hydra_suite.core.individual.pose.backends.sleap as sleap_mod

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "training_config.json").write_text("{}")
    export_dir = tmp_path / "model.onnx"
    export_dir.mkdir()
    (export_dir / "model.onnx").write_bytes(b"OLD-EXPORT")
    (export_dir / ".runtime_meta.json").write_text('{"signature": "stale"}')

    monkeypatch.setattr(
        sleap_mod,
        "_attempt_sleap_cli_export",
        lambda **kwargs: (False, "exporter blew up"),
    )
    monkeypatch.setattr(
        sleap_mod,
        "_attempt_sleap_python_export",
        lambda **kwargs: (False, "exporter blew up"),
    )

    with pytest.raises(RuntimeError, match="SLEAP auto-export failed"):
        sleap_mod.auto_export_sleap_model(_sleap_config(model_dir), "onnx")

    assert (export_dir / "model.onnx").read_bytes() == b"OLD-EXPORT"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "model",
        "model.onnx",
        "model.onnx.lock",
    ]


def test_failed_export_swap_restores_the_previous_export(tmp_path, monkeypatch):
    """A failed rename must not leave the model with no export at all."""
    import os

    import hydra_suite.core.individual.pose.backends.sleap as sleap_mod

    export_dir = tmp_path / "model.onnx"
    export_dir.mkdir()
    (export_dir / "model.onnx").write_bytes(b"OLD-EXPORT")
    staging = tmp_path / "model.onnx.tmp-test"
    staging.mkdir()
    (staging / "model.onnx").write_bytes(b"NEW-EXPORT")

    real_rename = os.rename
    calls = {"n": 0}

    def _flaky_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # staging -> export_dir
            raise OSError("simulated cross-device failure")
        return real_rename(src, dst)

    monkeypatch.setattr(sleap_mod.os, "rename", _flaky_rename)
    with pytest.raises(OSError):
        sleap_mod._swap_export_dir_into_place(staging, export_dir)

    assert (export_dir / "model.onnx").read_bytes() == b"OLD-EXPORT"
    assert not list(tmp_path.glob("*.old-*"))
