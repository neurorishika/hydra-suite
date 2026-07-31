"""Tests for the shared classifier backend."""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.runtime.resolver import ResolvedBackend


def test_error_hierarchy_importable():
    """ClassifierError hierarchy exports from the errors module with correct inheritance."""
    from hydra_suite.core.identity.classification.errors import (
        ClassifierConfigError,
        ClassifierError,
        ClassifierFormatError,
        ClassifierRuntimeError,
        HeadTailFormatError,
    )

    assert issubclass(ClassifierFormatError, ClassifierError)
    assert issubclass(ClassifierRuntimeError, ClassifierError)
    assert issubclass(ClassifierConfigError, ClassifierError)
    assert issubclass(HeadTailFormatError, ClassifierFormatError)

    # Instantiable with a message
    err = ClassifierFormatError("bad")
    assert str(err) == "bad"


def test_classifier_metadata_fields():
    """ClassifierMetadata is frozen and exposes canonical fields."""
    from hydra_suite.core.identity.classification.backend import ClassifierMetadata

    meta = ClassifierMetadata(
        arch="tinyclassifier",
        input_size=(224, 224),
        is_multihead=False,
        factor_names=["flat"],
        class_names_per_factor=[["a", "b"]],
        monochrome=False,
        source_path="/tmp/model.pth",
    )
    assert meta.arch == "tinyclassifier"
    assert meta.input_size == (224, 224)
    assert meta.is_multihead is False
    assert meta.factor_names == ["flat"]
    assert meta.class_names_per_factor == [["a", "b"]]

    # Frozen
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.arch = "yolo"


def test_backend_parses_tiny_flat_metadata(tiny_flat_headtail):
    """ClassifierBackend exposes metadata for a v2 tiny flat checkpoint without loading weights."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(
        str(tiny_flat_headtail), resolved=ResolvedBackend("torch", "cpu", False)
    )
    meta = backend.metadata
    assert meta.arch == "tinyclassifier"
    assert meta.input_size == (64, 64)
    assert meta.is_multihead is False
    assert meta.factor_names == ["flat"]
    assert meta.class_names_per_factor == [["up", "down", "left", "right", "unknown"]]
    assert meta.monochrome is False
    assert meta.source_path == str(tiny_flat_headtail)
    backend.close()


def test_backend_tiny_flat_predict_batch_shape(tiny_flat_headtail):
    """predict_batch returns per-crop per-factor probability vectors with correct shape."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(
        str(tiny_flat_headtail), resolved=ResolvedBackend("torch", "cpu", False)
    )
    crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
    out = backend.predict_batch(crops)
    assert isinstance(out, list) and len(out) == 3
    for per_crop in out:
        assert isinstance(per_crop, list) and len(per_crop) == 1  # K=1
        probs = per_crop[0]
        assert probs.shape == (5,)
        assert np.isfinite(probs).all()
        assert abs(probs.sum() - 1.0) < 1e-5
    backend.close()


def test_backend_tiny_preprocess_matches_training_path(tiny_flat_headtail):
    """Tiny backend preprocessing matches the tiny training/inference path."""
    import cv2

    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    crop = np.zeros((24, 40, 3), dtype=np.uint8)
    crop[..., 0] = 25
    crop[..., 1] = 125
    crop[..., 2] = 240

    backend = ClassifierBackend(
        str(tiny_flat_headtail), resolved=ResolvedBackend("torch", "cpu", False)
    )
    processed = backend._preprocess([crop])[0]

    rgb = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    expected = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0

    assert np.allclose(processed, expected, atol=1e-6)
    backend.close()


def test_backend_non_square_input_size_roundtrip(tiny_flat_nonsquare):
    """[H, W] serialization in checkpoint is preserved as (H, W) in memory."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(
        str(tiny_flat_nonsquare), resolved=ResolvedBackend("torch", "cpu", False)
    )
    assert backend.metadata.input_size == (256, 192)
    # predict_batch preprocesses to (H=256, W=192). Just smoke-test that it does not raise.
    crops = [np.zeros((100, 100, 3), dtype=np.uint8)]
    out = backend.predict_batch(crops)
    assert out[0][0].shape == (3,)
    backend.close()


def test_backend_parses_yolo_flat_metadata(yolo_flat_headtail):
    """YOLO classify .pt exposes 5-class flat metadata via backend."""
    pytest.importorskip("ultralytics")
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(
        str(yolo_flat_headtail), resolved=ResolvedBackend("torch", "cpu", False)
    )
    meta = backend.metadata
    assert meta.arch == "yolo"
    assert meta.is_multihead is False
    assert meta.factor_names == ["flat"]
    assert len(meta.class_names_per_factor[0]) == 5
    assert set(meta.class_names_per_factor[0]) == {
        "up",
        "down",
        "left",
        "right",
        "unknown",
    }
    backend.close()


def test_backend_yolo_flat_predict_batch_shape(yolo_flat_headtail):
    pytest.importorskip("ultralytics")
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(
        str(yolo_flat_headtail), resolved=ResolvedBackend("torch", "cpu", False)
    )
    crops = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    out = backend.predict_batch(crops)
    assert len(out) == 2
    for per_crop in out:
        assert len(per_crop) == 1
        probs = per_crop[0]
        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 1e-3
    backend.close()


def test_backend_yolo_flat_sidecar_metadata(tmp_path, yolo_flat_headtail):
    pytest.importorskip("ultralytics")
    import json
    import shutil

    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    model_path = tmp_path / "artifact.pt"
    shutil.copy2(str(yolo_flat_headtail), str(model_path))
    model_path.with_suffix(".v2meta.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "arch": "yolo",
                "factor_names": ["flat"],
                "class_names_per_factor": [["up", "down", "left", "right", "unknown"]],
                "input_size": [640, 640],
                "monochrome": True,
            }
        ),
        encoding="utf-8",
    )

    backend = ClassifierBackend(
        str(model_path), resolved=ResolvedBackend("torch", "cpu", False)
    )
    meta = backend.metadata
    assert meta.input_size == (640, 640)
    assert meta.monochrome is True
    assert meta.class_names_per_factor == [["up", "down", "left", "right", "unknown"]]
    backend.close()


def test_backend_torchvision_flat_metadata_and_inference(torchvision_flat_identity):
    """ClassifierBackend loads a torchvision flat v2 checkpoint and returns per-factor probs."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(
        str(torchvision_flat_identity), resolved=ResolvedBackend("torch", "cpu", False)
    )
    meta = backend.metadata
    assert meta.arch == "resnet18"
    assert meta.input_size == (64, 64)
    assert meta.is_multihead is False
    assert meta.class_names_per_factor == [["antA", "antB", "antC"]]

    crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    out = backend.predict_batch(crops)
    assert len(out) == 2
    for per_crop in out:
        assert len(per_crop) == 1
        assert per_crop[0].shape == (3,)
    backend.close()


def test_backend_parses_legacy_flat_torchvision_metadata(
    legacy_torchvision_flat_headtail,
):
    """ClassifierBackend rejects pre-v2 flat torchvision checkpoints."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    from hydra_suite.core.identity.classification.errors import ClassifierFormatError

    with pytest.raises(ClassifierFormatError):
        ClassifierBackend(
            str(legacy_torchvision_flat_headtail),
            resolved=ResolvedBackend("torch", "cpu", False),
        )


def test_backend_tiny_multi_metadata_and_inference(tiny_multi_identity):
    """ClassifierBackend parses multi-head metadata and splits logits per factor."""
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(
        str(tiny_multi_identity), resolved=ResolvedBackend("torch", "cpu", False)
    )
    meta = backend.metadata
    assert meta.is_multihead is True
    assert meta.factor_names == ["color", "shape"]
    assert meta.class_names_per_factor == [["r", "g", "b"], ["sq", "ci"]]

    crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    out = backend.predict_batch(crops)
    assert len(out) == 2
    for per_crop in out:
        assert len(per_crop) == 2  # K=2
        assert per_crop[0].shape == (3,)
        assert per_crop[1].shape == (2,)
        assert abs(per_crop[0].sum() - 1.0) < 1e-5
        assert abs(per_crop[1].sum() - 1.0) < 1e-5
    backend.close()


def test_backend_yolo_multihead_bundle(yolo_multihead_bundle):
    """ClassifierBackend loads .multihead.json manifests and runs each YOLO per factor."""
    pytest.importorskip("ultralytics")
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(
        str(yolo_multihead_bundle), resolved=ResolvedBackend("torch", "cpu", False)
    )
    meta = backend.metadata
    assert meta.arch == "yolo_multihead"
    assert meta.is_multihead is True
    assert meta.factor_names == ["color", "shape"]
    assert meta.class_names_per_factor == [["r", "g", "b"], ["sq", "ci"]]

    crops = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]
    out = backend.predict_batch(crops)
    for per_crop in out:
        assert len(per_crop) == 2
        assert per_crop[0].shape == (3,)
        assert per_crop[1].shape == (2,)
    backend.close()


def test_backend_generic_classifier_multihead_bundle(
    tmp_path, tiny_flat_subset, tiny_flat_headtail
):
    import shutil

    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    from hydra_suite.training.model_publish import write_classifier_multihead_manifest

    factor_a = tmp_path / "color.pth"
    factor_b = tmp_path / "heading.pth"
    shutil.copy2(str(tiny_flat_subset), str(factor_a))
    shutil.copy2(str(tiny_flat_headtail), str(factor_b))

    manifest = write_classifier_multihead_manifest(
        tmp_path / "bundle.multihead.json",
        factor_entries=[
            {"factor": "side", "path": factor_a, "class_names": ["left", "right"]},
            {
                "factor": "heading",
                "path": factor_b,
                "class_names": ["up", "down", "left", "right", "unknown"],
            },
        ],
        input_size=(64, 64),
        monochrome=False,
    )

    backend = ClassifierBackend(
        str(manifest), resolved=ResolvedBackend("torch", "cpu", False)
    )
    meta = backend.metadata
    assert meta.arch == "classifier_multihead"
    assert meta.factor_names == ["side", "heading"]
    assert meta.class_names_per_factor == [
        ["left", "right"],
        ["up", "down", "left", "right", "unknown"],
    ]

    out = backend.predict_batch([np.zeros((48, 48, 3), dtype=np.uint8)])
    assert len(out) == 1
    assert len(out[0]) == 2
    assert out[0][0].shape == (2,)
    assert out[0][1].shape == (5,)
    backend.close()


def test_backend_generic_multihead_bundle_dedupes_duplicate_factor_names(
    tmp_path, tiny_flat_subset, tiny_flat_headtail
):
    import json
    import shutil

    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    factor_a = tmp_path / "color.pth"
    factor_b = tmp_path / "heading.pth"
    shutil.copy2(str(tiny_flat_subset), str(factor_a))
    shutil.copy2(str(tiny_flat_headtail), str(factor_b))

    manifest = tmp_path / "duplicate.multihead.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "classifier_multihead_bundle",
                "factor_names": ["flat", "flat"],
                "factor_models": [
                    {
                        "factor": "flat",
                        "path": factor_a.name,
                        "class_names": ["left", "right"],
                    },
                    {
                        "factor": "flat",
                        "path": factor_b.name,
                        "class_names": ["up", "down", "left", "right", "unknown"],
                    },
                ],
                "input_size": [64, 64],
                "monochrome": False,
            }
        ),
        encoding="utf-8",
    )

    backend = ClassifierBackend(
        str(manifest), resolved=ResolvedBackend("torch", "cpu", False)
    )
    meta = backend.metadata
    assert meta.factor_names == ["flat", "flat_1"]

    out = backend.predict_batch([np.zeros((48, 48, 3), dtype=np.uint8)])
    assert len(out) == 1
    assert len(out[0]) == 2
    backend.close()


def test_backend_generic_multihead_bundle_preserves_onnx_runtime(
    tmp_path, tiny_flat_subset, tiny_flat_headtail, monkeypatch
):
    import shutil

    import hydra_suite.core.identity.classification.backend as backend_module
    from hydra_suite.core.identity.classification.backend import ClassifierBackend
    from hydra_suite.training.model_publish import write_classifier_multihead_manifest

    factor_a = tmp_path / "color.pth"
    factor_b = tmp_path / "heading.pth"
    shutil.copy2(str(tiny_flat_subset), str(factor_a))
    shutil.copy2(str(tiny_flat_headtail), str(factor_b))

    manifest = write_classifier_multihead_manifest(
        tmp_path / "bundle.multihead.json",
        factor_entries=[
            {"factor": "side", "path": factor_a, "class_names": ["left", "right"]},
            {
                "factor": "heading",
                "path": factor_b,
                "class_names": ["up", "down", "left", "right", "unknown"],
            },
        ],
        input_size=(64, 64),
        monochrome=False,
    )

    class FakeFactorBackend:
        def __init__(self, probs):
            self._probs = np.array(probs, dtype=np.float32)

        def predict_batch(self, crops):
            return [[self._probs.copy()] for _ in crops]

        def close(self):
            return None

    observed: dict[str, object] = {}

    def _fake_load(path: str, runtime: str):
        observed["path"] = path
        observed["runtime"] = runtime
        return [
            FakeFactorBackend([0.4, 0.6]),
            FakeFactorBackend([0.1, 0.2, 0.3, 0.15, 0.25]),
        ]

    def _fail_load_onnx(self):
        raise AssertionError("generic multi-head bundles should not load ONNX directly")

    monkeypatch.setattr(
        backend_module._ClassifierMultiheadBundleLoader,
        "load",
        staticmethod(_fake_load),
    )
    monkeypatch.setattr(ClassifierBackend, "_load_onnx", _fail_load_onnx)

    # The resolved backend is threaded verbatim to each factor child; TensorRT
    # is a representable ResolvedBackend (onnx_cpu is not, under Gen-2).
    resolved = ResolvedBackend("tensorrt", "cuda", False)
    backend = ClassifierBackend(str(manifest), resolved=resolved)
    out = backend.predict_batch([np.zeros((48, 48, 3), dtype=np.uint8)])

    assert observed["path"] == str(manifest)
    assert observed["runtime"] == resolved
    assert len(out) == 1
    assert len(out[0]) == 2
    assert out[0][0].shape == (2,)
    assert out[0][1].shape == (5,)
    backend.close()


def test_backend_yolo_multihead_bundle_preserves_export_runtime(
    yolo_multihead_bundle, monkeypatch
):
    import hydra_suite.core.identity.classification.backend as backend_module
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    class FakeFactorBackend:
        def __init__(self, probs):
            self._probs = np.array(probs, dtype=np.float32)

        def predict_batch(self, crops):
            return [[self._probs.copy()] for _ in crops]

        def close(self):
            return None

    observed: dict[str, object] = {}

    def _fake_load(path: str, runtime: str):
        observed["path"] = path
        observed["runtime"] = runtime
        return [FakeFactorBackend([0.3, 0.3, 0.4]), FakeFactorBackend([0.6, 0.4])]

    monkeypatch.setattr(
        backend_module._ClassifierMultiheadBundleLoader,
        "load",
        staticmethod(_fake_load),
    )

    resolved = ResolvedBackend("coreml", "mps", False)
    backend = ClassifierBackend(str(yolo_multihead_bundle), resolved=resolved)
    out = backend.predict_batch([np.zeros((64, 64, 3), dtype=np.uint8)])

    assert observed["path"] == str(yolo_multihead_bundle)
    assert observed["runtime"] == resolved
    assert len(out) == 1
    assert len(out[0]) == 2
    backend.close()


# NOTE: test_backend_tiny_onnx_lazy_derive was deleted in Task 4a. It asserted
# native-ONNX-on-CPU ("onnx_cpu"): the backend derives a .onnx peer and runs it
# through ONNX Runtime on CPU. Under ResolvedBackend that behavior is
# unrepresentable — the resolver never emits onnx_cpu, and (torch, cpu) loads the
# checkpoint natively without ever creating an ONNX peer — and unreachable in
# production, so there is no faithful ResolvedBackend translation of the test.


def test_backend_falls_back_to_native_torch_when_onnx_accelerator_missing(
    tiny_flat_headtail, monkeypatch
):
    import hydra_suite.core.identity.classification.backend as backend_module
    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    observed: dict[str, object] = {}

    class FakeModel:
        def __call__(self, batch):
            import torch

            return torch.zeros((batch.shape[0], 5), dtype=torch.float32)

    def _fake_load(path: str, device: str):
        observed["path"] = path
        observed["device"] = device
        return FakeModel()

    def _fail_load_onnx(self):
        raise AssertionError("backend should fall back before creating an ONNX session")

    monkeypatch.setattr(
        backend_module,
        "_available_onnx_provider_names",
        lambda: {"CPUExecutionProvider"},
    )
    monkeypatch.setattr(
        backend_module,
        "_native_accelerator_available",
        lambda resolved: resolved.device == "cuda",
    )
    monkeypatch.setattr(backend_module._TinyLoader, "load", staticmethod(_fake_load))
    monkeypatch.setattr(ClassifierBackend, "_load_onnx", _fail_load_onnx)

    backend = ClassifierBackend(
        str(tiny_flat_headtail), resolved=ResolvedBackend("tensorrt", "cuda", False)
    )
    out = backend.predict_batch([np.zeros((32, 32, 3), dtype=np.uint8)])

    assert observed["path"] == str(tiny_flat_headtail)
    assert observed["device"] == "cuda"
    assert len(out) == 1
    assert len(out[0]) == 1
    assert out[0][0].shape == (5,)
    backend.close()


def test_backend_tiny_monochrome_preprocess_matches_training_path(
    tiny_flat_monochrome,
):
    """Tiny monochrome checkpoints grayscale inputs without ImageNet normalization."""
    import cv2

    from hydra_suite.core.identity.classification.backend import ClassifierBackend

    backend = ClassifierBackend(
        str(tiny_flat_monochrome), ResolvedBackend("torch", "cpu", False)
    )
    assert backend.metadata.monochrome is True

    crop = np.zeros((28, 20, 3), dtype=np.uint8)
    crop[..., 0] = 10
    crop[..., 1] = 140
    crop[..., 2] = 250
    processed = backend._preprocess([crop])[0]

    rgb = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    expected = (
        cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32).transpose(2, 0, 1)
        / 255.0
    )

    assert np.allclose(processed, expected, atol=1e-6)
    backend.close()
