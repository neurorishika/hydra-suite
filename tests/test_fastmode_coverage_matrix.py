"""Coverage-matrix test: every classifier family the pipeline supports has a CoreML exporter.

This test proves that every family (tinyclassifier, torchvision, timm, yolo/OBB)
can produce a real .mlpackage on Apple Silicon.

Guards:
- ``pytest.importorskip("coremltools")`` — skips the whole module on non-Mac or when
  coremltools is absent.
- No Qt/GUI imports anywhere in this file.

Spec reference: §7 hard requirement — no classifier family is left without a CoreML
export path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Skip the entire module when coremltools is not available or platform is wrong.
pytest.importorskip("coremltools")

if sys.platform != "darwin":
    pytest.skip("CoreML export only runs on macOS", allow_module_level=True)

import torch.nn as nn  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal fixture model (no checkpoint file required)
# ---------------------------------------------------------------------------


class _MinimalCNN(nn.Module):
    """Tiny 3-layer conv-net used as a stand-in for all non-YOLO families."""

    def __init__(self, n_classes: int = 3, h: int = 32, w: int = 32):
        super().__init__()
        self.c = nn.Conv2d(3, 8, 3, padding=1)
        self.p = nn.AdaptiveAvgPool2d(1)
        self.f = nn.Linear(8, n_classes)
        self._h = h
        self._w = w

    def forward(self, x):
        return self.f(self.p(self.c(x)).flatten(1))


# ---------------------------------------------------------------------------
# Family: tinyclassifier
# ---------------------------------------------------------------------------


def test_tinyclassifier_family_exports_coreml(tmp_path):
    """TinyClassifier family: export_tiny_to_coreml produces a .mlpackage."""
    from hydra_suite.training.tiny_model import _build_tiny_classifier_class

    TinyClassifier = _build_tiny_classifier_class()
    model = TinyClassifier(
        n_classes=3,
        hidden_layers=0,
        hidden_dim=32,
        tiny_preset="small",
    ).eval()

    from hydra_suite.training.tiny_model import export_tiny_to_coreml

    out = tmp_path / "tiny.mlpackage"
    ckpt = {"input_size": [32, 32]}
    p = export_tiny_to_coreml(model, ckpt, out)

    assert p.exists(), f"TinyClassifier .mlpackage not found at {p}"
    assert p.is_dir(), ".mlpackage should be a directory bundle"
    assert str(p).endswith(".mlpackage")


# ---------------------------------------------------------------------------
# Family: torchvision (resnet18 representative)
# ---------------------------------------------------------------------------


def test_torchvision_resnet18_family_exports_coreml(tmp_path):
    """Torchvision family (resnet18 repr): export_torchvision_to_coreml produces .mlpackage."""
    from hydra_suite.training.torchvision_model import export_torchvision_to_coreml

    # Use the minimal CNN as the model body — the exporter only needs a callable
    # nn.Module; it does not care which torchvision backbone it originated from.
    # This avoids downloading pretrained weights during CI.
    model = _MinimalCNN(n_classes=3, h=32, w=32).eval()
    ckpt = {"input_size": (32, 32), "arch": "resnet18"}

    out = tmp_path / "resnet18.mlpackage"
    p = export_torchvision_to_coreml(model, ckpt, out)

    assert p.exists(), f"resnet18 .mlpackage not found at {p}"
    assert p.is_dir(), ".mlpackage should be a directory bundle"
    assert str(p).endswith(".mlpackage")


# ---------------------------------------------------------------------------
# Family: torchvision (efficientnet_b0 representative)
# ---------------------------------------------------------------------------


def test_torchvision_efficientnet_family_exports_coreml(tmp_path):
    """Torchvision family (efficientnet_b0 repr): export_torchvision_to_coreml produces .mlpackage."""
    from hydra_suite.training.torchvision_model import export_torchvision_to_coreml

    model = _MinimalCNN(n_classes=5, h=32, w=32).eval()
    ckpt = {"input_size": (32, 32), "arch": "efficientnet_b0"}

    out = tmp_path / "efficientnet_b0.mlpackage"
    p = export_torchvision_to_coreml(model, ckpt, out)

    assert p.exists(), f"efficientnet_b0 .mlpackage not found at {p}"
    assert p.is_dir(), ".mlpackage should be a directory bundle"
    assert str(p).endswith(".mlpackage")


# ---------------------------------------------------------------------------
# Family: timm (via torchvision exporter — same code path)
# ---------------------------------------------------------------------------


def test_timm_family_exports_coreml(tmp_path):
    """TIMM family: export_torchvision_to_coreml is the shared exporter for timm models."""
    from hydra_suite.training.torchvision_model import export_torchvision_to_coreml

    # TIMM models share the same export_torchvision_to_coreml path; only the
    # model construction differs (timm.create_model vs tvm.*).  Use the minimal
    # CNN to verify the exporter contract without a network download.
    model = _MinimalCNN(n_classes=4, h=32, w=32).eval()
    ckpt = {"input_size": (32, 32), "arch": "timm/efficientnet_b0"}

    out = tmp_path / "timm_efficientnet.mlpackage"
    p = export_torchvision_to_coreml(model, ckpt, out)

    assert p.exists(), f"timm .mlpackage not found at {p}"
    assert p.is_dir(), ".mlpackage should be a directory bundle"
    assert str(p).endswith(".mlpackage")


# ---------------------------------------------------------------------------
# Family: yolo / OBB (ultralytics)
# ---------------------------------------------------------------------------


def test_yolo_obb_family_exports_coreml(tmp_path):
    """YOLO/OBB family: ultralytics YOLO.export(format='coreml') produces .mlpackage.

    Uses yolov8n-obb.pt (auto-downloaded by ultralytics on first run).
    This test is the definitive proof that the YOLO family has a CoreML path.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        pytest.skip("ultralytics not installed — YOLO family coverage gap (known)")

    # Export in tmp_path so the .mlpackage lands there, not in cwd.
    orig_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        model = YOLO("yolov8n-obb.pt")
        export_result = model.export(
            format="coreml",
            imgsz=640,
            nms=False,
        )
        mlpackage_path = Path(export_result).expanduser().resolve()
    finally:
        os.chdir(orig_dir)

    assert (
        mlpackage_path.exists()
    ), f"YOLO/OBB .mlpackage not produced at {mlpackage_path}"
    assert mlpackage_path.suffix == ".mlpackage"
