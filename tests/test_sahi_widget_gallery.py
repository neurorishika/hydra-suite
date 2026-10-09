"""The SAHI widget gallery renders one PNG per role and per host."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from hydra_suite.widgets.slice_settings import ROLES  # noqa: E402

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "sahi_widget_gallery.py"

EXPECTED = (
    [f"role_{role}.png" for role in ROLES]
    + [f"role_{role}_advanced.png" for role in ROLES]
    + [
        "host_detectkit_training.png",
        "host_sam3_training.png",
        "host_inference_settings.png",
        "host_sam3_escalation.png",
        "host_sam2_escalation.png",
        "host_trackerkit.png",
        "host_trackerkit_custom.png",
    ]
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("sahi_widget_gallery", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gallery_writes_every_png(tmp_path, monkeypatch):
    # The tool points HYDRA_DATA_DIR / HYDRA_CONFIG_DIR at temp dirs; keep
    # that from leaking into the rest of the test session.
    monkeypatch.delenv("HYDRA_DATA_DIR", raising=False)
    monkeypatch.delenv("HYDRA_CONFIG_DIR", raising=False)
    written = _load_tool().main(["--out", str(tmp_path)])
    assert sorted(p.name for p in written) == sorted(EXPECTED)
    for name in EXPECTED:
        path = tmp_path / name
        assert path.is_file(), name
        assert path.stat().st_size > 1000, name
