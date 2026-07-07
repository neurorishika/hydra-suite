"""Tests for the single compute-tier selector introduced in Task 6."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from hydra_suite.runtime.resolver import PlatformInfo, available_tiers, tier_label


def test_tier_combo_lists_cuda_platform_tiers():
    p = PlatformInfo(has_cuda=True, has_mps=False)
    labels = [tier_label(t, p) for t in available_tiers(p)]
    assert labels == ["CPU", "GPU (CUDA)", "GPU-Fast (TensorRT)"]


def test_tier_combo_lists_mps_platform_tiers():
    p = PlatformInfo(has_cuda=False, has_mps=True)
    labels = [tier_label(t, p) for t in available_tiers(p)]
    assert labels == ["CPU", "GPU (Metal)", "GPU-Fast (CoreML)"]


def test_tier_combo_lists_cpu_only_platform():
    p = PlatformInfo(has_cuda=False, has_mps=False)
    labels = [tier_label(t, p) for t in available_tiers(p)]
    assert labels == ["CPU"]


def test_setup_panel_has_tier_combo():
    """SetupPanel exposes combo_runtime_tier and lbl_runtime_fallback."""
    from PySide6.QtWidgets import QApplication, QComboBox, QLabel

    QApplication.instance() or QApplication([])
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    mw = MainWindow()
    panel = mw._setup_panel
    assert hasattr(panel, "combo_runtime_tier"), "combo_runtime_tier missing"
    assert isinstance(panel.combo_runtime_tier, QComboBox)
    assert panel.combo_runtime_tier.count() >= 1
    assert hasattr(panel, "lbl_runtime_fallback"), "lbl_runtime_fallback missing"
    assert isinstance(panel.lbl_runtime_fallback, QLabel)
    assert not hasattr(
        panel, "combo_compute_runtime"
    ), "old combo_compute_runtime still present"
    assert not hasattr(
        panel, "combo_headtail_runtime"
    ), "old combo_headtail_runtime still present"
    assert not hasattr(
        panel, "combo_cnn_runtime"
    ), "old combo_cnn_runtime still present"
    mw.close()


def test_tier_combo_itemdata_are_tier_ids():
    """Each item in combo_runtime_tier stores the tier id as itemData."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from hydra_suite.runtime.resolver import available_tiers, detect_platform
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    mw = MainWindow()
    combo = mw._setup_panel.combo_runtime_tier
    platform = detect_platform()
    expected_tiers = available_tiers(platform)
    actual_data = [combo.itemData(i) for i in range(combo.count())]
    assert actual_data == expected_tiers
    mw.close()


def test_selected_runtime_tier_returns_tier_id():
    """_selected_runtime_tier() returns the current tier id."""
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    mw = MainWindow()
    tier = mw._selected_runtime_tier()
    assert tier in ("cpu", "gpu", "gpu_fast")
    mw.close()
