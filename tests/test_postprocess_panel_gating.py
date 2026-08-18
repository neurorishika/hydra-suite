"""Clean Results panel: sections must not be shown when they cannot apply."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def main_window(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


def _settle(qapp):
    for _ in range(6):
        qapp.processEvents()


def test_identity_postprocessing_follows_identity_classification(qapp, main_window):
    """The identity section used to be gated only on the auto-clean toggle, so
    it stayed on screen with identity classification off -- a whole block of
    controls that could not affect the run."""
    panel = main_window._postprocess_panel
    identity = main_window._identity_panel
    # Read through the advanced collapsible, which is collapsed by default:
    # isVisibleTo() would otherwise report False for the wrong reason.
    panel.clean_advanced.setExpanded(True)

    identity.g_identity.setChecked(False)
    _settle(qapp)
    assert panel.g_identity_postprocess.isVisibleTo(panel) is False

    identity.g_identity.setChecked(True)
    _settle(qapp)
    assert panel.g_identity_postprocess.isVisibleTo(panel) is True

    identity.g_identity.setChecked(False)
    _settle(qapp)
    assert panel.g_identity_postprocess.isVisibleTo(panel) is False


def test_identity_section_stays_hidden_when_cleaning_is_off(qapp, main_window):
    """Both gates must hold: identity on but cleaning off shows nothing."""
    panel = main_window._postprocess_panel
    identity = main_window._identity_panel
    panel.clean_advanced.setExpanded(True)

    identity.g_identity.setChecked(True)
    panel.enable_postprocessing.setChecked(False)
    _settle(qapp)
    assert panel.g_identity_postprocess.isVisibleTo(panel) is False

    panel.enable_postprocessing.setChecked(True)
    _settle(qapp)
    assert panel.g_identity_postprocess.isVisibleTo(panel) is True

    identity.g_identity.setChecked(False)
    _settle(qapp)


def test_pose_scored_relink_knobs_hidden_without_pose(qapp, main_window):
    """Both relink knobs score pose similarity; with pose off they are dead
    controls. The relinking toggle itself stays -- relinking still runs."""
    panel = main_window._postprocess_panel
    panel.clean_advanced.setExpanded(True)
    panel.enable_postprocessing.setChecked(True)
    _settle(qapp)

    assert main_window._is_pose_export_enabled() is False
    assert panel.relink_quality_row_widget.isVisibleTo(panel) is False
    assert panel.chk_enable_tracklet_relinking.isVisibleTo(panel) is True
