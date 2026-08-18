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

    # A source is required as well as the master toggle -- see
    # test_identity_needs_a_configured_source below.
    identity.g_identity.setChecked(True)
    identity.g_apriltags.setChecked(True)
    _settle(qapp)
    assert panel.g_identity_postprocess.isVisibleTo(panel) is True

    identity.g_identity.setChecked(False)
    identity.g_apriltags.setChecked(False)
    _settle(qapp)
    assert panel.g_identity_postprocess.isVisibleTo(panel) is False


def test_identity_section_stays_hidden_when_cleaning_is_off(qapp, main_window):
    """Both gates must hold: identity on but cleaning off shows nothing."""
    panel = main_window._postprocess_panel
    identity = main_window._identity_panel
    panel.clean_advanced.setExpanded(True)

    identity.g_identity.setChecked(True)
    identity.g_apriltags.setChecked(True)
    panel.enable_postprocessing.setChecked(False)
    _settle(qapp)
    assert panel.g_identity_postprocess.isVisibleTo(panel) is False

    panel.enable_postprocessing.setChecked(True)
    _settle(qapp)
    assert panel.g_identity_postprocess.isVisibleTo(panel) is True

    identity.g_identity.setChecked(False)
    identity.g_apriltags.setChecked(False)
    _settle(qapp)


def test_pose_scored_relink_knobs_hidden_without_pose(qapp, main_window):
    """Both relink knobs only score pose similarity, so with pose off neither
    value is ever read. The relinking toggle itself stays: scoring falls back
    to motion-only (predicted position, jump limit, heading gate) and the
    feature still runs."""
    panel = main_window._postprocess_panel
    panel.clean_advanced.setExpanded(True)
    panel.enable_postprocessing.setChecked(True)
    _settle(qapp)

    assert main_window._is_pose_export_enabled() is False
    assert panel.relink_quality_row_widget.isVisibleTo(panel) is False
    assert panel.chk_enable_tracklet_relinking.isVisibleTo(panel) is True


def test_relinking_hidden_when_the_post_pass_that_runs_it_is_off(qapp, main_window):
    """`relink_trajectories_with_pose` has exactly one call site, inside the
    interpolated post-pass. With nothing to trigger that pass the toggle
    cannot affect the run at all -- a dead control across far more configs
    than "pose is off"."""
    from hydra_suite.core.tracking import session_policy

    panel = main_window._postprocess_panel
    dataset = main_window._dataset_panel
    panel.clean_advanced.setExpanded(True)
    panel.enable_postprocessing.setChecked(True)

    dataset.chk_enable_individual_dataset.setChecked(True)
    panel._set_cleaning_section_state(True)
    _settle(qapp)
    config = main_window._config_orch.build_config_dict()
    assert session_policy.should_run_interpolated_postpass(config) is True
    assert panel.g_relinking.isVisibleTo(panel) is True

    # Nothing consumes the post-pass -> it never runs -> relinking cannot fire.
    dataset.chk_enable_individual_dataset.setChecked(False)
    panel._set_cleaning_section_state(True)
    _settle(qapp)
    config = main_window._config_orch.build_config_dict()
    assert session_policy.should_run_interpolated_postpass(config) is False
    assert panel.g_relinking.isVisibleTo(panel) is False

    dataset.chk_enable_individual_dataset.setChecked(True)
    panel._set_cleaning_section_state(True)
    _settle(qapp)


def test_pose_overlay_section_hidden_entirely_without_pose(qapp, main_window):
    """With pose off the subsection collapsed to a title, a dead checkbox and a
    line of explanation -- it looked like a setting but could not be set."""
    panel = main_window._postprocess_panel
    orch = main_window._session_orch
    panel.check_video_output.setChecked(True)
    _settle(qapp)

    assert orch._is_pose_inference_enabled() is False
    assert panel.g_video_pose_overlay.isVisibleTo(panel) is False

    # ...and it must come back when pose inference is actually available.
    original = orch._is_pose_inference_enabled
    try:
        orch._is_pose_inference_enabled = lambda *a, **k: True
        main_window._sync_video_pose_overlay_controls()
        _settle(qapp)
        assert panel.g_video_pose_overlay.isVisibleTo(panel) is True
        assert panel.check_video_show_pose.isEnabled() is True
    finally:
        orch._is_pose_inference_enabled = original
        main_window._sync_video_pose_overlay_controls()
        panel.check_video_output.setChecked(False)
        _settle(qapp)


def test_identity_needs_a_configured_source(qapp, main_window):
    """Identity on with neither AprilTags nor a CNN classifier produces no
    evidence at all -- `_selected_identity_method()` collapses to
    "none_disabled" and the run does no identity work. The master toggle alone
    must not reveal the post-processing section."""
    panel = main_window._postprocess_panel
    identity = main_window._identity_panel
    panel.clean_advanced.setExpanded(True)
    panel.enable_postprocessing.setChecked(True)

    identity.g_identity.setChecked(True)
    identity.g_apriltags.setChecked(False)
    _settle(qapp)
    assert main_window._has_identity_source() is False
    assert main_window._selected_identity_method() == "none_disabled"
    assert identity.lbl_no_identity_source.isVisibleTo(identity) is True
    assert panel.g_identity_postprocess.isVisibleTo(panel) is False

    identity.g_apriltags.setChecked(True)
    _settle(qapp)
    assert main_window._has_identity_source() is True
    assert identity.lbl_no_identity_source.isVisibleTo(identity) is False
    assert panel.g_identity_postprocess.isVisibleTo(panel) is True

    identity.g_identity.setChecked(False)
    identity.g_apriltags.setChecked(False)
    _settle(qapp)


def test_tracking_blocked_when_identity_has_no_source(qapp, main_window, monkeypatch):
    """The run would otherwise complete having done no identity work while the
    UI still claimed identity classification was on."""
    from PySide6.QtWidgets import QMessageBox

    identity = main_window._identity_panel
    orch = main_window._tracking_orch
    shown: list = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a[1]))
    )

    identity.g_identity.setChecked(False)
    _settle(qapp)
    assert orch._validate_identity_requirements("tracking") is True
    assert shown == []

    identity.g_identity.setChecked(True)
    identity.g_apriltags.setChecked(False)
    _settle(qapp)
    assert orch._validate_identity_requirements("tracking") is False
    assert shown == ["No Identity Source Configured"]

    shown.clear()
    identity.g_apriltags.setChecked(True)
    _settle(qapp)
    assert orch._validate_identity_requirements("tracking") is True
    assert shown == []

    identity.g_identity.setChecked(False)
    identity.g_apriltags.setChecked(False)
    _settle(qapp)
