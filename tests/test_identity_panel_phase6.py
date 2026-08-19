"""Phase 6 Task 6: Identity Models panel honesty — status line + calibration affordance.

Verifies:
- A verbatim status line under the master "Enable Identity Classification" toggle
  states that identity evidence is computed during inference and cached, and is
  available to both realtime and post-hoc paths.
- Each CNN classifier row exposes a calibration status label + a "Fit…" affordance.
- The unique-identifier tooltip no longer claims a legacy `IdentityAssignedLabel`
  column or an incorrect `CNN_<label>_Prob` pairing.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

STATUS_TEXT = (
    "Identity evidence is computed during inference and cached — "
    "available to both realtime and post-hoc."
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def main_window(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


def test_identity_panel_has_honest_status_line(main_window):
    """The exact status-line text is present among the panel's labels."""
    panel = main_window._identity_panel
    labels = panel.findChildren(QLabel)
    texts = [lbl.text() for lbl in labels]
    assert STATUS_TEXT in texts

    # It must live under the master toggle group, not some unrelated section.
    assert hasattr(panel, "lbl_identity_evidence_status")
    assert panel.lbl_identity_evidence_status.text() == STATUS_TEXT


def test_cnn_row_exposes_calibration_affordance(main_window):
    """A CNN row exposes a calibration status label and a Fit… button."""
    panel = main_window._identity_panel
    row = panel._add_cnn_classifier_row()
    try:
        assert hasattr(row, "lbl_calibration_status")
        assert isinstance(row.lbl_calibration_status, QLabel)
        assert row.lbl_calibration_status.text().startswith("Calibration:")

        assert hasattr(row, "btn_fit_calibration")
        assert isinstance(row.btn_fit_calibration, QPushButton)

        # No model selected yet — status should reflect that plainly.
        assert row.lbl_calibration_status.text() == "Calibration: —"

        # Toggling unique_identifier updates the status text even with no model.
        row.chk_unique_identifier.setChecked(True)
        assert row.lbl_calibration_status.text() == "Calibration: —"
    finally:
        panel._remove_cnn_classifier_row(row)


def test_unique_identifier_tooltip_uses_phase6_column_names(main_window):
    """Tooltip must reference the Phase-6 provenance column names, not legacy ones."""
    panel = main_window._identity_panel
    row = panel._add_cnn_classifier_row()
    try:
        tooltip = row.chk_unique_identifier.toolTip()
        assert "IdentityAssignedLabel" not in tooltip
        assert "IdentityRealtimeLabel" in tooltip
        assert "IdentityFinalLabel" in tooltip
        # CNN_<label>_Prob was previously (incorrectly) paired with _Class;
        # the real pairing is _Class/_Conf, with _Prob only per-class.
        assert "CNN_<label>_Conf" in tooltip
    finally:
        panel._remove_cnn_classifier_row(row)


def test_load_from_config_restores_non_identifying_classes(main_window):
    # No CNN model is registered in this fixture environment, so
    # to_config() returns None (it reads `meta` from the selected model and
    # bails out early with no selection -- see CNNClassifierRow.to_config).
    # This test covers only the load half of the round trip: whatever
    # to_config() would emit under "non_identifying_classes",
    # load_from_config() restores it onto row state. See
    # test_to_config_emits_factor_names_and_non_identifying_classes below
    # for the emission half, which requires a registered model.
    panel = main_window._identity_panel
    row = panel._add_cnn_classifier_row()
    other = panel._add_cnn_classifier_row()
    try:
        row.chk_unique_identifier.setChecked(True)
        row._non_identifying_classes = ["front:notag", "notag_notag"]
        cfg = {"non_identifying_classes": list(row._non_identifying_classes)}

        other.load_from_config(cfg)
        assert other._non_identifying_classes == ["front:notag", "notag_notag"]
    finally:
        panel._remove_cnn_classifier_row(row)
        panel._remove_cnn_classifier_row(other)


def test_to_config_emits_factor_names_and_non_identifying_classes(
    main_window, monkeypatch
):
    """to_config() must emit both new keys with the values the row holds.

    A model has to be "selected" for to_config() to run past its early
    return, so this stubs the CNN registry lookup with a fake multi-factor
    entry rather than depending on a real model being registered in the
    test environment.
    """
    panel = main_window._identity_panel
    fake_rel_path = "fake/classifier.pt"
    fake_meta = {
        "arch": "fake_arch",
        "classification_label": "colortag",
        "factor_names": ["front", "abdomen"],
        "class_names_per_factor": [["red", "blue", "notag"], ["dot", "notag"]],
        "scoring_mode": "atomic",
    }
    monkeypatch.setattr(
        panel, "_cnn_registry_by_path", lambda: {fake_rel_path: fake_meta}
    )
    row = panel._add_cnn_classifier_row()
    try:
        row._populate_model_combo()
        idx = row.combo_model.findData(fake_rel_path)
        assert idx >= 0, "stubbed registry entry did not appear in the combo"
        row.combo_model.setCurrentIndex(idx)
        row.chk_unique_identifier.setChecked(True)
        row._non_identifying_classes = ["front:notag", "notag_notag"]

        cfg = row.to_config()

        assert cfg is not None
        assert cfg["factor_names"] == ["front", "abdomen"]
        assert cfg["non_identifying_classes"] == ["front:notag", "notag_notag"]
    finally:
        panel._remove_cnn_classifier_row(row)


def test_to_config_omits_non_identifying_marks_when_untouched(main_window, monkeypatch):
    """An untouched row must still round-trip to today's equivalent config:
    factor_names present (harmless/new), non_identifying_classes empty."""
    panel = main_window._identity_panel
    fake_rel_path = "fake/classifier2.pt"
    fake_meta = {
        "classification_label": "colortag",
        "factor_names": ["flat"],
        "class_names_per_factor": [["red", "blue"]],
    }
    monkeypatch.setattr(
        panel, "_cnn_registry_by_path", lambda: {fake_rel_path: fake_meta}
    )
    row = panel._add_cnn_classifier_row()
    try:
        row._populate_model_combo()
        idx = row.combo_model.findData(fake_rel_path)
        row.combo_model.setCurrentIndex(idx)

        cfg = row.to_config()

        assert cfg["non_identifying_classes"] == []
        assert cfg["factor_names"] == ["flat"]
    finally:
        panel._remove_cnn_classifier_row(row)


def test_non_identifying_button_follows_unique_identifier(main_window):
    panel = main_window._identity_panel
    row = panel._add_cnn_classifier_row()
    # identity_content is disabled by default in a fresh MainWindow (no YOLO
    # detection method / identity master toggle selected); force it enabled
    # so isEnabled() below reflects the row's own toggled state, not an
    # unrelated ancestor gate.
    panel.identity_content.setEnabled(True)
    try:
        row.chk_unique_identifier.setChecked(False)
        assert not row.btn_non_identifying.isEnabled()
        row.chk_unique_identifier.setChecked(True)
        assert row.btn_non_identifying.isEnabled()
    finally:
        panel._remove_cnn_classifier_row(row)
