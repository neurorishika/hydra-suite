"""A CNN classifier row must not silently discard what a session recorded.

Three fields were lost on a load/save round trip, each in a different way:

* ``confidence`` was overwritten by the model's recommended threshold on every
  combo refresh -- and refreshes fire across ALL rows whenever any model is
  added or removed, so touching row B reverted row A and the next save
  persisted the revert.
* ``match_bonus`` / ``mismatch_penalty`` / ``calibration_temperature`` are read
  by the headless CLI (``core/inference/config.py``) but have no widget, so
  ``to_config`` never emitted them and a hand-edited config was normalised to
  defaults on the next GUI save.
* ``label`` was re-derived from the model registry, so renaming a model's
  classification_label silently renamed an existing session's output columns
  (``CNN_<label>_Class`` / ``_Conf``) in the GUI while the CLI, which reads the
  config, kept the old ones.

Same shape as the scoring_mode bug in test_identity_scoring_mode_roundtrip.py:
the GUI re-derives what the config already recorded.
"""

from __future__ import annotations

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

    window = MainWindow()
    yield window
    window.close()


@pytest.fixture
def row(main_window):
    widget = main_window._identity_panel._add_cnn_classifier_row()
    yield widget
    widget.deleteLater()


# ---------------------------------------------------------------- confidence


def _stub_recommendation(row, monkeypatch, value: float = 0.42) -> None:
    """Make the row's model offer a recommended threshold.

    Without this the refresh is a no-op (an unknown rel_path yields no
    recommendation), and a test asserting "the value survived" would pass
    even with the clobber present.
    """
    panel = main_window_of(row)
    monkeypatch.setattr(
        panel, "_classifier_recommended_confidence_threshold", lambda _meta: value
    )
    monkeypatch.setattr(panel, "_cnn_registry_entry", lambda _rel: {})


def test_a_loaded_threshold_survives_a_combo_refresh(row, monkeypatch):
    """The regression: refreshing the combo reverted a loaded threshold."""
    _stub_recommendation(row, monkeypatch)
    row.load_from_config({"confidence": 0.8})
    assert row.spin_confidence.value() == pytest.approx(0.8)

    row._update_verification_labels("some/model.pth")
    assert row.spin_confidence.value() == pytest.approx(
        0.8
    ), "a loaded threshold was replaced by the model's recommendation"


def test_a_user_edited_threshold_survives_a_combo_refresh(row, monkeypatch):
    _stub_recommendation(row, monkeypatch)
    row.spin_confidence.setValue(0.77)

    row._update_verification_labels("some/model.pth")
    assert row.spin_confidence.value() == pytest.approx(
        0.77
    ), "a user-edited threshold was replaced by the model's recommendation"


def test_one_rows_refresh_does_not_revert_another_rows_threshold(
    main_window, monkeypatch
):
    """The reported scenario, end to end across two rows."""
    panel = main_window._identity_panel
    row_a = panel._add_cnn_classifier_row()
    row_b = panel._add_cnn_classifier_row()
    try:
        monkeypatch.setattr(
            panel, "_classifier_recommended_confidence_threshold", lambda _meta: 0.42
        )
        monkeypatch.setattr(panel, "_cnn_registry_entry", lambda _rel: {})
        row_a.spin_confidence.setValue(0.8)

        # The real mechanism: _refresh_cnn_classifier_model_rows repopulates
        # EVERY row's combo after any model add/remove, and repopulating fires
        # currentIndexChanged -> _sync_model_ui -> _update_verification_labels.
        # Row A is a bystander to whatever the user did in row B.
        row_a._populate_model_combo()

        assert row_a.spin_confidence.value() == pytest.approx(
            0.8
        ), "a bystander row's threshold was reverted by an unrelated refresh"
    finally:
        row_a.deleteLater()
        row_b.deleteLater()


def test_an_untouched_row_still_takes_the_recommendation(row, monkeypatch):
    """The helpful default must survive: only CHOSEN values are protected."""
    panel = main_window_of(row)
    monkeypatch.setattr(
        panel,
        "_classifier_recommended_confidence_threshold",
        lambda _meta: 0.42,
    )
    monkeypatch.setattr(panel, "_cnn_registry_entry", lambda _rel: {})
    assert row._confidence_pinned is False

    row._update_verification_labels("some/model.pth")
    assert row.spin_confidence.value() == pytest.approx(0.42)
    assert row._confidence_pinned is False, "a recommendation is not a choice"


def test_user_switching_model_unpins_the_threshold(row, monkeypatch):
    panel = main_window_of(row)
    monkeypatch.setattr(panel, "_registry_has_cnn_entry", lambda _rel: True)
    monkeypatch.setattr(row, "_sync_model_ui", lambda: None)
    monkeypatch.setattr(
        row._main_window, "_sync_individual_analysis_mode_ui", lambda: None
    )
    row.load_from_config({"confidence": 0.8})
    assert row._confidence_pinned is True

    row.combo_model.addItem("other", "classification/identity/other.json")
    row._on_model_selected(row.combo_model.count() - 1)
    assert row._confidence_pinned is False


def main_window_of(row):
    return row._main_window._identity_panel


# ------------------------------------------------------------- passthrough


CLI_ONLY = {
    "match_bonus": 0.25,
    "mismatch_penalty": 0.75,
    "calibration_temperature": 1.7,
}


def test_cli_only_keys_are_captured_on_load(row):
    row.load_from_config(dict(CLI_ONLY, confidence=0.5))
    for key, value in CLI_ONLY.items():
        assert row._passthrough_cfg[key] == value


def test_owned_keys_are_not_captured_as_passthrough(row):
    row.load_from_config({"confidence": 0.5, "label": "x", "scoring_mode": "atomic"})
    assert not (set(row._passthrough_cfg) & row._OWNED_CONFIG_KEYS)


def test_owned_keys_win_over_an_echoed_value(row):
    """A stale echoed key must never shadow what the widgets say."""
    row.load_from_config({"confidence": 0.5})
    row._passthrough_cfg = {"confidence": 0.01, "match_bonus": 0.25}
    row.spin_confidence.setValue(0.9)

    entry = _to_config_with_stub_model(row)
    assert entry["confidence"] == pytest.approx(0.9)
    assert entry["match_bonus"] == 0.25


def _to_config_with_stub_model(row, *, meta: dict | None = None):
    """Drive to_config() with a selected model, without touching a registry."""
    row.combo_model.addItem("stub", "classification/identity/stub.json")
    row.combo_model.setCurrentIndex(row.combo_model.count() - 1)
    panel = main_window_of(row)
    original = panel._cnn_registry_entry
    panel._cnn_registry_entry = lambda _rel: dict(
        meta if meta is not None else {"classification_label": "registry_label"}
    )
    try:
        return row.to_config()
    finally:
        panel._cnn_registry_entry = original


def test_cli_only_keys_survive_the_round_trip(row):
    row.load_from_config(dict(CLI_ONLY, confidence=0.5))
    entry = _to_config_with_stub_model(row)
    for key, value in CLI_ONLY.items():
        assert entry[key] == value, f"{key} was dropped on save"


# ------------------------------------------------------------------- label


def test_a_saved_label_wins_over_the_registry(row):
    row.load_from_config({"label": "session_label"})
    entry = _to_config_with_stub_model(row)
    assert entry["label"] == "session_label"


def test_registry_label_applies_when_the_config_names_none(row):
    row.load_from_config({"confidence": 0.5})
    entry = _to_config_with_stub_model(row)
    assert entry["label"] == "registry_label"


def test_user_switching_model_drops_the_passthrough(row, monkeypatch):
    """A stale calibration_temperature must not follow the user to a new model.

    core/inference/config.py's _resolve_cnn_temperature gives an explicit
    entry value priority over the artifact's own fitted temperature, so
    carrying it across a model switch would silently run model B at model A's
    temperature -- while the panel's calibration status displayed B's fit.
    """
    panel = main_window_of(row)
    monkeypatch.setattr(panel, "_registry_has_cnn_entry", lambda _rel: True)
    monkeypatch.setattr(row, "_sync_model_ui", lambda: None)
    monkeypatch.setattr(
        row._main_window, "_sync_individual_analysis_mode_ui", lambda: None
    )
    row.load_from_config(dict(CLI_ONLY, confidence=0.5))
    assert row._passthrough_cfg

    row.combo_model.addItem("other", "classification/identity/other.json")
    row._on_model_selected(row.combo_model.count() - 1)

    assert row._passthrough_cfg == {}
    entry = _to_config_with_stub_model(row)
    for key in CLI_ONLY:
        assert key not in entry, f"{key} followed the user to a different model"


def test_a_label_conflict_is_logged_once(row, caplog):
    row.load_from_config({"label": "session_label"})
    with caplog.at_level("WARNING"):
        for _ in range(4):
            row._effective_label({"classification_label": "registry_label"})
    hits = [r for r in caplog.records if "label" in r.message]
    assert len(hits) == 1, f"expected one warning, got {len(hits)}"


def test_a_matching_label_is_not_logged(row, caplog):
    row.load_from_config({"label": "registry_label"})
    with caplog.at_level("WARNING"):
        row._effective_label({"classification_label": "registry_label"})
    assert not [r for r in caplog.records if "pins label" in r.message]


def test_user_switching_model_drops_the_label_pin(row, monkeypatch):
    panel = main_window_of(row)
    monkeypatch.setattr(panel, "_registry_has_cnn_entry", lambda _rel: True)
    monkeypatch.setattr(row, "_sync_model_ui", lambda: None)
    monkeypatch.setattr(
        row._main_window, "_sync_individual_analysis_mode_ui", lambda: None
    )
    row.load_from_config({"label": "session_label"})
    assert row._label_override == "session_label"

    row.combo_model.addItem("other", "classification/identity/other.json")
    row._on_model_selected(row.combo_model.count() - 1)
    assert row._label_override is None


def test_class_lists_are_never_pinned(row):
    """Artifact facts must track the model, not a stale session."""
    row.load_from_config(
        {
            "labels": ["stale"],
            "class_names_per_factor": [["stale"]],
            "factor_names": ["stale"],
        }
    )
    entry = _to_config_with_stub_model(
        row,
        meta={
            "classification_label": "registry_label",
            "class_names_per_factor": [["red", "blue"]],
            "factor_names": ["head"],
        },
    )
    assert entry["labels"] == ["red", "blue"]
    assert entry["class_names_per_factor"] == [["red", "blue"]]
    assert entry["factor_names"] == ["head"]
