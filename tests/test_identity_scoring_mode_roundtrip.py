"""A saved session's ``scoring_mode`` must survive a load/save round trip.

``scoring_mode`` decides how a multi-head classifier's per-factor posteriors
are aggregated -- atomic tuple compare vs per-head averaging
(``core/tracking/identity/evidence.py``) -- so it changes identity assignment.

Before this fix, ``CNNClassifierRow.load_from_config`` never read the key and
``to_config`` always re-derived it from the model registry. A config saying
``per_head_average`` therefore RAN as ``atomic`` in the GUI while the headless
CLI honoured the config (``core/inference/config.py``), and the next save
overwrote the user's value permanently. The divergence was invisible to the
equivalence harness, which compares CLI against CLI.

It was found only because it hid behind a key-set assertion in the
``get_parameters_dict`` characterization golden, which had been red for an
unrelated reason (a stale ``PIPELINE_DEPTH`` key) and so was never read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_CONFIG_DIR = REPO_ROOT / "tools" / "equivalence" / "fixtures" / "configs"

ATOMIC_META = {"scoring_mode": "atomic", "classification_label": "colortag"}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def main_window(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    window = MainWindow()
    yield window
    window.close()


# --------------------------------------------------------------------------
# Row-level: the pin, its precedence, and when it is dropped.
# --------------------------------------------------------------------------


def test_saved_scoring_mode_wins_over_the_registry(main_window):
    row = main_window._identity_panel._add_cnn_classifier_row()
    try:
        row.load_from_config({"scoring_mode": "per_head_average"})
        assert row._effective_scoring_mode(ATOMIC_META) == "per_head_average"
    finally:
        row.deleteLater()


def test_registry_applies_when_the_config_names_nothing(main_window):
    row = main_window._identity_panel._add_cnn_classifier_row()
    try:
        row.load_from_config({})
        assert row._effective_scoring_mode(ATOMIC_META) == "atomic"
        assert row._effective_scoring_mode({}) == "atomic"
        assert (
            row._effective_scoring_mode({"scoring_mode": "per_head_average"})
            == "per_head_average"
        )
    finally:
        row.deleteLater()


def test_unknown_saved_value_is_ignored_not_pinned(main_window):
    """The backend rejects anything outside the two known modes."""
    row = main_window._identity_panel._add_cnn_classifier_row()
    try:
        row.load_from_config({"scoring_mode": "bogus"})
        assert row._scoring_mode_override is None
        assert row._effective_scoring_mode(ATOMIC_META) == "atomic"
    finally:
        row.deleteLater()


def test_user_switching_model_drops_the_pin(main_window, monkeypatch):
    """A pin belongs to the model it was saved against.

    ``combo_model.activated`` fires only on user interaction, never on the
    programmatic ``setCurrentIndex`` a config restore performs -- so clearing
    here cannot undo a restore.

    The panel collaborators are stubbed because an unregistered rel_path sends
    ``_on_model_selected`` down the discovered-model annotation path, which
    opens a modal dialog and hangs a headless run.
    """
    panel = main_window._identity_panel
    row = panel._add_cnn_classifier_row()
    try:
        monkeypatch.setattr(panel, "_registry_has_cnn_entry", lambda _rel: True)
        monkeypatch.setattr(row, "_sync_model_ui", lambda: None)
        monkeypatch.setattr(
            main_window, "_sync_individual_analysis_mode_ui", lambda: None
        )
        row.load_from_config({"scoring_mode": "per_head_average"})
        assert row._scoring_mode_override == "per_head_average"

        row.combo_model.addItem("other model", "classification/identity/other.json")
        row._on_model_selected(row.combo_model.count() - 1)

        assert row._scoring_mode_override is None
        assert row._effective_scoring_mode(ATOMIC_META) == "atomic"
    finally:
        row.deleteLater()


def test_pin_is_dropped_even_on_the_discovered_model_path(main_window, monkeypatch):
    """That branch returns early; the pin must already be gone."""
    panel = main_window._identity_panel
    row = panel._add_cnn_classifier_row()
    try:
        monkeypatch.setattr(panel, "_registry_has_cnn_entry", lambda _rel: False)
        monkeypatch.setattr(panel, "_annotate_discovered_cnn_model", lambda _rel: None)
        monkeypatch.setattr(row, "_populate_model_combo", lambda: None)
        row.load_from_config({"scoring_mode": "per_head_average"})

        row.combo_model.addItem("discovered", "classification/identity/found.json")
        row._on_model_selected(row.combo_model.count() - 1)

        assert row._scoring_mode_override is None
    finally:
        row.deleteLater()


def test_add_new_sentinel_does_not_drop_the_pin(main_window, monkeypatch):
    """'Add new…' is not a model choice; it must not disturb the session."""
    panel = main_window._identity_panel
    row = panel._add_cnn_classifier_row()
    try:
        monkeypatch.setattr(panel, "_handle_add_new_cnn_identity_model", lambda: None)
        monkeypatch.setattr(row, "_populate_model_combo", lambda: None)
        row.load_from_config({"scoring_mode": "per_head_average"})

        row.combo_model.addItem("Add new…", "__add_new__")
        row._on_model_selected(row.combo_model.count() - 1)

        assert row._scoring_mode_override == "per_head_average"
    finally:
        row.deleteLater()


def test_disagreement_is_logged(main_window, caplog):
    row = main_window._identity_panel._add_cnn_classifier_row()
    try:
        row.load_from_config({"scoring_mode": "per_head_average"})
        with caplog.at_level("WARNING"):
            row._effective_scoring_mode(ATOMIC_META)
        assert any(
            "scoring_mode" in record.message for record in caplog.records
        ), "a session/registry disagreement must be visible in the log"
    finally:
        row.deleteLater()


def test_disagreement_is_logged_once_per_row(main_window, caplog):
    """to_config runs on every build_config_dict; one warning, not a stream."""
    row = main_window._identity_panel._add_cnn_classifier_row()
    try:
        row.load_from_config({"scoring_mode": "per_head_average"})
        with caplog.at_level("WARNING"):
            for _ in range(5):
                row._effective_scoring_mode(ATOMIC_META)
        hits = [r for r in caplog.records if "scoring_mode" in r.message]
        assert len(hits) == 1, f"expected one warning, got {len(hits)}"
    finally:
        row.deleteLater()


def test_reloading_a_config_re_arms_the_warning(main_window, caplog):
    """The once-flag must not silence a genuinely new session."""
    row = main_window._identity_panel._add_cnn_classifier_row()
    try:
        row.load_from_config({"scoring_mode": "per_head_average"})
        with caplog.at_level("WARNING"):
            row._effective_scoring_mode(ATOMIC_META)
        caplog.clear()
        row.load_from_config({"scoring_mode": "per_head_average"})
        with caplog.at_level("WARNING"):
            row._effective_scoring_mode(ATOMIC_META)
        assert [r for r in caplog.records if "scoring_mode" in r.message]
    finally:
        row.deleteLater()


def test_agreement_is_not_logged(main_window, caplog):
    row = main_window._identity_panel._add_cnn_classifier_row()
    try:
        row.load_from_config({"scoring_mode": "atomic"})
        with caplog.at_level("WARNING"):
            row._effective_scoring_mode(ATOMIC_META)
        assert not [r for r in caplog.records if "scoring_mode" in r.message]
    finally:
        row.deleteLater()


# --------------------------------------------------------------------------
# The oracle that would have caught the original bug: a real gate config,
# loaded into a real MainWindow, must round-trip its scoring_mode.
# --------------------------------------------------------------------------


def _fixture_scoring_modes() -> dict[str, str]:
    payload = json.loads((FIXTURES_CONFIG_DIR / "ant_cnn_identity.json").read_text())
    return {
        str(entry.get("label", "")): str(entry.get("scoring_mode", ""))
        for entry in payload.get("cnn_classifiers", [])
    }


def _registry_scoring_mode(main_window, rel_path_fragment: str) -> str | None:
    """The host registry's scoring_mode for the fixture's classifier, if known."""
    try:
        entries = main_window._identity_panel._cnn_registry_by_path()
    except Exception:
        return None
    for rel_path, meta in entries.items():
        if rel_path_fragment in rel_path:
            return str(meta.get("scoring_mode", "") or "")
    return None


def test_gate_config_scoring_mode_survives_load_and_save(main_window):
    expected = _fixture_scoring_modes()
    assert expected, "fixture must define at least one CNN classifier"
    assert "per_head_average" in expected.values(), (
        "this oracle is only meaningful while the fixture pins a NON-default "
        "scoring_mode; with every value at the 'atomic' default it would pass "
        "even with the bug present"
    )

    # This oracle only has teeth where the host registry DISAGREES with the
    # fixture: the bug re-derived scoring_mode from the registry, so on a
    # machine whose registry already records per_head_average the test would
    # pass with the bug present. Skip loudly rather than pass vacuously.
    registry_mode = _registry_scoring_mode(main_window, "colortag")
    if registry_mode is not None and registry_mode in expected.values():
        pytest.skip(
            "host registry records scoring_mode=%r, matching the fixture; this "
            "oracle cannot distinguish the bug on this machine" % registry_mode
        )

    config_path = FIXTURES_CONFIG_DIR / "ant_cnn_identity.json"
    main_window._config_orch._load_config_from_file(str(config_path), preset_mode=False)

    saved = main_window._config_orch.build_config_dict()
    round_tripped = {
        str(entry.get("label", "")): str(entry.get("scoring_mode", ""))
        for entry in saved.get("cnn_classifiers", [])
    }
    assert round_tripped == expected

    params = main_window._config_orch.get_parameters_dict()
    from_params = {
        str(entry.get("label", "")): str(entry.get("scoring_mode", ""))
        for entry in params.get("CNN_CLASSIFIERS", [])
    }
    assert from_params == expected, (
        "the GUI's params must agree with the saved config the headless CLI "
        "reads -- this is the GUI/CLI divergence the fix closes"
    )
