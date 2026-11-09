"""Widgetless config keys survive GUI load -> save and reach the engine
exactly as the CLI sees them (identity-evidence honesty + trust keys)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.config.identity_schema import (  # noqa: E402
    _EVIDENCE_FIELD_KEYS,
    EVIDENCE_ENGINE_KEYS,
    GUI_EVIDENCE_FIELDS,
    PostHocIdentityConfig,
)
from hydra_suite.trackerkit.engine_params import (  # noqa: E402
    RuntimeContext,
    build_engine_params,
)
from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402
from hydra_suite.trackerkit.gui.orchestrators.config import (  # noqa: E402
    CONFIG_ONLY_KEYS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    REPO_ROOT
    / "tools"
    / "equivalence"
    / "fixtures"
    / "configs"
    / "ant_cnn_identity.json"
)

NON_DEFAULT = {
    "identity_noncatalog_to_unknown": False,
    "fragment_min_absolute_posterior": 0.25,
    "fragment_absolute_seeding": True,
    "identity_evidence_trust": True,
    "identity_trust_isolation_bl": 1.5,
    "identity_trust_min_views": 2,
    "identity_trust_split_min_log_odds": 6.0,
    "identity_trust_contact_split": False,
    "identity_trust_contact_bl": 0.75,
    "identity_trust_contact_window": 12,
}
GUI_KEYS = {_EVIDENCE_FIELD_KEYS[f] for f in GUI_EVIDENCE_FIELDS}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def main_window(monkeypatch, qapp):
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def _config_file(tmp_path, extra):
    cfg = json.loads(FIXTURE.read_text())
    cfg.update(extra)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    return path, cfg


def test_every_evidence_key_is_either_a_widget_or_config_only():
    every = set(_EVIDENCE_FIELD_KEYS.values())
    assert GUI_KEYS <= every
    assert every - GUI_KEYS <= set(CONFIG_ONLY_KEYS)
    assert not (GUI_KEYS & set(CONFIG_ONLY_KEYS))


def test_load_save_round_trip_keeps_widgetless_keys(main_window, tmp_path):
    path, _ = _config_file(tmp_path, NON_DEFAULT)
    orch = main_window._config_orch
    orch._load_config_from_file(str(path), preset_mode=False)
    saved = orch.build_config_dict()
    for key, value in NON_DEFAULT.items():
        assert saved[key] == value, key


def test_gui_and_cli_params_agree_on_widgetless_keys(main_window, tmp_path):
    path, cfg = _config_file(tmp_path, NON_DEFAULT)
    orch = main_window._config_orch
    orch._load_config_from_file(str(path), preset_mode=False)
    gui = orch.get_parameters_dict()
    cli = build_engine_params(
        cfg,
        runtime=RuntimeContext(
            fps=10.0, total_frames=None, frame_width=None, frame_height=None
        ),
        advanced_config={},
    )
    for engine_key in EVIDENCE_ENGINE_KEYS.values():
        assert gui[engine_key] == cli[engine_key], engine_key
    assert gui["IDENTITY_EVIDENCE_TRUST"] is True
    assert gui["IDENTITY_TRUST_ISOLATION_BL"] == 1.5


def test_full_load_without_the_keys_reverts_to_defaults(main_window, tmp_path):
    path, _ = _config_file(tmp_path, NON_DEFAULT)
    orch = main_window._config_orch
    orch._load_config_from_file(str(path), preset_mode=False)
    plain = tmp_path / "plain.json"
    plain.write_text(FIXTURE.read_text())
    orch._load_config_from_file(str(plain), preset_mode=False)
    saved = orch.build_config_dict()
    # Config-only keys are not invented; widget keys fall back to defaults.
    assert not ((set(NON_DEFAULT) - GUI_KEYS) & set(saved))
    defaults = PostHocIdentityConfig()
    for f in GUI_EVIDENCE_FIELDS:
        assert saved[_EVIDENCE_FIELD_KEYS[f]] == getattr(defaults, f), f
    assert orch.get_parameters_dict()["IDENTITY_EVIDENCE_TRUST"] is True
