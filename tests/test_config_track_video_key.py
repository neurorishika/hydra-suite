"""The per-track-video flag survives a save->load round-trip.

NOTE: the implementation-plan brief for this task assumed (a) the canonical
config key was ``generate_individual_track_videos`` (matching the widget
attribute name ``chk_generate_individual_track_videos``), and (b) the load
site lived in ``_load_config_visualization``. Verification against the
actual code showed both assumptions stale:

- The canonical key already unified on both save (config.py ~1864-1866)
  and load (config.py ~1263-1269) sides is
  ``final_media_export_videos_enabled`` (introduced in 3210abb2, predating
  this branch), with ``generate_oriented_track_videos`` already accepted as
  a read-time legacy alias via ``_cfg_get``.
- The load site actually lives in ``_load_config_individual_analysis(self,
  cfg, get_cfg)``, not ``_load_config_visualization(self, get_cfg)``.

There is no load/save key asymmetry left to fix for this flag; these tests
lock in that existing, correct behavior as regression coverage for Task 4's
``should_export_final_media_videos`` predicate, which depends on a single
stable canonical key.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from tests.test_config_build_dict import qapp, qtbot_config_stub  # noqa: E402,F401


def _get_cfg_for(cfg):
    """Build a ``get_cfg`` callable matching ConfigOrchestrator._cfg_get's
    signature: (new_key, *legacy_keys, default=...)."""

    def get_cfg(*keys, default=None):
        for key in keys:
            if key in cfg:
                return cfg[key]
        return default

    return get_cfg


def test_track_video_flag_round_trips(monkeypatch, qtbot_config_stub):
    orch = qtbot_config_stub
    # _load_config_individual_analysis ends by calling _sync_individual_analysis_mode_ui,
    # which force-unchecks this checkbox when no head-tail model is configured (unrelated
    # gating business logic). Stub it out so this test isolates the key round-trip.
    monkeypatch.setattr(orch._mw, "_sync_individual_analysis_mode_ui", lambda: None)
    orch._panels.dataset.chk_generate_individual_track_videos.setChecked(True)
    cfg = orch.build_config_dict()
    assert cfg["final_media_export_videos_enabled"] is True
    # Re-load into a fresh widget state and confirm the checkbox comes back True.
    orch._panels.dataset.chk_generate_individual_track_videos.setChecked(False)
    orch._load_config_individual_analysis(cfg, _get_cfg_for(cfg))
    assert orch._panels.dataset.chk_generate_individual_track_videos.isChecked() is True


def test_legacy_oriented_key_still_loads(monkeypatch, qtbot_config_stub):
    orch = qtbot_config_stub
    monkeypatch.setattr(orch._mw, "_sync_individual_analysis_mode_ui", lambda: None)
    orch._panels.dataset.chk_generate_individual_track_videos.setChecked(False)
    legacy = {"generate_oriented_track_videos": True}
    orch._load_config_individual_analysis(legacy, _get_cfg_for(legacy))
    assert orch._panels.dataset.chk_generate_individual_track_videos.isChecked() is True
