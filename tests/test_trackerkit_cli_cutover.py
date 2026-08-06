"""Slice 4 CLI cutover: every session runs the direct (Qt-free) path."""

from __future__ import annotations

from pathlib import Path

import pytest

from hydra_suite.trackerkit.cli_config import TrackerCliSession, TrackerCliVideoProbe


def _make_session(**overrides) -> TrackerCliSession:
    base = dict(
        video_path="video.mp4",
        config_path=None,
        video_probe=TrackerCliVideoProbe(
            fps=30.0, total_frames=10, width=64, height=64
        ),
        config={},
        raw_csv_path="video_tracking.csv",
        final_csv_path="video_tracking_forward_processed.csv",
        params={"FPS": 30.0},
        save_confidence_metrics=False,
        use_cached_detections=False,
        enable_backward_tracking=False,
        enable_postprocessing=True,
        interpolation_method="None",
        interpolation_max_gap_seconds=0.0,
        heading_flip_max_burst=5,
        identity_method="none_disabled",
        enable_pose_extractor=False,
    )
    base.update(overrides)
    return TrackerCliSession(**base)


def test_supports_direct_run_true_for_pose_sessions():
    session = _make_session(enable_pose_extractor=True)
    assert session.supports_direct_run() is True


def test_supports_direct_run_true_for_identity_sessions():
    session = _make_session(identity_method="apriltags")
    assert session.supports_direct_run() is True


def test_supports_direct_run_true_for_plain_session():
    assert _make_session().supports_direct_run() is True


def test_cli_module_has_no_bridge_symbols():
    import hydra_suite.trackerkit.cli as cli

    for gone in (
        "_suppress_message_boxes",
        "_ensure_qapplication",
        "_prepare_video_session",
        "_run_one_tracking_session",
        "_run_bridge_tracking_session",
    ):
        assert not hasattr(cli, gone), f"bridge symbol still present: {gone}"


def test_cli_module_imports_no_qt():
    import ast
    from pathlib import Path

    import hydra_suite.trackerkit.cli as cli

    tree = ast.parse(Path(cli.__file__).read_text(), filename=cli.__file__)
    offenders = []
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom):
            mod = node.module
        elif isinstance(node, ast.Import):
            mod = ",".join(a.name for a in node.names)
        if mod and any(q in mod for q in ("PySide6", "QtCore", "QtWidgets", "QtGui")):
            offenders.append(f"{mod}:{node.lineno}")
    assert not offenders, "cli.py must be Qt-free: " + "; ".join(offenders)


def test_gui_headless_hooks_deleted():
    """No _headless_* references survive in the GUI after the bridge is gone."""
    import re
    from pathlib import Path

    import hydra_suite.trackerkit.gui as gui_pkg

    root = Path(gui_pkg.__file__).parent
    pattern = re.compile(
        r"_headless_tracking_mode|_headless_tracking_callback|_headless_session_error"
    )
    offenders = []
    for py in root.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{py.relative_to(root)}:{lineno}")
    assert not offenders, "GUI still references _headless_* hooks: " + "; ".join(
        offenders
    )


_REPO = Path(__file__).resolve().parents[1]
_FX = _REPO / "tools/equivalence/fixtures"

_BRIDGE_CLIPS = [
    ("emi_obb_identity.mp4", "emi_obb_identity.json"),
    ("ant_pose_headtail.mp4", "ant_pose_headtail.json"),
    ("ant_obb_sleap.mp4", "ant_obb_sleap.json"),
    ("ant_cnn_identity.mp4", "ant_cnn_identity.json"),
]


@pytest.mark.parametrize("clip_name,config_name", _BRIDGE_CLIPS)
def test_previously_bridged_clips_run_direct_no_mainwindow(
    clip_name, config_name, tmp_path, monkeypatch
):
    clip_src = _FX / "clips" / clip_name
    config = _FX / "configs" / config_name
    if not (clip_src.exists() and config.exists()):
        pytest.skip(f"fixture missing: {clip_name}")

    # Any attempt to construct MainWindow (the deleted bridge) is a hard failure.
    def _boom(*_a, **_k):
        raise AssertionError("MainWindow constructed - bridge path was taken")

    mw_mod = pytest.importorskip("hydra_suite.trackerkit.gui.main_window")
    monkeypatch.setattr(mw_mod, "MainWindow", _boom)

    clip = tmp_path / clip_name
    clip.write_bytes(clip_src.read_bytes())

    from hydra_suite.trackerkit.cli import run_tracking_cli

    code = run_tracking_cli([str(clip)], config_path=str(config))
    assert code == 0

    csvs = list(tmp_path.glob("*_forward_processed.csv")) + list(
        tmp_path.glob("*_final.csv")
    )
    assert csvs, f"{clip_name}: no output CSV"
    rows = sum(1 for _ in csvs[0].open())
    assert rows > 1, f"{clip_name}: CSV has only {rows} line(s)"
