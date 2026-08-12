"""Regression oracle: GUI ``get_parameters_dict()`` vs the shared Qt-free
``build_engine_params``.

The GUI's ``get_parameters_dict()`` is the correctness reference — it is the
widget scrape that produced the params the Slice-4/5 headless-cutover gate
validated byte-identical against the legacy pipeline. This test loads each of
the 7 gate-clip saved configs into a real, offscreen ``MainWindow``, and
asserts that ``build_engine_params(build_config_dict(), runtime=...)``
reproduces the GUI reference key-for-key on every key that isn't purely
cosmetic/runtime-overlay.

As of Task 6 this test PASSES: ``get_parameters_dict()`` was collapsed into a
thin wrapper over ``build_engine_params`` + a GUI-only display/runtime overlay,
so the two derivations agree on every non-display / non-runtime-overlay key by
construction. It is the durable anti-drift guard for the GUI/CLI param
unification (the Task-2 oracle that was RED through Tasks 3-5 and is now GREEN).

A second, always-passing diagnostic test dumps the per-clip diverging-key
sets (GUI-only / shared-only / value-mismatch) to
``.superpowers/sdd/2026-08-05-shared-engine-param-builder/keydiff-baseline.txt``
— the authoritative worklist for Tasks 3-5.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.trackerkit.engine_params import (  # noqa: E402
    RuntimeContext,
    build_engine_params,
)
from hydra_suite.trackerkit.gui.main_window import MainWindow  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_CONFIG_DIR = REPO_ROOT / "tools" / "equivalence" / "fixtures" / "configs"
KEYDIFF_BASELINE_PATH = (
    REPO_ROOT
    / ".superpowers"
    / "sdd"
    / "2026-08-05-shared-engine-param-builder"
    / "keydiff-baseline.txt"
)

CLIPS = [
    "fly_obb",
    "worm_bgsub",
    "emi_obb_identity",
    "ant_pose_headtail",
    "ant_obb_sleap",
    "ant_obb_sequential",
    "ant_cnn_identity",
]

# Bucket 3: purely cosmetic / display-only GUI state that has no engine
# meaning and is never expected to appear in the shared builder's output.
DISPLAY_ONLY_KEYS = {
    "SHOW_FG",
    "SHOW_BG",
    "SHOW_CIRCLES",
    "SHOW_ORIENTATION",
    "SHOW_YOLO_OBB",
    "SHOW_TRAJECTORIES",
    "SHOW_LABELS",
    "SHOW_STATE",
    "SHOW_KALMAN_UNCERTAINTY",
    "zoom_factor",
    "VISUALIZATION_FREE_MODE",
    "TRACKING_REALTIME_MODE",
    "TRACKING_WORKFLOW_MODE",
    "TRAJECTORY_COLORS",
}

# Bucket 4: runtime-overlay keys supplied via RuntimeContext (video/session
# facts, not config-derived) — excluded because this oracle intentionally
# builds a synthetic RuntimeContext with None/placeholder values for these.
#
# START_FRAME / END_FRAME are the frame-processing range: a video/session fact,
# not a config-derived value. The GUI reads them off the live spin boxes
# (config.py:2157-2158), which — with no video loaded in this oracle — read 0,
# while the shared builder derives END_FRAME from the runtime/probe total-frame
# count. Both denote "process the whole clip"; in a real GUI run the spin boxes
# are seeded from the loaded video's frame count, so Task 6's gui_runtime_context
# supplies them from the GUI just like the CLI supplies them from its probe. They
# are therefore treated as runtime overlay, not compared here.
RUNTIME_OVERLAY_KEYS = {
    "ROI_MASK",
    "INDIVIDUAL_PROPERTIES_CACHE_PATH",
    "INDIVIDUAL_DATASET_RUN_ID",
    "DATASET_OUTPUT_DIR",
    "FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR",
    "INDIVIDUAL_DATASET_OUTPUT_DIR",
    "INDIVIDUAL_DATASET_NAME",
    "START_FRAME",
    "END_FRAME",
}


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def main_window(monkeypatch, qapp):
    """A real, offscreen ``MainWindow`` with the advanced-config disk hooks
    stubbed out (same pattern as ``tests/test_config_build_dict.py``)."""
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(MainWindow, "_load_advanced_config", lambda self: {})
    window = MainWindow()
    try:
        yield window
    finally:
        window.close()


def load_gate_config_into(main_window: MainWindow, clip: str) -> None:
    """Load a gate-clip's saved config into ``main_window`` via the real
    config-load path (``_load_config_from_file``). No stubbing of the load
    path itself was needed: every fixture config has ``file_path: ""`` and
    ``roi_shapes: []``, so the video-probing / ROI-rehydration branches
    (which do touch disk) are simply not entered.
    """
    config_path = FIXTURES_CONFIG_DIR / f"{clip}.json"
    assert config_path.is_file(), f"missing gate fixture config: {config_path}"
    main_window._config_orch._load_config_from_file(str(config_path), preset_mode=False)


def gui_runtime_context(main_window: MainWindow) -> RuntimeContext:
    """Build a ``RuntimeContext`` for the oracle comparison.

    ``fps`` falls back to the loaded config's fps (which dominates inside
    ``build_engine_params`` anyway, so this can't cause a mismatch).
    ``total_frames``/``frame_width``/``frame_height`` only feed
    runtime-overlay keys and an fps-independent path excluded from the value
    comparison, so their exact values don't matter here. ``roi_mask`` and all
    output-dir/cache fields are left ``None`` (also excluded via
    ``RUNTIME_OVERLAY_KEYS``).
    """
    orch = main_window._config_orch
    cfg = orch.build_config_dict()
    fps = cfg.get("fps", 30.0) or 30.0
    total_frames = cfg.get("end_frame") or 100
    return RuntimeContext(
        fps=fps,
        total_frames=total_frames,
        frame_width=640,
        frame_height=480,
        roi_mask=None,
        dataset_output_dir=None,
        final_media_video_output_dir=None,
        individual_dataset_output_dir=None,
        individual_dataset_name=None,
        individual_dataset_run_id=None,
        individual_properties_cache_path=None,
    )


def _compare(reference: dict, shared: dict) -> dict:
    compared = (set(reference) | set(shared)) - DISPLAY_ONLY_KEYS - RUNTIME_OVERLAY_KEYS
    return {
        k: (reference.get(k, "∅"), shared.get(k, "∅"))
        for k in compared
        if reference.get(k, "∅") != shared.get(k, "∅")
    }


@pytest.mark.parametrize("clip", CLIPS)
def test_shared_builder_reproduces_gui_reference(main_window, clip):
    load_gate_config_into(main_window, clip)
    orch = main_window._config_orch
    reference = orch.get_parameters_dict()
    cfg = orch.build_config_dict()
    rt = gui_runtime_context(main_window)
    # Feed the builder the SAME advanced-config source the GUI uses
    # (``self._mw.advanced_config``) so ADVANCED_CONFIG is compared
    # apples-to-apples: both start from the identical base dict and overlay the
    # identical derived keys. Letting the builder default to
    # load_advanced_tracker_config() would pull in the on-disk advanced file,
    # which the GUI does not consult here (the oracle stubs _load_advanced_config
    # to {}).
    shared = build_engine_params(
        cfg, runtime=rt, advanced_config=main_window.advanced_config
    )
    diffs = _compare(reference, shared)
    assert not diffs, f"{clip}: {len(diffs)} keys diverge: {sorted(diffs)}"


def test_dump_keydiff_baseline(main_window):
    """Always-passing diagnostic: dumps per-clip diverging-key detail to
    ``keydiff-baseline.txt`` — the authoritative worklist for Tasks 3-5.
    Categorizes each diverging key as GUI-only, shared-only, or a genuine
    value mismatch (present, but unequal, on both sides).
    """
    lines = [
        "# GUI-reference vs shared-builder param key diff baseline",
        "# Generated by tests/test_gui_cli_param_equivalence.py::test_dump_keydiff_baseline",
        "# Authoritative worklist for Tasks 3-5 of the shared-engine-param-builder program.",
        "",
    ]
    for clip in CLIPS:
        load_gate_config_into(main_window, clip)
        orch = main_window._config_orch
        reference = orch.get_parameters_dict()
        cfg = orch.build_config_dict()
        rt = gui_runtime_context(main_window)
        shared = build_engine_params(
            cfg, runtime=rt, advanced_config=main_window.advanced_config
        )
        diffs = _compare(reference, shared)

        gui_only = sorted(k for k in diffs if k not in shared)
        shared_only = sorted(k for k in diffs if k not in reference)
        mismatched = sorted(k for k in diffs if k in reference and k in shared)

        lines.append(f"## {clip}  (total diverging: {len(diffs)})")
        lines.append(f"### GUI-only (missing from shared builder): {len(gui_only)}")
        for k in gui_only:
            lines.append(f"    {k} = {diffs[k][0]!r}")
        lines.append(
            f"### shared-only (spurious in shared builder): {len(shared_only)}"
        )
        for k in shared_only:
            lines.append(f"    {k} = {diffs[k][1]!r}")
        lines.append(
            f"### value-mismatch (present both sides, unequal): {len(mismatched)}"
        )
        for k in mismatched:
            lines.append(f"    {k}: reference={diffs[k][0]!r}  shared={diffs[k][1]!r}")
        lines.append("")

    KEYDIFF_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEYDIFF_BASELINE_PATH.write_text("\n".join(lines) + "\n")
    assert KEYDIFF_BASELINE_PATH.is_file()
