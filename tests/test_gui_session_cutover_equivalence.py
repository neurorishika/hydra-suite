"""The GUI post-tracking path yields the same final CSV as the CLI.

Regression guard for the Task-4/5 deletion: proves TrackingOrchestrator's
Qt-free ``SessionWorker`` -> ``TrackingSessionCore.run_post_tracking`` path
produces a byte-identical final CSV to ``run_tracking_cli`` when fed the
SAME raw tracking CSVs and the SAME ``config``/``params`` dicts the CLI
derives via ``cli_config.load_tracker_cli_session`` (config =
``load_tracker_cli_config``, params = ``build_tracking_parameters``). The
GUI orchestrator is driven directly (no MainWindow, no event loop): the
SessionWorker's ``start()`` is monkeypatched to call ``execute()``
synchronously.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

REPO = Path(__file__).resolve().parents[1]
FIXTURE_CLIPS_DIR = REPO / "tools/equivalence/fixtures/clips"
FIXTURE_CONFIGS_DIR = REPO / "tools/equivalence/fixtures/configs"
# Fixture configs are portable (pose_skeleton_file left blank); the equivalence
# runner injects a skeleton at run time (see tools/equivalence/runner.py
# build_config). Mirror that here for clips whose pose model needs keypoint
# names -- per tools/equivalence/fixtures/make_manifest.py's clip->skeleton map.
SKELETON_FILE = REPO / "tools/equivalence/fixtures/ooceraea_biroi.json"
CLIP_SKELETONS = {
    "ant_cnn_identity": SKELETON_FILE,
    "ant_pose_headtail": SKELETON_FILE,
}

# Clips covered end-to-end (GUI post-tracking path == CLI path):
#   - fly_obb: OBB detection, no identity/pose (baseline smoke clip)
#   - ant_cnn_identity: CNN identity classification + SLEAP pose (headtail)
#   - ant_pose_headtail: SLEAP pose path, orientation from pose
# Both ant_* clips need the `sleap` conda env (the SLEAP service spawns
# `conda run -n sleap`).
CLIPS = ["fly_obb", "ant_cnn_identity", "ant_pose_headtail"]


def _clip_paths(name: str) -> tuple[Path, Path]:
    return (FIXTURE_CLIPS_DIR / f"{name}.mp4", FIXTURE_CONFIGS_DIR / f"{name}.json")


def _materialize_config(name: str, config_path: Path, tmp_path: Path) -> Path:
    """Return a config path with pose_skeleton_file filled in, if needed.

    Fixture configs are intentionally portable (blank pose_skeleton_file); a
    concrete skeleton is normally supplied at run time (see
    tools/equivalence/runner.py). Write a materialized copy for clips that
    need it so pose keypoint_names resolve; pass the original path through
    unchanged otherwise.
    """
    skeleton = CLIP_SKELETONS.get(name)
    if skeleton is None:
        return config_path
    cfg = json.loads(config_path.read_text())
    cfg["pose_skeleton_file"] = str(skeleton)
    out_path = tmp_path / f"{name}_config.json"
    out_path.write_text(json.dumps(cfg))
    return out_path


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("clip_name", CLIPS)
def test_gui_run_session_worker_matches_cli(qapp, tmp_path, clip_name):
    clip_path, config_path = _clip_paths(clip_name)
    if not (clip_path.exists() and config_path.exists()):
        pytest.skip(
            f"{clip_name} fixture missing "
            "(run tools/equivalence/fixtures/fetch_fixtures.sh)"
        )
    config_path = _materialize_config(clip_name, config_path, tmp_path)

    from hydra_suite.trackerkit.cli import run_tracking_cli
    from hydra_suite.trackerkit.cli_config import load_tracker_cli_session
    from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator

    # 1) CLI run -> canonical final CSV (also the source of the SAME raw
    #    forward/backward CSVs and detection cache the GUI path re-reads).
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    cli_clip = cli_dir / f"{clip_name}.mp4"
    shutil.copy(clip_path, cli_clip)
    assert run_tracking_cli([str(cli_clip)], config_path=str(config_path)) == 0
    cli_final = next(cli_dir.glob("*_tracking_final.csv"))
    cli_forward_raw = next(cli_dir.glob("*_tracking_forward.csv"))
    cli_backward_raw = next(cli_dir.glob("*_tracking_backward.csv"))

    # 2) GUI post-tracking over the SAME raw CSVs the CLI produced (backward
    #    tracking is enabled in these configs, so both the "_forward" and
    #    "_backward" raw CSVs -- not just a single raw CSV -- must exist next
    #    to the GUI's raw_csv_path base, since TrackingSessionCore derives
    #    those two paths from `paths["raw_csv_path"]` itself).
    gui_dir = tmp_path / "gui"
    gui_dir.mkdir()
    gui_raw = gui_dir / f"{clip_name}_tracking.csv"
    shutil.copy(cli_forward_raw, gui_dir / f"{clip_name}_tracking_forward.csv")
    shutil.copy(cli_backward_raw, gui_dir / f"{clip_name}_tracking_backward.csv")

    # Derive `config`/`params` EXACTLY the way the CLI did: same function,
    # same video path (so the video-probe-derived FPS/frame-count params
    # match), same config file. This is the crux reconciliation -- a raw
    # `json.loads(config_path)` config dict is fine for `config` (it *is*
    # what `load_tracker_cli_config` returns), but `params` MUST be the
    # `build_tracking_parameters` derivation, not the raw config dict, or
    # the post-processing stages read wrong keys/units and diverge.
    cli_session = load_tracker_cli_session(str(cli_clip), config_path=str(config_path))
    config = cli_session.config
    params = cli_session.params

    # The "detection_cache_path" the GUI passes through `paths` is not the
    # InferenceRunner's own per-video cache file (that lives under a hidden
    # ``.inference_cache_<stem>/`` dir keyed purely off the video path/config,
    # not `use_cached_detections`) -- it is the anchor path
    # `plan_tracking_cache` resolves, which post-processing uses as a
    # directory anchor for identity/pose sidecar caches. Recompute it exactly
    # as `run_headless_tracking_session` does, from the SAME params/session,
    # so the GUI stub gets byte-identical `paths`.
    from hydra_suite.trackerkit.tracking_cache import plan_tracking_cache

    cache_plan = plan_tracking_cache(
        str(cli_clip),
        params=dict(params),
        preferred_output_dir=os.path.dirname(cli_session.raw_csv_path),
        use_cached_detections=cli_session.use_cached_detections,
    )
    detection_cache_path = cache_plan.detection_cache_path

    panels = SimpleNamespace(
        setup=SimpleNamespace(
            csv_line=SimpleNamespace(text=lambda: str(gui_raw)),
            file_line=SimpleNamespace(text=lambda: str(cli_clip)),
        ),
    )
    finalized = {"done": False}
    mw = SimpleNamespace(
        _stop_all_requested=False,
        _session_final_csv_path=None,
        _session_fps_list=[],
        current_detection_cache_path=detection_cache_path,
        current_individual_properties_cache_path=None,
        current_detected_properties_cache_path=None,
        get_parameters_dict=lambda: params,
        progress_bar=SimpleNamespace(
            setVisible=lambda *_: None, setValue=lambda *_: None
        ),
        progress_label=SimpleNamespace(
            setVisible=lambda *_: None, setText=lambda *_: None
        ),
    )
    orch = TrackingOrchestrator(main_window=mw, config=object(), panels=panels)
    orch._mw = mw
    orch._finalize_tracking_session_ui = lambda: finalized.__setitem__("done", True)
    orch._build_session_config = lambda: config  # RECONCILED: same dict the CLI used

    # run the SessionWorker synchronously by calling execute() rather than start()
    from hydra_suite.trackerkit.gui.workers import session_worker as sw_module

    original_start = sw_module.SessionWorker.start
    sw_module.SessionWorker.start = lambda self: self.execute()
    try:
        orch._run_session_worker()
    finally:
        sw_module.SessionWorker.start = original_start

    assert finalized["done"] is True
    gui_final = next(gui_dir.glob("*_tracking_final.csv"))
    pd.testing.assert_frame_equal(
        pd.read_csv(gui_final).reset_index(drop=True),
        pd.read_csv(cli_final).reset_index(drop=True),
    )
