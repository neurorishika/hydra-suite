"""Regression guard: final-media export roots must be resolvable from params alone.

The Slice-5 cutover (a218a93a) moved final-media export into
``TrackingSessionCore`` but no caller ever wrote the four ``paths`` keys the
stage reads -- ``individual_dataset_dir``, ``final_media_video_dir``,
``interpolated_roi_npz_path`` and ``source_video_fps``. Both roots resolved to
``None`` on every real run, so ``export_final_media`` logged "no image/video
output directory found" and returned before building the exporter.

The pre-existing ``test_session_export_chain`` fixtures hand-feed those keys, so
they could not catch it. These tests deliberately build the paths dict the way
production callers do (``gui/orchestrators/tracking.py:841`` and
``headless_tracking.py:298``) and assert the stage still resolves its roots.
"""

import pytest

import hydra_suite.core.tracking.session as session_mod
from hydra_suite.core.tracking.session import SessionCallbacks, TrackingSessionCore

PRODUCTION_CONFIG = {
    # Both gates on, under the YOLO-OBB detection method the policy requires.
    "detection_method": "yolo_obb",
    "enable_individual_dataset": True,
    "final_media_export_videos_enabled": True,
    "individual_background_color": [0, 0, 0],
}


def _production_params(tmp_path, run_id="20260825_160911"):
    """The output-context keys ``build_engine_params`` always emits."""
    return {
        "INDIVIDUAL_DATASET_OUTPUT_DIR": str(tmp_path / "ds" / "individual_crops"),
        "INDIVIDUAL_DATASET_NAME": "",
        "INDIVIDUAL_DATASET_RUN_ID": run_id,
        "FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR": str(tmp_path / "ds" / "oriented_videos"),
        "FPS": 25.0,
    }


def _production_service(tmp_path, params, extra_paths=None):
    """A service whose ``paths`` dict carries ONLY what real callers supply."""
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"x")
    paths = {
        "raw_csv_path": str(tmp_path / "raw.csv"),
        "detection_cache_path": str(cache),
        "individual_properties_cache_path": None,
        "detected_properties_cache_path": None,
    }
    paths.update(extra_paths or {})
    return TrackingSessionCore(
        video_path=str(tmp_path / "in.mp4"),
        config=dict(PRODUCTION_CONFIG),
        params=params,
        paths=paths,
        callbacks=SessionCallbacks(),
    )


@pytest.fixture
def captured_export(monkeypatch):
    seen = {}

    def _fake_export(**kwargs):
        seen.update(kwargs)
        return {"output_dir": "vids", "image_output_dir": "imgs"}

    monkeypatch.setattr(session_mod.media_export, "export_final_media", _fake_export)
    return seen


def test_media_export_roots_resolve_from_params(tmp_path, captured_export):
    """Both roots must be non-None with a production-shaped paths dict."""
    params = _production_params(tmp_path)
    svc = _production_service(tmp_path, params)
    final_csv = tmp_path / "final.csv"
    final_csv.write_text("TrajectoryID,X,Y,Theta,FrameID\n0,1,2,0,0\n")

    svc._run_final_media_export(str(final_csv))

    assert captured_export, "export_final_media was never called"
    assert captured_export["image_root"] is not None, (
        "image_root is None -> 'Skipping final canonical image export: "
        "no image output directory found.'"
    )
    assert captured_export["video_root"] is not None, (
        "video_root is None -> 'Skipping final media video export: "
        "no video output directory found.'"
    )
    # Roots must land in the per-run subfolder, matching the pre-cutover
    # _resolve_current_* helpers in main_window.py.
    assert captured_export["image_root"].name == "20260825_160911"
    assert captured_export["video_root"].name == "20260825_160911"


def test_media_export_fps_is_not_degenerate(tmp_path, captured_export):
    """fps=None becomes max(0.1, 0.0) -> 0.1 FPS videos in the exporter."""
    params = _production_params(tmp_path)
    svc = _production_service(tmp_path, params)
    final_csv = tmp_path / "final.csv"
    final_csv.write_text("TrajectoryID,X,Y,Theta,FrameID\n0,1,2,0,0\n")

    svc._run_final_media_export(str(final_csv))

    assert captured_export["fps"] == 25.0


def test_explicit_paths_override_params(tmp_path, captured_export):
    """A caller that DOES supply the keys keeps winning (back-compat)."""
    params = _production_params(tmp_path)
    override_img = tmp_path / "explicit_images"
    override_vid = tmp_path / "explicit_videos"
    svc = _production_service(
        tmp_path,
        params,
        extra_paths={
            "individual_dataset_dir": override_img,
            "final_media_video_dir": override_vid,
            "source_video_fps": 60.0,
        },
    )
    final_csv = tmp_path / "final.csv"
    final_csv.write_text("TrajectoryID,X,Y,Theta,FrameID\n0,1,2,0,0\n")

    svc._run_final_media_export(str(final_csv))

    assert str(captured_export["image_root"]) == str(override_img)
    assert str(captured_export["video_root"]) == str(override_vid)
    assert captured_export["fps"] == 60.0


def test_interpolated_roi_npz_path_is_captured_from_postpass(tmp_path, monkeypatch):
    """_run_interp_crops must not drop the payload's roi_npz_path."""
    params = _production_params(tmp_path)
    svc = _production_service(tmp_path, params)
    roi_npz = tmp_path / "interpolated_rois.npz"
    roi_npz.write_bytes(b"x")

    monkeypatch.setattr(
        session_mod,
        "run_interpolated_crops",
        lambda *a, **k: {"roi_npz_path": str(roi_npz)},
    )
    svc._run_interp_crops(str(tmp_path / "final.csv"))

    assert svc._interpolated_roi_npz_path == str(roi_npz)
