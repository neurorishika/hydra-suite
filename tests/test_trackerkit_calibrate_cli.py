"""The headless equivalent of the Calibrate button."""

import cv2
import numpy as np
import pytest

from hydra_suite.trackerkit.app import build_parser


def _write_tiny_video(
    path, *, n_frames: int = 8, size: tuple[int, int] = (16, 12)
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    try:
        for i in range(n_frames):
            frame = np.full((size[1], size[0], 3), i % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_track_exposes_the_apply_flag_and_not_the_old_modes():
    parser = build_parser()
    args = parser.parse_args(["track", "--video", "v.mp4", "--apply-tuned-inference"])
    assert args.apply_tuned_inference is True
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["track", "--video", "v.mp4", "--inference-autotune", "record"]
        )


def test_no_apply_flag_disables():
    parser = build_parser()
    args = parser.parse_args(
        ["track", "--video", "v.mp4", "--no-apply-tuned-inference"]
    )
    assert args.apply_tuned_inference is False


def test_calibrate_subcommand_exists_with_a_budget():
    parser = build_parser()
    args = parser.parse_args(
        ["calibrate", "--video", "v.mp4", "--budget-seconds", "600"]
    )
    assert args.command == "calibrate"
    assert args.budget_seconds == 600.0


def test_calibrate_refuses_fanout_flags():
    """Concurrent calibration on one box measures contention, not throughput."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["calibrate", "--video", "v.mp4", "--gpus", "auto"])


def test_clamp_frame_range_pins_out_of_range_bounds():
    """The shared clamp both ``track`` and ``calibrate`` derive bounds
    through -- a config carrying frame bounds stale from a different
    (e.g. longer) video must clamp identically on both paths, or the two
    fingerprint different (start_frame, end_frame) pairs (Task 10 review
    Finding 1)."""
    from hydra_suite.core.inference.config import clamp_frame_range

    # Both bounds far beyond a 50-frame video.
    assert clamp_frame_range(0, 999, total_video_frames=50) == (0, 49)
    assert clamp_frame_range(40, 999, total_video_frames=50) == (40, 49)
    # start_frame itself out of range.
    assert clamp_frame_range(999, None, total_video_frames=50) == (49, 49)
    # end_frame omitted entirely -> whole video.
    assert clamp_frame_range(0, None, total_video_frames=50) == (0, 49)
    # No frame count known (unseekable stream) -> no clamping.
    assert clamp_frame_range(0, 999, total_video_frames=None) == (0, 999)


def test_calibrate_and_track_derive_identical_context_inputs(tmp_path):
    """The one property the whole feature rests on: for the same
    video/config, ``calibrate_cli``'s derivation of the five values fed into
    ``build_autotune_context`` (cache_dir, start_frame, end_frame, realtime,
    use_cached_detections) must agree with what ``worker.py``'s equivalent
    block produces -- otherwise the two paths fingerprint different
    contexts, a calibration writes a profile under a key ``track`` never
    looks up, and the whole feature silently does nothing.

    This does not need models or the autotune planner: it exercises the
    REAL production derivation (``calibrate_cli.derive_context_inputs``,
    the exact function ``run_calibrate_cli`` calls) against the
    session/param/probe plumbing both paths share -- which is exactly where
    Finding 1 (unclamped explicit frame bounds) lived. Reimplementing the
    derivation inline in the test would not have caught that: the bug was
    two independent derivations silently drifting apart, so the test has to
    call the real one.
    """
    from hydra_suite.core.inference.config import clamp_frame_range
    from hydra_suite.trackerkit.calibrate_cli import derive_context_inputs
    from hydra_suite.trackerkit.cli_config import load_tracker_cli_session
    from hydra_suite.utils.video_artifacts import build_inference_cache_dir

    video_path = tmp_path / "clip.mp4"
    _write_tiny_video(video_path, n_frames=10)

    # A stale/hand-edited config: END_FRAME left over from a much longer
    # video than the one actually being processed now.
    config_data = {
        "start_frame": 0,
        "end_frame": 9999,
        "use_cached_detections": True,
        "tracking_realtime_mode": False,
    }

    session = load_tracker_cli_session(str(video_path), config_data=config_data)
    params = session.params
    probe = session.video_probe

    # --- What calibrate_cli.run_calibrate_cli actually derives. ---
    inputs = derive_context_inputs(
        str(video_path),
        params,
        probe,
        use_cached_detections=session.use_cached_detections,
    )
    calibrate_cache_dir = inputs.cache_dir
    calibrate_start, calibrate_end = inputs.start_frame, inputs.end_frame
    calibrate_realtime = inputs.realtime
    calibrate_use_cache = inputs.use_cached_detections

    # --- worker.py's equivalent block, reproduced with cv2's OWN frame
    # count (not the session's probe) -- this is the one input the two
    # paths read independently, so it is the one worth cross-checking
    # rather than assuming agreement. ---
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    assert cap.isOpened()
    try:
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    finally:
        cap.release()
    worker_start, worker_end = clamp_frame_range(
        params.get("START_FRAME", 0), params.get("END_FRAME", None), total_video_frames
    )
    # preview_mode/backward_mode are both False on a plain `track` run, so
    # effective_realtime_tracking_mode reduces to the requested flag.
    worker_realtime = bool(params.get("TRACKING_REALTIME_MODE", False))
    worker_use_cache = bool(session.use_cached_detections)
    worker_cache_dir = build_inference_cache_dir(str(video_path))

    assert calibrate_start == worker_start
    assert calibrate_end == worker_end
    # Pins the out-of-range END_FRAME actually clamping down to the real
    # video length, not silently passing 9999 through.
    assert calibrate_end == 9
    assert calibrate_realtime == worker_realtime
    assert calibrate_use_cache == worker_use_cache is True
    assert calibrate_cache_dir == worker_cache_dir


def test_calibrate_accepts_the_manual_field_flag():
    """I4: manual fields feed ``compute_baseline_digest`` -> the profile KEY.
    Without this flag, ``calibrate`` could not produce the key a
    ``track --inference-autotune-manual ...`` run looks up.
    """
    parser = build_parser()
    args = parser.parse_args(
        [
            "calibrate",
            "--video",
            "v.mp4",
            "--inference-autotune-manual",
            "pose_batch_size",
            "--inference-autotune-manual",
            "pipeline_depth",
        ]
    )
    assert args.inference_autotune_manual == ["pose_batch_size", "pipeline_depth"]


def test_calibrate_threads_manual_fields_through_the_same_override_as_track(
    monkeypatch, tmp_path
):
    """Parser existence is not enough: the flag must reach the config the
    session is built from, through the SAME ``apply_inference_autotune_override``
    helper ``track`` uses (batch_plan.py), or the two key differently.
    """
    from hydra_suite.trackerkit import calibrate_cli, cli_config

    video = tmp_path / "v.mp4"
    _write_tiny_video(video)

    captured = {}

    class _Stop(Exception):
        pass

    def fake_session(video_path, *, config_path=None, config_data=None, **_kwargs):
        captured["config_data"] = config_data
        raise _Stop()

    monkeypatch.setattr(cli_config, "load_tracker_cli_session", fake_session)

    with pytest.raises(_Stop):
        calibrate_cli.run_calibrate_cli(
            str(video),
            budget_seconds=60.0,
            inference_autotune_manual=["pose_batch_size"],
        )

    assert (
        captured["config_data"] is not None
    ), "manual fields must be merged into a config_data override, not dropped"
    assert captured["config_data"]["inference_autotune_manual_fields"] == [
        "pose_batch_size"
    ]
    # And it is the SAME merge `track` performs.
    expected = cli_config.apply_inference_autotune_override(
        cli_config.load_tracker_cli_config(None),
        manual_fields=("pose_batch_size",),
    )
    assert (
        captured["config_data"]["inference_autotune_manual_fields"]
        == expected["inference_autotune_manual_fields"]
    )
