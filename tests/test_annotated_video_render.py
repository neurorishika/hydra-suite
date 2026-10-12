"""End-to-end final annotated video render: scale, frame range, threads."""

from __future__ import annotations

import os
import threading
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.post import media_export as mx
from hydra_suite.utils import video_decoder as vd

av = pytest.importorskip("av")

from tests.test_video_decoder import _decode_index, _write_index_clip  # noqa: E402


def _df(n=40):
    return pd.DataFrame(
        {
            "TrajectoryID": [0] * n,
            "FrameID": list(range(n)),
            "X": [120.0] * n,
            "Y": [8.0] * n,
            "Theta": [0.0] * n,
        }
    )


def _params(start=0, end=None):
    return {
        "TRAJECTORY_COLORS": [(0, 255, 0)],
        "REFERENCE_BODY_SIZE": 6.0,
        "ADVANCED_CONFIG": {},
        "POSE_MIN_KPT_CONF_VALID": 0.2,
        "START_FRAME": start,
        "END_FRAME": end,
    }


def _config(**extra):
    cfg = {
        "video_show_labels": False,
        "video_show_orientation": True,
        "video_show_trails": False,
        "video_marker_size": 0.3,
        "video_text_scale": 0.5,
        "video_arrow_length": 0.7,
    }
    cfg.update(extra)
    return cfg


@pytest.fixture()
def clip(tmp_path):
    path = tmp_path / "in.mp4"
    _write_index_clip(path, n_frames=40, w=128, h=64)
    return path


def _read_output(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        frames = [f.to_ndarray(format="bgr24") for f in c.decode(s)]
        return frames, s.width, s.height, s.average_rate


def _decode_threads():
    return [t for t in threading.enumerate() if t.name == "video-decode"]


def test_default_scale_renders_half_size_mp4_with_every_frame(clip, tmp_path):
    out = tmp_path / "out.mp4"
    result = mx.render_annotated_video(
        trajectories_df=_df(),
        video_path=str(clip),
        output_path=str(out),
        params=_params(),
        config=_config(),  # no video_output_scale -> 0.5
    )
    assert result == str(out)
    frames, w, h, _ = _read_output(out)
    assert (w, h) == (64, 32)
    assert len(frames) == 40
    assert [_decode_index(f) for f in frames] == list(range(40))
    assert not _decode_threads()


def test_scale_one_renders_source_size(clip, tmp_path):
    out = tmp_path / "out.mp4"
    mx.render_annotated_video(
        trajectories_df=_df(),
        video_path=str(clip),
        output_path=str(out),
        params=_params(),
        config=_config(video_output_scale=1.0),
    )
    frames, w, h, _ = _read_output(out)
    assert (w, h) == (128, 64) and len(frames) == 40


@pytest.mark.parametrize("start,end", [(5, 14), (10, 39), (23, 30)])
def test_start_end_frame_range_is_preserved(clip, tmp_path, start, end):
    out = tmp_path / "out.mp4"
    mx.render_annotated_video(
        trajectories_df=_df(),
        video_path=str(clip),
        output_path=str(out),
        params=_params(start, end),
        config=_config(),
    )
    frames, _, _, _ = _read_output(out)
    assert [_decode_index(f) for f in frames] == list(range(start, end + 1))


def test_out_of_range_scale_is_a_loud_error(clip, tmp_path):
    out = tmp_path / "out.mp4"
    with pytest.raises(ValueError, match="video_output_scale"):
        mx.render_annotated_video(
            trajectories_df=_df(),
            video_path=str(clip),
            output_path=str(out),
            params=_params(),
            config=_config(video_output_scale=1.5),
        )
    assert not out.exists()


def test_cancel_stops_decode_thread_and_removes_partial_output(clip, tmp_path):
    out = tmp_path / "out.mp4"
    calls = {"n": 0}

    def _stop():
        calls["n"] += 1
        return calls["n"] > 3

    result = mx.render_annotated_video(
        trajectories_df=_df(),
        video_path=str(clip),
        output_path=str(out),
        params=_params(),
        config=_config(),
        should_stop=_stop,
    )
    assert result is None
    assert not out.exists()
    assert not _decode_threads()


class _BrokenReader:
    def __init__(self, start):
        self.i = start

    def read(self):
        if self.i >= 6:
            raise RuntimeError("boom at 6")
        self.i += 1
        return np.zeros((32, 64, 3), np.uint8)

    def close(self):
        pass


def test_decode_thread_failure_propagates_and_removes_partial_output(clip, tmp_path):
    out = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="boom at 6"):
        mx.render_annotated_video(
            trajectories_df=_df(),
            video_path=str(clip),
            output_path=str(out),
            params=_params(),
            config=_config(),
            decoder_candidates=[vd.DecoderCandidate("broken", _BrokenReader)],
        )
    assert not out.exists()
    assert not _decode_threads()


def test_fractional_fps_is_kept(tmp_path, monkeypatch):
    # Re-probe: other tests may leave the process-wide backend cache on
    # "opencv" (cv2 mp4v), which is not the PyAV path under test.
    from hydra_suite.utils import video_encoder

    monkeypatch.setattr(video_encoder, "_BACKEND_CACHE", None)
    src = tmp_path / "ntsc.mp4"
    c = av.open(str(src), mode="w")
    st = c.add_stream("libx264", rate=Fraction(30000, 1001))
    st.width, st.height, st.pix_fmt = 64, 64, "yuv420p"
    for _ in range(12):
        frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), "bgr24")
        for pkt in st.encode(frame):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    out = tmp_path / "out.mp4"
    mx.render_annotated_video(
        trajectories_df=_df(12),
        video_path=str(src),
        output_path=str(out),
        params=_params(),
        config=_config(video_output_scale=1.0),
    )
    _, _, _, rate = _read_output(out)
    assert rate == Fraction(30000, 1001)
    assert os.path.getsize(out) > 0
