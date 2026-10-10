"""Decode stage for the final annotated video (``utils/video_decoder.py``)."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
import pytest

from hydra_suite.utils import video_decoder as vd

# ── fakes ────────────────────────────────────────────────────────────────────


def _frame(idx, w=8, h=6):
    f = np.zeros((h, w, 3), np.uint8)
    f[0, 0, 0] = idx % 256
    f[0, 0, 1] = idx // 256
    return f


def _idx(frame):
    return int(frame[0, 0, 0]) + 256 * int(frame[0, 0, 1])


class _FakeReader:
    def __init__(self, start, n_total, *, fail_at=None, delay=0.0, log=None):
        self.next = start
        self.n_total = n_total
        self.fail_at = fail_at
        self.delay = delay
        self.closed = False
        self.log = log

    def read(self):
        if self.fail_at is not None and self.next >= self.fail_at:
            raise RuntimeError(f"decode error at {self.next}")
        if self.next >= self.n_total:
            return None
        if self.delay:
            time.sleep(self.delay * (1 + (self.next % 3)))
        f = _frame(self.next)
        self.next += 1
        return f

    def close(self):
        self.closed = True


def _candidate(name, opened, **kw):
    def _open(start):
        if kw.get("open_raises"):
            raise RuntimeError(f"{name} unavailable")
        r = _FakeReader(start, kw.get("n_total", 50), fail_at=kw.get("fail_at"))
        opened.append((name, start, r))
        return r

    return vd.DecoderCandidate(name, _open)


# ── ladder / probe ───────────────────────────────────────────────────────────


def test_falls_through_to_first_candidate_that_decodes_a_frame(caplog):
    opened = []
    cands = [
        _candidate("cuvid", opened, open_raises=True),
        _candidate("hwaccel", opened, fail_at=0),  # opens, first decode fails
        _candidate("pyav", opened),
        _candidate("opencv", opened),
    ]
    with caplog.at_level(logging.INFO, logger=vd.logger.name):
        src = vd.FrameSource(cands, start_frame=7, out_size=(8, 6))
    assert src.name == "pyav"
    assert [_idx(src.read()) for _ in range(3)] == [7, 8, 9]
    # The failed hwaccel reader was closed; opencv was never tried.
    assert [o[0] for o in opened] == ["hwaccel", "pyav"]
    assert opened[0][2].closed
    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(info) == 1
    assert "pyav" in info[0].getMessage() and "8x6" in info[0].getMessage()
    src.close()
    assert opened[1][2].closed


def test_all_candidates_fail_is_a_loud_error():
    opened = []
    cands = [
        _candidate("a", opened, open_raises=True),
        _candidate("b", opened, fail_at=0),
    ]
    with pytest.raises(RuntimeError, match="No video decoder"):
        vd.FrameSource(cands, start_frame=0, out_size=(8, 6))


def test_midstream_error_falls_back_from_the_current_frame_index():
    opened = []
    cands = [_candidate("hw", opened, fail_at=12), _candidate("sw", opened)]
    src = vd.FrameSource(cands, start_frame=5, out_size=(8, 6))
    got = [_idx(src.read()) for _ in range(15)]
    assert got == list(range(5, 20))  # no drop, no duplicate
    assert [(o[0], o[1]) for o in opened] == [("hw", 5), ("sw", 12)]
    assert src.name == "sw"


def test_midstream_error_on_last_candidate_raises():
    opened = []
    src = vd.FrameSource(
        [_candidate("only", opened, fail_at=3)], start_frame=0, out_size=(8, 6)
    )
    for _ in range(3):
        src.read()
    with pytest.raises(RuntimeError, match="decode error at 3"):
        src.read()


def test_eof_returns_none():
    opened = []
    src = vd.FrameSource(
        [_candidate("x", opened, n_total=2)], start_frame=0, out_size=(8, 6)
    )
    assert src.read() is not None and src.read() is not None
    assert src.read() is None


# ── threaded stage ───────────────────────────────────────────────────────────


def _slow_source_factory(n_total, delay=0.0005, fail_at=None):
    def _factory():
        cand = vd.DecoderCandidate(
            "fake",
            lambda start: _FakeReader(start, n_total, fail_at=fail_at, delay=delay),
        )
        return vd.FrameSource([cand], start_frame=0, out_size=(8, 6))

    return _factory


def test_threaded_reader_delivers_frames_in_order():
    reader = vd.ThreadedFrameReader(_slow_source_factory(60), max_frames=60, maxsize=3)
    reader.start()
    got = []
    while True:
        f = reader.get()
        if f is None:
            break
        got.append(_idx(f))
    reader.close()
    assert got == list(range(60))


def test_threaded_reader_stops_at_max_frames():
    reader = vd.ThreadedFrameReader(_slow_source_factory(60), max_frames=10)
    reader.start()
    got = []
    while (f := reader.get()) is not None:
        got.append(_idx(f))
    reader.close()
    assert got == list(range(10))


def test_decode_thread_exception_propagates_to_caller():
    reader = vd.ThreadedFrameReader(_slow_source_factory(60, fail_at=4), max_frames=60)
    reader.start()
    got = []
    with pytest.raises(RuntimeError, match="decode error at 4"):
        while (f := reader.get()) is not None:
            got.append(_idx(f))
    assert got == [0, 1, 2, 3]
    reader.close()


def test_close_stops_a_producer_blocked_on_a_full_queue():
    closed = []

    def _factory():
        def _open(start):
            r = _FakeReader(start, 10_000)
            closed.append(r)
            return r

        return vd.FrameSource(
            [vd.DecoderCandidate("x", _open)], start_frame=0, out_size=(8, 6)
        )

    reader = vd.ThreadedFrameReader(_factory, max_frames=10_000, maxsize=2)
    reader.start()
    assert reader.get() is not None
    time.sleep(0.05)  # producer now blocked on the full queue
    t0 = time.monotonic()
    reader.close()
    assert time.monotonic() - t0 < 2.0
    assert not reader.thread_alive()
    assert closed and closed[0].closed


# ── real decoders ────────────────────────────────────────────────────────────

_BITS = 8


def _write_index_clip(path, n_frames=40, w=128, h=64, gop=10, fps=25):
    """H.264 clip (B-frames, short GOP) whose frame index is drawn as 8 bit-blocks."""
    av = pytest.importorskip("av")
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.options = {"g": str(gop), "bf": "2", "keyint_min": str(gop)}
    bw = w // _BITS
    for i in range(n_frames):
        img = np.zeros((h, w, 3), np.uint8)
        for b in range(_BITS):
            if (i >> b) & 1:
                img[:, b * bw : (b + 1) * bw] = 255
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def _decode_index(img):
    w = img.shape[1]
    bw = w // _BITS
    return sum(
        1 << b
        for b in range(_BITS)
        if img[:, b * bw + bw // 4 : (b + 1) * bw - bw // 4].mean() > 128
    )


def _cv2_indices(path, start, n):
    cap = cv2.VideoCapture(str(path))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        out.append(_decode_index(f))
    cap.release()
    return out


def _source_indices(cands, start, n):
    src = vd.FrameSource(cands, start_frame=start, out_size=(128, 64))
    out = []
    try:
        for _ in range(n):
            f = src.read()
            if f is None:
                break
            assert f.dtype == np.uint8 and f.flags.c_contiguous
            assert f.shape == (64, 128, 3)
            out.append(_decode_index(f))
    finally:
        src.close()
    return out


@pytest.fixture(scope="module")
def index_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "index.mp4"
    _write_index_clip(path)
    return path


@pytest.mark.parametrize("start", [0, 5, 10, 23, 39])
@pytest.mark.parametrize("kind", ["pyav", "opencv", "hwaccel"])
def test_start_frame_alignment_matches_cv2(index_clip, start, kind):
    cands = {c.name: c for c in vd.default_decoder_candidates(str(index_clip), 128, 64)}
    if kind not in cands:
        pytest.skip(f"{kind} decoder not available here")
    expected = _cv2_indices(index_clip, start, 40)
    assert expected == list(range(start, 40))  # the clip itself is sane
    assert _source_indices([cands[kind]], start, 40) == expected


def test_default_ladder_order():
    names = [c.name for c in vd.default_decoder_candidates("x.mp4", 64, 48)]
    assert names[-2:] == ["pyav", "opencv"]
    if "cuvid" in names:
        assert names.index("cuvid") < names.index("hwaccel")
    if "hwaccel" in names:
        assert names.index("hwaccel") < names.index("pyav")


def test_downscaled_decode_is_output_size(index_clip):
    for c in vd.default_decoder_candidates(str(index_clip), 64, 32):
        if c.name not in ("pyav", "opencv"):
            continue
        src = vd.FrameSource([c], start_frame=3, out_size=(64, 32))
        f = src.read()
        src.close()
        assert f.shape == (32, 64, 3) and f.flags.c_contiguous
        assert _decode_index(f) == 3


def test_frame_to_bgr_handles_odd_source_and_padded_planes():
    av = pytest.importorskip("av")
    yy, xx = np.mgrid[0:51, 0:77]
    img = (
        np.stack([xx * 3, yy * 4, (xx + yy) * 2], axis=2).clip(0, 255).astype(np.uint8)
    )
    frame = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p")
    out = vd.frame_to_bgr(frame, 38, 24)
    assert out.shape == (24, 38, 3) and out.flags.c_contiguous
    even = av.VideoFrame.from_ndarray(img[:50, :76].copy(), format="bgr24").reformat(
        format="yuv420p"
    )
    full = vd.frame_to_bgr(even, 76, 50)
    assert full.shape == (50, 76, 3)
    # Same BT.601 limited-range conversion as swscale, within rounding.
    ref = even.reformat(format="bgr24").to_ndarray()
    assert np.abs(full.astype(int) - ref.astype(int)).max() <= 3


def test_pyav_start_frame_uses_keyframe_seek_not_full_rescan(index_clip, caplog):
    cands = {c.name: c for c in vd.default_decoder_candidates(str(index_clip), 128, 64)}
    with caplog.at_level(logging.DEBUG, logger=vd.logger.name):
        assert _source_indices([cands["pyav"]], 23, 3) == [23, 24, 25]
    assert not any("counting from 0" in r.getMessage() for r in caplog.records)


def test_unreliable_pts_falls_back_to_counting_from_stream_start(
    index_clip, monkeypatch
):
    monkeypatch.setattr(vd._PyAVReader, "_seek_frames", lambda self: None)
    cands = {c.name: c for c in vd.default_decoder_candidates(str(index_clip), 128, 64)}
    assert _source_indices([cands["pyav"]], 17, 40) == list(range(17, 40))
