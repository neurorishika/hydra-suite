import sys
from unittest import mock

import numpy as np
import pytest

# ── probe_video_backend ────────────────────────────────────────────────────────


def test_probe_returns_known_backend():
    """probe_video_backend() always returns one of the four known backend strings."""
    import hydra_suite.utils.video_encoder as ve

    ve._BACKEND_CACHE = None
    backend = ve.probe_video_backend()
    assert backend in ("nvenc", "videotoolbox", "pyav_software", "opencv")


def test_probe_result_is_cached():
    """Calling probe_video_backend() twice returns the same value."""
    import hydra_suite.utils.video_encoder as ve

    ve._BACKEND_CACHE = None
    b1 = ve.probe_video_backend()
    b2 = ve.probe_video_backend()
    assert b1 == b2


def test_probe_falls_back_to_opencv_when_av_unavailable():
    """_probe_backend() returns 'opencv' when av cannot be imported."""
    import hydra_suite.utils.video_encoder as ve

    saved = sys.modules.get("av")
    try:
        sys.modules["av"] = None  # type: ignore[assignment]
        ve._BACKEND_CACHE = None
        result = ve._probe_backend()
        assert result == "opencv"
    finally:
        ve._BACKEND_CACHE = None  # prevent cache state leaking to other tests
        if saved is None:
            sys.modules.pop("av", None)
        else:
            sys.modules["av"] = saved


# ── VideoEncoder write + release ──────────────────────────────────────────────


def test_video_encoder_opencv_creates_nonempty_file(tmp_path):
    """VideoEncoder(backend='opencv') writes a valid, non-empty MP4."""
    from hydra_suite.utils.video_encoder import VideoEncoder

    path = tmp_path / "out.mp4"
    enc = VideoEncoder(path, fps=30.0, width=64, height=64, backend="opencv")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    enc.write(frame)
    enc.write(frame)
    enc.release()
    assert path.exists()
    assert path.stat().st_size > 100


def test_video_encoder_context_manager(tmp_path):
    """Context manager calls release() on __exit__ and the file is written."""
    from hydra_suite.utils.video_encoder import VideoEncoder

    path = tmp_path / "ctx.mp4"
    with VideoEncoder(path, fps=25.0, width=32, height=32, backend="opencv") as enc:
        enc.write(np.zeros((32, 32, 3), dtype=np.uint8))
    assert path.exists()


def test_video_encoder_double_release_is_safe(tmp_path):
    """Calling release() twice must not raise."""
    from hydra_suite.utils.video_encoder import VideoEncoder

    path = tmp_path / "safe.mp4"
    enc = VideoEncoder(path, fps=10.0, width=16, height=16, backend="opencv")
    enc.release()
    enc.release()  # must not raise


def test_video_encoder_auto_backend_writes_file(tmp_path):
    """VideoEncoder with auto backend selection produces a readable file."""
    from hydra_suite.utils.video_encoder import VideoEncoder

    path = tmp_path / "auto.mp4"
    with VideoEncoder(path, fps=10.0, width=64, height=64) as enc:
        for _ in range(5):
            enc.write(np.zeros((64, 64, 3), dtype=np.uint8))
    assert path.exists()
    assert path.stat().st_size > 0


def test_video_encoder_pyav_write_uses_bgr_input(monkeypatch: pytest.MonkeyPatch):
    """PyAV path should ingest BGR frames directly without channel reversal."""
    import hydra_suite.utils.video_encoder as ve

    class _FakeFrame:
        def reformat(self, *, format):
            assert format == "yuv420p"
            return self

    calls = []

    class _FakeVideoFrame:
        @staticmethod
        def from_ndarray(arr, format):
            calls.append((arr.shape, arr.flags.c_contiguous, format))
            return _FakeFrame()

    monkeypatch.setitem(sys.modules, "av", mock.Mock(VideoFrame=_FakeVideoFrame))

    enc = ve.VideoEncoder.__new__(ve.VideoEncoder)
    enc._stream = mock.Mock()
    enc._stream.encode.return_value = [object()]
    enc._container = mock.Mock()
    enc._cv_writer = None

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    enc.write(frame)

    assert calls == [((32, 32, 3), True, "bgr24")]
    enc._container.mux.assert_called_once()


# ── NVENC session cap ─────────────────────────────────────────────────────────


def test_nvenc_cap_exceeded_falls_back_to_software(tmp_path):
    """When _NVENC_ACTIVE >= _NVENC_MAX, VideoEncoder silently downgrades from nvenc."""
    import hydra_suite.utils.video_encoder as ve

    saved_max, saved_active = ve._NVENC_MAX, ve._NVENC_ACTIVE
    try:
        ve._NVENC_MAX = 0  # cap at zero: any nvenc request must fall back
        ve._NVENC_ACTIVE = 0
        enc = ve.VideoEncoder(
            tmp_path / "capped.mp4",
            fps=10.0,
            width=16,
            height=16,
            backend="nvenc",
        )
        # Must have been downgraded; backend is no longer "nvenc"
        assert enc._backend != "nvenc"
        enc.release()
    finally:
        ve._NVENC_MAX = saved_max
        ve._NVENC_ACTIVE = saved_active


def test_nvenc_session_counter_decrements_on_release(tmp_path):
    """Releasing an NVENC encoder decrements _NVENC_ACTIVE."""
    import hydra_suite.utils.video_encoder as ve

    saved_cache = ve._BACKEND_CACHE
    saved_max, saved_active = ve._NVENC_MAX, ve._NVENC_ACTIVE
    try:
        ve._NVENC_MAX = 10
        ve._NVENC_ACTIVE = 0

        with mock.patch.object(ve.VideoEncoder, "_open", lambda self: None):
            enc = ve.VideoEncoder.__new__(ve.VideoEncoder)
            enc._backend = "nvenc"
            enc._used_nvenc = False
            enc._container = None
            enc._stream = None
            enc._cv_writer = None
            # Manually simulate what _open does for nvenc
            ve._NVENC_ACTIVE += 1
            enc._used_nvenc = True

            before = ve._NVENC_ACTIVE
            enc.release()
            assert ve._NVENC_ACTIVE == before - 1
    finally:
        ve._BACKEND_CACHE = saved_cache
        ve._NVENC_MAX = saved_max
        ve._NVENC_ACTIVE = saved_active


def test_tracking_worker_module_imports_video_encoder():
    """worker.py must import VideoEncoder from hydra_suite.utils.video_encoder."""
    import hydra_suite.core.tracking.worker as worker_mod

    assert hasattr(
        worker_mod, "VideoEncoder"
    ), "worker.py must have 'from hydra_suite.utils.video_encoder import VideoEncoder'"


def test_tracking_orchestrator_imports_video_encoder():
    """orchestrators/tracking.py must import VideoEncoder."""
    import hydra_suite.trackerkit.gui.orchestrators.tracking as orch_mod

    assert hasattr(orch_mod, "VideoEncoder"), (
        "orchestrators/tracking.py must have "
        "'from hydra_suite.utils.video_encoder import VideoEncoder'"
    )


def test_oriented_video_exporter_imports_video_encoder():
    """oriented_video.py must import VideoEncoder."""
    import hydra_suite.core.identity.dataset.oriented_video as ov_mod

    assert hasattr(ov_mod, "VideoEncoder"), (
        "oriented_video.py must have "
        "'from hydra_suite.utils.video_encoder import VideoEncoder'"
    )


def test_diagnostic_video_module_imports_video_encoder():
    """confidence_density.py must import VideoEncoder."""
    import hydra_suite.core.tracking.confidence.confidence_density as cd_mod

    assert hasattr(cd_mod, "VideoEncoder"), (
        "confidence_density.py must have "
        "'from hydra_suite.utils.video_encoder import VideoEncoder'"
    )
