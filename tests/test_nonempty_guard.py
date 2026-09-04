import numpy as np
import pytest

from hydra_suite.core.tracking.errors import TrackingSessionError
from hydra_suite.core.tracking.session import (
    csv_has_data_rows,
    detection_cache_has_detections,
    enforce_nonempty_forward,
)


def _write_reducer_marker(path: str):
    from pathlib import Path

    Path(path).write_text("executed")
    return object()


class _ExecutableReducer:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (_write_reducer_marker, (self.marker,))


class _ExpandingReducer:
    def __reduce__(self):
        return (bytearray, (512 * 1024 * 1024,))


def test_csv_has_data_rows(tmp_path):
    empty = tmp_path / "h.csv"
    empty.write_text("TrackID,X\n")
    assert csv_has_data_rows(str(empty)) is False
    full = tmp_path / "f.csv"
    full.write_text("TrackID,X\n1,2\n")
    assert csv_has_data_rows(str(full)) is True


def test_detection_cache_has_detections(tmp_path):
    p = tmp_path / "d.npz"
    np.savez(str(p), frame_0_meas=np.zeros((3, 3)))
    assert detection_cache_has_detections(str(p)) is True
    q = tmp_path / "e.npz"
    np.savez(str(q), frame_0_meas=np.zeros((0, 3)))
    assert detection_cache_has_detections(str(q)) is False


@pytest.mark.parametrize("payload_kind", ["executable", "expanding"])
def test_legacy_object_cache_is_rejected_before_pickle_load(
    tmp_path, monkeypatch, payload_kind
):
    import hydra_suite.core.tracking.session as session

    marker = tmp_path / "reducer-executed"
    value = (
        _ExecutableReducer(marker)
        if payload_kind == "executable"
        else _ExpandingReducer()
    )
    cache = tmp_path / f"{payload_kind}.npz"
    np.savez(cache, frame_0_meas=np.asarray([value], dtype=object))
    np_load_called = False
    real_load = session.np.load

    def recording_load(*args, **kwargs):
        nonlocal np_load_called
        np_load_called = True
        return real_load(*args, **kwargs)

    monkeypatch.setattr(session.np, "load", recording_load)
    assert detection_cache_has_detections(cache) is False
    assert np_load_called is False
    assert not marker.exists()


def test_enforce_raises_tracking_session_error(tmp_path):
    csv = tmp_path / "h.csv"
    csv.write_text("TrackID,X\n")  # header only
    cache = tmp_path / "d.npz"
    np.savez(str(cache), frame_0_meas=np.zeros((3, 3)))  # has detections
    with pytest.raises(TrackingSessionError):
        enforce_nonempty_forward(str(csv), str(cache))
