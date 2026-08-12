import numpy as np
import pytest

from hydra_suite.core.tracking.errors import TrackingSessionError
from hydra_suite.core.tracking.session import (
    csv_has_data_rows,
    detection_cache_has_detections,
    enforce_nonempty_forward,
)


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


def test_enforce_raises_tracking_session_error(tmp_path):
    csv = tmp_path / "h.csv"
    csv.write_text("TrackID,X\n")  # header only
    cache = tmp_path / "d.npz"
    np.savez(str(cache), frame_0_meas=np.zeros((3, 3)))  # has detections
    with pytest.raises(TrackingSessionError):
        enforce_nonempty_forward(str(csv), str(cache))
