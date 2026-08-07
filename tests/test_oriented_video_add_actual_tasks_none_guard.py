"""Guard: _add_actual_tasks must not crash when the detection cache has no
entry for a frame (get_frame -> None). Previously it fell through to the
legacy 12-tuple unpack and raised TypeError: cannot unpack non-iterable
NoneType."""

import types

from hydra_suite.core.individual.dataset.oriented_video import (
    FrameBundle,
    OrientedTrackVideoExporter,
)


class _NoneCache:
    def get_frame(self, frame_id):
        return None


def test_add_actual_tasks_handles_missing_frame_without_crashing():
    # Bypass __init__ (heavy) — the None-guard returns before touching self.
    exporter = OrientedTrackVideoExporter.__new__(OrientedTrackVideoExporter)
    rows = [
        types.SimpleNamespace(TrajectoryID=1, DetectionID=10),
        types.SimpleNamespace(TrajectoryID=2, DetectionID=11),
    ]
    missing = exporter._add_actual_tasks(
        _NoneCache(),
        frame_id=0,
        rows=rows,
        bundle=FrameBundle(),
        track_sizes={},
        track_theta_state={},
    )
    # No exception, and every unresolved row is reported as missing.
    assert missing["missing_detected_rows"] == len(rows)
