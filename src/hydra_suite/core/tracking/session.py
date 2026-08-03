"""Qt-free post-tracking session service (Slice 2: analysis chain)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from hydra_suite.core.tracking.errors import TrackingSessionError


def csv_has_data_rows(csv_path) -> bool:
    """Return True if *csv_path* has at least one data row beyond the header.

    O(1) in the file size -- reads only the header and the first data line.
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            fh.readline()  # header
            return bool(fh.readline().strip())
    except OSError:
        return False


def detection_cache_has_detections(detection_cache_path) -> bool:
    """Return True if the detection cache recorded at least one detection.

    Reads the raw ``.npz`` directly (no full ``DetectionCache`` instantiation /
    validation) and checks whether any per-frame ``frame_<N>_meas`` array is
    non-empty. Returns False on any read error or a missing cache, so the
    empty-output guard only fires when detections are *positively confirmed* --
    it can never false-positive on a legitimately empty clip.
    """
    try:
        with np.load(str(detection_cache_path), allow_pickle=True) as data:
            for key in data.files:
                if key.startswith("frame_") and key.endswith("_meas"):
                    arr = data[key]
                    if arr is not None and getattr(arr, "shape", (0,))[0] > 0:
                        return True
    except Exception:
        return False
    return False


def enforce_nonempty_forward(raw_csv_path, detection_cache_path) -> None:
    """Fail loud if the forward pass emitted an empty CSV despite detections.

    Invariant: if the detection cache contains detections, a *successful* forward
    tracking pass MUST produce at least one tracked row. A completed run that
    wrote a header-only CSV is a silent pipeline failure (e.g. a crashed
    pose/identity stage or an OOM that still returned exit 0). Raising here turns
    that into a loud abort instead of a valid-looking empty CSV that "falsely
    passes" downstream comparison/benchmark tooling.
    """
    if detection_cache_has_detections(detection_cache_path) and not csv_has_data_rows(
        raw_csv_path
    ):
        raise TrackingSessionError(
            "Forward tracking produced ZERO tracked rows even though the detection "
            "cache contains detections. This indicates a silent pipeline failure "
            "(e.g. a crashed pose/identity stage or an out-of-memory abort). "
            "Refusing to emit an empty tracking CSV. "
            f"csv={raw_csv_path} detection_cache={detection_cache_path}"
        )


def _noop1(_a) -> None:
    return None


def _noop2(_a, _b) -> None:
    return None


def _never() -> bool:
    return False


@dataclass
class SessionCallbacks:
    progress: Callable[[int, str], None] = _noop2
    status: Callable[[str], None] = _noop1
    warning: Callable[[str, str], None] = _noop2
    stage_changed: Callable[[str], None] = _noop1
    should_stop: Callable[[], bool] = _never


@dataclass
class SessionResult:
    success: bool
    final_csv_path: str | None
    rich_export_path: str | None
    media_paths: list[str]
    dataset_result: dict | None
    summary_lines: list[str]
    error: str | None


class TrackingSessionCore:
    """Owns the post-tracking analysis chain, Qt-free."""

    def __init__(self, *, video_path, config, params, paths, callbacks=None):
        self.video_path = video_path
        self.config = config
        self.params = params
        self.paths = paths
        self.callbacks = callbacks if callbacks is not None else SessionCallbacks()

    def run_post_tracking(
        self, forward_trajectories, backward_trajectories=None
    ) -> SessionResult:
        raise NotImplementedError  # wired in Task 8
