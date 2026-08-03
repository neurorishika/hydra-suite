"""Qt-free post-tracking session service (Slice 2: analysis chain)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


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
