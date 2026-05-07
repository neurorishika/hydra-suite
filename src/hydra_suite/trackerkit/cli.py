"""Minimal TrackerKit CLI runner for config-driven tracking sessions."""

from __future__ import annotations

import json
import logging
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from hydra_suite.trackerkit.cli_config import (
    load_tracker_cli_config,
    load_tracker_cli_session,
)
from hydra_suite.trackerkit.headless_tracking import run_headless_tracking_session
from hydra_suite.trackerkit.session_plan import build_batch_video_plan

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_message_boxes():
    from PySide6.QtWidgets import QMessageBox

    original = {
        "information": QMessageBox.information,
        "warning": QMessageBox.warning,
        "critical": QMessageBox.critical,
        "question": QMessageBox.question,
    }

    def _information(*_args, **_kwargs):
        return QMessageBox.StandardButton.Ok

    def _warning(*_args, **_kwargs):
        return QMessageBox.StandardButton.Ok

    def _critical(*_args, **_kwargs):
        return QMessageBox.StandardButton.Ok

    def _question(*_args, **_kwargs):
        return QMessageBox.StandardButton.Yes

    QMessageBox.information = _information
    QMessageBox.warning = _warning
    QMessageBox.critical = _critical
    QMessageBox.question = _question
    try:
        yield
    finally:
        QMessageBox.information = original["information"]
        QMessageBox.warning = original["warning"]
        QMessageBox.critical = original["critical"]
        QMessageBox.question = original["question"]


def _ensure_qapplication():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        return app, False
    app = QApplication([])
    app.setApplicationName("TrackerKit CLI")
    return app, True


def _prepare_video_session(
    main_window, video_path: str, config_path: str | None
) -> None:
    main_window._config_orch._setup_video_file(video_path, skip_config_load=True)
    if config_path:
        main_window._config_orch._load_config_from_file(config_path)


def _run_one_tracking_session(main_window, video_path: str) -> dict[str, Any]:
    from PySide6.QtCore import QEventLoop, QTimer

    result: dict[str, Any] = {}
    loop = QEventLoop()

    def _on_complete(payload: dict[str, Any]) -> None:
        result.update(payload)
        QTimer.singleShot(0, loop.quit)

    main_window._headless_tracking_callback = _on_complete
    main_window._headless_session_error = None
    main_window.start_tracking_on_video(video_path, backward_mode=False)
    if main_window.tracking_worker is None:
        raise RuntimeError(
            "Tracking did not start. Check the config and logs for details."
        )
    loop.exec()
    main_window._headless_tracking_callback = None
    return result


def _run_bridge_tracking_session(
    main_window,
    *,
    video_path: str,
    config_path: str | None,
) -> dict[str, Any]:
    _prepare_video_session(main_window, video_path, config_path)
    return _run_one_tracking_session(main_window, video_path)


def run_tracking_cli(
    video_paths: Sequence[str],
    *,
    config_path: str | None = None,
    keystone_override: bool = False,
) -> int:
    """Run one or more TrackerKit sessions from the CLI."""

    videos = [str(path).strip() for path in video_paths if str(path).strip()]
    if not videos:
        raise ValueError("At least one video path is required.")

    for video_path in videos:
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
    if config_path and not Path(config_path).is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    plan = build_batch_video_plan(
        videos,
        explicit_config_path=config_path,
        keystone_override=keystone_override,
    )
    if not plan:
        raise ValueError("No videos were resolved for tracking.")

    exit_code = 0
    with (
        tempfile.TemporaryDirectory(prefix="trackerkit-cli-") as tmpdir,
        _suppress_message_boxes(),
    ):
        tmpdir_path = Path(tmpdir)
        baseline_config_data: dict[str, Any] | None = None
        main_window = None
        owns_app = False

        try:
            for index, item in enumerate(plan, start=1):
                logger.info(
                    "Tracker CLI: preparing video %s/%s: %s",
                    index,
                    len(plan),
                    item.video_path,
                )
                effective_config_data = None
                if item.use_keystone_baseline and item.config_path is None:
                    effective_config_data = baseline_config_data or {}
                session = load_tracker_cli_session(
                    item.video_path,
                    config_path=(
                        item.config_path if effective_config_data is None else None
                    ),
                    config_data=effective_config_data,
                )

                if index == 1:
                    baseline_config_data = (
                        deepcopy(load_tracker_cli_config(item.config_path))
                        if item.config_path
                        else deepcopy(session.config)
                    )

                if session.supports_direct_run():
                    result = run_headless_tracking_session(session)
                else:
                    effective_config_path = item.config_path
                    if item.use_keystone_baseline and effective_config_path is None:
                        effective_config_path = str(
                            tmpdir_path / "keystone_config.json"
                        )
                        with open(
                            effective_config_path, "w", encoding="utf-8"
                        ) as handle:
                            json.dump(baseline_config_data or {}, handle, indent=2)
                    if main_window is None:
                        app, owns_app = _ensure_qapplication()
                        from hydra_suite.trackerkit.gui.main_window import MainWindow

                        main_window = MainWindow()
                        main_window.hide()
                        main_window._headless_tracking_mode = True
                    result = _run_bridge_tracking_session(
                        main_window,
                        video_path=item.video_path,
                        config_path=effective_config_path,
                    )

                if result.get("success"):
                    summary = " | ".join(result.get("lines", []))
                    logger.info("Tracker CLI completed: %s", summary)
                else:
                    error_message = result.get("error") or "Tracker session failed."
                    logger.error(
                        "Tracker CLI failed for %s: %s",
                        item.video_path,
                        error_message,
                    )
                    exit_code = 1
                    break
        finally:
            if main_window is not None:
                main_window._headless_tracking_mode = False
                main_window._headless_tracking_callback = None
                main_window.close()
            if owns_app:
                app.quit()

    return exit_code
