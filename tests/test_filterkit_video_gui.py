"""FilterKit video-source GUI integration tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from hydra_suite.filterkit.core import FilterKitCore
from hydra_suite.filterkit.gui.main_window import FilterKitWindow


def _write_video(path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 16))
    assert writer.isOpened()
    try:
        for value in (20, 160):
            writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    finally:
        writer.release()


def test_filterkit_window_exports_video_selection_with_provenance(
    tmp_path, monkeypatch
) -> None:
    QApplication.instance() or QApplication([])
    video_path = tmp_path / "recording.avi"
    _write_video(video_path)
    _, items = FilterKitCore().load_video(video_path)

    window = FilterKitWindow()
    try:
        assert window.load_dataset_root(video_path)
        assert window._source_kind == "video"
        assert not window.chk_preserve_full_frames.isEnabled()

        window.filtered_dataset = [items[1]]
        reads = 0
        original_read = window._preview_reader.read

        def count_reads(item):
            nonlocal reads
            reads += 1
            return original_read(item)

        monkeypatch.setattr(window._preview_reader, "read", count_reads)
        assert not window._preview_pixmap(items[1]).isNull()
        assert not window._preview_pixmap(items[1]).isNull()
        assert reads == 1
        window.pipeline_stats = {"loaded": len(items)}
        window._last_config = {"temporal_enabled": True, "temporal_interval": 2}
        monkeypatch.setattr(
            QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes
        )
        monkeypatch.setattr(QMessageBox, "information", lambda *_args: None)

        window.process_dataset()

        output_root = tmp_path / "recording_filterkit_output"
        manifest = json.loads(
            (output_root / "filterkit_video_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["source_video"] == str(video_path.resolve())
        assert manifest["selected_frame_indices"] == [1]
        assert (output_root / "images" / items[1]["filename"]).is_file()
    finally:
        window.close()


def test_filterkit_window_removes_export_if_transaction_cannot_be_written(
    tmp_path, monkeypatch
) -> None:
    QApplication.instance() or QApplication([])
    video_path = tmp_path / "recording.avi"
    _write_video(video_path)
    _, items = FilterKitCore().load_video(video_path)

    window = FilterKitWindow()
    try:
        assert window.load_dataset_root(video_path)
        window.filtered_dataset = [items[0]]
        failed_transaction = tmp_path / "transaction-failure.json"
        monkeypatch.setattr(
            window, "_transaction_path", lambda _path: failed_transaction
        )
        monkeypatch.setattr(
            QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes
        )
        monkeypatch.setattr(QMessageBox, "information", lambda *_args: None)
        monkeypatch.setattr(QMessageBox, "critical", lambda *_args: None)

        real_write_text = Path.write_text

        def fail_transaction_write(path, *args, **kwargs):
            if path == failed_transaction:
                raise OSError("simulated transaction write failure")
            return real_write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_transaction_write)
        window.process_dataset()

        assert not (tmp_path / "recording_filterkit_output").exists()
    finally:
        window.close()


def test_filterkit_window_releases_preview_video_on_source_change(tmp_path) -> None:
    QApplication.instance() or QApplication([])
    first_video = tmp_path / "first.avi"
    second_video = tmp_path / "second.avi"
    _write_video(first_video)
    _write_video(second_video)
    _, first_items = FilterKitCore().load_video(first_video)

    window = FilterKitWindow()
    try:
        assert window.load_dataset_root(first_video)
        assert not window._preview_pixmap(first_items[0]).isNull()
        first_source = next(iter(window._preview_reader._video_sources.values()))
        assert first_source._cap is not None

        assert window.load_dataset_root(second_video)
        assert first_source._cap is None
        assert window._preview_reader._video_sources == {}
    finally:
        window.close()
