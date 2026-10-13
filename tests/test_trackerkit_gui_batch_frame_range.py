"""The GUI's sequential batch: a video without its own config tracks whole."""

from __future__ import annotations

import json
from types import SimpleNamespace


def _orchestrator(tmp_path, keystone_override=False):
    from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator

    calls: list[tuple] = []
    keystone = tmp_path / "a.mp4"
    keystone.write_bytes(b"\x00")
    (tmp_path / "a_config.json").write_text(
        json.dumps({"start_frame": 100, "end_frame": 200})
    )
    main_window = SimpleNamespace(
        batch_videos=[str(keystone)],
        _setup_video_file=lambda fp, skip_config_load: calls.append(
            ("setup", fp, skip_config_load)
        ),
        _config_orch=SimpleNamespace(
            _load_config_from_file=lambda path: calls.append(("load", path))
        ),
        _reset_frame_range=lambda: calls.append(("reset",)),
    )
    panels = SimpleNamespace(
        setup=SimpleNamespace(
            chk_batch_keystone_override=SimpleNamespace(
                isChecked=lambda: keystone_override
            )
        )
    )
    return TrackingOrchestrator(main_window, SimpleNamespace(), panels), calls


def _video(tmp_path, name, own_range=None):
    video = tmp_path / f"{name}.mp4"
    video.write_bytes(b"\x00")
    if own_range is not None:
        start, end = own_range
        (tmp_path / f"{name}_config.json").write_text(
            json.dumps({"start_frame": start, "end_frame": end})
        )
    return str(video)


def test_borrowing_video_resets_to_the_whole_video_after_the_keystone_load(tmp_path):
    orch, calls = _orchestrator(tmp_path)
    orch._load_next_batch_video(_video(tmp_path, "b"))
    assert [c[0] for c in calls] == ["setup", "load", "reset"]
    assert calls[1][1].endswith("a_config.json")


def test_override_borrower_resets_too(tmp_path):
    orch, calls = _orchestrator(tmp_path, keystone_override=True)
    orch._load_next_batch_video(_video(tmp_path, "b", own_range=(5, 9)))
    assert [c[0] for c in calls] == ["setup", "load", "reset"]


def test_video_with_its_own_config_keeps_its_range(tmp_path):
    orch, calls = _orchestrator(tmp_path)
    orch._load_next_batch_video(_video(tmp_path, "b", own_range=(5, 9)))
    assert calls == [("setup", calls[0][1], False)]


def test_borrower_without_any_keystone_config_still_resets(tmp_path):
    orch, calls = _orchestrator(tmp_path)
    (tmp_path / "a_config.json").unlink()
    orch._load_next_batch_video(_video(tmp_path, "b"))
    assert [c[0] for c in calls] == ["setup", "reset"]
