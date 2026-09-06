"""plan_batch_jobs must hand every video the SAME config dict the pre-refactor
sequential loop in cli.py passed to load_tracker_cli_session."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from hydra_suite.trackerkit import batch_plan
from hydra_suite.trackerkit.batch_plan import BatchPlanError, plan_batch_jobs
from hydra_suite.trackerkit.cli_config import load_tracker_cli_config
from hydra_suite.trackerkit.session_plan import build_batch_video_plan


def _reference_loop(
    videos, *, config_path=None, keystone_override=False, sahi_profile=None
):
    """Verbatim copy of the pre-refactor cli.run_tracking_cli config resolution."""
    from hydra_suite.trackerkit.cli_config import apply_sahi_profile_override

    plan = build_batch_video_plan(
        videos, explicit_config_path=config_path, keystone_override=keystone_override
    )
    out = []
    baseline = None
    for index, item in enumerate(plan, start=1):
        effective = None
        if item.use_keystone_baseline and item.config_path is None:
            effective = baseline or {}
        if sahi_profile:
            base = (
                effective
                if effective is not None
                else load_tracker_cli_config(item.config_path)
            )
            effective = apply_sahi_profile_override(base, sahi_profile)
        # what load_tracker_cli_session would deepcopy into session.config
        cfg = (
            deepcopy(dict(effective))
            if effective is not None
            else load_tracker_cli_config(item.config_path)
        )
        if index == 1:
            baseline = (
                deepcopy(load_tracker_cli_config(item.config_path))
                if item.config_path
                else deepcopy(cfg)
            )
        out.append((item.video_path, cfg))
    return out


def _mk_video(directory: Path, name: str, cfg: dict | None = None) -> str:
    video = directory / f"{name}.mp4"
    video.write_bytes(b"\x00")
    if cfg is not None:
        (directory / f"{name}_config.json").write_text(json.dumps(cfg))
    return str(video)


@pytest.fixture(autouse=True)
def _no_sahi(monkeypatch):
    # apply_sahi_profile_override needs a real model sidecar; stub it to a
    # deterministic transform so the parity test exercises the plumbing.
    def _fake(cfg, profile):
        out = deepcopy(dict(cfg))
        out["SAHI_PROFILE"] = profile
        return out

    monkeypatch.setattr(batch_plan, "apply_sahi_profile_override", _fake)
    import hydra_suite.trackerkit.cli_config as cc

    monkeypatch.setattr(cc, "apply_sahi_profile_override", _fake)


@pytest.mark.parametrize("keystone_override", [False, True])
@pytest.mark.parametrize("sahi_profile", [None, "prof_a"])
def test_planner_matches_reference_loop_mixed_sidecars(
    tmp_path, keystone_override, sahi_profile
):
    a = _mk_video(tmp_path, "a", {"k": "keystone"})
    b = _mk_video(tmp_path, "b", {"k": "own_b"})
    c = _mk_video(tmp_path, "c")  # no sidecar
    videos = [a, b, c]
    ref = _reference_loop(
        videos, keystone_override=keystone_override, sahi_profile=sahi_profile
    )
    specs = plan_batch_jobs(
        videos, keystone_override=keystone_override, sahi_profile=sahi_profile
    )
    assert [(s.video_path, s.config) for s in specs] == ref
    assert [s.index for s in specs] == [1, 2, 3]


def test_planner_matches_reference_loop_explicit_config(tmp_path):
    a = _mk_video(tmp_path, "a")
    b = _mk_video(tmp_path, "b", {"k": "own_b"})
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"k": "explicit"}))
    ref = _reference_loop([a, b], config_path=str(explicit))
    specs = plan_batch_jobs([a, b], explicit_config_path=str(explicit))
    assert [(s.video_path, s.config) for s in specs] == ref
    assert specs[0].provenance == "explicit"
    assert specs[1].provenance == "keystone-baseline"  # explicit implies override


def test_planner_no_sidecars_uses_empty_baseline(tmp_path):
    a = _mk_video(tmp_path, "a")
    b = _mk_video(tmp_path, "b")
    specs = plan_batch_jobs([a, b])
    assert specs[0].config == {} and specs[1].config == {}
    assert specs[0].config_path is None


def test_planner_rejects_duplicate_video(tmp_path):
    a = _mk_video(tmp_path, "a")
    with pytest.raises(BatchPlanError):
        plan_batch_jobs([a, a])


def test_planner_rejects_same_stem_when_outputs_collide(tmp_path, monkeypatch):
    dir_x = tmp_path / "x"
    dir_y = tmp_path / "y"
    dir_x.mkdir()
    dir_y.mkdir()
    a = _mk_video(dir_x, "same")
    b = _mk_video(dir_y, "same")
    # Force both raw CSVs to the same path (read-only dir redirect scenario).
    monkeypatch.setattr(
        batch_plan,
        "_default_output_paths",
        lambda v: (str(tmp_path / "same_tracking.csv"), ""),
    )
    with pytest.raises(BatchPlanError):
        plan_batch_jobs([a, b])


def test_planner_does_not_mutate_returned_configs_across_jobs(tmp_path):
    a = _mk_video(tmp_path, "a", {"k": "keystone", "nested": {"v": 1}})
    b = _mk_video(tmp_path, "b")
    specs = plan_batch_jobs([a, b])
    specs[0].config["nested"]["v"] = 99
    assert specs[1].config["nested"]["v"] == 1
