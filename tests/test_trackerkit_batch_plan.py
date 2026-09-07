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


# --- Borrowed-config output paths -------------------------------------------
# A config that did NOT come from the video's own sidecar carries the
# KEYSTONE's absolute output paths. ``video_output_path`` is consumed by
# ``core/tracking/session.py::_run_annotated_video``, so leaving it alone makes
# every video in the batch render its overlay into ONE file.


def _sidecar_cfg(directory: Path, name: str) -> dict:
    """What the GUI's save_config writes: absolute per-video output paths."""
    return {
        "k": name,
        "file_path": str(directory / f"{name}.mp4"),
        "csv_path": str(directory / f"{name}_tracking.csv"),
        "video_output_enabled": True,
        "video_output_path": str(directory / f"{name}_tracking.mp4"),
    }


def test_inheriting_videos_render_beside_their_own_video(tmp_path):
    a = _mk_video(tmp_path, "a", _sidecar_cfg(tmp_path, "a"))
    b = _mk_video(tmp_path, "b")  # no sidecar: inherits the keystone
    c = _mk_video(tmp_path, "c")  # ditto
    specs = plan_batch_jobs([a, b, c])
    assert [s.provenance for s in specs] == [
        "own-sidecar",
        "keystone-baseline",
        "keystone-baseline",
    ]
    outs = [s.config["video_output_path"] for s in specs]
    assert len(set(outs)) == 3, outs
    for spec, name in zip(specs, ("a", "b", "c")):
        assert spec.config["video_output_path"] == str(
            tmp_path / f"{name}_tracking.mp4"
        )
        assert spec.config["csv_path"] == str(tmp_path / f"{name}_tracking.csv")
        assert spec.config["file_path"] == spec.video_path
        assert spec.config["video_output_enabled"] is True


def test_explicit_config_batch_renders_beside_each_video(tmp_path):
    a = _mk_video(tmp_path, "a")
    b = _mk_video(tmp_path, "b")
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps(_sidecar_cfg(tmp_path, "keystone")))
    specs = plan_batch_jobs([a, b], explicit_config_path=str(explicit))
    outs = [s.config["video_output_path"] for s in specs]
    assert outs == [
        str(tmp_path / "a_tracking.mp4"),
        str(tmp_path / "b_tracking.mp4"),
    ]


def test_keystone_override_rewrites_paths_for_every_borrower(tmp_path):
    a = _mk_video(tmp_path, "a", _sidecar_cfg(tmp_path, "a"))
    b = _mk_video(tmp_path, "b", _sidecar_cfg(tmp_path, "b"))
    specs = plan_batch_jobs([a, b], keystone_override=True)
    # b's own sidecar is ignored under override: it runs the keystone config,
    # but must still render to its own file.
    assert specs[1].config["k"] == "a"
    assert specs[1].config["video_output_path"] == str(tmp_path / "b_tracking.mp4")


def test_own_sidecar_paths_are_left_exactly_as_written(tmp_path):
    cfg = _sidecar_cfg(tmp_path, "b")
    cfg["video_output_path"] = str(tmp_path / "custom_name.mp4")
    a = _mk_video(tmp_path, "a", _sidecar_cfg(tmp_path, "a"))
    b = _mk_video(tmp_path, "b", cfg)
    specs = plan_batch_jobs([a, b])
    assert specs[1].config["video_output_path"] == str(tmp_path / "custom_name.mp4")


def test_absent_output_keys_are_never_invented(tmp_path):
    """A config with no ``video_output_path`` must not gain one: that would
    start rendering a video the sequential path never rendered."""
    a = _mk_video(tmp_path, "a", {"k": "keystone", "video_output_enabled": True})
    b = _mk_video(tmp_path, "b")
    specs = plan_batch_jobs([a, b])
    assert "video_output_path" not in specs[0].config
    assert "video_output_path" not in specs[1].config


def test_planner_rejects_two_videos_rendering_to_one_mp4(tmp_path):
    cfg_a = _sidecar_cfg(tmp_path, "a")
    cfg_b = _sidecar_cfg(tmp_path, "b")
    cfg_a["video_output_path"] = cfg_b["video_output_path"] = str(
        tmp_path / "shared.mp4"
    )
    a = _mk_video(tmp_path, "a", cfg_a)
    b = _mk_video(tmp_path, "b", cfg_b)
    with pytest.raises(BatchPlanError):
        plan_batch_jobs([a, b])


def test_divergence_from_the_reference_loop_is_exactly_the_output_paths(tmp_path):
    """The pre-refactor sequential loop had this same latent bug, so the parity
    oracle would only stay green by reproducing it. This documents the ONE
    deliberate divergence and pins its blast radius to the three keys that
    name per-video side outputs -- none of which reach the tracking engine
    (``build_engine_params`` never reads them and the session derives its CSV
    paths from the video path), so tracking output is unchanged.
    """
    a = _mk_video(tmp_path, "a", _sidecar_cfg(tmp_path, "a"))
    b = _mk_video(tmp_path, "b")
    ref = _reference_loop([a, b])
    specs = plan_batch_jobs([a, b])
    assert [s.video_path for s in specs] == [v for v, _ in ref]
    for (_, ref_cfg), spec in zip(ref, specs):
        differing = {
            key
            for key in set(ref_cfg) | set(spec.config)
            if ref_cfg.get(key) != spec.config.get(key)
        }
        assert differing <= {"file_path", "csv_path", "video_output_path"}, differing
    # ...and the reference loop really does collide, which is what we fixed.
    assert len({cfg.get("video_output_path") for _, cfg in ref}) == 1
