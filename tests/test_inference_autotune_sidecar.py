from __future__ import annotations

import json

import numpy as np
import pytest

from hydra_suite.core.inference.autotune.device import RuntimeResourceProbe
from hydra_suite.core.inference.autotune.models import InferenceTuningSettings
from hydra_suite.core.inference.autotune.search import TrialObservation
from hydra_suite.core.inference.autotune.sidecar import (
    MAX_REQUEST_BYTES,
    ContainedTrialExecutor,
    SidecarTrialSpec,
    apply_settings_to_params,
    restore_sidecar_params,
    write_sidecar_request,
)
from hydra_suite.core.inference.autotune.sidecar_child import _block_window
from hydra_suite.runtime.resource_budget import AcceleratorKind, ResourceObservation


def _resources():
    observation = ResourceObservation(
        total_host_bytes=64 * 1024**3,
        available_host_bytes=48 * 1024**3,
        accelerator_kind=AcceleratorKind.CPU,
    )
    probe = RuntimeResourceProbe(
        observation,
        "cpu",
        "CPU",
        "none",
        None,
        "test",
        False,
        False,
    )
    return observation, probe


def _settings():
    return InferenceTuningSettings(
        detection_batch_size=4,
        slice_tile_batch_size=2,
        pose_batch_size=8,
        headtail_batch_size=4,
        identity_batch_sizes=(("color", 16),),
        pipeline_depth=3,
    )


def test_candidate_param_overlay_is_detached_complete_and_disables_recursion():
    source = {
        "YOLO_OBB_MODE": "direct",
        "CNN_CLASSIFIERS": [
            {"label": "color", "batch_size": 1},
            {"label": "behavior", "batch_size": 3},
        ],
        "INFERENCE_AUTOTUNE_MODE": "automatic",
        "SLICE_TILE_BATCH_AUTOTUNE": True,
    }

    output = apply_settings_to_params(
        source, _settings(), runtime_artifact_batch_size=32
    )

    assert source["CNN_CLASSIFIERS"][0]["batch_size"] == 1
    assert output["INFERENCE_AUTOTUNE_MODE"] == "off"
    assert output["YOLO_BATCH_SIZE"] == 4
    assert output["PIPELINE_DEPTH"] == 3
    assert output["SLICE_TILE_BATCH_SIZE"] == 2
    assert output["SLICE_TILE_BATCH_AUTOTUNE"] is False
    assert output["POSE_BATCH_SIZE"] == 8
    assert output["HEADTAIL_BATCH_SIZE"] == 4
    assert [item["batch_size"] for item in output["CNN_CLASSIFIERS"]] == [16, 3]
    assert output["USE_CACHED_DETECTIONS"] is False
    assert output["INFERENCE_AUTOTUNE_ARTIFACT_BATCH_SIZE"] == 32


def test_request_stages_roi_as_relative_non_pickle_payload(tmp_path):
    observation, probe = _resources()
    spec = SidecarTrialSpec(
        video_path=tmp_path / "video.mp4",
        params={"ROI_MASK": np.ones((2, 3), dtype=np.uint8)},
        observation=observation,
        resource_probe=probe,
        start_frame=0,
        end_frame=20,
    )

    request = write_sidecar_request(
        tmp_path / "ipc",
        spec,
        _settings(),
        phase="full",
        field_name=None,
        block_index=0,
    )

    payload = json.loads(request.read_text(encoding="utf-8"))
    assert payload["params"]["ROI_MASK"] == {"__hydra_npy__": "ROI_MASK.npy"}
    restored = np.load(request.parent / "arrays" / "ROI_MASK.npy", allow_pickle=False)
    np.testing.assert_array_equal(restored, np.ones((2, 3), dtype=np.uint8))
    assert payload["maximum_frames"] == 128


def test_array_params_round_trip_by_key(tmp_path):
    """Every ndarray param survives the request round-trip, not just ROI_MASK."""
    observation, probe = _resources()
    labels = np.arange(6, dtype=np.uint16).reshape(2, 3)
    mask = labels > 0
    spec = SidecarTrialSpec(
        video_path=tmp_path / "video.mp4",
        params={"ARENA_LABELS": labels, "ROI_MASK": mask, "N_ARENAS": 1},
        observation=observation,
        resource_probe=probe,
        start_frame=0,
        end_frame=20,
    )

    request = write_sidecar_request(
        tmp_path / "ipc",
        spec,
        _settings(),
        phase="full",
        field_name=None,
        block_index=0,
    )

    payload = json.loads(request.read_text(encoding="utf-8"))
    restored = restore_sidecar_params(payload["params"], request.parent)
    np.testing.assert_array_equal(restored["ARENA_LABELS"], labels)
    assert restored["ARENA_LABELS"].dtype == labels.dtype
    np.testing.assert_array_equal(restored["ROI_MASK"], mask)
    assert restored["N_ARENAS"] == 1


def test_array_params_do_not_inflate_json_request_size(tmp_path):
    """MAX_REQUEST_BYTES bounds the JSON only; arrays live in sidecar .npy files."""
    observation, probe = _resources()
    # A 4K-resolution arena-label map (~16 MB as uint16), well past MAX_REQUEST_BYTES.
    labels = np.zeros((2160, 3840), dtype=np.uint16)
    spec = SidecarTrialSpec(
        video_path=tmp_path / "video.mp4",
        params={"ARENA_LABELS": labels},
        observation=observation,
        resource_probe=probe,
        start_frame=0,
        end_frame=20,
    )

    request = write_sidecar_request(
        tmp_path / "ipc",
        spec,
        _settings(),
        phase="full",
        field_name=None,
        block_index=0,
    )

    assert request.stat().st_size < MAX_REQUEST_BYTES
    restored = restore_sidecar_params(
        json.loads(request.read_text(encoding="utf-8"))["params"], request.parent
    )
    np.testing.assert_array_equal(restored["ARENA_LABELS"], labels)


def test_legacy_roi_only_request_still_restores(tmp_path):
    """A request staged by an older build (ROI-only, no arrays/ dir) still loads."""
    root = tmp_path / "ipc"
    root.mkdir()
    roi = np.ones((2, 3), dtype=np.uint8)
    np.save(root / "roi.npy", roi, allow_pickle=False)
    legacy_params = {"ROI_MASK": {"__hydra_roi_npy__": "roi.npy"}, "N_ARENAS": 1}

    restored = restore_sidecar_params(legacy_params, root)

    np.testing.assert_array_equal(restored["ROI_MASK"], roi)
    assert restored["N_ARENAS"] == 1


def test_array_param_key_cannot_escape_array_dir_on_write(tmp_path):
    """A key containing a path separator must not stage a file outside arrays/."""
    observation, probe = _resources()
    spec = SidecarTrialSpec(
        video_path=tmp_path / "video.mp4",
        params={"../escape": np.ones((2, 2), dtype=np.uint8)},
        observation=observation,
        resource_probe=probe,
        start_frame=0,
        end_frame=20,
    )

    with pytest.raises(ValueError, match="invalid array parameter key"):
        write_sidecar_request(
            tmp_path / "ipc",
            spec,
            _settings(),
            phase="full",
            field_name=None,
            block_index=0,
        )


def test_measurement_blocks_share_one_640_frame_cap(tmp_path):
    observation, probe = _resources()
    spec = SidecarTrialSpec(
        video_path=tmp_path / "video.mp4",
        params={},
        observation=observation,
        resource_probe=probe,
        start_frame=0,
        end_frame=1000,
    )
    caps = []
    for block in range(5):
        request = write_sidecar_request(
            tmp_path / f"ipc-{block}",
            spec,
            _settings(),
            phase="stage",
            field_name="detection_batch_size",
            block_index=block,
        )
        caps.append(json.loads(request.read_text())["maximum_frames"])

    assert caps == [128, 128, 128, 128, 128]
    assert sum(caps) == 640


def test_final_validation_uses_dedicated_selected_runtime_profile(tmp_path):
    observation, probe = _resources()
    spec = SidecarTrialSpec(
        video_path=tmp_path / "video.mp4",
        params={},
        observation=observation,
        resource_probe=probe,
        start_frame=0,
        end_frame=20,
        runtime_artifact_batch_size=32,
    )
    screen = write_sidecar_request(
        tmp_path / "screen",
        spec,
        _settings(),
        phase="stage",
        field_name="detection_batch_size",
        block_index=0,
    )
    final = write_sidecar_request(
        tmp_path / "final",
        spec,
        _settings(),
        phase="final_validation",
        field_name=None,
        block_index=0,
    )

    assert (
        json.loads(screen.read_text())["params"][
            "INFERENCE_AUTOTUNE_ARTIFACT_BATCH_SIZE"
        ]
        == 32
    )
    assert (
        "INFERENCE_AUTOTUNE_ARTIFACT_BATCH_SIZE"
        not in json.loads(final.read_text())["params"]
    )


def test_request_rejects_arbitrary_python_objects(tmp_path):
    observation, probe = _resources()
    spec = SidecarTrialSpec(
        video_path=tmp_path / "video.mp4",
        params={"callback": object()},
        observation=observation,
        resource_probe=probe,
        start_frame=0,
        end_frame=20,
    )
    with pytest.raises(TypeError, match="unsupported sidecar parameter"):
        write_sidecar_request(
            tmp_path / "ipc",
            spec,
            _settings(),
            phase="full",
            field_name=None,
            block_index=0,
        )


def test_measurement_windows_are_large_enough_to_amortize_model_load():
    """One contiguous window per block, at least 128 frames when available.

    Splitting a block into sub-windows costs one full model load per
    sub-window, which swamps the batch effect the tuner is trying to
    measure.  Representativeness comes from striping the five blocks
    across the clip, not from splitting each block.
    """

    window = _block_window(0, 999, 128, block_index=0, blocks=5)
    assert window[1] - window[0] + 1 == 128


def test_block_windows_are_striped_across_the_clip():
    windows = [_block_window(10, 1009, 128, block_index=i, blocks=5) for i in range(5)]

    assert windows[0][0] == 10
    assert windows[-1][1] == 1009
    assert all(end - start + 1 == 128 for start, end in windows)
    assert len(set(windows)) == 5


def test_block_window_clamps_to_a_short_range():
    assert _block_window(4, 19, 128, block_index=3, blocks=5) == (4, 19)


def test_oom_adaptation_is_two_retries_and_never_mislabels_reduced_result(
    tmp_path, monkeypatch
):
    observation, probe = _resources()
    executor = ContainedTrialExecutor(
        SidecarTrialSpec(
            video_path=tmp_path / "video.mp4",
            params={},
            observation=observation,
            resource_probe=probe,
            start_frame=0,
            end_frame=20,
        )
    )
    attempted = []

    def fake_once(settings, **_kwargs):
        attempted.append(settings.detection_batch_size)
        if settings.detection_batch_size > 2:
            return TrialObservation(
                settings, 0.0, 0.0, None, failure_class="accelerator-oom"
            )
        return TrialObservation(settings, 10.0, 2.0, None)

    monkeypatch.setattr(executor, "_run_once", fake_once)
    requested = _settings().with_value("detection_batch_size", 8)
    result = executor.run(
        requested,
        phase="stage",
        field_name="detection_batch_size",
        block_index=0,
        should_cancel=lambda: False,
    )

    assert attempted == [8, 4, 2]
    assert result.settings == requested
    assert result.failure_class == "accelerator-oom"
    reduced = requested.with_value("detection_batch_size", 2)
    cached = executor.run(
        reduced,
        phase="stage",
        field_name="detection_batch_size",
        block_index=0,
        should_cancel=lambda: False,
    )
    assert cached.settings == reduced
    assert cached.failure_class is None
    assert attempted == [8, 4, 2]
