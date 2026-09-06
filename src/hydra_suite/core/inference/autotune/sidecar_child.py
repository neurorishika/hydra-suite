"""Fresh-process full tracking trial used by the inference throughput tuner."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.inference.autotune.models import InferenceTuningSettings
from hydra_suite.core.inference.autotune.sidecar import (
    MAX_REQUEST_BYTES,
    SIDECAR_SCHEMA_VERSION,
)
from hydra_suite.core.tracking.session import SessionCallbacks, TrackingSessionCore
from hydra_suite.core.tracking.worker import TrackingEngineCore
from hydra_suite.data.csv_writer import CSVWriterThread


def _read_request(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        encoded = stream.read(MAX_REQUEST_BYTES + 1)
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("sidecar request exceeds its size cap")
    value = json.loads(encoded)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SIDECAR_SCHEMA_VERSION
    ):
        raise ValueError("unknown inference autotune sidecar request schema")
    return value


def _restore_params(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__hydra_roi_npy__"}:
            path = (root / str(value["__hydra_roi_npy__"])).resolve()
            if path.parent != root.resolve() or path.suffix != ".npy":
                raise ValueError("invalid staged ROI reference")
            return np.load(path, allow_pickle=False)
        return {str(key): _restore_params(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_params(item, root) for item in value]
    return value


def _header(identity_method: str, n_arenas: int) -> list[str]:
    columns = [
        "TrackID",
        "TrajectoryID",
        "Index",
        "X",
        "Y",
        "Theta",
        "FrameID",
        "State",
        "DetectionConfidence",
        "AssignmentConfidence",
        "PositionUncertainty",
        "DetectionID",
        *C.identity_realtime_columns(),
    ]
    if identity_method.strip().lower() == "apriltags":
        columns.extend(
            (
                "DetectedTagID",
                "DetectedTagLabel",
                "DetectedTagConf",
                "DetectedTagHamming",
            )
        )
    if n_arenas > 1:
        columns.append("arena_id")
    return columns


def _representative_windows(
    start: int, end: int, maximum_frames: int
) -> tuple[tuple[int, int], ...]:
    total = end - start + 1
    if total <= maximum_frames:
        return ((start, end),)
    per_window = max(8, maximum_frames // 3)
    per_window = min(per_window, total)
    starts = (
        start,
        max(start, start + (total - per_window) // 2),
        max(start, end - per_window + 1),
    )
    output = []
    for item in starts:
        window = (item, min(end, item + per_window - 1))
        if window not in output:
            output.append(window)
    return tuple(output)


def _profile_times(video_path: Path) -> tuple[float, float, dict[str, float]]:
    log_dir = video_path.parent / f"{video_path.stem}_logs"
    path = log_dir / "tracking_profile_forward.json"
    if not path.exists():
        return 0.0, 0.0, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        wall = float(payload.get("wall_clock_s", 0.0))
        phases = payload.get("phases", {})
        prepare = float(phases.get("initialization", {}).get("total_s", 0.0))
        cleanup = float(phases.get("cleanup", {}).get("total_s", 0.0))
        phase_seconds = {
            str(name): float(value.get("total_s", 0.0))
            for name, value in phases.items()
            if isinstance(value, dict)
        }
        return (
            max(1e-9, wall - prepare - cleanup),
            max(0.0, prepare),
            phase_seconds,
        )
    except (OSError, ValueError, TypeError, KeyError):
        return 0.0, 0.0, {}


def _run_window(
    *,
    video_path: Path,
    params: dict[str, Any],
    project_config: dict[str, Any],
    root: Path,
    label: str,
    start: int,
    end: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    float,
    float,
    dict[str, float],
    tuple[str, ...],
]:
    run_root = root / label
    run_root.mkdir(parents=True, exist_ok=False)
    raw_csv = run_root / "forward.csv"
    cache_dir = run_root / "inference-cache"
    cache_dir.mkdir()
    run_params = dict(params)
    run_params.update(
        {
            "START_FRAME": int(start),
            "END_FRAME": int(end),
            "ENABLE_PROFILING": True,
            "INFERENCE_AUTOTUNE_MODE": "off",
            "USE_CACHED_DETECTIONS": False,
            "DEBUG_MODE": True,
        }
    )
    identity_method = str(
        project_config.get(
            "identity_method", run_params.get("IDENTITY_METHOD", "none_disabled")
        )
    )
    writer = CSVWriterThread(
        str(raw_csv),
        header=_header(identity_method, int(run_params.get("N_ARENAS", 1))),
    )
    writer.start()
    captured: dict[str, Any] = {"success": False, "finished": False}

    def finished(success: object, _fps: object, _trajectories: object) -> None:
        captured["success"] = bool(success)
        captured["finished"] = True

    engine = TrackingEngineCore(
        str(video_path),
        csv_writer_thread=writer,
        video_output_path=None,
        backward_mode=False,
        detection_cache_path=str(cache_dir),
        preview_mode=False,
        use_cached_detections=False,
        inference_cache_dir=cache_dir,
        on_finished=finished,
    )
    engine.set_parameters(run_params)
    try:
        engine.run_tracking()
    finally:
        writer.stop()
        writer.join(timeout=10)
    if not captured["finished"] or not captured["success"]:
        raise RuntimeError("calibration tracking pass did not finish successfully")
    forward = pd.read_csv(raw_csv)
    if forward.empty:
        raise RuntimeError("calibration tracking pass produced no rows")

    session_config = dict(project_config)
    session_config.update(
        {
            "enable_backward_tracking": False,
            "enable_dataset_generation": False,
            "enable_individual_dataset": False,
            "enable_individual_image_save": False,
            "final_media_export_videos_enabled": False,
            "video_output_enabled": False,
            "interpolation_method": str(
                project_config.get("interpolation_method", "none")
            ),
            "heading_flip_max_burst": int(
                project_config.get("heading_flip_max_burst", 3)
            ),
            "identity_method": identity_method,
        }
    )
    service = TrackingSessionCore(
        video_path=str(video_path),
        config=session_config,
        params=run_params,
        paths={
            "raw_csv_path": str(raw_csv),
            "final_csv_path": str(run_root / "final.csv"),
            "detection_cache_path": str(cache_dir),
            "individual_properties_cache_path": getattr(
                engine, "individual_properties_cache_path", None
            ),
            "detected_properties_cache_path": getattr(
                engine, "detected_properties_cache_path", None
            ),
        },
        callbacks=SessionCallbacks(),
    )
    result = service.run_post_tracking(forward)
    if not result.success or not result.final_csv_path:
        raise RuntimeError(result.error or "calibration post-tracking pass failed")
    final_path = Path(result.rich_export_path or result.final_csv_path)
    final = pd.read_csv(final_path)
    if final.empty:
        raise RuntimeError("calibration final tracking output produced no rows")
    steady, prepare, phase_seconds = _profile_times(video_path)
    artifact_prepare = float(engine.inference_runtime_artifact_prepare_seconds)
    return (
        forward,
        final,
        steady,
        max(prepare, artifact_prepare),
        phase_seconds,
        tuple(engine.inference_runtime_artifact_ids),
    )


def _stage_evidence(
    settings: InferenceTuningSettings,
    phase_seconds: dict[str, float],
    steady_seconds: float,
) -> dict[str, float]:
    detection = phase_seconds.get("batched_detection", 0.0)
    if detection <= 0:
        detection = sum(
            phase_seconds.get(name, 0.0)
            for name in ("yolo_obb_inference", "sequential_obb_inference")
        )
    pose = sum(
        phase_seconds.get(name, 0.0) for name in ("pose_inference", "precompute_pose")
    )
    headtail = phase_seconds.get("headtail_inference", 0.0)
    identity = phase_seconds.get("precompute_cnn_identity", 0.0)
    values: dict[str, float] = {
        "detection_batch_size": detection,
        "pipeline_depth": max(0.0, steady_seconds * 0.01),
    }
    if settings.slice_tile_batch_size is not None:
        values["slice_tile_batch_size"] = detection
    if settings.pose_batch_size is not None:
        values["pose_batch_size"] = pose
    if settings.headtail_batch_size is not None:
        values["headtail_batch_size"] = headtail
    for label, _batch in settings.identity_batch_sizes:
        values[f"identity_batch_size:{label}"] = identity
    return values


def _atomic_result(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(request_path: Path) -> None:
    request_path = request_path.expanduser().resolve()
    request_root = request_path.parent
    request = _read_request(request_path)
    # Candidate and duplicate-baseline trials start from the same stochastic
    # state; block randomization belongs to scheduling, never model semantics.
    random.seed(0)
    np.random.seed(0)
    settings = InferenceTuningSettings.from_dict(request["settings"])
    params = _restore_params(request["params"], request_root)
    project_config = params.get("INFERENCE_AUTOTUNE_PROJECT_CONFIG", {})
    if not isinstance(project_config, dict):
        project_config = {}
    source = Path(str(request["video_path"])).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"calibration video is unavailable: {source}")
    output = request_root / str(request["output_dir"])
    if output.resolve().parent != request_root.resolve():
        raise ValueError("calibration output escaped its private root")
    output.mkdir(parents=True, exist_ok=False)
    private_video = output / f"source{source.suffix or '.mp4'}"
    private_video.symlink_to(source)

    start = int(request["start_frame"])
    end = int(request["end_frame"])
    maximum_frames = int(request["maximum_frames"])
    windows = _representative_windows(start, end, maximum_frames)
    # Warm allocator/framework state with at least eight real frames and try to
    # provide three detector calls. The 128-frame cap wins for very large batch
    # values; those candidates report the smaller count and cannot be promoted.
    warmup_frames = min(
        maximum_frames,
        end - start + 1,
        max(8, 3 * settings.detection_batch_size),
    )
    _run_window(
        video_path=private_video,
        params=params,
        project_config=project_config,
        root=output,
        label="warmup",
        start=start,
        end=start + warmup_frames - 1,
    )
    warmup_calls = min(3, math.ceil(warmup_frames / settings.detection_batch_size))

    forwards = []
    finals = []
    steady_seconds = 0.0
    prepare_seconds = 0.0
    measured_frames = 0
    phase_seconds: dict[str, float] = {}
    artifact_ids: set[str] = set()
    wall_started = time.perf_counter()
    for index, (window_start, window_end) in enumerate(windows):
        forward, final, steady, prepare, window_phases, window_artifacts = _run_window(
            video_path=private_video,
            params=params,
            project_config=project_config,
            root=output,
            label=f"measure-{index}",
            start=window_start,
            end=window_end,
        )
        forwards.append(forward)
        finals.append(final)
        steady_seconds += steady
        prepare_seconds += prepare
        measured_frames += window_end - window_start + 1
        for name, seconds in window_phases.items():
            phase_seconds[name] = phase_seconds.get(name, 0.0) + seconds
        artifact_ids.update(window_artifacts)
    wall_seconds = time.perf_counter() - wall_started
    if steady_seconds <= 0:
        steady_seconds = wall_seconds
    forward_csv = output / "forward.csv"
    final_csv = output / "final.csv"
    pd.concat(forwards, ignore_index=True).to_csv(forward_csv, index=False)
    pd.concat(finals, ignore_index=True).to_csv(final_csv, index=False)
    stage_evidence = _stage_evidence(settings, phase_seconds, steady_seconds)
    field_name = request.get("field_name")
    measured_stage_seconds = (
        stage_evidence.get(str(field_name), steady_seconds)
        if request.get("phase") == "stage"
        else steady_seconds
    )
    if measured_stage_seconds <= 0:
        measured_stage_seconds = steady_seconds
    _atomic_result(
        output / "result.json",
        {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "settings": settings.to_dict(),
            "throughput": measured_frames / steady_seconds,
            "stage_seconds": measured_stage_seconds,
            "prepare_seconds": prepare_seconds,
            "measured_frames": measured_frames,
            "warmup_calls": warmup_calls,
            "warmup_frames": warmup_frames,
            "forward_csv": "output/forward.csv",
            "final_csv": "output/final.csv",
            "artifact_ids": sorted(artifact_ids),
            "stage_shares": stage_evidence,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        run(args.request)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
