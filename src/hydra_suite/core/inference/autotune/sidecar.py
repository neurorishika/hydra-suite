"""Contained fresh-process trial executor for full tracking confirmation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from hydra_suite.paths import get_data_dir
from hydra_suite.runtime.process_supervisor import (
    ContainmentPlan,
    ExitKind,
    SupervisedSidecar,
    WorkloadStillOwnedError,
)
from hydra_suite.runtime.resource_budget import AcceleratorKind, ResourceObservation
from hydra_suite.runtime.resource_limits import (
    ProcessMemoryLimits,
    build_limited_launch,
)

from .device import RuntimeResourceProbe, cuda_used_memory_bytes
from .equivalence import CalibrationOutputs
from .measure import MeasurementProtocol
from .models import (
    DEFAULT_CALIBRATION_BUDGET_SECONDS,
    MAXIMUM_CALIBRATION_BUDGET_SECONDS,
    MINIMUM_CALIBRATION_BUDGET_SECONDS,
    InferenceTuningSettings,
)
from .search import TrialObservation

logger = logging.getLogger(__name__)

SIDECAR_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MEASUREMENT_BLOCKS = 5
# The child has no MeasurementProtocol; the warmup-call minimum it must satisfy
# travels in the request so there is exactly one source of truth for it.
DEFAULT_WARMUP_CALLS = MeasurementProtocol().warmup_calls


def _frames_for_block(maximum_frames: int, block_index: int) -> int:
    """Distribute one phase's frame cap across its required five blocks."""

    quotient, remainder = divmod(int(maximum_frames), MEASUREMENT_BLOCKS)
    return max(8, quotient + (1 if block_index % MEASUREMENT_BLOCKS < remainder else 0))


@dataclass(frozen=True, slots=True)
class SidecarTrialSpec:
    video_path: str | Path
    params: Mapping[str, Any]
    observation: ResourceObservation
    resource_probe: RuntimeResourceProbe
    start_frame: int
    end_frame: int
    warmup_calls: int = DEFAULT_WARMUP_CALLS
    budget_seconds: float = DEFAULT_CALIBRATION_BUDGET_SECONDS
    # The validator's ceiling: a cold TensorRT engine build can take most of it.
    per_trial_timeout_seconds: float = 120.0
    # Per-PHASE frame cap, divided across MEASUREMENT_BLOCKS blocks by
    # _frames_for_block. 640 gives each of the five blocks >= 128 frames, which
    # is what amortizes a model load. Distinct from
    # MeasurementProtocol.maximum_frames, which is the per-candidate COMPLETION
    # threshold and must stay at the per-block value.
    maximum_frames: int = 640
    runtime_artifact_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("invalid calibration frame range")
        if not (
            MINIMUM_CALIBRATION_BUDGET_SECONDS
            <= self.budget_seconds
            <= MAXIMUM_CALIBRATION_BUDGET_SECONDS
        ):
            raise ValueError(
                "calibration budget must be between "
                f"{MINIMUM_CALIBRATION_BUDGET_SECONDS:g} and "
                f"{MAXIMUM_CALIBRATION_BUDGET_SECONDS:g} seconds"
            )
        if not 5 <= self.per_trial_timeout_seconds <= 120:
            raise ValueError("trial timeout must be between 5 and 120 seconds")
        if not 8 <= self.maximum_frames <= 2048:
            raise ValueError("calibration frame cap must be between 8 and 2048")
        if self.runtime_artifact_batch_size is not None and not (
            1 <= self.runtime_artifact_batch_size <= 64
        ):
            raise ValueError("runtime artifact batch size must be between 1 and 64")


def apply_settings_to_params(
    params: Mapping[str, Any],
    settings: InferenceTuningSettings,
    *,
    runtime_artifact_batch_size: int | None = None,
) -> dict[str, Any]:
    """Return candidate engine params without mutating the session snapshot."""

    result = deepcopy(dict(params))
    result["INFERENCE_AUTOTUNE_MODE"] = "off"
    result["YOLO_BATCH_SIZE"] = settings.detection_batch_size
    result["PIPELINE_DEPTH"] = settings.pipeline_depth
    if runtime_artifact_batch_size is not None:
        result["INFERENCE_AUTOTUNE_ARTIFACT_BATCH_SIZE"] = int(
            runtime_artifact_batch_size
        )
    if settings.slice_tile_batch_size is not None:
        if str(result.get("YOLO_OBB_MODE", "direct")).lower() == "sequential":
            result["YOLO_SEQ_STAGE1_SLICE_TILE_BATCH_SIZE"] = (
                settings.slice_tile_batch_size
            )
            result["YOLO_SEQ_STAGE1_SLICE_TILE_BATCH_AUTOTUNE"] = False
        else:
            result["SLICE_TILE_BATCH_SIZE"] = settings.slice_tile_batch_size
            result["SLICE_TILE_BATCH_AUTOTUNE"] = False
    if settings.headtail_batch_size is not None:
        result["HEADTAIL_BATCH_SIZE"] = settings.headtail_batch_size
    if settings.pose_batch_size is not None:
        result["POSE_BATCH_SIZE"] = settings.pose_batch_size
    identity_batches = dict(settings.identity_batch_sizes)
    classifiers = []
    for raw in result.get("CNN_CLASSIFIERS", ()):
        classifier = deepcopy(dict(raw))
        label = str(classifier.get("label", "cnn_identity"))
        if label in identity_batches:
            classifier["batch_size"] = identity_batches[label]
        classifiers.append(classifier)
    result["CNN_CLASSIFIERS"] = classifiers
    # Candidate work is disposable and must not consume or promote production
    # result generations or emit unrelated media/dataset artifacts.
    result.update(
        {
            "USE_CACHED_DETECTIONS": False,
            "ENABLE_PROFILING": True,
            "VIDEO_OUTPUT_ENABLED": False,
            "ENABLE_DATASET_GENERATION": False,
            "ENABLE_INDIVIDUAL_DATASET": False,
            "ENABLE_INDIVIDUAL_IMAGE_SAVE": False,
            "ENABLE_CONFIDENCE_DENSITY_MAP": False,
            "FINAL_MEDIA_EXPORT_VIDEOS_ENABLED": False,
            "DEBUG_MODE": True,
        }
    )
    return result


def _json_value(value: Any, *, array_dir: Path, key: str = "") -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if not key:
            raise TypeError("array parameters require a key to be serialized")
        array_dir.mkdir(parents=True, exist_ok=True)
        name = f"{key}.npy"
        # Symmetric guard to the read-side path-escape check below: a key
        # containing a path separator or ".." must never be able to stage a
        # file outside array_dir.
        resolved_array_dir = array_dir.resolve()
        target = (resolved_array_dir / name).resolve()
        if target.parent != resolved_array_dir:
            raise ValueError(f"invalid array parameter key: {key!r}")
        np.save(target, value, allow_pickle=False)
        return {"__hydra_npy__": name}
    if isinstance(value, Mapping):
        return {
            str(item_key): _json_value(item, array_dir=array_dir, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item, array_dir=array_dir, key=key) for item in value]
    raise TypeError(
        f"unsupported sidecar parameter type for {key}: {type(value).__name__}"
    )


def restore_sidecar_params(value: Any, root: Path) -> Any:
    """Restore ndarray params staged by ``_json_value`` under ``root``.

    Accepts both the current per-key ``arrays/`` layout (``__hydra_npy__``)
    and the legacy ROI-only layout (``__hydra_roi_npy__``, staged directly
    under ``root``), so an in-flight request written by an older build still
    loads.
    """

    if isinstance(value, Mapping):
        if set(value) == {"__hydra_npy__"}:
            array_dir = (root / "arrays").resolve()
            path = (array_dir / str(value["__hydra_npy__"])).resolve()
            if path.parent != array_dir or path.suffix != ".npy":
                raise ValueError("invalid staged array reference")
            return np.load(path, allow_pickle=False)
        if set(value) == {"__hydra_roi_npy__"}:
            resolved_root = root.resolve()
            path = (resolved_root / str(value["__hydra_roi_npy__"])).resolve()
            if path.parent != resolved_root or path.suffix != ".npy":
                raise ValueError("invalid staged ROI reference")
            return np.load(path, allow_pickle=False)
        return {
            str(item_key): restore_sidecar_params(item, root)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [restore_sidecar_params(item, root) for item in value]
    return value


def write_sidecar_request(
    root: Path,
    spec: SidecarTrialSpec,
    settings: InferenceTuningSettings,
    *,
    phase: str,
    field_name: str | None,
    block_index: int,
) -> Path:
    """Stage bounded JSON IPC and separate non-pickled ndarray payloads."""

    root.mkdir(parents=True, exist_ok=False)
    array_dir = root / "arrays"
    params = apply_settings_to_params(
        spec.params,
        settings,
        # Screens share one wide profile. The authoritative final pass omits
        # the override so it builds/loads the dedicated selected profile.
        runtime_artifact_batch_size=(
            None if phase == "final_validation" else spec.runtime_artifact_batch_size
        ),
    )
    payload = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "video_path": str(Path(spec.video_path).expanduser().resolve()),
        "params": _json_value(params, array_dir=array_dir),
        "settings": settings.to_dict(),
        "phase": str(phase),
        "field_name": field_name,
        "block_index": int(block_index),
        "start_frame": int(spec.start_frame),
        "end_frame": int(spec.end_frame),
        "maximum_frames": _frames_for_block(spec.maximum_frames, block_index),
        # The child places its single contiguous window using the block index
        # and the block count, so the five blocks stripe across the clip.
        "measurement_blocks": MEASUREMENT_BLOCKS,
        "warmup_calls": int(spec.warmup_calls),
        "output_dir": "output",
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("inference autotune sidecar request exceeds its size cap")
    request_path = root / "request.json"
    with request_path.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return request_path


def _read_relative_path(root: Path, raw: str) -> Path:
    candidate = (root / raw).resolve()
    if candidate.parent != (root / "output").resolve():
        raise ValueError("sidecar result path escaped its private output directory")
    return candidate


def read_sidecar_result(
    root: Path,
    settings: InferenceTuningSettings,
    *,
    peak_tree_rss_bytes: int,
    peak_accelerator_bytes: int,
) -> TrialObservation:
    path = root / "output" / "result.json"
    with path.open("rb") as stream:
        encoded = stream.read(MAX_RESULT_BYTES + 1)
    if len(encoded) > MAX_RESULT_BYTES:
        raise ValueError("inference autotune sidecar result exceeds its size cap")
    raw = json.loads(encoded)
    if raw.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ValueError("unknown inference autotune sidecar result schema")
    returned = InferenceTuningSettings.from_dict(raw["settings"])
    if returned != settings:
        raise ValueError("sidecar returned results for another candidate")
    forward = _read_relative_path(root, str(raw["forward_csv"]))
    final = _read_relative_path(root, str(raw["final_csv"]))
    outputs = CalibrationOutputs.from_csvs(forward, final)
    return TrialObservation(
        settings=settings,
        throughput=float(raw["throughput"]),
        stage_seconds=float(raw["stage_seconds"]),
        outputs=outputs,
        host_peak_bytes=max(int(raw.get("host_peak_bytes", 0)), peak_tree_rss_bytes),
        accelerator_peak_bytes=max(
            int(raw.get("accelerator_peak_bytes", 0)), peak_accelerator_bytes
        ),
        queue_high_water_bytes=int(raw.get("queue_high_water_bytes", 0)),
        frame_buffer_high_water_bytes=int(raw.get("frame_buffer_high_water_bytes", 0)),
        thermal_c=(
            float(raw["thermal_c"]) if raw.get("thermal_c") is not None else None
        ),
        warmup_calls=int(raw["warmup_calls"]),
        warmup_frames=int(raw["warmup_frames"]),
        prepare_seconds=float(raw.get("prepare_seconds", 0.0)),
        measured_frames=int(raw.get("measured_frames", 0)),
        artifact_ids=tuple(map(str, raw.get("artifact_ids", ()))),
        stage_shares=tuple(
            sorted(
                (str(name), float(value))
                for name, value in dict(raw.get("stage_shares", {})).items()
            )
        ),
    )


class ContainedTrialExecutor:
    """Execute every trial in a fresh supervised process and private cache root."""

    def __init__(self, spec: SidecarTrialSpec) -> None:
        self.spec = spec
        self._started = time.monotonic()

    def run(
        self,
        settings: InferenceTuningSettings,
        *,
        phase: str,
        field_name: str | None,
        block_index: int,
        should_cancel: Callable[[], bool],
    ) -> TrialObservation:
        first = self._run_once(
            settings,
            phase=phase,
            field_name=field_name,
            block_index=block_index,
            should_cancel=should_cancel,
        )
        # This candidate is an honest, unambiguous failure. A previous
        # implementation ran an extra reduced-pressure probe here and cached
        # its (successful) result for a LATER query of the reduced settings
        # vector at the "same" block_index. That broke the measurement
        # protocol's paired-block structure: `deterministic_block_order`
        # interleaves candidates so every candidate's sample for a given
        # block is measured at a matching point in the schedule (same
        # wall-clock/thermal/contention conditions); splicing in a
        # measurement taken out-of-turn, while probing a DIFFERENT failed
        # candidate, silently violated that pairing. If the coordinate
        # search independently wants to evaluate a reduced settings vector,
        # it measures it itself, in its own correct block position.
        return first

    def _run_once(
        self,
        settings: InferenceTuningSettings,
        *,
        phase: str,
        field_name: str | None,
        block_index: int,
        should_cancel: Callable[[], bool],
    ) -> TrialObservation:
        if should_cancel():
            return self._failure(settings, "cancelled")
        remaining = self.spec.budget_seconds - (time.monotonic() - self._started)
        if remaining <= 0:
            return self._failure(settings, "budget_expired")
        parent = get_data_dir() / "inference_tuning_scratch"
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="trial-", dir=parent))
        # write_sidecar_request requires an uncreated target so use a child of
        # the securely-created parent and never broaden cleanup beyond it.
        request_root = root / "ipc"
        sidecar = None
        cleanup_allowed = True
        try:
            request = write_sidecar_request(
                request_root,
                self.spec,
                settings,
                phase=phase,
                field_name=field_name,
                block_index=block_index,
            )
            limits = self._limits()
            probe = self.spec.resource_probe
            launch = build_limited_launch(
                (
                    sys.executable,
                    "-m",
                    "hydra_suite.core.inference.autotune.sidecar_child",
                    "--request",
                    str(request),
                ),
                limits,
                accelerator_kind=self.spec.observation.accelerator_kind,
                accelerator_device_uuid=(
                    probe.device_uuid
                    if self.spec.observation.accelerator_kind is AcceleratorKind.CUDA
                    else None
                ),
                accelerator_pci_bus_id=None,
            )
            plan = ContainmentPlan(
                launch=launch,
                job_name="inference-throughput-autotune",
                minimum_system_available_bytes=max(
                    1024**3, int(self.spec.observation.total_host_bytes * 0.15)
                ),
            )
            sidecar = SupervisedSidecar(
                plan,
                accelerator_probe=(
                    (lambda: cuda_used_memory_bytes(probe.device_uuid))
                    if self.spec.observation.accelerator_kind is AcceleratorKind.CUDA
                    else None
                ),
            )
            timeout = min(self.spec.per_trial_timeout_seconds, max(0.1, remaining))
            deadline = time.monotonic() + timeout
            while sidecar.process is not None and sidecar.process.poll() is None:
                if should_cancel():
                    sidecar.cancel()
                    sidecar = None
                    return self._failure(settings, "cancelled")
                if time.monotonic() >= deadline:
                    sidecar.cancel()
                    sidecar = None
                    return self._failure(settings, "timeout")
                time.sleep(0.05)
            supervised = sidecar.wait(timeout=10.0)
            sidecar = None
            if supervised.classified_exit.kind is not ExitKind.SUCCESS:
                return self._failure(settings, supervised.classified_exit.kind.value)
            return read_sidecar_result(
                request_root,
                settings,
                peak_tree_rss_bytes=supervised.peak_tree_rss_bytes,
                peak_accelerator_bytes=int(supervised.peak_accelerator_bytes or 0),
            )
        except WorkloadStillOwnedError:
            cleanup_allowed = False
            raise
        except Exception as exc:
            logger.warning("Contained inference trial failed: %s", exc, exc_info=True)
            if sidecar is not None:
                try:
                    sidecar.cancel()
                except WorkloadStillOwnedError:
                    cleanup_allowed = False
                    raise
                except Exception:
                    logger.exception("Failed to tear down inference trial sidecar")
            return self._failure(settings, type(exc).__name__)
        finally:
            if cleanup_allowed:
                shutil.rmtree(root, ignore_errors=True)

    def _limits(self) -> ProcessMemoryLimits:
        observation = self.spec.observation
        reserve = max(1024**3, int(observation.total_host_bytes * 0.15))
        hard = max(512 * 1024**2, observation.available_host_bytes - reserve)
        soft = max(256 * 1024**2, int(hard * 0.9))
        return ProcessMemoryLimits(
            soft_host_bytes=min(soft, hard),
            hard_host_bytes=hard,
            mps_high_watermark_ratio=(
                0.7 if observation.accelerator_kind is AcceleratorKind.MPS else None
            ),
        )

    @staticmethod
    def _failure(
        settings: InferenceTuningSettings, failure_class: str
    ) -> TrialObservation:
        return TrialObservation(
            settings=settings,
            throughput=0.0,
            stage_seconds=0.0,
            outputs=None,
            failure_class=failure_class,
        )
