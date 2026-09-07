"""Shared fixture helpers for the inference-autotune test suite.

Every autotune test file should import shared constructions from here rather
than redefining them locally -- keeps exactly one definition of each helper
alive as the remediation programme (see
``.superpowers/sdd/2026-09-07-inference-autotuner-review-remediation``) adds
tests across several files.

This module starts intentionally small: only the helpers Task 1 (the sidecar
child e2e anchor) needs. Later tasks add more helpers here as they need them
-- never redefine one locally that already has a home in this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra_suite.core.inference.autotune.device import RuntimeResourceProbe
from hydra_suite.core.inference.autotune.fingerprint import (
    AcceleratorFingerprint,
    DetectorFingerprint,
    FrameFingerprint,
    ModelArtifactFingerprint,
    PipelineFingerprint,
    SchemaFingerprint,
    SliceFingerprint,
    SoftwareFingerprint,
    SystemFingerprint,
    TuningProfileKey,
    WorkloadFingerprint,
)
from hydra_suite.core.inference.autotune.models import InferenceTuningSettings
from hydra_suite.runtime.resource_budget import AcceleratorKind, ResourceObservation


def _key() -> TuningProfileKey:
    """A representative, internally-consistent ``TuningProfileKey``.

    Lifted from ``tests/test_inference_autotune_search.py::_key`` -- the
    exact fingerprint values don't matter for most tests, only that the
    tuple is well-formed and stable across calls.
    """

    return TuningProfileKey(
        SchemaFingerprint(),
        SystemFingerprint("host", "linux", "cpu", 8, 32 * 1024**3),
        AcceleratorFingerprint("gpu", "gpu", "8.9", 1_000),
        SoftwareFingerprint(
            "1", "c", "py", "torch", "fp16", "d", "c", "u", "t", "p", "y", "s"
        ),
        (ModelArtifactFingerprint("detector", "a" * 64),),
        FrameFingerprint(10, 10, 3, "bgr8", 1.0, "ffmpeg"),
        DetectorFingerprint("obb", "obb", (), 0.2, 0.7, 8, "direct"),
        SliceFingerprint(False, 0, 0, 0.2, 0.2, False, "none", "none"),
        PipelineFingerprint(("detector", "pose"), "forward", "batch", ()),
        WorkloadFingerprint(8, 8, 8, 8, 8, ("10x10",)),
    )


def _settings(
    det: int = 1, pose: int = 1, identity: int = 1
) -> InferenceTuningSettings:
    """A representative ``InferenceTuningSettings``.

    Lifted from ``tests/test_inference_autotune_search.py::_settings``.
    """

    return InferenceTuningSettings(
        detection_batch_size=det,
        pose_batch_size=pose,
        identity_batch_sizes=(("animal", identity),),
        pipeline_depth=2,
    )


def _resource_probe(
    *, accelerator_kind: AcceleratorKind = AcceleratorKind.CPU
) -> tuple[ResourceObservation, RuntimeResourceProbe]:
    """A CPU-tier ``(ResourceObservation, RuntimeResourceProbe)`` pair.

    Lifted from ``tests/test_inference_autotune_sidecar.py::_resources``; the
    sidecar spec needs both to build a real containment plan.
    """

    observation = ResourceObservation(
        total_host_bytes=64 * 1024**3,
        available_host_bytes=48 * 1024**3,
        accelerator_kind=accelerator_kind,
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


def make_roi_params(video_path: Path) -> dict[str, Any]:
    """Engine params for the ``fly_obb`` fixture WITH a whole-frame ROI.

    A non-empty ``roi_shapes`` makes ``build_engine_params`` emit
    ``ARENA_LABELS`` (a uint16 ndarray) alongside ``ROI_MASK`` -- the exact
    shape that broke sidecar request serialization (finding B1). The
    rectangle covers the entire frame: an ROI smaller than the frame
    suppresses detections and yields a header-only CSV, which would fail
    later stages of this test for the wrong reason, so the real frame size
    is read via ``cv2`` rather than hardcoded.

    Built via ``build_tracking_parameters`` + ``probe_video`` -- the exact
    path the CLI uses -- so this fixture cannot silently drift from what a
    real headless run does.
    """

    import json

    from hydra_suite.trackerkit.cli_config import build_tracking_parameters, probe_video

    fixtures_root = Path(__file__).resolve().parents[1] / "tools/equivalence/fixtures"
    config = json.loads((fixtures_root / "configs/fly_obb.json").read_text())

    probe = probe_video(str(video_path))
    config["roi_shapes"] = [
        {
            "mode": "include",
            "arena_id": 0,
            "type": "rectangle",
            "points": [[0, 0], [probe.width, probe.height]],
        }
    ]
    return build_tracking_parameters(config, video_probe=probe)
