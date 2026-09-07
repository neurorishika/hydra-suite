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

from hydra_suite.core.inference.autotune.candidates import (
    AdmissionContext,
    CandidatePlanner,
    MemoryCostModel,
)
from hydra_suite.core.inference.autotune.coordinator import AutotuneRequest
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
from hydra_suite.core.inference.autotune.models import (
    CandidateEvidence,
    EquivalenceVerdict,
    InferenceTuningProfile,
    InferenceTuningSettings,
    ProfileState,
)
from hydra_suite.core.inference.autotune.store import InferenceTuningProfileStore
from hydra_suite.runtime.resource_budget import (
    AcceleratorKind,
    ResourceObservation,
    ResourcePolicy,
)


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


def _planner(
    *, free: int = 10_000, cost=None, cached_fields=frozenset()
) -> CandidatePlanner:
    """A representative ``CandidatePlanner``.

    Lifted from ``tests/test_inference_autotune_search.py::_planner``.
    """

    observation = ResourceObservation(
        total_host_bytes=100_000,
        available_host_bytes=100_000,
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_name="gpu",
        total_accelerator_bytes=10_000,
        available_accelerator_bytes=free,
    )
    return CandidatePlanner(
        AdmissionContext(
            observation,
            frame_bytes=1,
            crop_count_p95=8,
            hard_maxima=(
                ("detection_batch_size", 4),
                ("pose_batch_size", 4),
                ("identity_batch_size:animal", 8),
                ("pipeline_depth", 2),
            ),
            cost=cost or MemoryCostModel(),
            policy=ResourcePolicy(
                reserve_host_bytes=0,
                reserve_host_fraction=0,
                accelerator_safety_fraction=1,
            ),
            cached_fields=cached_fields,
        )
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
    # Production ROI shapes recognize ONLY "circle" or "polygon" (see
    # engine_params._fill_shape); there is no "rectangle" type, and geometry
    # is read from "params", never "points". A rectangle is represented as a
    # 4-point polygon, matching every real emitter in arena_geometry.py.
    config["roi_shapes"] = [
        {
            "mode": "include",
            "arena_id": 0,
            "type": "polygon",
            "params": [
                [0, 0],
                [probe.width, 0],
                [probe.width, probe.height],
                [0, probe.height],
            ],
        }
    ]
    return build_tracking_parameters(config, video_probe=probe)


def record_mode_request_with_validated_cache(
    tmp_path: Path,
) -> tuple[InferenceTuningProfileStore, AutotuneRequest]:
    """A record-mode request whose store already holds a VALIDATED profile.

    ``selected`` differs from ``baseline`` in ``detection_batch_size`` so a
    test can prove a record-mode cache hit keeps the *baseline* effective
    settings rather than silently applying the tuned vector (finding B2:
    record mode must never apply, on any run).

    Modeled on the construction in
    ``tests/test_inference_autotune_search.py::test_record_only_persists_but_does_not_apply``
    (run-1 coverage); this helper covers run 2 -- a cache hit.
    """

    key = _key()
    baseline = _settings(det=1)
    selected = _settings(det=4)
    profile = InferenceTuningProfile(
        key.digest[:24],
        key,
        baseline,
        baseline,
        selected,
        selected,
        (
            CandidateEvidence(
                selected,
                (120.0,) * 5,
                stage_seconds_samples=(0.5,) * 5,
                warmup_calls=3,
                warmup_frames=8,
                equivalence=EquivalenceVerdict(True),
            ),
        ),
        ProfileState.VALIDATED,
        "winner",
    )
    store = InferenceTuningProfileStore(tmp_path)
    store.save(profile)
    request = AutotuneRequest(
        key,
        baseline,
        _planner(),
        mode="record",
    )
    return store, request
