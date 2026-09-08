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

import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from hydra_suite.core.inference.autotune.candidates import (
    AdmissionContext,
    CandidatePlanner,
    MemoryCostModel,
)
from hydra_suite.core.inference.autotune.coordinator import (
    AutotuneCoordinator,
    AutotuneRequest,
)
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


def _key(**overrides: Any) -> TuningProfileKey:
    """A representative, internally-consistent ``TuningProfileKey``.

    Lifted from ``tests/test_inference_autotune_search.py::_key`` -- the
    exact fingerprint values don't matter for most tests, only that the
    tuple is well-formed and stable across calls. ``overrides`` are applied
    via ``dataclasses.replace`` (e.g. ``_key(baseline_digest="...")`` or
    ``_key(workload=WorkloadFingerprint(...))``).
    """

    key = TuningProfileKey(
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
    return replace(key, **overrides) if overrides else key


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


def _profile(key: TuningProfileKey | None = None) -> InferenceTuningProfile:
    """A representative VALIDATED ``InferenceTuningProfile`` for ``key``.

    ``selected`` differs from ``baseline`` in ``detection_batch_size`` so
    tests can tell a reused/tuned vector apart from the baseline fallback.
    """

    key = key or _key()
    baseline = _settings(det=1)
    selected = _settings(det=4)
    evidence = CandidateEvidence(
        selected,
        (100.0,) * 5,
        stage_seconds_samples=(0.5,) * 5,
        warmup_calls=3,
        warmup_frames=8,
        accelerator_peak_bytes=1024,
        equivalence=EquivalenceVerdict(True),
    )
    return InferenceTuningProfile(
        profile_id=key.digest[:24],
        key=key,
        baseline=baseline,
        requested=baseline,
        admitted=selected,
        selected=selected,
        candidates=(evidence,),
        state=ProfileState.VALIDATED,
        selection_reason="validated_throughput_gain",
        created_at_unix_ns=1,
        last_validation_unix_ns=1,
    )


def _store_with_validated_profile(
    tmp_path: Path, *, keyed_on_max_targets: bool = False
) -> tuple[InferenceTuningProfileStore, TuningProfileKey]:
    """A store holding one VALIDATED profile, optionally keyed run-1-style.

    ``keyed_on_max_targets=True`` mirrors run 1 of a brand-new video: no
    detection cache exists yet, so the workload bucket is the
    ``MAX_TARGETS`` fallback rather than a measurement (S2).
    """

    workload = WorkloadFingerprint.from_counts(
        8, (8,), (8,), ("10x10",), density_is_estimated=keyed_on_max_targets
    )
    key = _key(workload=workload)
    profile = _profile(key)
    store = InferenceTuningProfileStore(tmp_path)
    store.save(profile)
    return store, key


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


def calibrate_request_with_validated_cache(
    tmp_path: Path,
) -> tuple[InferenceTuningProfileStore, AutotuneRequest]:
    """A calibrate-mode request whose store already holds a VALIDATED profile.

    ``selected`` differs from ``baseline`` in ``detection_batch_size`` so a
    test can prove a calibrate-mode cache hit is served from the store
    (``cache_hit``) rather than re-measuring.

    Modeled on the construction in
    ``tests/test_inference_autotune_search.py::test_calibrate_persists_a_validated_profile``
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
        mode="calibrate",
    )
    return store, request


def equivalence_frame(
    rows: int = 1,
    *,
    frame_ids: "Any | None" = None,
    detection_ids: "Any | None" = None,
    track_ids: "Any | None" = None,
    x: "Any" = 1.0,
    y: "Any" = 2.0,
    theta: "Any" = 0.0,
    state: "Any" = "confirmed",
    extra: "dict[str, Any] | None" = None,
    drop: "tuple[str, ...]" = (),
):
    """A minimal tracking-CSV-shaped frame for correctness-gate tests.

    Column names match the real exported schema the gate sees (``FrameID``,
    ``DetectionID``, ``TrackID``, ``State``, ``X``, ``Y``, ``Theta``) so tests
    exercise the production code paths rather than invented names.
    """

    import pandas as pd

    def _column(value, default_range=False):
        if value is None and default_range:
            return list(range(rows))
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value] * rows

    data: "dict[str, Any]" = {
        "FrameID": _column(frame_ids, default_range=True),
        "DetectionID": _column(detection_ids, default_range=True),
        "TrackID": _column(track_ids if track_ids is not None else 1),
        "State": _column(state),
        "X": _column(x),
        "Y": _column(y),
        "Theta": _column(theta),
    }
    for name in drop:
        data.pop(name, None)
    if extra:
        for name, value in extra.items():
            data[name] = _column(value)
    return pd.DataFrame(data)


def equivalence_outputs(frame):
    """Wrap one frame as both the forward and final calibration outputs."""

    from hydra_suite.core.inference.autotune.equivalence import CalibrationOutputs

    return CalibrationOutputs(frame.copy(), frame.copy())


def make_request_inputs(
    tmp_path: Path | None = None,
    *,
    kind: AcceleratorKind = AcceleratorKind.CUDA,
    execution_mode: str = "batch",
    available_accelerator_bytes: int = 48 * 1024**3,
) -> dict[str, Any]:
    """Keyword arguments for ``build_tracking_autotune_request``.

    Defaults to a CUDA batch context with an admissible baseline; override
    ``kind``/``execution_mode``/``available_accelerator_bytes`` to exercise
    the eligibility-split cases. Follows the construction pattern in
    ``tests/test_inference_autotune_integration.py`` (``_config``/
    ``TrackingRunContext``/``build_tracking_autotune_request``) rather than
    inventing a new one.
    """

    import tempfile

    from hydra_suite.core.inference.autotune.integration import TrackingRunContext
    from hydra_suite.core.inference.config import (
        CNNConfig,
        HeadTailConfig,
        InferenceConfig,
        OBBConfig,
        OBBDirectConfig,
        PoseConfig,
        PoseYOLOConfig,
        SliceConfig,
    )

    tmp_path = tmp_path or Path(tempfile.mkdtemp())

    def _model(name: str, payload: bytes) -> str:
        path = tmp_path / name
        path.write_bytes(payload)
        return str(path)

    detector = _model("detector.pt", b"detector")
    headtail = _model("headtail.pt", b"headtail")
    pose = _model("pose.pt", b"pose")
    identity = _model("identity.pt", b"identity")
    config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(
                detector,
                slice=SliceConfig(
                    enabled=True,
                    geometry_mode="custom",
                    slice_width=512,
                    slice_height=384,
                    tile_batch_size=2,
                ),
            ),
            target_classes=[0],
            max_detections=25,
        ),
        headtail=HeadTailConfig(headtail, batch_size=8),
        cnn_phases=[CNNConfig("color", identity, batch_size=8)],
        pose=PoseConfig(backend="yolo", yolo=PoseYOLOConfig(pose, batch_size=8)),
        detection_batch_size=2,
        pipeline_depth=2,
        runtime_tier="cpu",
    )
    context = TrackingRunContext(
        video_path=tmp_path / "video.mp4",
        params={"INFERENCE_AUTOTUNE_MODE": "automatic", "MAX_TARGETS": 25},
        frame_width=1200,
        frame_height=900,
        execution_mode=execution_mode,
    )
    if kind is AcceleratorKind.CUDA:
        observation = ResourceObservation(
            total_host_bytes=64 * 1024**3,
            available_host_bytes=48 * 1024**3,
            accelerator_kind=kind,
            accelerator_name="accelerator",
            total_accelerator_bytes=48 * 1024**3,
            available_accelerator_bytes=available_accelerator_bytes,
        )
    else:
        # MPS uses unified host memory and CPU has no separate accelerator
        # pool -- neither may carry total/available_accelerator_bytes.
        observation = ResourceObservation(
            total_host_bytes=64 * 1024**3,
            available_host_bytes=48 * 1024**3,
            accelerator_kind=kind,
        )
    return {
        "config": config,
        "context": context,
        "observation": observation,
        "backend": "torch",
        "device_identity": ("cpu", "CPU", "none", 0),
    }


class _FakeProfileStore:
    """In-memory ``InferenceTuningProfileStore`` stand-in for intent tests.

    Records every ``save``/``claim`` call so a test can assert a ``lookup``
    request never writes to, or single-flight-claims, the store -- a lookup
    is read-only by construction, and this is how that gets proven rather
    than assumed.
    """

    def __init__(self, profile: InferenceTuningProfile | None = None) -> None:
        self._profile = profile
        self.saves: list[InferenceTuningProfile] = []
        self.claims: list[TuningProfileKey] = []

    def load(self, key: TuningProfileKey) -> InferenceTuningProfile | None:
        return self._profile

    def save(self, profile: InferenceTuningProfile) -> None:
        self.saves.append(profile)
        self._profile = profile

    @contextmanager
    def claim(self, key: TuningProfileKey, *, timeout_seconds: float = 2.0):
        self.claims.append(key)

        class _Claim:
            acquired = True

        yield _Claim()


class _StubExecutor:
    """A trivial always-succeeds trial executor for intent tests that only
    care about which branch ``resolve`` took, not about search dynamics."""

    def run(self, settings, *, phase, field_name, block_index, should_cancel):
        from hydra_suite.core.inference.autotune.search import TrialObservation

        return TrialObservation(
            settings,
            100.0,
            0.5,
            equivalence_outputs(equivalence_frame()),
            warmup_calls=3,
            warmup_frames=8,
            measured_frames=10,
        )


_UNSET = object()


def make_incomplete_profile(
    key: TuningProfileKey | None = None,
) -> InferenceTuningProfile:
    """An INCOMPLETE (negative-cache) profile, freshly timestamped so it
    still falls inside the 24h retry window."""

    key = key or _key()
    baseline = _settings()
    now = time.time_ns()
    return InferenceTuningProfile(
        profile_id=key.digest[:24],
        key=key,
        baseline=baseline,
        requested=baseline,
        admitted=baseline,
        selected=baseline,
        candidates=(),
        state=ProfileState.INCOMPLETE,
        selection_reason="budget_expired",
        created_at_unix_ns=now,
        last_validation_unix_ns=now,
        invalidation_reason="a prior calibration attempt did not complete",
    )


def make_request(
    *, mode: str = "lookup", eligible: bool = True, **overrides: Any
) -> AutotuneRequest:
    """A representative ``AutotuneRequest`` for the given ``mode``."""

    fields = {
        "key": _key(),
        "baseline": _settings(),
        "planner": _planner(),
        "mode": mode,
        "eligible": eligible,
    }
    fields.update(overrides)
    return AutotuneRequest(**fields)


def make_coordinator(
    *,
    trial_executor: Any = _UNSET,
    cached_state: ProfileState | None = None,
    cached_profile: InferenceTuningProfile | None = None,
) -> tuple[AutotuneCoordinator, _FakeProfileStore]:
    """An ``AutotuneCoordinator`` wired to a ``_FakeProfileStore``.

    ``cached_profile`` wins if given; otherwise ``cached_state`` builds a
    representative VALIDATED or INCOMPLETE profile. ``trial_executor``
    defaults to a working stub -- pass ``None`` explicitly to exercise the
    "no executor configured" path.
    """

    profile = cached_profile
    if profile is None and cached_state is not None:
        if cached_state is ProfileState.VALIDATED:
            profile = _profile()
        elif cached_state is ProfileState.INCOMPLETE:
            profile = make_incomplete_profile()
        else:
            raise ValueError(f"unsupported cached_state: {cached_state!r}")
    store = _FakeProfileStore(profile)
    executor = _StubExecutor() if trial_executor is _UNSET else trial_executor
    coordinator = AutotuneCoordinator(store, trial_executor=executor)
    return coordinator, store
