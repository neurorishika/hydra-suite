"""Policy coordinator for cache reuse, tuning single-flight, and safe fallback."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from .candidates import CandidatePlanner
from .fingerprint import TuningProfileKey
from .measure import MeasurementProtocol
from .models import (
    InferenceRuntimeOverlay,
    InferenceTuningProfile,
    InferenceTuningSettings,
    ProfileState,
)
from .search import CoordinateSearch, TrialExecutor
from .store import InferenceTuningProfileStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AutotuneRequest:
    """All exact-key policy, admission, execution, and callback inputs."""

    key: TuningProfileKey
    baseline: InferenceTuningSettings
    planner: CandidatePlanner
    mode: str = "off"  # off, record, automatic
    manual_fields: frozenset[str] = frozenset()
    budget_seconds: float = 600.0
    singleflight_wait_seconds: float = 2.0
    eligible: bool = True
    allow_cached_reuse: bool = True
    eligibility_reason: str | None = None
    stage_shares: tuple[tuple[str, float], ...] = ()
    should_cancel: Callable[[], bool] = lambda: False
    status_callback: Callable[[str], None] = lambda _message: None

    def __post_init__(self) -> None:
        if self.mode not in {"off", "record", "automatic"}:
            raise ValueError("autotune mode must be off, record, or automatic")
        unknown = self.manual_fields - set(self.baseline.field_names())
        if unknown:
            raise ValueError(
                f"manual tuning fields are inactive or unknown: {sorted(unknown)}"
            )
        if self.budget_seconds <= 0 or self.singleflight_wait_seconds < 0:
            raise ValueError("autotune timing bounds are invalid")


@dataclass(frozen=True, slots=True)
class ResolveResult:
    """The immutable runtime decision plus any profile evidence used."""

    overlay: InferenceRuntimeOverlay
    profile: InferenceTuningProfile | None = None
    key_digest: str | None = None


class AutotuneCoordinator:
    """One public resolve operation around an otherwise ordinary runner."""

    def __init__(
        self,
        store: InferenceTuningProfileStore,
        *,
        trial_executor: TrialExecutor | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.trial_executor = trial_executor
        self.monotonic = monotonic

    def resolve(self, request: AutotuneRequest) -> ResolveResult:
        """Reuse, tune, record, or safely fall back for one run request."""
        if request.mode == "off":
            return ResolveResult(
                InferenceRuntimeOverlay.baseline(
                    request.baseline,
                    status="disabled",
                    reason="automatic inference tuning is disabled",
                )
            )
        cached = self.store.load(request.key)
        if (
            request.allow_cached_reuse
            and request.mode != "record"
            and cached is not None
            and cached.state is ProfileState.VALIDATED
        ):
            return self._reuse(request, cached, status="cache_hit")
        if (
            request.mode == "record"
            and cached is not None
            and cached.state is ProfileState.VALIDATED
        ):
            return ResolveResult(
                InferenceRuntimeOverlay.baseline(
                    request.baseline,
                    status="recorded",
                    reason="validated profile already recorded; record-only mode kept configured settings",
                ),
                cached,
            )
        if not request.eligible:
            return ResolveResult(
                InferenceRuntimeOverlay.baseline(
                    request.baseline,
                    status="deferred_due_to_contention",
                    reason=request.eligibility_reason
                    or "live resource eligibility failed",
                )
            )
        if self.trial_executor is None:
            return ResolveResult(
                InferenceRuntimeOverlay.baseline(
                    request.baseline,
                    status="unavailable",
                    reason="no contained calibration executor is configured",
                )
            )

        with self.store.claim(
            request.key, timeout_seconds=request.singleflight_wait_seconds
        ) as claim:
            if not claim.acquired:
                return ResolveResult(
                    InferenceRuntimeOverlay.baseline(
                        request.baseline,
                        status="singleflight_wait_timeout",
                        reason="another process is tuning this exact profile",
                    )
                )
            # The winning process may have promoted while this process waited.
            cached = self.store.load(request.key)
            if (
                request.allow_cached_reuse
                and request.mode != "record"
                and cached is not None
                and cached.state is ProfileState.VALIDATED
            ):
                return self._reuse(request, cached, status="cache_hit_after_wait")
            if (
                request.mode == "record"
                and cached is not None
                and cached.state is ProfileState.VALIDATED
            ):
                return ResolveResult(
                    InferenceRuntimeOverlay.baseline(
                        request.baseline,
                        status="recorded",
                        reason="validated profile already recorded; record-only mode kept configured settings",
                    ),
                    cached,
                )
            try:
                protocol = MeasurementProtocol(budget_seconds=request.budget_seconds)
                search = CoordinateSearch(
                    request.planner,
                    self.trial_executor,
                    protocol=protocol,
                    monotonic=self.monotonic,
                )
                result = search.run(
                    request.baseline,
                    manual_fields=request.manual_fields,
                    stage_shares=dict(request.stage_shares),
                    should_cancel=request.should_cancel,
                    status_callback=request.status_callback,
                )
            except Exception as exc:
                logger.exception("Inference throughput calibration failed safely")
                return ResolveResult(
                    InferenceRuntimeOverlay.baseline(
                        request.baseline,
                        status="fallback",
                        reason=f"calibration failed: {type(exc).__name__}",
                    )
                )
            if not result.completed:
                return ResolveResult(
                    InferenceRuntimeOverlay.baseline(
                        request.baseline,
                        status=(
                            "cancelled" if result.reason == "cancelled" else "fallback"
                        ),
                        reason=result.reason,
                    )
                )
            now = time.time_ns()
            profile = InferenceTuningProfile(
                profile_id=request.key.digest[:24],
                key=request.key,
                baseline=request.baseline,
                requested=request.baseline,
                admitted=request.planner.admit(result.selected).settings,
                selected=result.selected,
                candidates=result.evidence,
                state=ProfileState.VALIDATED,
                selection_reason=result.reason,
                rejected=result.rejected,
                calibration_summary=(
                    ("candidate_count", len(result.evidence)),
                    (
                        "configured_target_count",
                        request.key.workload.configured_target_count,
                    ),
                    (
                        "detections_p50_bucket",
                        request.key.workload.detections_p50_bucket,
                    ),
                    (
                        "detections_p95_bucket",
                        request.key.workload.detections_p95_bucket,
                    ),
                    ("crops_p50_bucket", request.key.workload.crops_p50_bucket),
                    ("crops_p95_bucket", request.key.workload.crops_p95_bucket),
                    (
                        "canonical_crop_geometries",
                        request.key.workload.canonical_crop_geometries,
                    ),
                    (
                        "measured_frames",
                        sum(item.measured_frames for item in result.evidence),
                    ),
                ),
                created_at_unix_ns=now,
                last_validation_unix_ns=now,
            )
            self.store.save(profile)
            if request.mode == "record":
                return ResolveResult(
                    InferenceRuntimeOverlay.baseline(
                        request.baseline,
                        status="recorded",
                        reason="validated profile recorded; record-only mode kept configured settings",
                    ),
                    profile,
                )
            return self._reuse(request, profile, status="calibrated")

    def _reuse(
        self,
        request: AutotuneRequest,
        profile: InferenceTuningProfile,
        *,
        status: str,
    ) -> ResolveResult:
        selected = profile.selected
        for field_name in request.manual_fields:
            selected = selected.with_value(
                field_name, request.baseline.value_for(field_name)
            )
        successful = tuple(
            evidence.settings
            for evidence in profile.candidates
            if evidence.failure_class is None
            and evidence.equivalence is not None
            and evidence.equivalence.passed
            # search.py: "stage screens never authorize a winner" -- a
            # stage-only pass (or the "unknown"/pre-remediation default) must
            # never down-admit a production setting either. Only evidence
            # that actually ran the full pipeline may authorize a winner.
            and evidence.phase == "full"
        )
        decision = request.planner.down_admit(selected, successful, request.baseline)
        effective = decision.settings if decision.admitted else request.baseline
        sources = []
        for field_name in request.baseline.field_names():
            if field_name in request.manual_fields:
                source = "manual"
            elif effective.value_for(field_name) == profile.selected.value_for(
                field_name
            ):
                source = "validated_profile"
            elif effective.value_for(field_name) == request.baseline.value_for(
                field_name
            ):
                source = "configured_fallback"
            else:
                source = "validated_down_admission"
            sources.append((field_name, source))
        reason = profile.selection_reason
        if effective != selected:
            reason = (
                f"live down-admission: {decision.reason or 'smaller validated setting'}"
            )
        if not decision.admitted:
            reason = (
                f"baseline fallback: {decision.reason or 'resource admission failed'}"
            )
        overlay = InferenceRuntimeOverlay(
            requested=request.baseline,
            admitted=decision.settings,
            effective=effective,
            field_sources=tuple(sources),
            status=status,
            reason=reason,
            profile_id=profile.profile_id,
        )
        return ResolveResult(overlay, profile)
