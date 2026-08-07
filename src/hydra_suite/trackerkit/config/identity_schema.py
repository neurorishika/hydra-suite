"""Typed identity configuration for TrackerKit.

Single source of truth for identity state. ``build_engine_params`` derives the
flat ``IDENTITY_*`` engine params from this object (Phase 1); later phases
convert consumers to read it directly. Fields marked "reserved" are persisted
for round-trip stability but are not emitted into engine params until the phase
that wires them (calibration → Phase 2, robustness → Phase 3, smoothing /
changepoint / independent post-hoc toggle → Phases 5/6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class SlotLockConfig:
    swap_min_frames: int = 8
    swap_conf_margin: float = 0.2
    rejoin_velocity_budget: float = 1.5
    rejoin_dist_floor: float | None = None


@dataclass
class RealtimeIdentityConfig:
    enabled: bool = True  # master identity-in-tracking gate
    bayesian_cost_enabled: bool = False  # online decoder (ANDs with `enabled`)
    association_weight: float = 1.0
    rejoin_threshold: float = 0.5
    commit_threshold: float = 0.85
    display_threshold: float = 0.6
    transition_epsilon: float = 0.02
    unknown_prior: float = 0.05
    swap_enabled: bool = True
    slot_lock: SlotLockConfig = field(default_factory=SlotLockConfig)


@dataclass
class PostHocIdentityConfig:
    postprocess_mode: str = "Fragment Solver"
    fragment_solver_enabled: bool = False
    disagree_min_run: int = 5
    gates_trajectory_structure: bool = True
    # Reserved (Phases 5/6):
    enabled: bool = False  # independent of realtime; wired in Phase 6
    smoothing_enabled: bool = False
    changepoint_enabled: bool = False
    fragment_min_frames: int = 0
    ambiguity_margin: float = 0.0


@dataclass
class RobustnessConfig:
    # Reserved (Phase 3): no engine key emitted in Phase 1.
    per_frame_evidence_cap: float = 0.0
    prob_floor: float = 0.0
    source_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class IdentityModelConfig:
    kind: str = "cnn"  # "cnn" | "apriltag" | "color_tag"
    name: str = ""
    path: str | None = None
    unique_identifier: bool = False
    factors: tuple[str, ...] = ()
    # Reserved (Phase 2): fitted temperature + signature.
    calibration: dict[str, Any] | None = None


@dataclass
class IdentityConfig:
    enabled: bool = True  # master identity classification on
    models: list[IdentityModelConfig] = field(default_factory=list)
    calibration_required: bool = False  # reserved (Phase 2 gate)
    realtime: RealtimeIdentityConfig = field(default_factory=RealtimeIdentityConfig)
    posthoc: PostHocIdentityConfig = field(default_factory=PostHocIdentityConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    @classmethod
    def from_engine_config(
        cls,
        cfg: Any,
        advanced: Any,
        *,
        cfg_get: Callable[..., Any],
    ) -> "IdentityConfig":
        """Build from the persisted snake_case config, reproducing the exact
        derivations in ``engine_params.py:604-635, 1132-1178``.

        ``cfg_get(cfg, key, default)`` is injected so this stays independent of
        ``engine_params`` internals; the caller passes the module's ``_cfg_get``.
        """
        enable_postprocessing = bool(cfg_get(cfg, "enable_postprocessing", True))
        enable_in_tracking = bool(cfg_get(cfg, "enable_identity_in_tracking", True))
        online = enable_in_tracking and bool(
            cfg_get(cfg, "enable_identity_online_decoder", False)
        )

        saved_mode = cfg_get(cfg, "identity_postprocess_mode", None)
        if saved_mode is None:
            saved_mode = "Fragment Solver"
        saved_mode = str(saved_mode)
        mode = saved_mode if enable_postprocessing else "None"
        fragment_solver = enable_postprocessing and saved_mode == "Fragment Solver"

        realtime = RealtimeIdentityConfig(
            enabled=enable_in_tracking,
            bayesian_cost_enabled=online,
            association_weight=float(cfg_get(cfg, "identity_weight", 1.0)),
            rejoin_threshold=float(cfg_get(cfg, "identity_rejoin_threshold", 0.5)),
            commit_threshold=float(cfg_get(cfg, "identity_commit_threshold", 0.85)),
            display_threshold=float(cfg_get(cfg, "identity_display_threshold", 0.6)),
            transition_epsilon=float(cfg_get(cfg, "identity_transition_epsilon", 0.02)),
            unknown_prior=float(cfg_get(cfg, "identity_unknown_prior", 0.05)),
            swap_enabled=bool(cfg_get(cfg, "enable_identity_swap_correction", True)),
            slot_lock=SlotLockConfig(
                swap_min_frames=int(cfg_get(cfg, "identity_swap_min_frames", 8)),
                swap_conf_margin=float(advanced.get("identity_swap_conf_margin", 0.2)),
                rejoin_velocity_budget=float(
                    advanced.get("identity_rejoin_velocity_budget", 1.5)
                ),
                rejoin_dist_floor=advanced.get("identity_rejoin_dist_floor", None),
            ),
        )
        posthoc = PostHocIdentityConfig(
            postprocess_mode=mode,
            fragment_solver_enabled=fragment_solver,
            disagree_min_run=int(cfg_get(cfg, "identity_disagree_min_run", 5)),
            gates_trajectory_structure=bool(
                cfg_get(cfg, "identity_gates_trajectory_structure", True)
            ),
        )
        return cls(realtime=realtime, posthoc=posthoc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityConfig":
        d = dict(data)
        rt = dict(d.get("realtime", {}))
        sl = dict(rt.pop("slot_lock", {}) or {})
        realtime = RealtimeIdentityConfig(slot_lock=SlotLockConfig(**sl), **rt)
        posthoc = PostHocIdentityConfig(**dict(d.get("posthoc", {})))
        robustness = RobustnessConfig(**dict(d.get("robustness", {})))
        models = [IdentityModelConfig(**dict(m)) for m in d.get("models", [])]
        return cls(
            enabled=bool(d.get("enabled", True)),
            models=models,
            calibration_required=bool(d.get("calibration_required", False)),
            realtime=realtime,
            posthoc=posthoc,
            robustness=robustness,
        )
