"""Typed identity configuration for TrackerKit.

Single source of truth for identity state. ``build_engine_params`` derives the
flat ``IDENTITY_*`` engine params from this object (Phase 1); later phases
convert consumers to read it directly. Fields marked "reserved" are persisted
for round-trip stability but are not emitted into engine params until the phase
that wires them (calibration → Phase 2, robustness → Phase 3, independent
post-hoc toggle → Phase 5, smoothing / changepoint → Phase 6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class SlotLockConfig:
    swap_min_frames: int = 8
    swap_conf_margin: float = 0.2
    # Defaults below match the values online.py hardcoded before 2026-09-06,
    # so emitting them is behaviour-neutral. `override_margin` is the
    # commit-revision gate audited as F4: it can only ever block a revision
    # when `override_margin > 2 * commit_threshold - 1` (both confidences are
    # entries of the same normalised simplex). At the shipped defaults
    # (0.5 vs 2*0.85-1 = 0.70) it is vacuous, and the decoder now says so.
    min_frames: int = 30
    strength: float = 0.9
    override_margin: float = 0.5
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
    # Independent post-hoc master gate (Phase 5 Task 6): whether offline
    # identity post-processing runs at all. Mirrors `enable_postprocessing`
    # and is deliberately NOT ANDed with realtime (`RealtimeIdentityConfig.
    # enabled`) -- post-hoc identity reads the evidence cache + final
    # trajectories and works regardless of the realtime setting.
    enabled: bool = True
    # Forward-backward smoothing over the cached identity evidence
    # (Phase 6 Task 7). Default True preserves the pre-Task-7 behavior of
    # the fragment solver (which always smoothed when a cache was
    # present); unchecking it makes the solver use raw per-frame evidence.
    smoothing_enabled: bool = True
    # Evidence honesty (2026-10 identity-evidence fixes). See
    # docs/user-guide/post-processing.md "Honest identity evidence".
    noncatalog_to_unknown: bool = True
    # Off by default (failed the six-clip wrong-label tolerance); 0.3 + True
    # is the validated "on" setting.
    fragment_min_absolute_posterior: float = 0.0
    fragment_absolute_seeding: bool = False
    # Evidence-trust layer (core/individual/identity/evidence_trust.py).
    # Distances in body lengths, calibrated per video. No GUI controls yet.
    evidence_trust_enabled: bool = False
    trust_isolation_bl: float = 1.0
    trust_prior_isolation_bl: float = 2.0
    trust_area_tolerance: float = 0.35
    trust_min_aspect_fraction: float = 0.65
    trust_min_prior_detections: int = 200
    trust_keypoint_conf: float = 0.5
    trust_min_keypoint_fraction: float = 0.75
    trust_view_move_bl: float = 0.5
    trust_view_turn_deg: float = 30.0
    trust_min_views: int = 1
    trust_min_support: float = 0.3
    trust_split_min_log_odds: float = 8.0
    trust_split_min_views: int = 3
    trust_split_min_confidence: float = 0.5
    # With the trust layer on, a fragment only ever takes its own best label
    # (FRAGMENT_TOP_K forced to 1); if that label is held it stays unknown
    # rather than falling back to a lookalike runner-up.
    trust_top_label_only: bool = True
    # Reserved (Phase 6):
    changepoint_enabled: bool = False
    fragment_min_frames: int = 0
    ambiguity_margin: float = 0.0


_EVIDENCE_FIELD_KEYS: dict[str, str] = {
    # PostHocIdentityConfig field -> persisted snake_case config key
    "noncatalog_to_unknown": "identity_noncatalog_to_unknown",
    "fragment_min_absolute_posterior": "fragment_min_absolute_posterior",
    "fragment_absolute_seeding": "fragment_absolute_seeding",
    "evidence_trust_enabled": "identity_evidence_trust",
    "trust_isolation_bl": "identity_trust_isolation_bl",
    "trust_prior_isolation_bl": "identity_trust_prior_isolation_bl",
    "trust_area_tolerance": "identity_trust_area_tolerance",
    "trust_min_aspect_fraction": "identity_trust_min_aspect_fraction",
    "trust_min_prior_detections": "identity_trust_min_prior_detections",
    "trust_keypoint_conf": "identity_trust_keypoint_conf",
    "trust_min_keypoint_fraction": "identity_trust_min_keypoint_fraction",
    "trust_view_move_bl": "identity_trust_view_move_bl",
    "trust_view_turn_deg": "identity_trust_view_turn_deg",
    "trust_min_views": "identity_trust_min_views",
    "trust_min_support": "identity_trust_min_support",
    "trust_split_min_log_odds": "identity_trust_split_min_log_odds",
    "trust_split_min_views": "identity_trust_split_min_views",
    "trust_split_min_confidence": "identity_trust_split_min_confidence",
    "trust_top_label_only": "identity_trust_top_label_only",
}

# Persisted config keys that have no TrackerKit widget. The GUI carries their
# values from the loaded file through build_config_dict (so GUI Save does not
# erase them and GUI runs see the same values as `trackerkit track` on the
# same file). See trackerkit.gui.orchestrators.config.CONFIG_ONLY_KEYS.
CONFIG_ONLY_KEYS: tuple[str, ...] = tuple(_EVIDENCE_FIELD_KEYS.values())

# PostHocIdentityConfig field -> engine-params key
EVIDENCE_ENGINE_KEYS: dict[str, str] = {
    "noncatalog_to_unknown": "IDENTITY_NONCATALOG_TO_UNKNOWN",
    "fragment_min_absolute_posterior": "FRAGMENT_MIN_ABSOLUTE_POSTERIOR",
    "fragment_absolute_seeding": "FRAGMENT_ABSOLUTE_SEEDING",
    "evidence_trust_enabled": "IDENTITY_EVIDENCE_TRUST",
    **{
        f: "IDENTITY_" + f.upper()
        for f in _EVIDENCE_FIELD_KEYS
        if f.startswith("trust_")
    },
}


def _evidence_fields_from_cfg(cfg: Any, cfg_get: Callable[..., Any]) -> dict:
    """The evidence-honesty / trust fields of ``PostHocIdentityConfig``,
    read from the persisted config with the dataclass defaults."""
    defaults = PostHocIdentityConfig()
    out: dict[str, Any] = {}
    for fname, key in _EVIDENCE_FIELD_KEYS.items():
        default = getattr(defaults, fname)
        raw = cfg_get(cfg, key, default)
        out[fname] = type(default)(raw) if raw is not None else default
    return out


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
    non_identifying_classes: tuple[str, ...] = ()
    """Classes/composites this model declares non-identifying.

    Forms: ``"notag"`` (that class in any of this model's axes),
    ``"front:notag"`` (that class in the named factor), ``"notag_notag"``
    (that whole composite display label). Excluded from the identity
    catalog entirely -- see the design doc.

    NOT the live path. ``IdentityConfig.from_engine_config`` never populates
    ``models``, so nothing reads this field at runtime today. The marks
    actually reach the engine as part of each ``cnn_classifiers`` entry:
    ``identity_panel.CNNClassifierRow.to_config`` -> saved config ->
    ``engine_params.build_engine_params``'s verbatim ``cnn_classifiers``
    passthrough -> ``CNN_CLASSIFIERS`` -> ``resolve.non_identifying_marks``.
    This field exists so a future ``models``-populating loader round-trips
    them; keep the two in sync if that loader is ever written.
    """
    # Reserved (Phase 2): fitted temperature + signature.
    calibration: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.non_identifying_classes = tuple(
            str(c) for c in (self.non_identifying_classes or ())
        )


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
                min_frames=int(advanced.get("identity_slot_lock_min_frames", 30)),
                strength=float(advanced.get("identity_slot_lock_strength", 0.9)),
                override_margin=float(
                    advanced.get("identity_slot_lock_override_margin", 0.5)
                ),
            ),
        )
        posthoc = PostHocIdentityConfig(
            postprocess_mode=mode,
            fragment_solver_enabled=fragment_solver,
            disagree_min_run=int(cfg_get(cfg, "identity_disagree_min_run", 5)),
            gates_trajectory_structure=bool(
                cfg_get(cfg, "identity_gates_trajectory_structure", True)
            ),
            enabled=enable_postprocessing,
            smoothing_enabled=bool(cfg_get(cfg, "enable_identity_smoothing", True)),
            changepoint_enabled=bool(cfg_get(cfg, "enable_pelt_splitting", False)),
            **_evidence_fields_from_cfg(cfg, cfg_get),
        )
        calibration_required = bool(cfg_get(cfg, "calibration_required", False))
        return cls(
            calibration_required=calibration_required,
            realtime=realtime,
            posthoc=posthoc,
        )

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
