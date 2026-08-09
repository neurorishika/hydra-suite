from hydra_suite.trackerkit.config.identity_schema import IdentityConfig


def _get(cfg, key, default=None):
    return cfg.get(key, default)


def test_from_engine_config_maps_scalar_keys():
    cfg = {
        "enable_postprocessing": True,
        "enable_identity_in_tracking": True,
        "enable_identity_online_decoder": True,
        "identity_postprocess_mode": "Fragment Solver",
        "identity_weight": 0.7,
        "identity_commit_threshold": 0.9,
        "identity_display_threshold": 0.55,
        "identity_transition_epsilon": 0.03,
        "identity_unknown_prior": 0.04,
        "identity_rejoin_threshold": 0.6,
        "enable_identity_swap_correction": False,
        "identity_swap_min_frames": 10,
        "identity_disagree_min_run": 7,
        "identity_gates_trajectory_structure": False,
    }
    advanced = {
        "identity_swap_conf_margin": 0.25,
        "identity_rejoin_velocity_budget": 2.0,
        "identity_rejoin_dist_floor": 3.0,
    }
    ic = IdentityConfig.from_engine_config(cfg, advanced, cfg_get=_get)

    assert ic.realtime.enabled is True
    assert ic.realtime.bayesian_cost_enabled is True
    assert ic.realtime.association_weight == 0.7
    assert ic.realtime.commit_threshold == 0.9
    assert ic.realtime.display_threshold == 0.55
    assert ic.realtime.transition_epsilon == 0.03
    assert ic.realtime.unknown_prior == 0.04
    assert ic.realtime.rejoin_threshold == 0.6
    assert ic.realtime.swap_enabled is False
    assert ic.realtime.slot_lock.swap_min_frames == 10
    assert ic.realtime.slot_lock.swap_conf_margin == 0.25
    assert ic.realtime.slot_lock.rejoin_velocity_budget == 2.0
    assert ic.realtime.slot_lock.rejoin_dist_floor == 3.0
    assert ic.posthoc.postprocess_mode == "Fragment Solver"
    assert ic.posthoc.fragment_solver_enabled is True
    assert ic.posthoc.disagree_min_run == 7
    assert ic.posthoc.gates_trajectory_structure is False


def test_online_decoder_gated_by_master_switch():
    # bridge rule: online decoder ANDs with enable_identity_in_tracking.
    cfg = {"enable_identity_in_tracking": False, "enable_identity_online_decoder": True}
    ic = IdentityConfig.from_engine_config(cfg, {}, cfg_get=_get)
    assert ic.realtime.bayesian_cost_enabled is False


def test_postprocess_mode_falls_back_to_fragment_solver_when_key_absent():
    cfg = {"enable_postprocessing": True}
    ic = IdentityConfig.from_engine_config(cfg, {}, cfg_get=_get)
    assert ic.posthoc.postprocess_mode == "Fragment Solver"
    assert ic.posthoc.fragment_solver_enabled is True


def test_postprocess_mode_none_when_postprocessing_off():
    cfg = {
        "enable_postprocessing": False,
        "identity_postprocess_mode": "Fragment Solver",
    }
    ic = IdentityConfig.from_engine_config(cfg, {}, cfg_get=_get)
    assert ic.posthoc.postprocess_mode == "None"
    assert ic.posthoc.fragment_solver_enabled is False


def test_defaults_match_builder_defaults():
    ic = IdentityConfig.from_engine_config({}, {}, cfg_get=_get)
    assert ic.realtime.enabled is True
    assert ic.realtime.bayesian_cost_enabled is False
    assert ic.realtime.association_weight == 1.0
    assert ic.realtime.commit_threshold == 0.85
    assert ic.realtime.display_threshold == 0.6
    assert ic.realtime.transition_epsilon == 0.02
    assert ic.realtime.unknown_prior == 0.05
    assert ic.realtime.rejoin_threshold == 0.5
    assert ic.realtime.swap_enabled is True
    assert ic.realtime.slot_lock.swap_min_frames == 8
    assert ic.realtime.slot_lock.swap_conf_margin == 0.2
    assert ic.realtime.slot_lock.rejoin_velocity_budget == 1.5
    assert ic.realtime.slot_lock.rejoin_dist_floor is None
    assert ic.posthoc.disagree_min_run == 5
    assert ic.posthoc.gates_trajectory_structure is True


def test_roundtrip():
    ic = IdentityConfig.from_engine_config({}, {}, cfg_get=_get)
    assert IdentityConfig.from_dict(ic.to_dict()) == ic


def test_posthoc_enabled_true_by_default_independent_of_realtime():
    # Task 6: the post-hoc master toggle defaults True and is NOT gated on
    # `enable_identity_in_tracking` -- offline identity post-processing must
    # keep working whether or not realtime identity influence is on.
    ic = IdentityConfig.from_engine_config({}, {}, cfg_get=_get)
    assert ic.posthoc.enabled is True

    cfg_realtime_off = {"enable_identity_in_tracking": False}
    ic_off = IdentityConfig.from_engine_config(cfg_realtime_off, {}, cfg_get=_get)
    assert ic_off.realtime.enabled is False
    assert ic_off.posthoc.enabled is True

    cfg_realtime_on = {"enable_identity_in_tracking": True}
    ic_on = IdentityConfig.from_engine_config(cfg_realtime_on, {}, cfg_get=_get)
    assert ic_on.realtime.enabled is True
    assert ic_on.posthoc.enabled is True


def test_posthoc_enabled_tracks_enable_postprocessing_only():
    # posthoc.enabled mirrors `enable_postprocessing` -- the one flag that
    # governs whether offline post-processing runs at all -- not any
    # realtime-identity flag.
    cfg = {"enable_postprocessing": False}
    ic = IdentityConfig.from_engine_config(cfg, {}, cfg_get=_get)
    assert ic.posthoc.enabled is False
