"""TDD tests for Task 2: ``debug_mode`` config field -> ``DEBUG_MODE`` engine
param in ``build_engine_params`` (and its derived ``ENABLE_PROFILING`` /
``EXPORT_CONFIDENCE_DENSITY_VIDEO`` flags).

Rule (verbatim from the plan): an absent ``debug_mode`` key means
Debug/legacy -- ``DEBUG_MODE = bool(config.get("debug_mode", True))``. When
the ``debug_mode`` key IS present, ``ENABLE_PROFILING`` and
``EXPORT_CONFIDENCE_DENSITY_VIDEO`` equal ``DEBUG_MODE``; when it is ABSENT,
they keep their stored values (backward compat).
"""

from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params


def _rt():
    # Minimal CPU-shaped runtime context sufficient for param building --
    # same probe shape used by tests/test_get_parameters_dict_characterization.py.
    return RuntimeContext(
        fps=100.0,
        total_frames=500,
        frame_width=640,
        frame_height=480,
    )


def test_debug_mode_absent_defaults_to_debug_and_keeps_stored_flags():
    cfg = {"enable_profiling": False, "export_confidence_density_video": False}
    params = build_engine_params(cfg, runtime=_rt())
    assert params["DEBUG_MODE"] is True
    assert params["ENABLE_PROFILING"] is False
    assert params["EXPORT_CONFIDENCE_DENSITY_VIDEO"] is False


def test_debug_mode_true_derives_flags_on():
    params = build_engine_params({"debug_mode": True}, runtime=_rt())
    assert params["DEBUG_MODE"] is True
    assert params["ENABLE_PROFILING"] is True
    assert params["EXPORT_CONFIDENCE_DENSITY_VIDEO"] is True


def test_debug_mode_false_derives_flags_off():
    params = build_engine_params(
        {"debug_mode": False, "enable_profiling": True}, runtime=_rt()
    )
    assert params["DEBUG_MODE"] is False
    assert params["ENABLE_PROFILING"] is False
    assert params["EXPORT_CONFIDENCE_DENSITY_VIDEO"] is False
