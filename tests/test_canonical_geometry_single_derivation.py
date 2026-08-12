"""F7a: canonical_geometry_from_params is the sole params->geometry derivation.

core/inference/config.py used to re-inline CanonicalGeometry.from_reference(...)
in build_inference_config_from_params and hardcode the same magic defaults
(20.0 / 2.0 / 1.3) in _default_canonical_geometry's fallback. Both now route
through canonical_geometry_from_params, so this test pins the equivalence.
"""

from hydra_suite.core.canonicalization.geometry import canonical_geometry_from_params
from hydra_suite.core.inference.config import (
    _default_canonical_geometry,
    build_inference_config_from_params,
)


def test_config_derivation_matches_helper():
    params = {
        "REFERENCE_BODY_SIZE": 33.0,
        "RESIZE_FACTOR": 1.5,
        "ADVANCED_CONFIG": {"reference_aspect_ratio": 2.4, "canonical_margin": 1.5},
    }
    g_helper = canonical_geometry_from_params(params)
    g_config = build_inference_config_from_params(params).canonical
    assert g_helper.to_dict() == g_config.to_dict()


def test_default_fallback_matches_empty_params_helper():
    g_default = _default_canonical_geometry()
    g_helper = canonical_geometry_from_params({})
    assert g_default.to_dict() == g_helper.to_dict()
    # Pin the documented magic defaults: from_reference(20.0*1.0, 2.0, 1.3).
    assert g_default.to_dict() == {
        "canvas_wh": list(g_default.canvas_wh),
        "margin": 1.3,
        "aspect_ratio": 2.0,
        "schema_version": 1,
    }
