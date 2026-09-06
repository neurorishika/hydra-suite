"""Identity guard for the scoring primitives promoted out of `semantic/`.

`core/inference/shape_prior.py` and `core/inference/match_geometry.py` are the
one definition of these names; `core/inference/semantic/shape_prior.py` and
`core/inference/semantic/calibration.py` re-export them for backward
compatibility. This file follows the same pattern as
`tests/test_sam3_detection_quality.py::test_the_parity_tool_reuses_the_core_implementation`
for `detection_metrics.py`: asserting `is`, not just "importable" or
"equal", is what makes a future copy-paste (forking the logic instead of
sharing it) fail the suite instead of silently drifting.
"""

from __future__ import annotations


def test_shape_prior_reexports_are_the_same_objects():
    from hydra_suite.core.inference import shape_prior as core_shape_prior
    from hydra_suite.core.inference.semantic import shape_prior as semantic_shape_prior

    assert semantic_shape_prior.AreaBand is core_shape_prior.AreaBand
    assert semantic_shape_prior.fit_area_band is core_shape_prior.fit_area_band
    assert semantic_shape_prior.in_band is core_shape_prior.in_band
    assert semantic_shape_prior.match_quality is core_shape_prior.match_quality
    assert semantic_shape_prior.MIN_MATCH_QUALITY is core_shape_prior.MIN_MATCH_QUALITY
    assert semantic_shape_prior.aspect_ratio is core_shape_prior.aspect_ratio
    assert semantic_shape_prior.polygon_area is core_shape_prior.polygon_area


def test_calibration_matcher_reexports_are_the_same_objects():
    from hydra_suite.core.inference import match_geometry
    from hydra_suite.core.inference.semantic import calibration

    assert calibration.representative_point is match_geometry.representative_point
    assert calibration.match_one_to_one is match_geometry.match_one_to_one
    assert calibration._contains is match_geometry._contains
    assert calibration._contains is match_geometry.contains
    assert calibration._is_finite is match_geometry._is_finite
    assert (
        calibration._pole_of_inaccessibility is match_geometry._pole_of_inaccessibility
    )
    assert calibration._vertex_mean is match_geometry._vertex_mean


def test_match_geometry_uses_the_shared_shape_prior_not_a_copy():
    from hydra_suite.core.inference import match_geometry, shape_prior

    assert match_geometry.AreaBand is shape_prior.AreaBand
    assert match_geometry.in_band is shape_prior.in_band
    assert match_geometry.match_quality is shape_prior.match_quality
    assert match_geometry.MIN_MATCH_QUALITY is shape_prior.MIN_MATCH_QUALITY
