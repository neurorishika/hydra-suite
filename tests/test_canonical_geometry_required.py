"""F7b: `geometry` must be a required parameter on every crop-consuming stage,
with no module-level fallback `CanonicalGeometry` left to silently diverge
from the caller's project-wide canonical geometry.
"""

import inspect

import pytest

from hydra_suite.core.identity.classification import headtail as classification_headtail
from hydra_suite.core.inference.stages import cnn, headtail, pose


@pytest.mark.parametrize("fn", [cnn.run_cnn, headtail.run_headtail, pose.run_pose])
def test_geometry_has_no_default(fn):
    assert (
        inspect.signature(fn).parameters["geometry"].default is inspect.Parameter.empty
    )


@pytest.mark.parametrize(
    "fn", [cnn.run_cnn_batch, headtail.run_headtail_batch, pose.run_pose_batch]
)
def test_batch_geometry_has_no_default(fn):
    assert (
        inspect.signature(fn).parameters["geometry"].default is inspect.Parameter.empty
    )


def test_no_module_default_geometry():
    assert not hasattr(cnn, "_DEFAULT_CANONICAL_GEOMETRY")
    assert not hasattr(headtail, "_DEFAULT_CANONICAL_GEOMETRY")
    assert not hasattr(pose, "_DEFAULT_CANONICAL_GEOMETRY")
    assert not hasattr(classification_headtail, "_DEFAULT_CANONICAL_GEOMETRY")


def test_headtail_analyzer_geometry_has_no_default():
    sig = inspect.signature(classification_headtail.HeadTailAnalyzer.__init__)
    assert sig.parameters["geometry"].default is inspect.Parameter.empty


def test_headtail_analyzer_no_longer_accepts_fallback_dial_params():
    """The self-built ``from_reference(20.0, reference_aspect_ratio,
    canonical_margin)`` fallback path is deleted entirely -- so are the
    constructor knobs that only existed to feed it.
    """
    sig = inspect.signature(classification_headtail.HeadTailAnalyzer.__init__)
    assert "reference_aspect_ratio" not in sig.parameters
    assert "canonical_margin" not in sig.parameters
