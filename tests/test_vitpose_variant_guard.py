import pytest

from hydra_suite.posekit.core.vitpose_checkpoints import check_variant_available


def test_available_variant_passes():
    check_variant_available("b")  # vitpose-b-coco is in the catalog


def test_missing_variant_raises_with_guidance():
    with pytest.raises(ValueError) as ei:
        check_variant_available("h")
    msg = str(ei.value).lower()
    assert "h" in msg and "browse" in msg  # points user to Browse a checkpoint
