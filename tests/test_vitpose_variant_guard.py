import pytest

from hydra_suite.posekit.core.vitpose_checkpoints import check_variant_available


def test_available_variant_passes():
    for v in ("b", "l", "h"):  # AP-10K B / L / H are in the catalog
        check_variant_available(v)


def test_missing_variant_raises_with_guidance():
    with pytest.raises(ValueError) as ei:
        check_variant_available("s")  # no ViTPose-S in the catalog
    msg = str(ei.value).lower()
    assert "s" in msg and "browse" in msg  # points user to Browse a checkpoint
