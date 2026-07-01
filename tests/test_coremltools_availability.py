from hydra_suite.utils import gpu_utils


def test_coremltools_flag_exists_and_is_bool():
    assert isinstance(gpu_utils.COREMLTOOLS_AVAILABLE, bool)
