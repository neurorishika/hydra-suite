from tests.helpers.tiny_clip import _CNN_LABEL, run_pipeline_to_caches


def test_depth1_is_deterministic_across_runs(tmp_path):
    a = run_pipeline_to_caches(tmp_path / "a", depth=1)
    b = run_pipeline_to_caches(tmp_path / "b", depth=1)

    # Confirm all expected cache types were written so a future regression that
    # silently stops writing them fails here rather than silently passing.
    expected_keys = {
        "detection.npz",
        "headtail.npz",
        f"cnn_{_CNN_LABEL}.npz",
        "pose.npz",
    }
    assert expected_keys.issubset(
        a.keys()
    ), f"Missing cache files: {expected_keys - a.keys()}"

    assert a == b
