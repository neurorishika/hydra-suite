from tests.helpers.tiny_clip import run_pipeline_to_caches


def test_depth1_is_deterministic_across_runs(tmp_path):
    a = run_pipeline_to_caches(tmp_path / "a", depth=1)
    b = run_pipeline_to_caches(tmp_path / "b", depth=1)
    assert a == b
