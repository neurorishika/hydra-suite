from hydra_suite.core.tracking.session import _user_mode_intermediate_paths


def test_intermediate_paths_enumerated():
    paths = _user_mode_intermediate_paths(base="/out/clip", ext=".csv")
    assert "/out/clip_final.csv" in paths
    assert "/out/clip_forward.csv" in paths
    assert "/out/clip_backward.csv" in paths
    assert "/out/clip_forward_processed.csv" in paths
    assert "/out/clip_final_with_individual.csv" in paths
    # the clean deliverable must NOT be in the delete set
    assert "/out/clip_tracks.csv" not in paths
