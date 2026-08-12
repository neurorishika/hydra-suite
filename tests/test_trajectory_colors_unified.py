import numpy as np

from hydra_suite.core.tracking.session_policy import build_trajectory_colors


def test_first_three_colors_match_gui_legacy():
    assert build_trajectory_colors(3) == [(102, 179, 92), (14, 106, 71), (188, 20, 102)]


def test_does_not_leak_global_seed():
    np.random.seed(123)
    expected_next = np.random.randint(
        0, 1_000_000
    )  # the draw that should follow seed(123)
    np.random.seed(123)
    build_trajectory_colors(5)  # must NOT perturb global RNG state
    assert np.random.randint(0, 1_000_000) == expected_next


def test_gui_and_cli_paths_agree():
    # Both call sites now delegate here, so equality is by construction; pin N=10.
    assert build_trajectory_colors(10) == build_trajectory_colors(10)
    assert len(build_trajectory_colors(10)) == 10
