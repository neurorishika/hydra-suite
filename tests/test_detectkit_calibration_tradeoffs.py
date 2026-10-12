"""Tradeoff views retain row identity and use strict Pareto dominance."""

from types import SimpleNamespace


def test_frontier_ties_and_dominance():
    from hydra_suite.detectkit.gui.widgets.calibration_tradeoffs import frontier_rows

    # Equal coordinates are separate selectable measurements. Equal cost with
    # worse quality is dominated, as is slower execution with equal quality.
    values = [(0, 1, 0.8), (1, 1, 0.8), (2, 1, 0.7), (3, 2, 0.8), (4, 3, 0.9)]
    assert frontier_rows(values) == {0, 1, 4}
    assert frontier_rows([]) == set()


def test_plot_excludes_failed_unknown_and_nonfinite_measurements():
    from hydra_suite.detectkit.gui.widgets.calibration_tradeoffs import plot_values

    def point(time, recall=0.8, failed="", frames=4):
        return SimpleNamespace(
            seconds_per_frame=time,
            tiles_per_frame=4,
            failed_reason=failed,
            score=SimpleNamespace(recall=recall, frames=frames),
        )

    points = [
        point(1),
        point(0),
        point(float("nan")),
        point(2, failed="failed"),
        point(3, frames=0),
        point(4, recall=float("inf")),
    ]
    assert plot_values(points, "seconds_per_frame", "recall") == [(0, 1, 0.8)]
    assert plot_values(points, "tiles_per_frame", "recall")[0] == (0, 4, 0.8)


def test_precision_recall_frontier_maximizes_both_axes():
    from hydra_suite.detectkit.gui.widgets.calibration_tradeoffs import frontier_rows

    assert frontier_rows([(0, -0.9, 0.8), (1, -0.8, 0.7), (2, -0.7, 0.9)]) == {0, 2}


def test_frontier_matches_pairwise_definition_on_dense_grid():
    import random

    from hydra_suite.detectkit.gui.widgets.calibration_tradeoffs import frontier_rows

    random.seed(17)
    values = [(row, random.randrange(8), random.randrange(8)) for row in range(150)]
    expected = {
        row
        for row, x, y in values
        if not any(a <= x and b >= y and (a < x or b > y) for _, a, b in values)
    }
    assert frontier_rows(values) == expected
