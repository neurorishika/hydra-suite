import numpy as np
import pytest

from hydra_suite.classkit.core.al.label_review import (
    LabelReviewModel,
    label_probabilities,
    predicted_labels,
)

LABELS = [["a", "b"], ["x", "y"]]


def _marginals():
    # image: 0 confident a_x, 1 confident b_y, 2 unsure, 3 confident a_y
    fa = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.95, 0.05]])
    fb = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9]])
    return [fa, fb]


def test_label_probabilities_product_of_marginals():
    labels = ["a_x", "a_x", None, "zzz"]
    p = label_probabilities(labels, _marginals(), LABELS)
    assert p[0] == pytest.approx(0.81)
    assert p[1] == pytest.approx(0.1 * 0.2)
    assert np.isnan(p[2]) and np.isnan(p[3])  # unlabeled / not in the scheme


def test_label_probabilities_prefer_joint_column():
    joint = {"a_x": np.array([0.3, 0.0, 0.0, 0.0])}
    p = label_probabilities(["a_x", "b_y", None, None], _marginals(), LABELS, joint)
    assert p[0] == pytest.approx(0.3)
    assert p[1] == pytest.approx(0.9 * 0.8)  # not in joint: marginals


def test_predicted_labels_marginals_and_joint():
    assert predicted_labels(_marginals(), LABELS)[:2] == ["a_x", "b_y"]
    joint = {"a_x": np.array([0.1, 0.7]), "b_y": np.array([0.9, np.nan])}
    assert predicted_labels([], [], joint) == ["b_y", "a_x"]
    nan_row = [np.array([[np.nan, np.nan]]), np.array([[0.5, 0.5]])]
    assert predicted_labels(nan_row, LABELS) == [None]


def test_mismatch_and_collisions_raise():
    with pytest.raises(ValueError):
        label_probabilities(["a_x"], _marginals(), LABELS)
    with pytest.raises(ValueError):
        label_probabilities(
            [None, None],
            [np.ones((2, 2)), np.ones((2, 2))],
            [["a_b", "a"], ["c", "b_c"]],
        )


def test_items_sorted_lowest_agreement_first_unscored_last():
    labels = ["a_x", "a_x", "a_x", "b_y", "a_x"]
    agreement = np.array([0.9, 0.05, np.nan, 0.8, 0.4])
    m = LabelReviewModel(labels, agreement=agreement)
    assert m.items("a_x") == [1, 4, 0, 2]
    assert LabelReviewModel(labels).items("a_x") == [0, 1, 2, 4]


def test_source_filter_and_disagree_filter_and_summary():
    labels = ["a_x", "a_x", "a_x", "b_y"]
    machine = [False, True, False, True]
    predicted = ["a_x", "b_y", None, "b_y"]
    m = LabelReviewModel(labels, machine, predicted=predicted)
    assert m.items("a_x", "human") == [0, 2]
    assert m.items("a_x", "machine") == [1]
    assert m.items("a_x", disagree_only=True) == [1]
    assert m.summary() == {"a_x": (3, 1), "b_y": (1, 0)}
    assert m.summary("machine") == {"a_x": (1, 1), "b_y": (1, 0)}
    assert m.label_names("human") == ["a_x"]


def test_remove_and_undo_round_trip():
    labels = ["a_x", "a_x", "b_y", None]
    m = LabelReviewModel(labels)
    r1 = m.remove([1, 1, 3])  # duplicates and unlabeled ignored
    assert r1.entries == [(1, "a_x")]
    m.remove([0, 2])
    assert m.labels == [None, None, None, None] and m.removed_count() == 3
    assert m.items("a_x") == []
    assert m.undo().entries == [(0, "a_x"), (2, "b_y")]
    assert m.labels == ["a_x", None, "b_y", None]
    m.undo()
    assert m.labels == ["a_x", "a_x", "b_y", None]
    assert m.undo() is None
    assert m.remove([3]) is None and not m.history


def test_model_length_mismatch_raises():
    with pytest.raises(ValueError):
        LabelReviewModel(["a"], machine_mask=[True, False])
