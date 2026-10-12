import numpy as np

from hydra_suite.classkit.core.train.metrics import compute_metrics


def test_class_names_follow_class_ids_when_classes_are_absent():
    names = ["a_a", "b_b", "c_c", "d_d", "e_e"]
    # Only classes 2 and 4 occur; names must not shift to a_a / b_b.
    y_true = np.array([2, 2, 4, 4])
    y_pred = np.array([2, 4, 4, 4])
    m = compute_metrics(y_pred, y_true, class_names=names)
    assert [c.class_name for c in m.per_class] == ["c_c", "e_e"]
    assert [c.class_id for c in m.per_class] == [2, 4]
    assert m.per_class[0].recall == 0.5 and m.per_class[1].recall == 1.0


def test_prediction_only_class_is_named():
    m = compute_metrics(
        np.array([0, 3]), np.array([0, 0]), class_names=["x_x", "y_y", "z_z", "w_w"]
    )
    assert [c.class_name for c in m.per_class] == ["x_x", "w_w"]
