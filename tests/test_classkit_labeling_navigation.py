from types import SimpleNamespace

from hydra_suite.classkit.gui.main_window import MainWindow


def _stub(candidates):
    calls = []
    return SimpleNamespace(
        explorer_mode="labeling",
        candidate_indices=list(candidates),
        _labeling_navigation_scope="pool",
        selected_point_index=None,
        hover_locked=False,
        status=SimpleNamespace(showMessage=lambda *_: None),
        request_preview_for_index=lambda *a, **k: calls.append(a),
        request_update_explorer_selection=lambda *_: None,
    )


def test_click_keeps_pool_scope_when_batch_active():
    w = _stub([3, 5, 9])
    MainWindow.on_explorer_point_clicked(w, 7)  # a point outside the batch
    assert w._labeling_navigation_scope == "pool"
    assert w.selected_point_index == 7


def test_click_uses_database_scope_without_batch():
    w = _stub([])
    MainWindow.on_explorer_point_clicked(w, 7)
    assert w._labeling_navigation_scope == "database"


def test_labeling_proposal_ignores_verified_and_unknown():
    labels = [None, "left"]
    w = SimpleNamespace(
        selected_point_index=0,
        image_paths=["a", "b"],
        image_labels=labels,
        _review_status_for_index=lambda i: {},
        _review_prediction_for_index=lambda i: {"label": "right"},
    )
    assert MainWindow._labeling_prediction_for_selected(w) == "right"
    w.selected_point_index = 1  # already labeled: nothing to approve
    assert MainWindow._labeling_prediction_for_selected(w) is None
    w.selected_point_index = 0
    w._review_prediction_for_index = lambda i: {"label": "unknown"}
    assert MainWindow._labeling_prediction_for_selected(w) is None
    w._review_status_for_index = lambda i: {"label": "left", "verified": False}
    assert MainWindow._labeling_prediction_for_selected(w) == "left"
