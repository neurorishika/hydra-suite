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
