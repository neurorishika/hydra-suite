import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QKeySequence  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from hydra_suite.classkit.gui.dialogs.shortcut_editor import (  # noqa: E402
    ShortcutEditorDialog,
)
from hydra_suite.classkit.gui.shortcut_rules import (  # noqa: E402
    global_shortcut_conflicts,
    normalize_key,
    reserved_key_map,
    scheme_key_conflicts,
    scheme_key_map,
)

DEFAULTS = dict(ShortcutEditorDialog.DEFAULT_SHORTCUTS)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_approve_reject_default_to_plus_minus():
    assert DEFAULTS["Approve review label"] == "+"
    assert DEFAULTS["Reject review label"] == "-"


def test_normalize_is_case_and_form_insensitive():
    assert normalize_key("y") == normalize_key("Y") == "Y"
    assert normalize_key("ctrl+z") == normalize_key("Ctrl+Z")
    assert normalize_key("") == ""


def test_scheme_keys_cannot_reuse_global_or_builtin_keys():
    reserved = reserved_key_map(DEFAULTS)
    factors = [{"name": "f", "labels": ["a", "b", "c"], "shortcuts": ["+", "0", "x"]}]
    problems = scheme_key_conflicts(factors, reserved)
    assert len(problems) == 2
    assert any("'+'" in p for p in problems) and any("'0'" in p for p in problems)


def test_duplicate_within_factor_blocked_but_across_factors_allowed():
    same_factor = [{"name": "f", "labels": ["a", "b"], "shortcuts": ["Q", "q"]}]
    assert scheme_key_conflicts(same_factor, {})
    two_factors = [
        {"name": "t1", "labels": ["a"], "shortcuts": ["B"]},
        {"name": "t2", "labels": ["a"], "shortcuts": ["B"]},
    ]
    assert scheme_key_conflicts(two_factors, {}) == []


def test_global_bindings_cannot_collide_with_scheme_or_each_other():
    scheme_keys = scheme_key_map(
        [{"name": "tag", "labels": ["yellow"], "shortcuts": ["Y"]}]
    )
    ok = dict(DEFAULTS)
    assert global_shortcut_conflicts(ok, scheme_keys) == []
    bad = dict(DEFAULTS, **{"Approve review label": "Y", "Reject review label": "Y"})
    problems = global_shortcut_conflicts(bad, scheme_keys)
    assert any("tag/yellow" in p for p in problems)
    assert any("bound to both" in p for p in problems)


def test_shortcut_dialog_refuses_conflicting_key(app, monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")
    )
    dlg = ShortcutEditorDialog(scheme_keys={"Y": "tag/yellow"})
    dlg._key_edits["Approve review label"].setKeySequence(QKeySequence("Y"))
    dlg._validate_and_accept()
    assert shown and dlg.result() == 0  # warned, not accepted
    dlg._key_edits["Approve review label"].setKeySequence(QKeySequence("+"))
    dlg._validate_and_accept()
    assert dlg.result() == 1


def test_class_editor_refuses_label_key_that_is_a_global_shortcut(app, monkeypatch):
    from hydra_suite.classkit.gui.dialogs.class_editor import ClassEditorDialog

    shown = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")
    )
    scheme = {
        "name": "s",
        "factors": [{"name": "f", "labels": ["a", "b"], "shortcut_keys": ["+", "B"]}],
        "training_modes": [],
    }
    dlg = ClassEditorDialog(
        scheme_dict=scheme, reserved_keys=reserved_key_map(DEFAULTS)
    )
    dlg._do_accept()
    assert shown and "'+'" in shown[0] and dlg.result() == 0
