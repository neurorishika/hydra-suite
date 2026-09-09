"""Session-level entry points for autotune: shared context, lookup, calibrate."""

from hydra_suite.core.inference.autotune import session


def test_module_exposes_the_three_entry_points():
    assert callable(session.build_autotune_context)
    assert callable(session.lookup)
    assert callable(session.calibrate)


def test_session_module_is_qt_free():
    """core/ must never import Qt. Guard it at the module source level."""
    from pathlib import Path

    src = Path(session.__file__).read_text(encoding="utf-8")
    assert "PySide6" not in src
    assert "QtCore" not in src


def test_build_context_is_pure_and_does_not_mutate_caller_params():
    """The ephemeral params dict must be a copy: worker.py:147 does dict(params)
    and then mutates. A caller's dict must never gain autotune-only keys."""
    from pathlib import Path

    src = Path(session.__file__).read_text(encoding="utf-8")
    assert "ephemeral_params = dict(params)" in src
