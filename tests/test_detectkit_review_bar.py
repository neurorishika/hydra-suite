import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.panels.review_bar import ReviewBar  # noqa: E402


@pytest.fixture
def bar():
    """A ReviewBar on a live QApplication, cleaned up after the test.

    NO pytest-qt: it is not installed, and no test in this repo uses it --
    every existing `qtbot` argument is a plain no-op default (see
    tests/test_semantic_calibration.py:467). This is the repo's own Qt
    pattern (tests/test_detectkit_canvas.py:93-96).
    """
    QApplication.instance() or QApplication([])
    widget = ReviewBar()
    yield widget
    widget.deleteLater()


def _fired(signal):
    """Record emissions of a Qt signal in a list."""
    seen = []
    signal.connect(lambda *args: seen.append(args))
    return seen


def test_the_bar_is_hidden_until_a_review_is_set(bar):
    assert bar.isHidden()


def test_setting_a_review_shows_the_bar_and_the_counter(bar):
    bar.set_review_state(
        "sam3", "prompt 'ant'", decided=23, total=140, can_rethreshold=True
    )

    assert not bar.isHidden()
    assert "23/140" in bar.progress_text()
    assert "sam3" in bar.summary_text()
    assert "ant" in bar.summary_text()


def test_clearing_hides_the_bar(bar):
    bar.set_review_state("sam2", "sam2.1_hiera_large", 0, 10, can_rethreshold=False)

    bar.clear_review_state()

    assert bar.isHidden()


def test_rethreshold_is_offered_only_when_the_producer_supports_it(bar):
    bar.set_review_state("sam2", "v", 0, 10, can_rethreshold=False)
    assert not bar.rethreshold_button().isEnabled()

    bar.set_review_state("sam3", "prompt 'ant'", 0, 10, can_rethreshold=True)
    assert bar.rethreshold_button().isEnabled()


@pytest.mark.parametrize(
    "button_name,signal_name",
    [
        ("accept_overwrite_button", "accept_overwrite_requested"),
        ("accept_add_new_button", "accept_add_new_requested"),
        ("reject_button", "reject_requested"),
        ("accept_all_button", "accept_all_requested"),
        ("reject_all_button", "reject_all_requested"),
        ("next_undecided_button", "next_undecided_requested"),
        ("revert_button", "revert_requested"),
    ],
)
def test_each_button_emits_its_signal(bar, button_name, signal_name):
    bar.set_review_state("sam2", "v", 0, 10, can_rethreshold=False)
    seen = _fired(getattr(bar, signal_name))

    getattr(bar, button_name)().click()

    assert len(seen) == 1


def test_a_complete_review_says_so(bar):
    bar.set_review_state("sam2", "v", 10, 10, can_rethreshold=False)

    assert "complete" in bar.progress_text().lower()
