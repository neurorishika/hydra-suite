import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from types import SimpleNamespace

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


@pytest.fixture
def window():
    """A DetectKitMainWindow on a live QApplication, cleaned up after.

    NO pytest-qt (not installed; no test in this repo uses it -- see the
    fixture note above). These are the FIRST tests in the suite to construct
    a DetectKitMainWindow, so run this file on its own the first time and
    confirm it exits cleanly rather than aborting. If construction crashes
    the interpreter, fall back to testing the handlers as unbound functions
    against a SimpleNamespace stub and say so in the commit message.
    """
    from hydra_suite.detectkit.gui import main_window as mw

    QApplication.instance() or QApplication([])
    win = mw.DetectKitMainWindow()
    yield win
    win.deleteLater()


def test_accepting_a_frame_refreshes_both_layers(window, monkeypatch, tmp_path):
    """Accept must redraw GT (the change landed) and staged (it is decided).

    Directly, not incidentally: a selection-preserving refresh would
    otherwise leave the accepted proposal on screen with nothing asking for
    a redraw.
    """
    from hydra_suite.data.al.merge import MergeMode
    from hydra_suite.detectkit.gui import main_window as mw

    refreshed: list = []
    monkeypatch.setattr(
        window, "_refresh_overlays", lambda keys=None: refreshed.append(keys)
    )
    monkeypatch.setattr(window, "_current_staged_rel", lambda: "a.txt")
    # SimpleNamespace, not object(): _on_review_accept ends by calling
    # _offer_finish_if_complete -> is_complete, which reads staged_review.
    monkeypatch.setattr(
        window, "_current_source_obj", lambda: SimpleNamespace(staged_review=None)
    )
    monkeypatch.setattr(window, "_save_current_project", lambda: None)
    monkeypatch.setattr(window, "_sync_review_bar", lambda: None)
    monkeypatch.setattr(mw, "accept_frame", lambda *a, **k: None)

    window._on_review_accept(MergeMode.ADD_NEW)

    assert refreshed and set(refreshed[-1]) == {"gt", "staged"}


def test_next_undecided_selects_the_first_frame_without_a_decision(window, monkeypatch):
    from hydra_suite.detectkit.gui import main_window as mw

    monkeypatch.setattr(mw, "staged_frames", lambda root: ["a.txt", "b.txt", "c.txt"])
    monkeypatch.setattr(mw, "read_decisions", lambda root: {"a.txt": "rejected"})
    selected: list = []
    monkeypatch.setattr(
        window._dataset_panel,
        "select_image_by_relative_label",
        lambda rel: selected.append(rel),
    )
    monkeypatch.setattr(window, "_current_staged_root", lambda: "/tmp/staging")

    window._on_review_next_undecided()

    assert selected == ["b.txt"]


def test_empty_review_state_shows_only_discard(bar):
    """A zero-frame review (staging dir deleted/moved/missing) cannot be
    decided, finished, or reverted -- offer only Discard, disable every
    decision control so nothing else can be clicked into a dead end.
    """
    bar.set_review_state(
        "sam3", "prompt 'ant'", decided=0, total=0, can_rethreshold=True
    )
    bar.set_empty_review_state("sam3", "prompt 'ant'")

    assert not bar.isHidden()
    assert bar.discard_button().isVisible()
    for button_name in (
        "accept_overwrite_button",
        "accept_add_new_button",
        "reject_button",
        "next_undecided_button",
        "accept_all_button",
        "reject_all_button",
        "revert_button",
    ):
        assert not getattr(bar, button_name)().isEnabled()
    assert not bar.rethreshold_button().isVisible()


def test_discard_button_emits_its_signal(bar):
    bar.set_empty_review_state("sam2", "v")
    seen = _fired(bar.discard_requested)

    bar.discard_button().click()

    assert len(seen) == 1


def test_set_review_state_after_empty_state_re_enables_and_hides_discard(bar):
    """A source-switch back to a normal review must not carry over the
    empty-state's disabled controls or visible Discard button."""
    bar.set_empty_review_state("sam2", "v")

    bar.set_review_state("sam2", "v", decided=1, total=5, can_rethreshold=False)

    assert not bar.discard_button().isVisible()
    assert bar.accept_overwrite_button().isEnabled()


def test_clearing_after_empty_state_hides_discard_too(bar):
    bar.set_empty_review_state("sam2", "v")

    bar.clear_review_state()

    assert bar.isHidden()
    assert not bar.discard_button().isVisible()


def test_sync_review_bar_shows_empty_state_for_a_zero_frame_review(
    window, monkeypatch, tmp_path
):
    """A source whose review has no staged frames (missing staging dir,
    e.g. after the project was moved to another machine -- staged_path is
    stored absolute) must render the escape-hatch state, not a normal
    0/0 progress bar.
    """
    from types import SimpleNamespace

    review = SimpleNamespace(
        producer="sam3",
        prompt="ant",
        producer_variant="sam3",
        staged_path=str(tmp_path / "gone"),
    )
    monkeypatch.setattr(
        window,
        "_current_source_obj",
        lambda: SimpleNamespace(staged_review=review),
    )
    shown: list = []
    monkeypatch.setattr(
        window._review_bar,
        "set_empty_review_state",
        lambda producer, detail: shown.append((producer, detail)),
    )
    monkeypatch.setattr(
        window._review_bar,
        "set_review_state",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    window._sync_review_bar()

    assert shown == [("sam3", "prompt 'ant'")]


def test_discard_clears_a_review_whose_staging_dir_does_not_exist(
    window, monkeypatch, tmp_path
):
    """The escape hatch for BLOCKER 2: a review whose staged_path is gone
    must be discardable through the handler, ending with
    `source.staged_review is None` and no exception.
    """
    from hydra_suite.detectkit.gui.models import OBBSource, StagedReview

    src_root = tmp_path / "sources" / "orig"
    (src_root / "images").mkdir(parents=True)
    (src_root / "labels").mkdir(parents=True)
    source = OBBSource(path=str(src_root), name="orig", level="obb")
    source.staged_review = StagedReview(
        staged_path=str(tmp_path / "artifacts" / "pending_escalations" / "gone"),
        target_level="polygon",
        producer="sam3",
    )
    monkeypatch.setattr(window, "_current_source_obj", lambda: source)
    monkeypatch.setattr(window, "_save_current_project", lambda: None)
    monkeypatch.setattr(window, "_sync_review_bar", lambda: None)
    monkeypatch.setattr(window._dataset_panel, "refresh_sources", lambda project: None)

    window._on_review_discard()

    assert source.staged_review is None
