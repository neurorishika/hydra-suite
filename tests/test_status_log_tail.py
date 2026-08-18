"""Tests for the status-bar live log tail (``hydra_suite.widgets.status_log``)."""

import logging
import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from hydra_suite.widgets.status_log import StatusLogTail  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def tail(qapp):
    """A tail attached to a throwaway window and a private logger."""
    window = QMainWindow()
    window.resize(800, 200)
    logger = logging.getLogger("test_status_log_tail")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    obj = StatusLogTail(window, logger=logger)
    yield obj, logger, window
    obj.detach()
    window.close()


def test_shows_latest_message(tail):
    obj, logger, _window = tail
    logger.info("first message")
    logger.info("second message")
    obj._refresh()
    assert obj._label.toolTip() == "second message"


def test_below_level_is_ignored(tail):
    obj, logger, _window = tail
    logger.info("visible")
    logger.debug("hidden")
    obj._refresh()
    assert obj._label.toolTip() == "visible"


def test_warning_and_error_are_prefixed(tail):
    obj, logger, _window = tail
    logger.warning("careful")
    obj._refresh()
    assert obj._label.toolTip() == "⚠ careful"
    logger.error("broken")
    obj._refresh()
    assert obj._label.toolTip() == "✖ broken"


def test_multiline_message_collapses_to_first_line(tail):
    obj, logger, _window = tail
    logger.info("headline\ndetail line\nmore detail")
    obj._refresh()
    assert obj._label.toolTip() == "headline"


def test_blank_message_does_not_replace_previous(tail):
    obj, logger, _window = tail
    logger.info("kept")
    logger.info("   ")
    obj._refresh()
    assert obj._label.toolTip() == "kept"


def test_long_message_is_elided_but_tooltip_is_full(tail):
    obj, logger, window = tail
    window.show()
    QApplication.processEvents()
    message = "x" * 5000
    logger.info(message)
    obj._refresh()
    assert obj._label.toolTip() == message
    assert len(obj._label.text()) < len(message)


def test_logging_from_a_worker_thread_is_captured(tail):
    obj, logger, _window = tail

    def work():
        logger.info("from worker")

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    obj._refresh()
    assert obj._label.toolTip() == "from worker"


def test_detach_removes_the_handler(tail):
    obj, logger, _window = tail
    logger.info("before detach")
    obj.detach()
    assert obj._handler not in logger.handlers
    logger.info("after detach")
    obj._refresh()
    assert obj._label.toolTip() == "before detach"
    obj.detach()  # idempotent


def test_detach_stops_the_timer(tail):
    obj, _logger, _window = tail
    assert obj._timer.isActive()
    obj.detach()
    assert not obj._timer.isActive()
