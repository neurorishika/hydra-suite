import logging
from unittest.mock import MagicMock

from hydra_suite.core.inference.stages._resource_close import close_backend_resource


def test_close_backend_resource_none_is_noop():
    close_backend_resource(None)  # must not raise


def test_close_backend_resource_prefers_close_over_release():
    resource = MagicMock(spec=["release", "close"])
    close_backend_resource(resource)
    resource.close.assert_called_once()
    resource.release.assert_not_called()


def test_close_backend_resource_falls_back_to_release():
    resource = MagicMock(spec=["release"])
    close_backend_resource(resource)
    resource.release.assert_called_once()


def test_close_backend_resource_swallows_errors():
    resource = MagicMock(spec=["close"])
    resource.close.side_effect = RuntimeError("boom")
    close_backend_resource(resource)  # must not raise


def test_close_backend_resource_logs_failure_at_warning(caplog):
    resource = MagicMock(spec=["close"])
    resource.close.side_effect = RuntimeError("boom")
    with caplog.at_level(
        logging.WARNING, logger="hydra_suite.core.inference.stages._resource_close"
    ):
        close_backend_resource(resource)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_close_backend_resource_noop_for_bare_object():
    class NoCloseNoRelease:
        pass

    close_backend_resource(NoCloseNoRelease())  # must not raise
