from __future__ import annotations

from hydra_suite.core.tracking.worker import should_build_bgsub_detection_cache


def test_forward_full_run_builds_cache():
    assert (
        should_build_bgsub_detection_cache(preview_mode=False, backward_mode=False)
        is True
    )


def test_backward_run_builds_cache_for_reading():
    assert (
        should_build_bgsub_detection_cache(preview_mode=False, backward_mode=True)
        is True
    )


def test_preview_run_does_not_build_cache():
    assert (
        should_build_bgsub_detection_cache(preview_mode=True, backward_mode=False)
        is False
    )
