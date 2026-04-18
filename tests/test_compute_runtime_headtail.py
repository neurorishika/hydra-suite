"""Regression test: head_tail pipeline exposes the same runtime set as cnn_identity."""

from __future__ import annotations


def test_headtail_has_cnn_identity_runtime_set():
    from hydra_suite.runtime.compute_runtime import supported_runtimes_for_pipeline

    cnn_set = set(supported_runtimes_for_pipeline("cnn_identity"))
    headtail_set = set(supported_runtimes_for_pipeline("head_tail"))
    # Head-tail uses the identical capability table as cnn_identity.
    assert cnn_set == headtail_set
    # And must at minimum include cpu.
    assert "cpu" in headtail_set
