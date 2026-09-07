"""S3: the profile key must fingerprint the baseline and the manual fields.

Without this, a profile tuned from one starting point (or one search space)
is reused for a request that never made the same comparison -- see
``.superpowers/sdd/2026-09-07-inference-autotuner-review-remediation``.
"""

from __future__ import annotations

from hydra_suite.core.inference.autotune.fingerprint import compute_baseline_digest

from .autotune_helpers import _key, _settings


def test_a_different_baseline_is_a_different_key():
    """A profile tuned from batch 1 must not be reused by a batch-8 project."""

    a = _key(baseline_digest=compute_baseline_digest(_settings(det=1)))
    b = _key(baseline_digest=compute_baseline_digest(_settings(det=8)))
    assert a.digest != b.digest


def test_same_baseline_is_the_same_key():
    """Two projects with an identical baseline must land on the same key."""

    a = _key(baseline_digest=compute_baseline_digest(_settings(det=1)))
    b = _key(baseline_digest=compute_baseline_digest(_settings(det=1)))
    assert a.digest == b.digest


def test_pinning_a_manual_field_is_a_different_key():
    """A project pinning pose_batch_size searched a different space."""

    baseline = _settings()
    unpinned = _key(baseline_digest=compute_baseline_digest(baseline, ()))
    pinned = _key(
        baseline_digest=compute_baseline_digest(baseline, ("pose_batch_size",))
    )
    assert unpinned.digest != pinned.digest


def test_manual_field_set_order_does_not_matter():
    baseline = _settings()
    a = compute_baseline_digest(baseline, ("pose_batch_size", "detection_batch_size"))
    b = compute_baseline_digest(baseline, ("detection_batch_size", "pose_batch_size"))
    assert a == b
