"""S3: the profile key must fingerprint the baseline and the manual fields.

Without this, a profile tuned from one starting point (or one search space)
is reused for a request that never made the same comparison -- see
``.superpowers/sdd/2026-09-07-inference-autotuner-review-remediation``.
"""

from __future__ import annotations

import sys

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


# ---- a probe that raises must not take the whole autotuner down -------------


def test_a_raising_cudnn_probe_degrades_to_absent(monkeypatch):
    """``torch.backends.cudnn.version()`` raises on a cuDNN version mismatch.

    Measured on a real CUDA box: PyTorch compiled against cuDNN (9, 19, 0)
    with runtime (9, 1x) makes that call raise ``RuntimeError`` -- while CUDA
    tracking itself works fine (that host produced a full byte-identical
    equivalence matrix). The probe caught only ``ImportError``, so the
    exception propagated out of ``default_software_fingerprint`` and killed
    the entire autotune preflight: the tuner could never run on such a host,
    degrading to a "fallback" overlay on every run with an opaque reason.

    A fingerprint FIELD is not worth a feature. An unreadable probe reads
    "absent" -- which is itself a fingerprint value, so two hosts that differ
    here still get different profiles only when something else differs. That
    is the same trade ``_torch_cuda_version`` already documents.
    """

    import types

    from hydra_suite.core.inference.autotune import fingerprint as fingerprint_mod

    class _Raises:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def version():
            raise RuntimeError(
                "cuDNN version incompatibility: PyTorch was compiled against "
                "(9, 19, 0) but found runtime version (9, 12, 0)"
            )

    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(cudnn=_Raises()),
        version=types.SimpleNamespace(cuda="12.8"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert fingerprint_mod._torch_cudnn_version() == "absent"
    # And the fingerprint it feeds must still be constructible.
    software = fingerprint_mod.default_software_fingerprint(
        backend="torch",
        precision="fp16",
        driver="unknown",
        cuda="12.8",
        cudnn="unknown",
    )
    assert software.cudnn


def test_a_raising_cuda_version_probe_degrades_to_absent(monkeypatch):
    import types

    from hydra_suite.core.inference.autotune import fingerprint as fingerprint_mod

    class _Boom:
        def __getattr__(self, _name):
            raise RuntimeError("torch.version is unavailable")

    fake_torch = types.SimpleNamespace(version=_Boom())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert fingerprint_mod._torch_cuda_version() == "absent"
