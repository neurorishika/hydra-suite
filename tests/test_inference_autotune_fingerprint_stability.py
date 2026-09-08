"""The profile key must not depend on transient machine state."""

from hydra_suite.core.inference.autotune import session
from tests.autotune_helpers import make_tensorrt_context

# 64 MiB is deliberately tight, not the 2 GiB a first draft of this test
# used: at 2 GiB the admission-filtered candidate space already tops out at
# the same value (128) as at 48 GiB for this fixture's cost model, so a test
# using 2 GiB would pass even with the bug still present (verified: reverting
# ``static_max_for`` to ``values_for`` in ``session.calibrate`` still passed
# at 2 GiB). At 64 MiB, admission filters the live-memory-filtered candidate
# space down to 4 (vs. 128 unfiltered / at 48 GiB) -- so this genuinely
# exercises the bug this task fixes.
LOW_VRAM = 64 * 1024**2
HIGH_VRAM = 48 * 1024**3


def test_tensorrt_digest_identical_under_memory_pressure():
    """gpu_fast folds an artifact batch size into tensorrt_profile_id. If that
    size is admission-filtered it tracks free VRAM, so the same machine running
    the same config yields a different profile key when it happens to be busy --
    and a calibrated profile becomes permanently unfindable."""
    low = session.calibration_key_digest(
        make_tensorrt_context(available_accelerator_bytes=LOW_VRAM)
    )
    high = session.calibration_key_digest(
        make_tensorrt_context(available_accelerator_bytes=HIGH_VRAM)
    )
    assert low == high


def test_non_tensorrt_digest_also_stable():
    low = session.calibration_key_digest(
        make_tensorrt_context(backend="torch", available_accelerator_bytes=LOW_VRAM)
    )
    high = session.calibration_key_digest(
        make_tensorrt_context(backend="torch", available_accelerator_bytes=HIGH_VRAM)
    )
    assert low == high
