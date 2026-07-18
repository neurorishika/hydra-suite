"""Regression tests for the legacy detector-batching vestige removal.

The 'Legacy Detector Batching' group box claimed to affect only cache build /
preview / benchmark. It was wrong in three directions, and its 'Frame batch'
spin was in fact the TensorRT engine batch control on fixed runtimes. These
tests pin the corrected wiring.
"""


def test_trt_batch_follows_detection_batch_on_fixed_runtime():
    """On TensorRT/CoreML the engine's max batch must equal the detection
    batch it will actually be fed -- previously it came from the legacy
    'Frame batch' spin inside a box that claimed not to affect tracking."""
    from hydra_suite.trackerkit.gui.orchestrators.config import (
        resolve_tensorrt_max_batch_size,
    )

    assert (
        resolve_tensorrt_max_batch_size(detection_batch_size=8, fixed_runtime=True) == 8
    )


def test_trt_batch_is_one_on_non_fixed_runtime():
    """When the runtime is not TensorRT/CoreML, TensorRT is off and the value
    is inert. Pin it to a stable 1 rather than a stale widget value."""
    from hydra_suite.trackerkit.gui.orchestrators.config import (
        resolve_tensorrt_max_batch_size,
    )

    assert (
        resolve_tensorrt_max_batch_size(detection_batch_size=8, fixed_runtime=False)
        == 1
    )


def test_trt_batch_clamps_to_at_least_one():
    from hydra_suite.trackerkit.gui.orchestrators.config import (
        resolve_tensorrt_max_batch_size,
    )

    assert (
        resolve_tensorrt_max_batch_size(detection_batch_size=0, fixed_runtime=True) == 1
    )


def test_no_src_module_reads_the_removed_batching_keys():
    """Completion gate for the whole legacy-batching-removal effort (Task 7).

    Expected to still FAIL after this task -- the legacy widgets/keys are
    only re-sourced here, not yet deleted.
    """
    import subprocess

    result = subprocess.run(
        [
            "grep",
            "-rn",
            "spin_yolo_batch_size\\|spin_tensorrt_batch",
            "src/hydra_suite/trackerkit/gui/orchestrators/",
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""


def test_worker_module_no_longer_reads_enable_yolo_batching():
    """The key is gone from config; worker.py must not read it back.

    A stale advanced_config.get('enable_yolo_batching') would silently read
    False for every project saved after this change, flipping the prefetcher
    on for everyone -- the exact regression this plan avoids.
    """
    from pathlib import Path

    import hydra_suite.core.tracking.worker as worker_mod

    source = Path(worker_mod.__file__).read_text()
    assert "enable_yolo_batching" not in source
