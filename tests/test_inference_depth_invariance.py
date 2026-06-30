import pytest

from tests.helpers.tiny_clip import _CNN_LABEL, run_pipeline_to_caches


def test_depth1_is_deterministic_across_runs(tmp_path):
    a = run_pipeline_to_caches(tmp_path / "a", depth=1)
    b = run_pipeline_to_caches(tmp_path / "b", depth=1)

    # Confirm all expected cache types were written so a future regression that
    # silently stops writing them fails here rather than silently passing.
    expected_keys = {
        "detection.npz",
        "headtail.npz",
        f"cnn_{_CNN_LABEL}.npz",
        "pose.npz",
    }
    assert expected_keys.issubset(
        a.keys()
    ), f"Missing cache files: {expected_keys - a.keys()}"

    assert a == b


def test_depth1_equals_depth2(tmp_path):
    """The whole point: depth=2 (double buffer) output is byte-identical to depth=1.

    The tiny-clip stages are deterministic, so equal cache hashes prove the
    producer/consumer concurrency (decode+OBB ahead of crops/stages/scatter)
    introduced no ordering, batching, or GPU-race differences. If this fails the
    depth=2 path is not concurrency-safe.
    """
    a = run_pipeline_to_caches(tmp_path / "d1", depth=1)
    b = run_pipeline_to_caches(tmp_path / "d2", depth=2)
    assert a == b


def test_depth2_stage_exception_propagates_and_cleans_up():
    """A stage failure under depth=2 must re-raise (not hang) and close the writer.

    The supervisor sets the stop flag, drains/joins the producer, then flushes +
    closes the cache writer before re-raising. We assert the exception surfaces
    promptly and the (async) CacheWriter is closed so its worker thread exits.
    """
    from hydra_suite.core.inference.cache.writer import CacheWriter
    from hydra_suite.core.inference.pipeline import (
        BatchWindow,
        Pipeline,
        PipelineStages,
    )
    from hydra_suite.core.inference.runtime import RuntimeContext

    class _Boom(RuntimeError):
        pass

    # CPU runtime: handoff/await_handoff are no-ops; no real GPU work.
    runtime = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )
    writer = CacheWriter({}, [], async_mode=True)

    pipe = Pipeline.__new__(Pipeline)
    pipe._requested_depth = 2
    pipe.stages = PipelineStages(
        config=type("C", (), {})(),
        obb_models=None,
        headtail_model=None,
        cnn_models=[],
        pose_model=None,
        apriltag_model=None,
    )
    pipe.runtime = runtime
    pipe.cache_writer = writer
    pipe.depth = 2
    pipe.queue_bound = 1
    pipe._window_size = 2
    pipe._test_stage = None

    # OBB succeeds (producer); the consumer-side stage raises.
    def ok_obb(window):
        return [object() for _ in window.frames]

    def boom(window, raw_list):
        raise _Boom("stage exploded")

    pipe._run_obb_for_window = ok_obb  # type: ignore[assignment]
    pipe._process_obb_results = boom  # type: ignore[assignment]

    frames = [(i, object()) for i in range(6)]
    with pytest.raises(_Boom, match="stage exploded"):
        pipe.run(iter(frames), range(0, 6))

    # Writer was flushed + closed by the supervisor: a second close is a no-op
    # and submitting raises (closed), proving the worker thread was stopped.
    with pytest.raises(RuntimeError, match="closed"):
        writer.submit(object())  # type: ignore[arg-type]


def test_depth2_producer_exception_propagates_without_hang():
    """An OBB (producer) failure must surface to the caller and not deadlock."""
    from hydra_suite.core.inference.cache.writer import CacheWriter
    from hydra_suite.core.inference.pipeline import Pipeline, PipelineStages
    from hydra_suite.core.inference.runtime import RuntimeContext

    runtime = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )
    writer = CacheWriter({}, [], async_mode=True)

    pipe = Pipeline.__new__(Pipeline)
    pipe._requested_depth = 2
    pipe.stages = PipelineStages(
        config=type("C", (), {})(),
        obb_models=None,
        headtail_model=None,
        cnn_models=[],
        pose_model=None,
        apriltag_model=None,
    )
    pipe.runtime = runtime
    pipe.cache_writer = writer
    pipe.depth = 2
    pipe.queue_bound = 1
    pipe._window_size = 2
    pipe._test_stage = None

    def boom_obb(window):
        raise ValueError("decode/OBB failed")

    pipe._run_obb_for_window = boom_obb  # type: ignore[assignment]

    frames = [(i, object()) for i in range(6)]
    with pytest.raises(ValueError, match="decode/OBB failed"):
        pipe.run(iter(frames), range(0, 6))
