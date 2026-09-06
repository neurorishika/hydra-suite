import threading
import time

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


def test_depths_1_2_4_byte_identical(tmp_path):
    """depth 1, 2 and 4 must all yield byte-identical caches.

    Extends ``test_depth1_equals_depth2`` to a depth>2 (deep-prefetch) case.
    depth>2 only deepens the producer's prefetch runway (bounded queue of
    ``depth-1`` windows); a single in-order consumer still pulls windows in
    strict ascending frame order and writes caches in-order, so the per-type
    cache bytes (detection + headtail + cnn + pose) must be identical to both
    depth=1 (synchronous) and depth=2 (double buffer).
    """
    h = {d: run_pipeline_to_caches(tmp_path / f"d{d}", depth=d) for d in (1, 2, 4)}
    assert h[1] == h[2] == h[4]


def test_depth4_uses_deep_prefetch_queue(tmp_path):
    """Prove depth=4 actually ran the deep-prefetch path (not silent depth=1).

    Two independent checks so a future regression cannot hide:

    1. A depth=4 ``Pipeline`` derives ``queue_bound == depth - 1 == 3`` (and
       ``depth`` is not clamped to 2).
    2. End-to-end, the tiny-clip depth=4 run constructs its hand-off
       ``queue.Queue`` with ``maxsize == 3`` — captured by patching the Queue
       used inside the pipeline module. depth=1 would never build a Queue, so a
       captured maxsize of 3 proves depth=4 took effect through the real runner.
    """
    from unittest.mock import patch

    from hydra_suite.core.inference import pipeline as pipeline_mod
    from hydra_suite.core.inference.pipeline import Pipeline, PipelineStages

    # (1) Construction-level: depth takes effect, queue bound scales with depth.
    stages = PipelineStages(
        config=type("C", (), {"detection_batch_size": 2})(),
        obb_models=None,
        headtail_model=None,
        cnn_models=[],
        pose_model=None,
        apriltag_model=None,
    )
    pipe = Pipeline(stages, runtime=None, cache_writer=None, depth=4)
    assert pipe.depth == 4
    assert pipe.queue_bound == 3

    # (2) End-to-end: capture the maxsize the real pipeline uses at depth=4.
    captured_maxsizes: list[int] = []
    real_queue = pipeline_mod.queue.Queue

    def _spy_queue(*args, **kwargs):
        q = real_queue(*args, **kwargs)
        captured_maxsizes.append(q.maxsize)
        return q

    with patch.object(pipeline_mod.queue, "Queue", side_effect=_spy_queue):
        run_pipeline_to_caches(tmp_path / "spy", depth=4)

    assert 3 in captured_maxsizes, (
        "depth=4 did not build a hand-off queue with maxsize=3 "
        f"(captured: {captured_maxsizes}) — depth may have silently degraded"
    )


def test_depth2_stage_exception_propagates_and_cleans_up():
    """A stage failure under depth=2 must re-raise (not hang) and close the writer.

    The supervisor sets the stop flag, drains/joins the producer, then flushes +
    closes the cache writer before re-raising. We assert the exception surfaces
    promptly and the (async) CacheWriter is closed so its worker thread exits.
    """
    from hydra_suite.core.inference.cache.writer import CacheWriter
    from hydra_suite.core.inference.pipeline import Pipeline, PipelineStages
    from hydra_suite.core.inference.runtime import RuntimeContext

    class _Boom(RuntimeError):
        pass

    # CPU runtime: handoff/await_handoff are no-ops; no real GPU work.
    runtime = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )
    writer = CacheWriter({}, [], async_mode=True)

    pipe = Pipeline.__new__(Pipeline)
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

    pipe._run_detection_for_window = ok_obb  # type: ignore[assignment]
    pipe._process_obb_results = boom  # type: ignore[assignment]

    frames = [(i, object()) for i in range(6)]
    with pytest.raises(_Boom, match="stage exploded"):
        pipe.run(iter(frames), range(0, 6))

    # Writer was flushed + closed by the supervisor: a second close is a no-op
    # and writing raises (closed), proving the worker thread was stopped.
    with pytest.raises(RuntimeError, match="closed"):
        writer.write_detection(0, object())  # type: ignore[arg-type]


def test_depth2_producer_exception_propagates_without_hang():
    """An OBB (producer) failure must surface to the caller and not deadlock."""
    from hydra_suite.core.inference.cache.writer import CacheWriter
    from hydra_suite.core.inference.pipeline import Pipeline, PipelineStages
    from hydra_suite.core.inference.runtime import RuntimeContext

    runtime = RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        tensor_on_cuda=False,
    )
    writer = CacheWriter({}, [], async_mode=True)

    pipe = Pipeline.__new__(Pipeline)
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

    pipe._run_detection_for_window = boom_obb  # type: ignore[assignment]

    frames = [(i, object()) for i in range(6)]
    with pytest.raises(ValueError, match="decode/OBB failed"):
        pipe.run(iter(frames), range(0, 6))


def test_depth2_consumer_error_unblocks_full_producer_queue_without_thread_leak():
    """A consumer failure must cancel both data and sentinel queue puts."""
    from hydra_suite.core.inference.pipeline import Pipeline

    pipe = Pipeline.for_test(window_size=1, depth=2, stage=lambda window: [])
    producer_has_filled_queue = threading.Event()

    def fast_obb(window):
        producer_has_filled_queue.set()
        return [object()]

    def fail_consumer(window, raw_list):
        # Let the producer fill its one-slot queue and block on its next put.
        assert producer_has_filled_queue.wait(1.0)
        time.sleep(0.05)
        raise RuntimeError("consumer failed with producer queue full")

    pipe._run_detection_for_window = fast_obb  # type: ignore[assignment]
    pipe._process_obb_results = fail_consumer  # type: ignore[assignment]

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="consumer failed"):
        pipe.run([(i, object()) for i in range(100)], range(100), range_total=100)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert not any(
        thread.name == "pipeline-obb-producer" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_depth2_decodes_next_window_while_detector_is_blocked():
    """Decode has its own producer and can overlap a slow SAHI detector."""
    from hydra_suite.core.inference.pipeline import Pipeline

    pipe = Pipeline.for_test(window_size=1, depth=2, stage=lambda window: [])
    second_frame_requested = threading.Event()
    release_detector = threading.Event()

    class _Source:
        def __iter__(self):
            yield 0, object()
            second_frame_requested.set()
            yield 1, object()

        def close(self):
            pass

    calls = 0

    def slow_detection(window):
        nonlocal calls
        calls += 1
        if calls == 1:
            # The decoder must request frame 1 while OBB is still working on 0.
            assert second_frame_requested.wait(1.0)
            assert release_detector.wait(2.0)
        return [object()]

    pipe._run_detection_for_window = slow_detection  # type: ignore[assignment]
    pipe._process_obb_results = lambda window, raw: []  # type: ignore[assignment]

    runner = threading.Thread(
        target=lambda: pipe.run(_Source(), range(2), range_total=2), daemon=True
    )
    runner.start()
    assert second_frame_requested.wait(1.0)
    release_detector.set()
    runner.join(timeout=2.0)
    assert not runner.is_alive()


def test_depth2_decoder_queue_is_bounded_and_source_is_closed_on_consumer_error():
    """A failed consumer cannot leave a blocked decoder or reader behind."""
    from hydra_suite.core.inference import pipeline as pipeline_mod
    from hydra_suite.core.inference.pipeline import Pipeline

    real_queue = pipeline_mod.queue.Queue
    maxsizes: list[int] = []

    def spy_queue(*args, **kwargs):
        q = real_queue(*args, **kwargs)
        maxsizes.append(q.maxsize)
        return q

    closed = threading.Event()

    class _Source:
        def __iter__(self):
            for idx in range(100):
                yield idx, object()

        def close(self):
            closed.set()

    pipe = Pipeline.for_test(window_size=1, depth=2, stage=lambda window: [])
    pipe._run_detection_for_window = lambda window: [object()]  # type: ignore[assignment]
    pipe._process_obb_results = (  # type: ignore[assignment]
        lambda window, raw: (_ for _ in ()).throw(RuntimeError("consumer boom"))
    )

    from unittest.mock import patch

    with patch.object(pipeline_mod.queue, "Queue", side_effect=spy_queue):
        with pytest.raises(RuntimeError, match="consumer boom"):
            pipe.run(_Source(), range(100), range_total=100)

    # First queue is the one-window decode staging queue; the second remains
    # the configured detector-result runway for strict in-order consumption.
    assert maxsizes[:2] == [1, 1]
    assert closed.is_set()
    assert not any(
        thread.name in {"pipeline-decode-producer", "pipeline-obb-producer"}
        and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_depth2_decode_exception_propagates_and_closes_source():
    """Reader exceptions are reported from the decoder, not hidden by a sentinel."""
    from hydra_suite.core.inference.pipeline import Pipeline

    closed = threading.Event()

    class _Source:
        def __iter__(self):
            yield 0, object()
            raise OSError("decode failed")

        def close(self):
            closed.set()

    pipe = Pipeline.for_test(window_size=1, depth=2, stage=lambda window: [])
    pipe._run_detection_for_window = lambda window: [object()]  # type: ignore[assignment]
    pipe._process_obb_results = lambda window, raw: []  # type: ignore[assignment]

    with pytest.raises(OSError, match="decode failed"):
        pipe.run(_Source(), range(2), range_total=2)
    assert closed.is_set()


def test_run_does_not_return_while_detection_call_still_owns_model():
    """Python cannot kill inference safely; model ownership waits for its exit."""
    from hydra_suite.core.inference.pipeline import Pipeline

    pipe = Pipeline.for_test(window_size=1, depth=2, stage=lambda window: [])
    detection_entered = threading.Event()
    release_detection = threading.Event()
    outcome: list[BaseException] = []
    detection_calls = 0

    def blocked_detection(window):
        nonlocal detection_calls
        detection_calls += 1
        if detection_calls == 1:
            return [object()]
        detection_entered.set()
        assert release_detection.wait(5.0)
        return [object()]

    pipe._run_detection_for_window = blocked_detection  # type: ignore[assignment]

    def fail_after_second_detection(window, raw):
        # The consumer owns only the first raw window while the detector starts
        # its second call. This explicitly exercises the teardown guarantee
        # without relying on the old decode+OBB single-producer scheduling.
        assert detection_entered.wait(1.0)
        raise RuntimeError("consumer failed")

    pipe._process_obb_results = fail_after_second_detection  # type: ignore[assignment]

    def invoke():
        try:
            pipe.run([(0, object()), (1, object())], range(2), range_total=2)
        except Exception as exc:  # noqa: BLE001 - asserting supervisor outcome
            outcome.append(exc)

    caller = threading.Thread(target=invoke, name="pipeline-test-caller")
    caller.start()
    assert detection_entered.wait(1.0)
    time.sleep(2.5)
    assert caller.is_alive(), "run returned while its producer still owned the model"
    assert any(
        thread.name == "pipeline-obb-producer" and thread.is_alive()
        for thread in threading.enumerate()
    )

    release_detection.set()
    caller.join(timeout=2.0)
    assert not caller.is_alive()
    assert len(outcome) == 1 and "consumer failed" in str(outcome[0])
    assert not any(
        thread.name == "pipeline-obb-producer" and thread.is_alive()
        for thread in threading.enumerate()
    )
