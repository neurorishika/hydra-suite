# Task 9 Report: CUDA Stream-Sync Chokepoint on RuntimeContext

## Event-Association Approach

**Chosen approach:** module-level `WeakKeyDictionary` (`_HANDOFF_EVENTS`) keyed by tensor.

**Why:**
- `RuntimeContext` is a frozen dataclass, so `self` cannot carry mutable per-tensor state.
- Setting `tensor._handoff_event = ...` is tempting but fails for tensors that don't accept dynamic attributes (many internal PyTorch tensors do not).
- `WeakKeyDictionary` keys on the tensor object identity (by reference) — when the tensor is garbage-collected (e.g. consumer is done), the entry vanishes automatically, preventing memory leaks.
- No thread-safety issue: `handoff` is called by the producer before the tensor is shared, and `await_handoff` is called by the consumer after receiving it — there is no concurrent access to the same key.
- The dict is only written to when `cuda_mode and tensor_on_cuda` — CPU/MPS paths never touch it.

**CUDA API usage (untested locally, CUDA box required):**
```python
event = torch.cuda.Event()
event.record(torch.cuda.current_stream())  # in handoff — producer side
torch.cuda.current_stream().wait_event(event)  # in await_handoff — consumer side
```
`stream.wait_event(event)` inserts a GPU-side barrier so the consumer stream will stall until the producer's `event.record()` point is reached — standard cross-stream synchronization pattern.

## TDD Evidence

### Step 1 & 2 — test written, fails
```
$ PYTHONPATH=src ... pytest tests/test_inference_stream_sync.py -v
ImportError: cannot import name '_HANDOFF_EVENTS' from 'hydra_suite.core.inference.runtime'
```

### Step 3 — implementation added

Modified `src/hydra_suite/core/inference/runtime.py`:
- Added `import weakref` and `TYPE_CHECKING` guard for torch type hints
- Added module-level `_HANDOFF_EVENTS: WeakKeyDictionary`
- Added `RuntimeContext.handoff(tensor)` method
- Added `RuntimeContext.await_handoff(tensor)` method

### Step 4 — all new tests pass
```
$ PYTHONPATH=src ... pytest tests/test_inference_stream_sync.py -v
tests/test_inference_stream_sync.py::test_handoff_is_identity_on_cpu PASSED
tests/test_inference_stream_sync.py::test_handoff_does_not_attach_state_on_cpu PASSED
tests/test_inference_stream_sync.py::test_await_handoff_without_prior_handoff_is_safe PASSED
tests/test_inference_stream_sync.py::test_handoff_returns_tensor_unchanged_on_cpu PASSED
4 passed
```

### Pre-existing failure count unchanged
```
$ PYTHONPATH=src ... pytest tests/test_inference_runtime.py tests/test_inference_stream_sync.py
1 failed (test_cpu_config_produces_cpu_mode — pre-existing MPS device detection issue), 9 passed
```

## Files Changed

- `src/hydra_suite/core/inference/runtime.py` — added `_HANDOFF_EVENTS`, `handoff`, `await_handoff`
- `tests/test_inference_stream_sync.py` — created (4 tests)

## Concerns About Untested CUDA Path

1. **`WeakKeyDictionary` on CUDA tensors:** PyTorch CUDA tensors support `hash()` (they use object id), so `WeakKeyDictionary` keying should work. However, if a tensor is `.detach()`-ed or wrapped in any way that produces a new Python object, the consumer would look up the wrong key and silently miss the event (safe no-op, but not synchronized). Callers must pass the *same Python object* through both `handoff` and `await_handoff`.

2. **Multi-stream / multi-device:** The implementation records on `current_stream()` in `handoff` and waits on `current_stream()` in `await_handoff`. If the producer and consumer are on different CUDA devices, `current_stream()` refers to different devices' default streams — `wait_event` would need to be called on the correct consuming device's stream. Since the codebase targets `cuda:0` only (single device), this is acceptable for now but should be noted for multi-GPU expansion.

3. **Thread safety of WeakKeyDictionary:** CPython's GIL protects single-item dict operations, so concurrent writes from separate producer threads to different keys are safe. If two threads `handoff` the *same* tensor simultaneously (unlikely in this pipeline's single-producer model), the second write would overwrite the first event. This is benign for the expected usage pattern.

4. **`await_handoff` without `handoff`:** Confirmed safe no-op — `_HANDOFF_EVENTS.get(tensor)` returns `None` and no CUDA API is called.
