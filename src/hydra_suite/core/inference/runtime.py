from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import InferenceConfig

if TYPE_CHECKING:
    import torch as _torch

    from hydra_suite.runtime.resolver import ResolvedBackend

# Module-level map from a tensor's id() to its recorded CUDA event.
#
# IMPORTANT: the key is ``id(tensor)``, NOT the tensor itself. A torch tensor
# cannot be used as a dict key here because ``Tensor.__eq__`` returns an
# element-wise tensor (not a bool), so any dict key comparison — which
# ``WeakKeyDictionary.get``/``__setitem__`` perform — raises "Boolean value of
# Tensor with more than one value is ambiguous". The producer keeps the tensor
# alive (it sits in the producer→consumer queue) between ``handoff`` and
# ``await_handoff``, so its id is stable across the handoff; ``await_handoff``
# pops the entry so the map never accumulates and an id is never reused while an
# entry is live. On CPU/MPS paths this dict is never touched.
_HANDOFF_EVENTS: "dict[int, object]" = {}


@dataclass(frozen=True)
class RuntimeContext:
    cuda_mode: bool
    device: str  # "cuda:0", "mps", or "cpu"
    use_nvdec: bool  # cuda_mode AND NVDEC available
    # True ONLY for native PyTorch "cuda" runtime; onnx_cuda and tensorrt use
    # GPU compute but return CPU numpy arrays from inference calls.
    tensor_on_cuda: bool = False
    # True when the resolver selected CoreML (Apple gpu_fast + artifact available).
    coreml_mode: bool = False
    # True when runtime_tier is "gpu" or "gpu_fast" -- i.e. the caller asked
    # for GPU acceleration, independent of whether the host actually has one.
    # `cuda_mode`/`device` alone cannot distinguish "cpu" from "gpu" tier on a
    # non-CUDA host: both resolve device to the same MPS-or-CPU value (see
    # `_cpu_or_mps_device`), so a stage with no CUDA/ONNX/TensorRT/CoreML
    # story of its own (e.g. bg-sub, which only ever chooses between Numba
    # CPU, CuPy CUDA, and PyTorch MPS) needs this explicit signal to honor a
    # "cpu" tier request rather than opportunistically using MPS.
    #
    # WARNING: this defaults to False. Any hand-built ``RuntimeContext``
    # (tests, GUI workers, `api.py`) MUST set it explicitly -- forgetting it
    # silently forces tier-agnostic stages like bg-sub onto CPU even when the
    # rest of the context describes a live GPU. The `__post_init__` guard
    # below catches the `cuda_mode`/`coreml_mode` cases, but it CANNOT catch a
    # hand-built context that only sets ``device="mps"`` (or "cuda:0") without
    # `cuda_mode`/`coreml_mode` -- that combination passes the guard silently
    # while still being wrong if `requested_gpu` was meant to be True.
    requested_gpu: bool = False
    # The ResolvedBackend produced by RuntimeResolver.resolve() during
    # from_config(). None for hand-built contexts (tests, GUI workers) that
    # don't go through the resolver. Additive: no existing reader consumes
    # this yet.
    resolved: "ResolvedBackend | None" = None

    def __post_init__(self) -> None:
        # cuda_mode/coreml_mode are only ever selected on the "gpu"/"gpu_fast"
        # tiers, so either one implies requested_gpu. Guarding here turns a
        # silent wrong-default on a hand-built context (which would force
        # tier-agnostic stages like bg-sub onto CPU despite a live GPU) into a
        # loud error at construction.
        if (self.cuda_mode or self.coreml_mode) and not self.requested_gpu:
            raise ValueError(
                "RuntimeContext with cuda_mode/coreml_mode set must also set "
                "requested_gpu=True — GPU backends are only selected on the "
                "'gpu'/'gpu_fast' tiers. A hand-built context that omits it "
                "silently forces bg-sub onto CPU."
            )

    def handoff(self, tensor: "_torch.Tensor") -> "_torch.Tensor":
        """Producer-side stream-sync: record a CUDA event on the current stream.

        On CUDA, records a ``torch.cuda.Event`` on ``torch.cuda.current_stream()``
        and stores it in the module-level ``_HANDOFF_EVENTS`` map keyed by
        ``id(tensor)``.  The consumer must call :meth:`await_handoff` before
        reading the tensor to ensure the GPU work is complete on its stream.

        On CPU/MPS this is an identity no-op — the tensor is returned unchanged
        and no state is written to ``_HANDOFF_EVENTS``.

        Args:
            tensor: The tensor being handed off from the producer thread.

        Returns:
            The same tensor object (never a copy).
        """
        if self.cuda_mode and self.tensor_on_cuda:
            import torch

            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            _HANDOFF_EVENTS[id(tensor)] = event
        return tensor

    def await_handoff(self, tensor: "_torch.Tensor") -> "_torch.Tensor":
        """Consumer-side stream-sync: wait on the event recorded by :meth:`handoff`.

        On CUDA, looks up the event stored by :meth:`handoff` and calls
        ``current_stream.wait_event(event)`` so that the consuming stream does
        not read the tensor until all producer-side GPU work is done.  If no
        event was recorded for this tensor (e.g. it was produced without
        going through :meth:`handoff`), this is a safe no-op.

        On CPU/MPS this is always an identity no-op.

        Args:
            tensor: The tensor received by the consumer thread.

        Returns:
            The same tensor object (never a copy).
        """
        if self.cuda_mode and self.tensor_on_cuda:
            import torch

            event = _HANDOFF_EVENTS.pop(id(tensor), None)
            if event is not None:
                torch.cuda.current_stream().wait_event(event)  # type: ignore[arg-type]
        return tensor

    @staticmethod
    def from_config(config: InferenceConfig) -> "RuntimeContext":
        from hydra_suite.runtime.resolver import RuntimeResolver, detect_platform

        platform = detect_platform()
        resolver = RuntimeResolver(config.runtime_tier, platform)
        # Resolve OBB stage to determine backend type (artifact_available defaults
        # to True so CoreML is selected on Apple gpu_fast when artifacts exist).
        resolved = resolver.resolve("obb")
        # gpu_native: "torch" backend means native PyTorch (keeps tensors on CUDA).
        # "tensorrt" and "coreml" backends return CPU numpy from inference calls.
        gpu_native = resolved.backend == "torch"
        coreml_mode = resolved.backend == "coreml"
        cuda_mode = config.runtime_tier in ("gpu", "gpu_fast") and platform.has_cuda
        tensor_on_cuda = cuda_mode and gpu_native
        if cuda_mode:
            device = _cuda_device_available()
            nvdec = _nvdec_available()
        else:
            device = _cpu_or_mps_device()
            nvdec = False
        requested_gpu = config.runtime_tier in ("gpu", "gpu_fast")
        return RuntimeContext(
            cuda_mode=cuda_mode,
            device=device,
            use_nvdec=nvdec,
            tensor_on_cuda=tensor_on_cuda,
            coreml_mode=coreml_mode,
            requested_gpu=requested_gpu,
            resolved=resolved,
        )


def resolved_backend_for(runtime: RuntimeContext) -> "ResolvedBackend":
    """Return the context's ResolvedBackend, deriving one when it is absent.

    ``from_config`` always attaches ``resolved``. Hand-built contexts (tests,
    GUI workers, ``api.py``) may leave it ``None``; for those we reconstruct the
    exact ``(backend, device)`` the resolver would have produced from the
    tier-derived context flags — the inverse of the legacy tier→compute-runtime-string
    map, so predicate rewrites keyed off the returned ``ResolvedBackend`` stay
    faithful for every producible combo.
    """
    from hydra_suite.runtime.resolver import ResolvedBackend

    if runtime.resolved is not None:
        return runtime.resolved
    if runtime.cuda_mode:
        if runtime.tensor_on_cuda:
            return ResolvedBackend("torch", "cuda", False)
        return ResolvedBackend("tensorrt", "cuda", False)
    if runtime.coreml_mode:
        return ResolvedBackend("coreml", "mps", False)
    if runtime.device == "mps":
        return ResolvedBackend("torch", "mps", False)
    return ResolvedBackend("torch", "cpu", False)


def _cuda_device_available() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA runtime requested but no CUDA device is available. "
            "Check your CUDA installation or switch to a CPU-group runtime."
        )
    return "cuda:0"


def _cpu_or_mps_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _nvdec_available() -> bool:
    # Operator kill-switch: HYDRA_DISABLE_NVDEC forces the CPU decode path even
    # on a CUDA box with PyNvVideoCodec installed.  Useful when NVDEC is flaky
    # for a given GPU/codec/resolution (e.g. an RTX 6000 Ada rejects clips whose
    # per-frame macroblock count exceeds its NVDEC limit), and required for
    # apples-to-apples equivalence runs (legacy uses CPU decode on the
    # PyTorch-CUDA path, so both sides must decode identically).
    import os

    if os.environ.get("HYDRA_DISABLE_NVDEC"):
        return False

    # NvdecFrameReader uses PyNvVideoCodec + cupy, not torchvision's video
    # backend — probe the libraries the reader actually imports.  (CUDA-only;
    # validate on mehek.  make_frame_source already falls back if construction
    # fails, but the probe should test the right deps.)
    try:
        import cupy  # noqa: F401
        import PyNvVideoCodec  # noqa: F401

        return True
    except Exception:
        return False
