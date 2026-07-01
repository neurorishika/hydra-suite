from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import ComputeRuntime, InferenceConfig

if TYPE_CHECKING:
    import torch as _torch

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
    default_runtime: ComputeRuntime
    # True ONLY for native PyTorch "cuda" runtime; onnx_cuda and tensorrt use
    # GPU compute but return CPU numpy arrays from inference calls.
    tensor_on_cuda: bool = False

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
        # gpu_native: "torch" backend means native PyTorch (keeps tensors on CUDA).
        # "tensorrt" backend (gpu_fast tier) returns CPU numpy from inference calls.
        gpu_native = resolver.resolve("obb").backend == "torch"
        cuda_mode = config.runtime_tier in ("gpu", "gpu_fast") and platform.has_cuda
        tensor_on_cuda = cuda_mode and gpu_native
        if cuda_mode:
            device = _cuda_device_available()
            nvdec = _nvdec_available()
        else:
            device = _cpu_or_mps_device()
            nvdec = False
        default: ComputeRuntime = "cuda" if cuda_mode else "cpu"
        return RuntimeContext(
            cuda_mode=cuda_mode,
            device=device,
            use_nvdec=nvdec,
            default_runtime=default,
            tensor_on_cuda=tensor_on_cuda,
        )


def runtime_to_compute_runtime(runtime: RuntimeContext) -> "ComputeRuntime":
    """Translate a RuntimeContext (derived from runtime_tier) to a compute_runtime string.

    Used by stage loaders to get the single-string runtime expected by existing
    backend factories, without reading the deprecated per-stage compute_runtime fields.
    """
    if runtime.cuda_mode:
        if runtime.tensor_on_cuda:
            return "cuda"  # native torch GPU tier
        return "tensorrt"  # gpu_fast tier: TensorRT returns CPU numpy
    if runtime.device == "mps":
        return "mps"
    return "cpu"


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
