from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import CUDA_RUNTIMES, ComputeRuntime, InferenceConfig

if TYPE_CHECKING:
    import torch as _torch

# Module-level WeakKeyDictionary mapping tensors to their recorded CUDA events.
# Using weakref so that tensors are not kept alive by the event registry.
# On CPU/MPS paths, this dict is never written to.
_HANDOFF_EVENTS: "weakref.WeakKeyDictionary[_torch.Tensor, object]" = (
    weakref.WeakKeyDictionary()
)


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
        and stores it in the module-level ``_HANDOFF_EVENTS`` WeakKeyDictionary
        keyed by *tensor*.  The consumer must call :meth:`await_handoff` before
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
            _HANDOFF_EVENTS[tensor] = event
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

            event = _HANDOFF_EVENTS.get(tensor)
            if event is not None:
                torch.cuda.current_stream().wait_event(event)  # type: ignore[arg-type]
        return tensor

    @staticmethod
    def from_config(config: InferenceConfig) -> "RuntimeContext":
        runtimes = config._collect_all_runtimes()
        cuda_mode = bool(runtimes & CUDA_RUNTIMES)
        # tensor_on_cuda: only native PyTorch "cuda" leaves model outputs as
        # live CUDA device tensors. ONNX Runtime CUDAExecutionProvider and
        # TensorRT both produce CPU numpy from the inference call.
        tensor_on_cuda = "cuda" in runtimes and not (
            runtimes & {"onnx_cuda", "tensorrt"}
        )
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
    try:
        import torchvision

        return torchvision.get_video_backend() == "cuda"
    except Exception:
        return False
