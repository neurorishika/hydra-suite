from __future__ import annotations

from dataclasses import dataclass

from .config import CUDA_RUNTIMES, ComputeRuntime, InferenceConfig


@dataclass(frozen=True)
class RuntimeContext:
    cuda_mode: bool
    device: str  # "cuda:0", "mps", or "cpu"
    use_nvdec: bool  # cuda_mode AND NVDEC available
    default_runtime: ComputeRuntime
    # True ONLY for native PyTorch "cuda" runtime; onnx_cuda and tensorrt use
    # GPU compute but return CPU numpy arrays from inference calls.
    tensor_on_cuda: bool = False

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
