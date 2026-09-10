#!/usr/bin/env python3
"""Verify CUDA runtime libraries and accelerated runtime packages in CUDA envs."""

from __future__ import annotations

import glob
import importlib.metadata as importlib_metadata
import os
import sys


def _prepend_cuda_library_dirs(prefix: str) -> None:
    search_dirs = [
        os.path.join(prefix, "targets", "x86_64-linux", "lib"),
        os.path.join(prefix, "lib"),
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    merged = [directory for directory in search_dirs if os.path.isdir(directory)]
    if existing:
        merged.append(existing)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def _verify_onnxruntime_import() -> int:
    try:
        import onnxruntime as ort
    except Exception as exc:
        print(
            f"ERROR: Failed to import onnxruntime after install: {exc}",
            file=sys.stderr,
        )
        return 1

    if not hasattr(ort, "get_available_providers"):
        module_path = getattr(ort, "__file__", None) or getattr(ort, "__path__", None)
        print(
            "ERROR: Imported onnxruntime does not expose get_available_providers(). "
            f"Loaded module path: {module_path}",
            file=sys.stderr,
        )
        print(
            "This usually means the ONNX Runtime wheel is incomplete or another install left a broken namespace package behind.",
            file=sys.stderr,
        )
        return 1

    try:
        providers = list(ort.get_available_providers() or [])
    except Exception as exc:
        print(
            f"ERROR: onnxruntime imported but provider discovery failed: {exc}",
            file=sys.stderr,
        )
        return 1

    if "CUDAExecutionProvider" not in providers:
        print(
            "ERROR: ONNX Runtime GPU installed, but CUDAExecutionProvider is missing.",
            file=sys.stderr,
        )
        print(f"Providers reported: {providers}", file=sys.stderr)
        return 1

    print(f"ONNX Runtime providers: {providers}")
    return 0


def _installed_distribution_names() -> set[str]:
    names: set[str] = set()
    try:
        for dist in importlib_metadata.distributions():
            name = str(dist.metadata.get("Name") or "").strip().lower()
            if name:
                names.add(name.replace("_", "-"))
    except Exception:
        return set()
    return names


# CUDA component wheels that ship under two naming schemes: the CUDA 12 family
# is suffixed (`nvidia-cublas-cu12`) while the CUDA 13 family dropped the
# suffix (`nvidia-cublas`). Both install into site-packages/nvidia/, so having
# one of each leaves mismatched headers and shared objects in a single tree --
# CuPy then compiles against the wrong CUDA headers and background subtraction
# silently falls back to CPU.
_NVIDIA_COMPONENTS = frozenset(
    {
        "cublas",
        "cuda-cupti",
        "cuda-nvrtc",
        "cuda-runtime",
        "cudnn",
        "cufft",
        "cufile",
        "curand",
        "cusolver",
        "cusparse",
        "cusparselt",
        "nccl",
        "nvjitlink",
        "nvshmem",
        "nvtx",
    }
)


def _nvidia_wheel_families(dist_names: set[str]) -> dict[str, set[str]]:
    """Map "12"/"13" -> the installed nvidia-* component wheels in that family."""
    families: dict[str, set[str]] = {"12": set(), "13": set()}
    for name in dist_names:
        if not name.startswith("nvidia-"):
            continue
        stem = name[len("nvidia-") :]
        if stem.endswith("-cu12") and stem[: -len("-cu12")] in _NVIDIA_COMPONENTS:
            families["12"].add(name)
        elif stem.endswith("-cu13") and stem[: -len("-cu13")] in _NVIDIA_COMPONENTS:
            families["13"].add(name)
        elif stem in _NVIDIA_COMPONENTS:
            # Unsuffixed component wheels are the CUDA 13 naming scheme.
            families["13"].add(name)
    return families


def _verify_nvidia_wheel_families() -> int:
    families = _nvidia_wheel_families(_installed_distribution_names())
    if not (families["12"] and families["13"]):
        return 0

    print(
        "ERROR: Mixed CUDA 12 and CUDA 13 component wheels are installed in the "
        "active environment.",
        file=sys.stderr,
    )
    for major in ("12", "13"):
        print(
            f"  CUDA {major}: {', '.join(sorted(families[major]))}",
            file=sys.stderr,
        )
    print(
        "Both families unpack into site-packages/nvidia/, so headers and shared "
        "objects from the wrong CUDA version end up on the search path.",
        file=sys.stderr,
    )
    print(
        "Do NOT uninstall individual nvidia-* packages to clean this up: they "
        "share files, and a partial uninstall removes cuDNN sublibraries that "
        "the surviving packages still need (every convolution then fails with "
        "CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED). Reinstall the whole stack "
        "instead:",
        file=sys.stderr,
    )
    print(
        "  pip list | awk '/^nvidia-/{print $1}' | xargs -r pip uninstall -y\n"
        '  rm -rf "$CONDA_PREFIX"/lib/python*/site-packages/nvidia\n'
        "  make install-cuda CUDA_MAJOR=<12|13>",
        file=sys.stderr,
    )
    return 1


def _driver_max_cuda() -> "tuple[int, int] | None":
    """Highest CUDA version the installed NVIDIA driver supports, if known."""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    first = (out.stdout or "").strip().splitlines()
    if not first:
        return None
    try:
        major = int(first[0].strip().split(".")[0])
    except (ValueError, IndexError):
        return None
    # Driver -> maximum supported CUDA runtime, per NVIDIA's compatibility table.
    if major >= 580:
        return (13, 0)
    if major >= 525:
        return (12, 8)
    if major >= 450:
        return (11, 8)
    return (0, 0)


def _verify_torch_cuda() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"ERROR: Failed to import torch after install: {exc}", file=sys.stderr)
        return 1

    build = torch.version.cuda or "none"
    if not torch.cuda.is_available():
        # device_count() can be non-zero here: it counts what the driver
        # enumerates, not what this torch build can actually drive. Reporting it
        # makes the "GPUs are visible but unusable" case obvious.
        print(
            f"ERROR: torch {torch.__version__} imported, but "
            "torch.cuda.is_available() is False "
            f"(torch.cuda.device_count() reports {torch.cuda.device_count()}).",
            file=sys.stderr,
        )
        print(f"This torch is a CUDA {build} build.", file=sys.stderr)
        driver_max = _driver_max_cuda()
        if driver_max is not None and driver_max != (0, 0):
            print(
                f"The installed driver supports CUDA up to {driver_max[0]}.{driver_max[1]}. "
                f"Reinstall with `make install-cuda CUDA_MAJOR={driver_max[0]}`.",
                file=sys.stderr,
            )
        else:
            print(
                "Check `nvidia-smi` and reinstall with the CUDA_MAJOR matching the "
                "driver's reported CUDA version.",
                file=sys.stderr,
            )
        return 1

    # is_available() only proves the driver handshake. Run a real convolution:
    # it is the cheapest call that loads the cuDNN sublibraries, which a
    # partially-uninstalled nvidia-* stack leaves broken.
    try:
        import torch.nn.functional as F

        x = torch.randn(1, 3, 32, 32, device="cuda")
        w = torch.randn(8, 3, 3, 3, device="cuda")
        torch.cuda.synchronize()
        F.conv2d(x, w)
        torch.cuda.synchronize()
    except Exception as exc:
        print(
            f"ERROR: torch CUDA is available, but a test convolution failed: {exc}",
            file=sys.stderr,
        )
        print(
            "This usually means the nvidia-* CUDA component wheels are damaged or "
            "mixed across CUDA versions. Reinstall the whole stack (see the "
            "mixed-family instructions above) rather than uninstalling individual "
            "nvidia-* packages.",
            file=sys.stderr,
        )
        return 1

    driver_max = _driver_max_cuda()
    driver_note = (
        f", driver supports up to CUDA {driver_max[0]}.{driver_max[1]}"
        if driver_max is not None and driver_max != (0, 0)
        else ""
    )
    print(
        f"torch {torch.__version__} (CUDA {build}) is available: "
        f"{torch.cuda.device_count()} device(s), test convolution OK{driver_note}."
    )
    return 0


def _verify_cupy() -> int:
    """CuPy powers the GPU background-subtraction path."""
    try:
        import cupy as cp
    except ModuleNotFoundError:
        print("CuPy is not installed; skipping GPU background-subtraction check.")
        return 0
    except Exception as exc:
        print(f"ERROR: Failed to import cupy after install: {exc}", file=sys.stderr)
        return 1

    # CuPy compiles kernels at runtime against whichever CUDA headers it finds
    # in site-packages/nvidia/. A stale directory from the other CUDA family
    # makes this raise while everything else in the environment still works --
    # and the tracker degrades to CPU background subtraction with only an INFO
    # log line to show for it.
    try:
        a = cp.random.rand(64, 64, dtype=cp.float32)
        float((a @ a).sum())
    except Exception as exc:
        print(
            f"ERROR: CuPy imported, but a test kernel failed to compile or run: {exc}",
            file=sys.stderr,
        )
        print(
            "GPU background subtraction would silently fall back to CPU. This is "
            "usually leftover headers from the other CUDA family under "
            "site-packages/nvidia/ -- reinstall the CUDA stack (see above).",
            file=sys.stderr,
        )
        return 1

    print(f"CuPy {cp.__version__} test kernel OK.")
    return 0


def _verify_tensorrt_import() -> int:
    dist_names = _installed_distribution_names()
    cuda_variants = {
        name for name in dist_names if name in {"tensorrt-cu12", "tensorrt-cu13"}
    }

    if len(cuda_variants) > 1:
        variants = ", ".join(sorted(cuda_variants))
        print(
            "ERROR: Mixed TensorRT CUDA wheel families detected in the active environment: "
            f"{variants}",
            file=sys.stderr,
        )
        print(
            "Uninstall all TensorRT wheels, then rerun `make install-cuda CUDA_MAJOR=<12|13>` \
to reinstall a single matching family.",
            file=sys.stderr,
        )
        return 1

    try:
        import tensorrt as trt
    except Exception as exc:
        print(
            f"ERROR: Failed to import tensorrt after install: {exc}",
            file=sys.stderr,
        )
        return 1

    logger = trt.Logger(trt.Logger.ERROR)
    try:
        builder = trt.Builder(logger)
    except Exception as exc:
        print(
            "ERROR: TensorRT imported, but builder initialization failed: " f"{exc}",
            file=sys.stderr,
        )
        if cuda_variants:
            print(
                f"Installed TensorRT CUDA family: {next(iter(cuda_variants))}",
                file=sys.stderr,
            )
        print(
            "This usually means the TensorRT wheel family does not match the active CUDA install, \
or stale TensorRT packages are still present in the environment.",
            file=sys.stderr,
        )
        return 1

    if builder is None:
        print(
            "ERROR: TensorRT builder initialization returned None.",
            file=sys.stderr,
        )
        return 1

    variant_label = next(iter(cuda_variants), "unknown")
    print(
        f"TensorRT builder initialized successfully: {trt.__version__} ({variant_label})"
    )
    return 0


def main() -> int:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        print("ERROR: CONDA_PREFIX is not set.", file=sys.stderr)
        return 1

    _prepend_cuda_library_dirs(prefix)

    search_dirs = [
        os.path.join(prefix, "lib"),
        os.path.join(prefix, "targets", "x86_64-linux", "lib"),
    ]
    required_libs = [
        "libcublasLt.so.12",
        "libcudart.so.12",
        "libcurand.so.10",
        "libcufft.so.11",
        "libcudnn.so.9",
    ]

    missing = []
    for lib_name in required_libs:
        found = False
        for directory in search_dirs:
            matches = glob.glob(os.path.join(directory, lib_name))
            matches += glob.glob(os.path.join(directory, f"{lib_name}.*"))
            if matches:
                found = True
                break
        if not found:
            missing.append(lib_name)

    if missing:
        print(
            "ERROR: Missing CUDA runtime libraries required by ONNX Runtime:",
            file=sys.stderr,
        )
        for lib_name in missing:
            print(f"  - {lib_name}", file=sys.stderr)
        print(
            "Run `mamba env update -f environment-cuda.yml --prune` or install the "
            "missing package(s), then reactivate the environment.",
            file=sys.stderr,
        )
        return 1

    # Order matters: the mixed-family check explains the root cause behind most
    # torch/CuPy failures below, so report it first.
    families_status = _verify_nvidia_wheel_families()
    if families_status != 0:
        return families_status

    torch_status = _verify_torch_cuda()
    if torch_status != 0:
        return torch_status

    import_status = _verify_onnxruntime_import()
    if import_status != 0:
        return import_status

    tensorrt_status = _verify_tensorrt_import()
    if tensorrt_status != 0:
        return tensorrt_status

    cupy_status = _verify_cupy()
    if cupy_status != 0:
        return cupy_status

    print(
        "CUDA runtime self-check passed for torch, ONNX Runtime GPU, TensorRT and CuPy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
