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

    import_status = _verify_onnxruntime_import()
    if import_status != 0:
        return import_status

    tensorrt_status = _verify_tensorrt_import()
    if tensorrt_status != 0:
        return tensorrt_status

    print("CUDA runtime self-check passed for ONNX Runtime GPU and TensorRT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
