# Conda Environments and Makefile Reference

This page documents the conda environment files and Makefile targets used in the [developer install path](installation.md#conda-pip-developer-install).

## Environment files

| File | Env name | Platform | What conda provides |
| ---- | -------- | -------- | ----------------- |
| `environment.yml` | `hydra` | All | Python, NumPy, SciPy, PySide6, Qt6, OpenCV, Numba |
| `environment-mps.yml` | `hydra-mps` | macOS M1-M4 | Same as CPU |
| `environment-cuda.yml` | `hydra-cuda` | Linux/Windows (NVIDIA) | Same + CUDA 12 runtime libs (cublas, cudnn, curand, cufft) |
| `environment-rocm.yml` | `hydra-rocm` | Linux (AMD) | Same as CPU (ROCm is system-level) |

The conda environments provide **system libraries only** (Qt, OpenGL, CUDA runtime). Python packages are installed separately via `make install-*`, which runs `uv pip install -r requirements-*.txt`.

## Requirements files

| File | Inherits | Adds (beyond pyproject.toml) |
| ---- | -------- | ---------------------------- |
| `requirements.txt` | `-e .` | `torch`, `torchvision` (CPU) |
| `requirements-mps.txt` | `-e .` | `torch`, `torchvision`, `torchaudio`, `onnxruntime` |
| `requirements-cuda.txt` | `-e .` | `torch`, `torchvision`, `torchaudio`, `onnxruntime-gpu` |
| `requirements-cuda12.txt` | `-r requirements-cuda.txt` | `--extra-index-url .../cu128`, `tensorrt-cu12-*`, `cupy-cuda12x` |
| `requirements-cuda13.txt` | `-r requirements-cuda.txt` | `--extra-index-url .../cu130`, `tensorrt-cu13-*`, `cupy-cuda13x` |
| `requirements-rocm.txt` | `-e .` | `torch` (ROCm), `cupy-rocm-6-0`, `onnxruntime` |
| `requirements-dev.txt` | (standalone) | pytest, black, flake8, mypy, build, twine, etc. |

All requirements files include `-e .` which installs the package in editable mode, pulling base dependencies from `pyproject.toml`. This means dependencies are declared once — in `pyproject.toml` — and requirements files only add what `pyproject.toml` cannot express (torch index URLs, GPU-specific wheels).

For runtime-specific GPU installs, prefer the Makefile targets over direct `pip install -e .[extra]`. The CUDA install targets reset conflicting `onnxruntime` and TensorRT wheel families before reinstalling the correct variant for the selected platform, which avoids mixed-provider installs and broken CUDA backend selection.

## ONNX Runtime and CUDA compatibility

`onnxruntime-gpu==1.24.1` links against CUDA 12 user-space libraries (`libcublasLt.so.12`, `libcudart.so.12`, `libcurand.so.10`). This is handled differently by each install path:

- **pip path:** PyTorch's CUDA wheel installs `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc. as pip dependencies and preloads them via `ctypes` at import time. ONNX Runtime finds them in the same process.
- **conda path:** `environment-cuda.yml` installs CUDA 12 runtime libs via conda packages, including `libcurand`. `make install-cuda` writes `LD_LIBRARY_PATH` activation hooks.

Both approaches work on CUDA 13 systems — CUDA 12 user-space libs coexist with a CUDA 13 driver.

TensorRT follows the same rule: install exactly one CUDA wheel family. `requirements-cuda12.txt` installs the CUDA 12 TensorRT wheels, and `requirements-cuda13.txt` installs the CUDA 13 family. Mixing both families in one env can leave `import tensorrt` working while builder initialization fails at runtime.

## Makefile targets

### Setup (creates conda environment)

```bash
make setup            # CPU
make setup-mps        # Apple Silicon
make setup-cuda       # NVIDIA CUDA
make setup-rocm       # AMD ROCm
```

### Install (pip packages into activated environment)

```bash
make install                      # CPU
make install-mps                  # Apple Silicon
make install-cuda CUDA_MAJOR=13  # NVIDIA CUDA 13
make install-cuda CUDA_MAJOR=12  # NVIDIA CUDA 12
make install-rocm                 # AMD ROCm
make install-dev                  # Dev tools (formatting, linting, testing, publishing)
```

### Update (refresh both conda and pip)

```bash
make env-update                      # CPU
make env-update-mps                  # Apple Silicon
make env-update-cuda CUDA_MAJOR=13  # NVIDIA CUDA
make env-update-rocm                 # AMD ROCm
```

### Remove

```bash
make env-remove         # CPU
make env-remove-mps     # Apple Silicon
make env-remove-cuda    # NVIDIA CUDA
make env-remove-rocm    # AMD ROCm
```

### Other useful targets

```bash
make help              # Full command catalog
make pytest            # Run tests
make format            # Format code (autopep8 → black → isort)
make lint              # Lint at moderate severity
make build             # Build wheel and sdist for PyPI
make publish-test      # Upload to Test PyPI
make publish           # Upload to real PyPI
```
