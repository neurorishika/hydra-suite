# Conda Environments and Makefile Reference

This page documents the conda environment files and Makefile targets used in the [developer install path](installation.md#conda-pip-developer-install).

## Environment files

| File | Env name | Platform | What conda provides |
| ---- | -------- | -------- | ----------------- |
| `environment.yml` | `hydra` | All | Python, NumPy, SciPy, PySide6, Qt6, OpenCV, Numba |
| `environment-mps.yml` | `hydra-mps` | macOS M1-M4 | Same as CPU |
| `environment-cuda.yml` | `hydra-cuda` | Linux/Windows (NVIDIA) | Same + CUDA 12 runtime libs (cublas, cudnn, curand, cufft) |

The conda environments provide **system libraries only** (Qt, OpenGL, CUDA runtime). Python packages are installed separately via `make install-*`, which runs `uv pip install -r requirements-*.txt`.

## Requirements files

| File | Inherits | Adds (beyond pyproject.toml) |
| ---- | -------- | ---------------------------- |
| `requirements.txt` | `-e .` | `torch`, `torchvision` (CPU) |
| `requirements-mps.txt` | `-e .` | `torch`, `torchvision`, `torchaudio`, `onnxruntime` |
| `requirements-cuda.txt` | `-e .` | `torch`, `torchvision`, `torchaudio`, `onnxruntime-gpu` |
| `requirements-cuda12.txt` | `-r requirements-cuda.txt` | `--extra-index-url .../cu128`, `tensorrt-cu12-*`, `cupy-cuda12x` |
| `requirements-cuda13.txt` | `-r requirements-cuda.txt` | `--extra-index-url .../cu130`, `tensorrt-cu13-*`, `cupy-cuda13x` |
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
```

### Install (pip packages into activated environment)

```bash
make install                      # CPU
make install-mps                  # Apple Silicon
make install-cuda CUDA_MAJOR=13  # NVIDIA CUDA 13
make install-cuda CUDA_MAJOR=12  # NVIDIA CUDA 12
make install-dev                  # Dev tools (formatting, linting, testing, publishing)
```

`CUDA_MAJOR` must match the driver: 13 needs driver 580+, otherwise use 12.
Check with `nvidia-smi --query-gpu=driver_version --format=csv,noheader`. See
[Choosing `CUDA_MAJOR`](installation.md#choosing-cuda_major) — a mismatch leaves
`torch.cuda.is_available()` `False` while GPUs still appear in
`device_count()`.

Each profile pins torch/torchvision/torchaudio to explicit `+cu128` / `+cu130`
builds in `requirements-cuda12.txt` / `requirements-cuda13.txt`. Those local
version tags are what force the wheels to come from the PyTorch index rather
than PyPI; do not relax them to bare version pins.

### Update (refresh both conda and pip)

```bash
make env-update                      # CPU
make env-update-mps                  # Apple Silicon
make env-update-cuda CUDA_MAJOR=13  # NVIDIA CUDA
```

### Remove

```bash
make env-remove         # CPU
make env-remove-mps     # Apple Silicon
make env-remove-cuda    # NVIDIA CUDA
```

## Fresh-box conda/mamba initialisation (non-interactive shells)

Two things a first-time provisioning session must get right that are easy to
miss because they only bite in a **non-interactive** shell (e.g. a spawned
subprocess, `ssh host 'cmd'`, or a service worker) rather than an interactive
login shell:

1. **conda's init block must sit above Ubuntu's interactivity guard** in
   `~/.bashrc` — the guard is the line that looks like
   `case $- in *i*) ;; *) return;; esac`. If conda's `>>> conda initialize >>>`
   block is *below* that guard, a non-interactive shell returns before conda
   ever gets onto `PATH`. The SLEAP service launches `conda run -n sleap`
   from exactly such a shell; if `conda` is unreachable there, the run does
   not error loudly — it produces **empty CSVs**, which then falsely compare
   as `EQUIVALENT` against a baseline in `tools/equivalence/`. Verify with
   `ssh <host> 'conda run -n sleap python -c "import sleap"'` — no
   `source ~/.bashrc` first — to prove the non-interactive path actually
   works.
2. **`MAMBA_ROOT_PREFIX` must be set before `mamba.sh` is sourced.** Without
   it, mamba prints an 11-line `WARNING` block to **stdout** on every
   non-interactive shell invocation. This is invisible in an interactive
   terminal but corrupts the SAM3 sidecar's stdout protocol, which parses the
   child process's stdout line-by-line for `@@HYDRA_SAM3_PROGRESS@@` records.

Both were found missing on a from-scratch Ubuntu box during provisioning
(2026-09-07); the fix is in `~/.bashrc`, not in this repo, since it's a
per-machine shell-init concern.

## SAM3 training prerequisite: Hugging Face authentication

`make setup-sam3-train` / `tools/setup_sam3_train_env.sh` build the `sam3`
package and its dependencies, but SAM3 LoRA training itself additionally
needs a Hugging Face credential on the machine that runs it — this is
**separate from and not satisfied by** copying a local `sam3.pt` checkpoint.
`build_sam3_image_model` fetches the model config from the **gated**
`facebook/sam3` Hugging Face repo at build time, so on a fresh box with no
credential, training (and even the auto-batch probe) refuses to start:

```bash
hf auth login      # paste a token from a HF account that has accepted the
                    # licence at https://huggingface.co/facebook/sam3
# or: export HF_TOKEN=... / HUGGING_FACE_HUB_TOKEN=...
```

`src/hydra_suite/training/sam3_lora/preflight.py` checks for this up front
and refuses with a message that already names the fix (`hf auth login`) —
if you see a bare `GatedRepoError: 401` instead, you are hitting the
credential check too late in the pipeline; file that as a regression.

## SAM3 environment recipe — use the script, not prose

The canonical, tested recipe for the `sam3` training sidecar env is
`tools/setup_sam3_train_env.sh` (invoked by `make setup-sam3-train`). Do not
re-derive or re-copy the pip install list into other docs — it drifts. Two
of its pins are load-bearing and easy to accidentally drop if hand-rolling
the env instead of running the script: `scipy<1.14` and
`opencv-python-headless<4.12` — both would otherwise pull in `numpy>=2`,
silently breaking the `numpy<2` pin the whole env exists to hold. See the
comments in that script for the full rationale per pin.

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
