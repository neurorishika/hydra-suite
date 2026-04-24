# Installation

There are three ways to install HYDRA Suite, depending on your situation:

| Method | When to use |
|--------|-------------|
| [**pip install**](#pip-install-from-pypi) | You want to use the software. One command, no git needed. |
| [**pip install from GitHub**](#pip-install-from-github) | You want the latest unreleased version, or the package isn't on PyPI yet. |
| [**conda + pip (developer)**](#conda-pip-developer-install) | You're developing the code. Editable install, Makefile workflows. |

All three methods support full GPU acceleration. See the [platform matrix](#platform-matrix) for what's available on each platform.

---

## Platform matrix

| Feature | CPU | MPS (Apple Silicon) | CUDA (NVIDIA) |
|---------|-----|---------------------|---------------|
| YOLO detection / pose | CPU | GPU | GPU |
| TensorRT acceleration | No | No | Yes (NVIDIA only) |
| ONNX Runtime | CPU | CPU | GPU |
| CuPy background subtraction | No | No (NVIDIA only) | GPU |
| Platforms | All | macOS M1-M4 | Linux, Windows |
| System requirements | Python 3.11+ | Python 3.11+ | Python 3.11+ |

---

## pip install from PyPI

Requires only Python 3.11+. No conda, no git clone.

### CPU (all platforms)

```bash
pip install hydra-suite
```

### Apple Silicon / MPS (macOS M1-M4)

```bash
pip install torch torchvision
pip install "hydra-suite[mps]"
```

MPS is built into the standard PyTorch macOS wheel. GPU acceleration works for YOLO inference and neural network operations.

### NVIDIA GPU / CUDA (Linux, Windows)

```bash
# Step 1: Install PyTorch from PyTorch's own index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Step 2: Install hydra-suite with CUDA extras
pip install "hydra-suite[cuda]"
```

This installs everything: ONNX Runtime GPU, TensorRT, CuPy, ONNX export tools.

**CUDA version selection:**

| Your CUDA version | PyTorch `--index-url` | Package extra |
|---|---|---|
| 12.6 | `.../cu126` | `[cuda]` or `[cuda12]` |
| 12.8 | `.../cu128` | `[cuda]` or `[cuda12]` |
| 13.0 | `.../cu130` | `[cuda13]` |

`[cuda]` is an alias for `[cuda12]`. For CUDA 13 systems:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "hydra-suite[cuda13]"
```

**You do not need the CUDA toolkit installed.** PyTorch's pip wheel bundles NVIDIA CUDA runtime libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-curand-cu12`, etc.) as pip dependencies and preloads them at import time, so ONNX Runtime and TensorRT find them automatically.

### Why is GPU install two commands?

PyTorch GPU builds are hosted on PyTorch's own server, not PyPI. Python packaging standards (PEP 621) have no way to specify per-dependency index URLs. This is how every ML project handles it: you install torch first, then the package. The second command will **not** overwrite your existing torch.

---

## pip install from GitHub

Install the latest code directly from the repository without cloning. Useful when:

- The package isn't on PyPI yet
- You need a fix that hasn't been released
- A collaborator wants to try the latest version

### CPU

```bash
pip install "hydra-suite @ git+https://github.com/neurorishika/hydra-suite.git"
```

### With GPU extras

```bash
# MPS
pip install torch torchvision
pip install "hydra-suite[mps] @ git+https://github.com/neurorishika/hydra-suite.git"

# CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "hydra-suite[cuda] @ git+https://github.com/neurorishika/hydra-suite.git"

# CUDA 13
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "hydra-suite[cuda13] @ git+https://github.com/neurorishika/hydra-suite.git"
```

### Install a specific branch or tag

```bash
# Specific branch
pip install "hydra-suite @ git+https://github.com/neurorishika/hydra-suite.git@main"

# Specific tag
pip install "hydra-suite @ git+https://github.com/neurorishika/hydra-suite.git@v1.0.0"
```

### Upgrading a GitHub install

```bash
pip install --upgrade --force-reinstall --no-deps \
    "hydra-suite @ git+https://github.com/neurorishika/hydra-suite.git"
```

---

## conda + pip (developer install)

For active development. Uses conda for system libraries (Qt, OpenGL) and pip for Python packages. Provides editable install so code changes are live immediately.

### Step 1: Clone

```bash
git clone https://github.com/neurorishika/hydra-suite.git
cd hydra-suite
```

### Step 2: Create environment and install

Pick your platform:

```bash
# CPU
make setup
conda activate hydra
make install

# Apple Silicon (MPS)
make setup-mps
conda activate hydra-mps
make install-mps

# NVIDIA GPU (CUDA)
make setup-cuda
conda activate hydra-cuda
make install-cuda CUDA_MAJOR=13     # or CUDA_MAJOR=12
```

### Step 3: Install dev tools (optional)

```bash
make install-dev
```

This adds formatting, linting, testing, and publishing tools. Run `make help` for the full command catalog.

### How it works

The conda environment provides system libraries and Python. The `make install-*` targets run `uv pip install -r requirements-*.txt`, which installs:

1. **torch** with the correct GPU variant and index URL
2. **`-e .`** — an editable install that pulls all `pyproject.toml` dependencies

Base dependencies (numpy, scipy, PySide6, ultralytics, etc.) are declared once in `pyproject.toml`. The requirements files only add what `pyproject.toml` cannot express (torch GPU index URLs). This means there is **one source of truth** for dependencies.

### Development workflow

```bash
conda activate hydra-mps   # or your env
# Edit code — changes are live immediately (editable install)
make pytest                               # run tests
make format && make lint                  # format and lint before committing
```

---

## After installation

### Verify

```bash
hydra --help
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())"
```

For CUDA, also check ONNX Runtime:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include 'CUDAExecutionProvider'
```

### Launch

```bash
hydra              # HYDRA Suite launcher
trackerkit         # TrackerKit tracking GUI
posekit            # PoseKit pose-labeling GUI
classkit           # ClassKit labeler
detectkit          # DetectKit training
filterkit          # FilterKit tool
refinekit          # RefineKit proofreading
```

### Data directories

User data is stored in platform-appropriate locations (not inside the package):

| Data | macOS | Linux | Windows |
|------|-------|-------|---------|
| Config / presets | `~/Library/Application Support/hydra-suite/` | `~/.config/hydra-suite/` | `%LOCALAPPDATA%\Kronauer Lab\hydra-suite\` |
| Models | same `/models/` | same `/models/` | same `\models\` |
| Training runs | same `/training/` | same `/training/` | same `\training\` |

Default config presets and skeleton definitions are bundled with the package and seeded to your config directory on first run.

### Customizing data directories

Override the default locations with environment variables:

| Variable | What it overrides | Default |
|----------|------------------|---------|
| `HYDRA_DATA_DIR` | Models, training runs | `platformdirs` user data dir |
| `HYDRA_CONFIG_DIR` | Presets, skeletons, advanced config | `platformdirs` user config dir |

Examples:

```bash
# Use a shared lab network drive for models
export HYDRA_DATA_DIR=/mnt/lab-shared/hydra-data
hydra

# Use a project-specific config
HYDRA_CONFIG_DIR=./my-project-config mat

# Check where everything currently points
python -c "from hydra_suite.paths import print_paths; print_paths()"
```

All sub-applications (TrackerKit, PoseKit, DetectKit, ClassKit, RefineKit, FilterKit) use the same `hydra_suite.paths` module, so they all respect these overrides and share the same data directories.

### Programmatic access from other tools

Scripts and notebooks can access the same paths:

```python
from hydra_suite.paths import get_models_dir, get_presets_dir, get_skeleton_dir

print(get_models_dir())        # where trained models are stored
print(get_presets_dir())       # where config presets live
print(get_skeleton_dir())      # where skeleton definitions live
```

### Migrating from an older repo checkout

If you had models in `<repo>/models/` or training data in `<repo>/training/`:

```bash
python -m hydra_suite.paths_migrate /path/to/repo --dry-run  # preview what would be copied
python -m hydra_suite.paths_migrate /path/to/repo            # copy files
```

---

## Updating

### pip (PyPI)

```bash
pip install --upgrade hydra-suite
```

### pip (GitHub)

```bash
pip install --upgrade --force-reinstall --no-deps \
    "hydra-suite @ git+https://github.com/neurorishika/hydra-suite.git"
```

### conda + pip (developer)

```bash
conda activate hydra-mps  # or your env
git pull
mamba env update -f environment-mps.yml --prune   # if conda deps changed
make install-mps                                  # reinstall pip packages
```

---

## Troubleshooting

### `CUDAExecutionProvider` missing from ONNX Runtime

- Ensure you installed with `[cuda]` or `[cuda13]` extra
- Ensure PyTorch is imported before ONNX Runtime (PyTorch preloads CUDA libs)
- Verify: `python -c "import torch; import onnxruntime as ort; print(ort.get_available_providers())"`
- For a direct ONNX Runtime session check, use: `python -c "import onnxruntime as ort; s=ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider']); print(s.get_providers())"`
- If you see `libcurand.so.10: cannot open shared object file`, update the conda env: `conda install -n hydra-cuda -c nvidia -c conda-forge libcurand=10.*`
- **conda path:** re-run `make install-cuda CUDA_MAJOR=13` (or `CUDA_MAJOR=12`) and reactivate the environment to refresh hooks; the Makefile now runs a CUDA runtime self-check and fails fast if ONNX Runtime libraries are still missing

### PySide6 / Qt errors on Linux (pip only)

pip-installed PySide6 may need system libraries:

```bash
sudo apt install libgl1-mesa-glx libegl1 libxcb-xinerama0  # Ubuntu/Debian
```

The conda path handles this automatically.

### CuPy compilation fails

CuPy builds from source on first install if no prebuilt wheel is found. Ensure CUDA dev packages are installed.

### Fresh reinstall

```bash
# pip
pip uninstall hydra-suite
pip install "hydra-suite[cuda]"

# conda
conda env remove -n hydra-cuda
make setup-cuda
conda activate hydra-cuda
make install-cuda CUDA_MAJOR=13
```

---

## Related docs

- [Environments](environments.md) — conda environment matrix and Makefile reference
- [Integrations](integrations.md) — SLEAP, X-AnyLabeling setup
- [Publishing to PyPI](../developer-guide/publishing.md) — releasing new versions
