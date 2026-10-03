# Installation

HYDRA Suite installs with **one command on every platform**: `python install.py`
from a checkout. The installer detects your platform (and, on NVIDIA machines,
your driver's CUDA version), builds a conda env or venv, installs the matching
PyTorch build plus everything in `pyproject.toml`, and finishes by running
`hydra doctor` to prove the result works.

!!! warning "HYDRA Suite is not on PyPI"

    `pip install hydra-suite` does **not** work — the package has not been
    published to PyPI. Install from a checkout (below) or from the git URL
    (see [Install without a checkout](#install-without-a-checkout)).

The page is a decision tree:

1. [Pick your platform](#1-pick-your-platform) and check its prerequisites.
2. [Run the installer](#2-run-the-installer) — one command.
3. [Add optional extras](#3-optional-add-ons) — SAM3 training, SLEAP, dev/docs tools.
4. [Verify](#verify) with `hydra doctor`.

---

## 1. Pick your platform

| Your machine | Tier the installer picks | What you get |
|---|---|---|
| Apple Silicon Mac (M1–M4) | `mps` | PyTorch MPS, ONNX Runtime with CoreML provider, `coremltools` |
| Linux or Windows with an NVIDIA GPU | `cuda` (CUDA 12 or 13, from the driver) | CUDA PyTorch, ONNX Runtime GPU, TensorRT, CuPy, PyNvVideoCodec |
| Anything else (Intel Mac, Linux/Windows without NVIDIA) | `cpu` | CPU PyTorch, ONNX Runtime CPU |

Every tier also gets SAM2 and SAM3 **inference** (DetectKit escalation) by
default. See [Platform Notes](platforms.md) for which features run on which
device.

### Prerequisites

All platforms need:

- **Python ≥ 3.9** to *run* the installer (base conda, system `python3`, or
  `py -3` on Windows). The installer only uses the standard library. The env it
  builds gets Python 3.13 by default (`--python` to change it); HYDRA itself
  needs Python 3.11–3.13.
- **git** — CLIP (SAM3's text encoder) and the AprilTag fork install from git.
- **conda/mamba (recommended, optional)** — if `conda` or `mamba` is on `PATH`
  the installer builds a conda env that also supplies `git`, `cmake` and a C
  toolchain. Without conda it builds a venv instead (see
  [venv vs conda vs current](#venv-vs-conda-vs-current-env)).
- **ffmpeg** — nothing to do. The `imageio-ffmpeg` wheel bundles a binary on
  every platform, and a system `ffmpeg` on `PATH` takes precedence.

Plus, per platform:

=== "macOS"

    - Xcode Command Line Tools, for the AprilTag C build:
      `xcode-select --install` (accept the licence with
      `sudo xcodebuild -license` if prompted).

=== "Linux"

    - With conda: nothing else — the env includes `c-compiler`/`cmake`.
    - Without conda: a C compiler and `cmake` from your package manager.
    - For the Qt GUIs on a minimal install you may need system libraries, e.g.
      on Ubuntu/Debian:

        ```bash
        sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 \
            libdbus-1-3 libfontconfig1 libxcb-cursor0
        ```

    - NVIDIA: a working driver (`nvidia-smi` must run). You do **not** need
      the CUDA toolkit — the CUDA user-space libraries come from pip wheels.

=== "Windows"

    - Python from python.org, the Microsoft Store, or Miniforge/Anaconda.
    - git for Windows (<https://git-scm.com>), or `conda install git`.
    - **Visual Studio Build Tools** with the *Desktop development with C++*
      workload, for the AprilTag C build. If you do not need AprilTag identity,
      skip it with `--skip-apriltag` and you need no compiler at all.
    - `cmake` (installed into the conda env automatically; `pip install cmake`
      or the VS Build Tools CMake component for a venv).
    - NVIDIA: a working driver (`nvidia-smi` must run).
    - Conda is optional.

    !!! note "Windows status"

        Windows is a supported target and CI builds and verifies a **CPU**
        install with `install.py` on Windows for Python 3.11, 3.12 and 3.13.
        A Windows **CUDA** install has not yet been verified on real hardware.
        Please report problems.

---

## 2. Run the installer

```bash
git clone https://github.com/neurorishika/hydra-suite.git
cd hydra-suite
python install.py            # Windows: py -3 install.py
```

That is the whole install. When it finishes it prints how to activate the env:

```bash
conda activate hydra-mps     # or hydra / hydra-cuda; venv users get an activate path
```

Preview without changing anything:

```bash
python install.py --dry-run
```

### What the installer does

| Step | Name (for `--only`) | What happens |
|---|---|---|
| 1 | `env` | Create (or update) the conda env / venv. Conda supplies only `python`, `pip`, `git`, `cmake`, and the C/C++ compilers (no compilers on Windows — MSVC comes from VS Build Tools). |
| 2 | `uv` | Install `uv` into the env (`--no-uv` falls back to pip). |
| 3 | `conflicts` | Remove package variants that must not coexist for this tier (the other `onnxruntime`/`onnxruntime-gpu`, GUI `opencv` builds, the other TensorRT family). |
| 4 | `torch` | Install the pinned `torch`/`torchvision` pair from the PyTorch index for the tier (`cpu`, `cu128` or `cu130`; PyPI on macOS). |
| 5 | `hydra` | Install `hydra-suite[<tier>,sam]` from the checkout (editable), with the [tested pins](#tested-pins) applied. |
| 6 | `clip` | Install the pinned ultralytics CLIP fork (SAM3's text encoder; git-only, so it cannot be a pyproject dependency). |
| 7 | `apriltag` | Build the Kronauer lab AprilTag fork (`Social-Evolution-and-Behavior/apriltag` @ `c43a9b6`) from source, statically linked. Skipped with `--skip-apriltag`. |
| 8 | `libomp` | Apple Silicon only: de-duplicate the OpenMP runtime so torch and other OpenMP users load one copy. |
| 9 | `sam3-train`, `sleap` | Optional sidecar envs — only with `--with-sam3-train` / `--with-sleap`. |
| 10 | `doctor` | Run `hydra doctor` for the tier. Skipped with `--skip-doctor`. |

`--only clip,apriltag` re-runs individual steps against an existing env.

### CUDA: auto-detection and override

On a machine with an NVIDIA GPU the installer reads the driver version from
`nvidia-smi` and picks the CUDA build the driver can actually run:

| Driver version | CUDA build | PyTorch wheels | Extra |
|---|---|---|---|
| 580 or newer | CUDA 13 | `cu130` | `cuda13` |
| older than 580 | CUDA 12 | `cu128` | `cuda12` |

Override it when you need to:

```bash
python install.py --cuda 12        # or 13; CUDA_MAJOR=12 in the environment also works
python install.py --cuda none      # ignore the GPU, install the CPU tier
```

An explicit `--cuda 13` on a driver older than 580 is honoured but warns: that
build imports fine, but `torch.cuda.is_available()` is `False`. (Note that
`torch.cuda.device_count()` still reports the GPUs in that state — do not be
reassured by it.) On a Linux/Windows machine with no working `nvidia-smi`,
`python install.py` simply installs the CPU tier; forcing `--tier cuda` there
without `--cuda 12|13` is refused.

### The `make` equivalents

The Makefile targets keep their old names but are thin wrappers around
`install.py`. They split the install into two steps — create the env, then
install into the **activated** env:

```bash
# Apple Silicon
make setup-mps && conda activate hydra-mps && make install-mps

# NVIDIA (CUDA major auto-detected; CUDA_MAJOR=12|13 overrides)
make setup-cuda && conda activate hydra-cuda && make install-cuda

# CPU
make setup && conda activate hydra && make install
```

`make setup*` only creates the env (`--create-only`); `make install*` runs
`install.py --target current` with the env's Python. See
[Environments and Makefile](environments.md) for the full target list.

### Choosing a tier explicitly

```bash
python install.py --tier cpu       # e.g. a CPU-only env on a GPU box
python install.py --tier cuda --cuda 12
python install.py --env my-hydra   # custom env name (default: hydra / hydra-mps / hydra-cuda)
```

### venv vs conda vs current env

`--target` decides where packages go:

| `--target` | When | Notes |
|---|---|---|
| `auto` (default) | — | `conda` if conda/mamba is on `PATH`, else `venv`. |
| `conda` | Recommended | Named env (`hydra`, `hydra-mps`, `hydra-cuda`); conda supplies `git` and the build toolchain. Sidecar envs (SAM3 training, SLEAP) require conda. |
| `venv` | No conda | Created at `<checkout>/.venv-<name>` (e.g. `.venv-hydra-cuda`), or at the path given by `--env some/path`. The installer itself must run on Python ≥ 3.11 because it becomes the venv's Python. |
| `current` | You manage the env | Installs into the interpreter running `install.py` (Python ≥ 3.11). This is what `make install*` uses. |

### Tested pins

By default a fresh install gets the exact versions in `constraints/`
(`base.txt` on every tier, plus `cuda12.txt` or `cuda13.txt` on CUDA), so
tracking output does not drift every time an upstream package releases.
`pyproject.toml` declares the allowed *ranges*; `constraints/` records the
*tested* versions. Torch is pinned separately in `install.py`
(`TORCH_VERSION`/`TORCHVISION_VERSION`).

```bash
python install.py --latest                 # ignore constraints/, take the newest compatible releases
python install.py --constraints my.lock    # add your own lock on top of the tested pins
```

To bump the tested pins (maintainers):

```bash
python tools/lock_constraints.py                          # re-resolve, keep existing pins
python tools/lock_constraints.py --upgrade-package numpy  # bump one package
```

then re-run the equivalence gates before committing — see
[Dependency and CUDA Updates](../developer-guide/dependency-cuda-updates.md).

### Install without a checkout

You can install straight from GitHub without cloning:

```bash
curl -O https://raw.githubusercontent.com/neurorishika/hydra-suite/main/install.py
python install.py --source git+https://github.com/neurorishika/hydra-suite@main
```

`--source` also accepts a checkout directory or a built wheel.

!!! warning "No tested pins without a checkout"

    The tested pins live in `constraints/` next to `install.py`. A lone
    downloaded `install.py` has no `constraints/` beside it, so a no-checkout
    install resolves the newest compatible releases. Pass a lock file with
    `--constraints FILE` (e.g. a copy of `constraints/base.txt`) to pin.

---

## 3. Optional add-ons

| Add-on | Command | Make target | Notes |
|---|---|---|---|
| SAM3 LoRA training sidecar | `python install.py --with-sam3-train` | `make setup-sam3-train` | **CUDA only** (compute capability ≥ 8.0, bf16); refused on other tiers. Builds conda env `hydra-sam3`. See [SAM2 and SAM3](../user-guide/sam-install-and-run.md#sam3-training-optional-cuda-only). |
| SLEAP pose sidecar | `python install.py --with-sleap` | `make setup-sleap` | Builds conda env `sleap` with `sleap==1.6.2` and the torch build matching this host. See [Integrations](integrations.md#sleap-integration-trackerkit-posekit). |
| Dev tools | `python install.py --dev` | `make install-dev` | pytest, black, flake8, mypy, build, twine, … (`[dev]` extra). |
| Docs tools | `python install.py --docs` | `make docs-install` | mkdocs and plugins (`[docs]` extra). |

Flags combine, e.g. `python install.py --dev --with-sleap`. When the main env
already exists, the installer updates it in place and then adds the sidecar.
The `make setup-*` sidecar targets run only the sidecar steps.

---

## Verify

The installer runs this for you; run it again any time:

```bash
hydra doctor                 # or: python -m hydra_suite.runtime.doctor, or: make doctor
```

`hydra doctor` is headless (no Qt window, no downloads) and checks:

- Python version, and that every kit's GUI module imports
- that exactly **one** package owns OpenCV's `cv2/`
- macOS: that only one OpenMP runtime loads in every import order
- torch for the tier — MPS availability, or on CUDA a real convolution plus a
  check that the CUDA 12/13 wheel families are not mixed
- ONNX Runtime providers (CoreML on MPS); TensorRT and CuPy on CUDA;
  `coremltools` on MPS
- SAM2, SAM3 inference packages (including the correct CLIP fork), and whether
  the gated SAM3 checkpoint is downloaded (a warning, not a failure)
- AprilTag and `ffmpeg` (warnings)
- the optional SAM3 training and SLEAP sidecars

Each warning or failure prints the command that fixes it. The exit code is
non-zero only when a check **fails**.

```bash
hydra doctor --json                  # machine-readable
hydra doctor --tier cuda             # check against an expected tier
hydra doctor --quick                 # skip the slow GUI-import check
hydra doctor --require-sam3-train    # fail if the training sidecar is unusable
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

---

!!! note "Linux needs glibc 2.34 or newer"
    The tested pins include PySide6 6.11, whose Linux wheels are
    `manylinux_2_34` (RHEL/Rocky 9, Ubuntu 22.04 and newer). Older
    distributions (RHEL 8, Ubuntu 20.04) have no compatible wheel.

### Verification status (2026-10-03)

What has actually been run, not just documented. "CI" is the
`Install smoke` workflow (`install.py --target current --cuda none`, then
`hydra doctor` and a test subset) on every push.

| Install path | Linux CPU | macOS (Apple Silicon) | Windows CPU | Linux CUDA 12 | Linux CUDA 13 | Windows CUDA |
| --- | --- | --- | --- | --- | --- | --- |
| `python install.py` (pip, current env) | CI, py3.11–3.13 | CI, py3.11–3.13 (mps packages) | CI, py3.11–3.13 | — | — | not yet verified (no box) |
| `python install.py` (conda env) | — | fresh env, doctor all OK (MPS) | — | fresh env on diptera (driver 570, auto-detected cu128), doctor 0 failures | fresh env on mehek (driver 595, auto-detected cu130), doctor all OK | not yet verified |
| `make setup` + `make install` | CI (conda + make) | CI (conda + make) | n/a (use `install.py`) | — | — | n/a |
| SAM2 escalation (real frames) | — | MPS + CPU | — | — | CUDA | — |
| SAM3 escalation (real frames) | — | MPS + CPU | — | — | CUDA | — |
| SAM3 training sidecar + 1-epoch run | n/a | n/a (refused by design) | n/a | — | `--with-sam3-train`, trained + exported | not yet verified |

Tracking output: an environment built by `install.py` with the tested pins
produces the same trajectories as the previous conda envs — byte-identical on
CUDA (mehek, `fly_obb` + `worm_bgsub`) and identical positions/headings/IDs on
MPS, where only float32 rounding in `PositionUncertainty` and
`AssignmentConfidence` differs (pip scipy links Apple Accelerate, conda scipy
OpenBLAS; relative difference ≤ 5e-6).

## Updating and recreating

```bash
git pull
python install.py --update       # upgrade packages in the existing env
python install.py --recreate     # delete and rebuild the conda env from scratch
```

- **`--update`** refreshes the conda packages and re-installs `hydra-suite`
  with `--upgrade` (still within the tested pins unless you pass `--latest`).
- **`--recreate`** removes the conda env and builds it again. For a venv,
  delete the `.venv-<name>` directory and re-run.

!!! warning "Recreate environments built before the pyproject-only layout"

    Older envs were built from `environment*.yml` files that installed Qt,
    OpenCV and CUDA libraries from conda, plus `requirements*.txt` files that
    no longer exist. Updating such an env in place leaves conda- and
    pip-installed copies fighting over the same files. `hydra doctor` flags
    the most common symptom — more than one package (or a conda package)
    owning `cv2/` — and tells you to run `python install.py --recreate`.

---

## Data directories

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
| `HYDRA_MODELS_DIR` | **Only** the models root (`model_registry.json` + published models), leaving engine artifacts and calibration profiles on the host data dir | `<HYDRA_DATA_DIR>/models` |

Examples:

```bash
# Use a shared lab network drive for models
export HYDRA_DATA_DIR=/mnt/lab-shared/hydra-data
hydra

# Use a project-specific config
HYDRA_CONFIG_DIR=./my-project-config trackerkit

# Run a packaged job's models without relocating engine caches or
# calibration profiles (set automatically by a packed job's run.sh)
export HYDRA_MODELS_DIR=/home/rutalab/jobs/colony_A/models

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

## Troubleshooting

Start with `hydra doctor` — most failures name their own fix. Common cases:

### `torch.cuda.is_available()` is `False` on a machine with GPUs

Almost always a CUDA-major mismatch: a CUDA 13 build on a driver older than
580. Check `python -c "import torch; print(torch.version.cuda)"` against the
[driver table](#cuda-auto-detection-and-override), then rebuild with the right
build:

```bash
python install.py --recreate --cuda 12     # or 13
```

### Mixed CUDA 12 / CUDA 13 wheels (`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`, CuPy falls back to CPU)

The CUDA component wheels ship under two naming schemes — `nvidia-cublas-cu12`
(CUDA 12) and plain `nvidia-cublas` (CUDA 13) — and both unpack into the same
`site-packages/nvidia/` tree. Switching CUDA major in place can leave both
families installed. Symptoms surface far from the cause: convolutions fail
with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`, or CuPy compiles against stale
headers and GPU background subtraction silently falls back to CPU.

**Do not uninstall individual `nvidia-*` packages** — they share files, and a
partial uninstall removes cuDNN sublibraries the survivors still need.
`hydra doctor` detects the mixed state; rebuild the env:

```bash
python install.py --recreate      # add --cuda 12|13 to override detection
```

### `CUDAExecutionProvider` missing from ONNX Runtime

ONNX Runtime GPU loads its CUDA libraries from the pip `nvidia-*` wheels via
`onnxruntime.preload_dlls()` — there is no `LD_LIBRARY_PATH` hook and no CUDA
library from conda any more. If the provider is missing, the env usually has
the wrong `onnxruntime` variant or a mixed wheel family; `hydra doctor --tier
cuda` reports which, and `python install.py --recreate` fixes both.

### AprilTag build fails

The fork is a C extension. Install a compiler (Xcode CLT on macOS, VS Build
Tools on Windows, `c-compiler`/`cmake` on Linux) and re-run just that step:

```bash
python install.py --only apriltag
```

Or install without it: `python install.py --skip-apriltag` (AprilTag identity
is then unavailable; `hydra doctor` reports it as a warning).

### SAM3 says the `clip` package is the wrong fork

SAM3 needs the ultralytics CLIP fork, not `openai/CLIP`. Re-run
`python install.py --only clip`. See
[SAM2 and SAM3](../user-guide/sam-install-and-run.md#troubleshooting).

### `OMP Error #15` on macOS

Two OpenMP runtimes were loaded. Run `python -m hydra_suite.runtime.macos_libomp`
(or `make configure-mps-libs`); `hydra doctor` checks for this.

---

## Related docs

- [Environments and Makefile](environments.md) — how environments are composed, Makefile reference
- [Platform Notes](platforms.md) — feature × device matrix
- [Integrations](integrations.md) — SLEAP, X-AnyLabeling setup
- [SAM2 and SAM3: install and run](../user-guide/sam-install-and-run.md)
- [Dependency and CUDA Updates](../developer-guide/dependency-cuda-updates.md) — bumping pins
