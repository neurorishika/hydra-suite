# Install single-source-of-truth + cross-platform installer — design

**Status:** Approved direction (user, 2026-10-03); implementation on `feat/install-ssot`.
**Checklist:** `docs/superpowers/plans/2026-10-03-install-paths-and-sam-todo.md`.

## Goals

1. `pyproject.toml` is the only place Python dependencies are declared.
2. One pure-Python installer works on Linux, macOS and Windows. Every `make` install target wraps it.
3. CUDA major version is detected from the driver; the user can override it.
4. SAM2 and SAM3 *inference* install by default on every tier.
5. SAM3 *training* is an opt-in sidecar: CUDA only, refused loudly elsewhere, installable without a repo checkout.
6. Tracking output is byte-identical before and after (proven by equivalence runs across the old and new envs).

## Non-goals

- Publishing to PyPI. `hydra-suite` is **not on PyPI** (the JSON API returns 404). Docs stop claiming `pip install hydra-suite`. Publishing is an outward-facing release decision left to the user.
- Training SAM3 on MPS or CPU.
- The manual Windows+CUDA gate, which waits for a box.

## 1. Installer: `install.py` (repo root)

**Bootstrap constraints**
- stdlib only; Python ≥ 3.9.
- Runs from *outside* the target env, e.g. base conda's Python, a system `python3`, or `py -3` on Windows.
- It must never import `hydra_suite`.
- The installer is a **single file, `install.py`**, holding the pure logic (platform detection, CUDA mapping, plan building) and the step runner. Unit tests import it by file path (`tests/test_installer_*.py`).

**CLI**

```
python install.py [--tier auto|cpu|mps|cuda] [--cuda auto|12|13|none]
                  [--target conda|venv|current] [--env NAME] [--python 3.13]
                  [--source PATH_OR_URL] [--update] [--create-only]
                  [--with-sam3-train] [--with-sleap] [--dev] [--docs]
                  [--skip-apriltag] [--skip-doctor] [--dry-run] [--yes]
```

**Tier resolution** (`resolve_tier`)
- An explicit `--tier` wins.
- Otherwise:
  - `darwin`+`arm64` → `mps`.
  - `nvidia-smi` present → `cuda`.
  - Anything else → `cpu`.
- `--cuda none` forces `cpu` on a CUDA host.

**CUDA resolution** (`resolve_cuda_major`)
- Parse the driver version from `nvidia-smi --query-gpu=driver_version --format=csv,noheader`.
- Driver major ≥ 580 → 13; otherwise → 12.
- An explicit `--cuda 12|13` overrides the detected value. It warns when it exceeds what the driver supports, because that is the silent `is_available()==False` trap.
- `CUDA_MAJOR` from make arrives as `--cuda`.
- The same mapping replaces the private `_driver_max_cuda` in `verify_cuda_runtime.py`. That file moves into the package (see §4).

**Targets**
- `conda` (default when conda/mamba is on PATH): create or update the env from the slim yml, then install with `<env>/bin/python` (or `<env>\python.exe`).
- `venv`: `python -m venv` with a host Python that must be ≥ 3.11 (checked).
- `current`: the interpreter running the installer. Used by CI and by `make install*`, which run inside the activated env.

Default env names are unchanged (`hydra`, `hydra-mps`, `hydra-cuda`).

**Steps.** Each step is a pure `Step(description, argv)` in a plan. `--dry-run` prints the plan.
1. Env: create or update (skipped for `current`).
2. Bootstrap `uv` into the env via `python -m pip install uv`. Fall back to `pip` if that fails.
3. Uninstall conflicting families:
   - `onnxruntime` vs `onnxruntime-gpu`
   - tensorrt cu12 vs cu13
   - all opencv variants except `opencv-python-headless`
4. **Torch first**, with `--index-url`, never `--extra-index-url`. Versions come from the `TORCH_PINS` table (`torch==2.11.0`, `torchvision==0.26.0`; `+cu128`/`+cu130` on CUDA). Index:
   - `cpu` on Linux/Windows → `whl/cpu`
   - `mps` → PyPI
   - `cuda` → `whl/cu128` / `whl/cu130`

   This removes `--index-strategy unsafe-best-match` and the iopath/sam2 resolver failure.
5. `uv pip install -e "<source>[<tier-extra>,sam,(dev),(docs)]"` with an **overrides file** that removes `opencv-python` (`opencv-python; sys_platform == 'never'`). This leaves exactly one `cv2` owner (headless). Without `uv`, run the same `pip` command, then force-reinstall headless.
6. CLIP from git at the pinned commit `CLIP_REF`. A loud error if `git` is missing.
7. AprilTag fork: clone + cmake build at the pinned commit. The cmake prefix and Python paths are computed per OS. Failure is an **error** that names the missing prerequisite (cmake / C compiler / MSVC Build Tools) and the `--skip-apriltag` escape.
8. Platform fixups:
   - MPS: the libomp de-duplication (logic from `tools/configure_mps_libs.py`, generalised to whichever duplicate pair exists). It no-ops when only one libomp is present.
   - CUDA: no conda library hook. ORT uses the pip `nvidia-*-cu12` wheels via `onnxruntime.preload_dlls()` (§3).
9. Optional sidecars: `--with-sam3-train` (§5), `--with-sleap` (§6).
10. `hydra doctor` (§4), unless `--skip-doctor`.

`--update` re-runs steps 1–10 with upgrade semantics.

**Source resolution**
- Default `--source`: the directory containing `install.py`, installed editable.
- A git URL or a wheel path is also accepted, which is how a no-checkout install works (`python install.py --source git+https://github.com/neurorishika/hydra-suite@main` after downloading only `install.py`).

## 2. pyproject as single source of truth

**Core deps added**

| Dependency | Why |
|---|---|
| `torch>=2.4`, `torchvision>=0.16` | Declared explicitly |
| `optuna`, `optunahub`, `cmaes`, `umap-learn`, `pyyaml`, `scikit-image`, `pillow`, `einops` | Everything `src/` imports at top level that only the ymls provided — the authoritative list comes from the import-declared test (§7) |
| `imageio-ffmpeg` | `ffmpeg` binary fallback on venv/Windows |

`opencv-python-headless` stays.

**Removed:** `torchaudio` (never imported).

**Extras**

| Extra | Contents |
|---|---|
| `cpu` | `onnxruntime`, `onnx`, `onnxslim`, `onnxscript`, `av` |
| `mps` | `cpu` + `coremltools>=8` |
| `cuda12` | `onnxruntime-gpu[cuda,cudnn]`, `cupy-cuda12x`, `tensorrt-cu12`(+`-bindings`,`-libs`; Linux/Windows markers), `PyNvVideoCodec`, onnx tools, `av` |
| `cuda13` | Same with cu13 packages. `cupy-cuda13x` is stable on PyPI now (14.2.0), so no `--pre`/custom index. ORT stays cu12 user-space via its own extra |
| `cuda` | Alias → `cuda12` (unchanged) |
| `sam` | `ftfy`, `regex` (CLIP is installed by the installer — PEP 508 direct refs break `twine upload`) |
| `sam3` | Alias → `sam` (compat) |
| `sam3-train` | Unchanged (sidecar only) |
| `dev` | Absorbs `requirements-dev.txt` |
| `docs` | Absorbs `requirements-docs.txt` |

**Deleted files:** `requirements.txt`, `requirements-mps.txt`, `requirements-cuda.txt`, `requirements-cuda12.txt`, `requirements-cuda13.txt`, `requirements-dev.txt`, `requirements-docs.txt`. A test asserts that none come back. The docs workflow installs the `docs` extra's requirement list (read via `tomllib`) without the heavy core deps.

**Conda ymls** (`environment.yml`, `-mps`, `-cuda`) shrink to `python=3.13`, `pip`, `ffmpeg`, `git`, `cmake`, `c-compiler`, `cxx-compiler`.
- No numpy/opencv/PySide6/qt6-main/pyqtwebengine/optuna/numba.
- The CUDA-12 user-space libraries leave the CUDA yml, replaced by ORT's pip `[cuda,cudnn]` extra plus `preload_dlls()`.
- **Fallback, if that fails verification on mehek or diptera:** keep only those libraries in `environment-cuda.yml` as a documented exception.

## 3. Runtime code changes required by the slimmer envs

- **ORT CUDA libraries.** Before the first CUDA/TensorRT `InferenceSession`, call `onnxruntime.preload_dlls()` when it exists. Wrap it in the existing `onnx_providers` path, idempotently and without failing. This replaces the `LD_LIBRARY_PATH` activate.d hook (`configure-cuda-ort`), which was Linux-only.
- **ffmpeg binary.** Add a helper `utils/ffmpeg.py::ffmpeg_exe()` that tries `shutil.which("ffmpeg")` first and then `imageio_ffmpeg.get_ffmpeg_exe()`. The trackerkit crop path uses it.
- **libomp.** `hydra doctor` imports `torch`+`cv2`+`sklearn`+`numba` in both orders in a subprocess. The installer's MPS fixup handles any duplicate pair. `KMP_DUPLICATE_LIB_OK` is **not** introduced on the main path.

## 4. `hydra doctor`

- **Entry point.** `hydra` points at `hydra_suite.launcher.cli:main`. That module dispatches `doctor` (and `--version`) **before** any Qt import; otherwise it imports `launcher.app` and runs the GUI exactly as today.
- **Location.** The checks live in `hydra_suite/runtime/doctor.py`. The CUDA checks are moved from `verify_cuda_runtime.py`, with the `CONDA_PREFIX` exit and the `x86_64-linux` hard-coding removed. The root `verify_cuda_runtime.py` becomes a thin shim that calls doctor's CUDA section, so `tests/test_verify_cuda_runtime.py` keeps working (tests are updated to the new module path).
- **Check result shape.** Each check is a `Check(name, status ∈ {ok, warn, fail, skip}, detail, fix)`.
- **Exit code.** Nonzero iff any check is `fail`. `--json` prints machine-readable output.

**Checks**
1. Python version.
2. Core imports: every kit's app module, imported headless, with Qt imports allowed but no `QApplication` created.
3. Single `cv2` distribution.
4. libomp import-order check (MPS/macOS).
5. Torch build vs tier.
   - CUDA tier: CUDA available + driver compatible + nvidia wheel families consistent.
   - MPS tier: `torch.backends.mps.is_available()`.
6. ONNX Runtime and its providers per tier.
7. TensorRT / CuPy imports (CUDA tier).
8. CoreML (MPS).
9. SAM2 import.
10. SAM3 inference packages (ultralytics + clip + ftfy).
11. SAM3 checkpoint state: present / missing / gated-needs-login. A **warn** with the exact `hf auth login` and licence URL fix.
12. AprilTag import (warn).
13. SLEAP env and SAM3 training sidecar: info/warn only, never fail.

## 5. SAM3 training sidecar (opt-in, CUDA only)

**Installer `--with-sam3-train`**
- Refuses unless `tier == cuda`. The exit code is nonzero and the message is explicit.
- The torch index follows the same resolved CUDA major. This replaces `tools/setup_sam3_train_env.sh`, which is deleted, and `make setup-sam3-train` becomes a wrapper.

**hydra-suite inside the sidecar, without a repo checkout.** The installer reads the main env's `hydra-suite` `direct_url.json` and installs the **same source** with `--no-deps`:
- editable dir → `-e dir`
- git URL → same URL@commit
- wheel → same wheel

The sidecar must match the main env's code (sentinel protocol), so it is never PyPI. The availability probe gains a skew check: sidecar `hydra_suite` origin + version vs the main env's. A mismatch is reported as a reason with a re-run command.

**Messages and gates**
- The `availability.py` install hint points to the user guide section and `python install.py --with-sam3-train`.
- Precision: `SUPPORTED_PRECISIONS` is narrowed to `("bf16",)` so it matches the CLI's actual rejection of fp32.
- The GUI text "~32 GB" is replaced by the measured 12 GiB, sourced from the preflight constant.
- On a non-CUDA host the training panel says "SAM3 training requires an NVIDIA GPU (CUDA)". When the sidecar is missing, it says "not installed — run …". Inference is unaffected in both cases.

## 6. SLEAP sidecar

- `--with-sleap` creates the `sleap` conda env with `sleap==1.6.2` and the torch index matching the resolved CUDA (cu128/cu130, cpu, mps). `make setup-sleap` wraps it.
- At runtime the SLEAP service checks the `sleap-nn` version in the child. Anything other than `0.1.x` raises a clear error naming the pin.

## 7. Tests and CI

**Unit tests (no env needed)**
- Installer tier/CUDA mapping with mocked `nvidia-smi` (570 → 12, 580 → 13, 595 → 13, absent → cpu, override warn).
- Plan building per tier/OS (dry-run golden), Windows path layout, sidecar refusal on non-CUDA.
- Source resolution from `direct_url.json`.

**`test_dependencies_declared.py`**
- AST-scan top-level imports in `src/`.
- Map each import to a distribution with `packages_distributions()`, plus a static fallback map for when a package is not installed.
- Assert every third-party import is declared in core or an extra.
- Allowlist the intentionally external modules: `sleap`, `sleap_nn`, `sam3`, `clip`, `apriltag`, `tensorrt`, `cupy`, `coremltools`, `PyNvVideoCodec`, `decord`.

**`test_no_requirements_files.py`**

**GitHub Actions `install-smoke.yml`**
- Matrix: `ubuntu-latest`, `macos-14`, `windows-latest` × Python 3.11 / 3.12 / 3.13.
- Steps: `python install.py --target current --cuda none --skip-doctor` (apriltag included, to prove Windows builds) → `hydra doctor --json` → headless import of all kits → `pytest` on the installer, declared-dependency and doctor tests plus a fast non-GPU subset.
- Ubuntu needs apt `libegl1 libxkbcommon0 libgl1 libdbus-1-3`; `QT_QPA_PLATFORM=offscreen` is set.

## 8. Equivalence: proving install changes don't move tracking output

`run_matrix.sh` compares two `src` trees in one env. This change instead swaps the **env** under the same src.

**Protocol**
1. **Baseline.** In the existing env (`hydra-mps`; `hydra-cuda` on mehek), run `tools/equivalence/runner.py` for `fly_obb` and `worm_bgsub` with branch-base src. Write the output to `OUT_OLD`. The env is never rebuilt in place.
2. **Candidate.** Build a fresh env under a **new name** (`hydra-mps-ssot`; `hydra-cuda-ssot` on mehek) with the installer. Run the same clips with the same src into `OUT_NEW`.
3. **Compare.** Use `tools/equivalence/compare.py` across `OUT_OLD`/`OUT_NEW` for `_forward.csv` and `_tracking_final.csv`. Assert row counts > 1.

**Known risk.** `cv2` decoding changes from a conda/pip mixture to pip `opencv-python-headless` (a different bundled FFmpeg). If pixels differ, the CSVs will differ. That is a genuine decision for the user, not something to paper over.

## 9. Make wrappers

- `BOOT_PY ?= python3`.
- `setup`/`setup-mps`/`setup-cuda` → `install.py --target conda --tier X --create-only`.
- `install`/`install-mps`/`install-cuda` → `install.py --target current --tier X [--cuda $(CUDA_MAJOR)]`. `CUDA_MAJOR` has no default (auto).
- `env-update*` → `--target conda --tier X --update`.
- `install-dev`/`docs-install` → `--target current --dev` / `--docs` (extras only).
- `install-apriltag-fork`/`install-sam3-clip`/`configure-mps-libs`/`configure-cuda-ort` stay as names and call the matching installer `--only <step>`. `configure-cuda-ort` becomes a no-op that prints the doctor result.
- `setup-sam3-train` → `--with-sam3-train --only sam3-train`; `setup-sleap` is new.

## 10. Rollout order

1. Installer pure logic + unit tests.
2. CI workflow.
3. **Ask the user to push** (Windows evidence).
4. pyproject/yml/requirements.
5. `hydra doctor`.
6. Fresh MPS env + equivalence.
7. Runtime changes.
8. SAM items.
9. Docs.
10. mehek/diptera gates.
11. Adversarial review.
12. Merge.
