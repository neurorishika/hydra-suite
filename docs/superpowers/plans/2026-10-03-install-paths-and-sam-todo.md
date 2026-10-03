# Install paths + SAM2/SAM3 availability — to-do checklist

**Status:** Checklist with decisions agreed (2026-10-03). Nothing implemented yet. Next step: design spec → worktree.
It comes from two read-only audits. Items marked (verified) were confirmed locally; the rest came from reading code and need a run to confirm.

## Decisions (agreed)

- **Single source of truth (SSOT):** `pyproject.toml` extras are the only place Python dependencies are declared.
  - A cross-platform Python installer (no bash) does the work.
  - `make` targets become thin wrappers around it, so existing users see no change.
  - Conda only provides what pip can't: Python, ffmpeg, compilers, and the sidecar envs.
  - The `requirements*.txt` files are deleted or generated from pyproject.
- **Windows:** fully supported. Every installer path and runtime feature must work on it and be tested in CI.
- **CUDA version:** auto-detected from the driver (≥580 → cu130, otherwise cu128). The user can override with `--cuda 12|13|none` / `CUDA_MAJOR=`.
- **SAM inference** (SAM2 + SAM3 escalation, base and published checkpoints) installs by default on every device: CPU, MPS, CUDA, Windows.
- **SAM3 training:** a separate opt-in (`--with-sam3-train` / `make setup-sam3-train`), CUDA only. It refuses on non-CUDA hosts with a clear message, and never installs implicitly.

## Phase 1 — Installer + pyproject as SSOT

- [ ] Write a design spec: installer CLI surface, extras layout, conda's reduced role, migration for existing `make` users.
- [ ] Create a cross-platform installer, e.g. `python -m hydra_suite.install` or `tools/install.py`, with these flags:
  - platform detection
  - `--cuda auto|12|13|none`
  - `--with-sam3-train`
  - `--dev`
  - `--docs`
  - `--dry-run`, which prints the resolved plan
- [ ] Driver detection uses `nvidia-smi` and reuses `_driver_max_cuda()` (verify_cuda_runtime.py:161). It picks the matching torch index URL (cu128 / cu130 / cpu).
- [ ] Restructure the extras:
  - core
  - `[cpu]`
  - `[mps]`
  - `[cuda12]` / `[cuda13]`
  - `[sam]`, the inference extra included by default: SAM2, ultralytics SAM3, pinned CLIP
  - `[dev]`
  - `[docs]`

  The `[sam3-train]` extra stays out of the main env, because it is sidecar-only.
- [ ] Rewire the `make setup*` / `install*` / `env-update*` targets to call the installer, keeping target names and `CUDA_MAJOR` passthrough.
- [ ] Shrink the conda ymls to Python + ffmpeg + compilers. Drop the duplicate numpy/opencv/PySide6/numba, and drop `pyqtwebengine` (unused, pulls PyQt5) (verified).
- [ ] Delete `requirements*.txt`, or generate them from pyproject and add a CI check that they match.
- [ ] Add a packaged `hydra doctor` built from `verify_cuda_runtime.py`. It must work without `CONDA_PREFIX`, on Windows, and on aarch64.

## Phase 2 — Dependency correctness

- [ ] Fix the opencv conflict: pyproject uses `opencv-python-headless`, ultralytics uses `opencv-python`, and conda adds its own `opencv`. All three write to `cv2/` (verified).
- [ ] Fix the duplicate PySide6 (conda 6.10.2 + pip 6.11.0) (verified). Remove the `--upgrade` that overwrites conda-owned packages.
- [ ] Add `optuna`, `optunahub`, `umap-learn` and `pyyaml` to pyproject. Pip users crash when opening the parameter helper (verified via grep).
- [ ] Declare `torch`/`torchvision` explicitly. Fix the pyproject:107 comment.
- [ ] CPU path: CPU torch on Linux and Windows (cpu index URL), plus `onnxruntime`.
- [ ] MPS: `coremltools` is in the extra but missing from `make install-mps` (verified). Keep `torchvision>=0.16`.
- [ ] Drop `torchaudio`. It is never imported and could pull a newer torch over cu128.
- [ ] CUDA extras: add `PyNvVideoCodec`. Reconcile `onnxruntime-gpu` pins. Resolve the cupy-cuda13x `--pre` / custom-index handling.
- [ ] Reconcile the duplicated pins: hnswlib/usearch/annoy appear in three places.

## Phase 3 — Windows + platform tooling

- [ ] Replace the bash-only logic (`$prefix/bin/python` at makefile:71/101, `uname`, hard-coded `x86_64-linux`, Linux `.so` names) with installer code that handles Windows paths.
- [ ] AprilTag fork: provide a prebuilt wheel per platform (including Windows), or give the installer a build step that checks for a compiler and fails loudly. Document the prerequisites (MSVC Build Tools on Windows; Xcode CLT plus licence on macOS).
- [ ] MPS: the installer always applies the `configure-mps-libs` libomp fix. Today `env-update-mps` skips it.
- [ ] Pin the CLIP git ref. Replace the silent `|| true` resets with loud failures.
- [ ] SLEAP sidecar:
  - Add an installer step / `make setup-sleap` that uses the same CUDA auto-detect and pins `sleap==1.6.2`.
  - Add a runtime sleap-nn version check.
  - Make Windows work.
  - Fix the Python 3.11 vs 3.13 contradiction in the recipe.
- [ ] Windows runtime audit: `conda run` (already Windows-aware), the subprocess/sidecar launchers, and host containment, which falls back to watchdog-only (resource_limits.py:124-127). Confirm that fallback is acceptable.

## Phase 4 — SAM inference on every device

- [ ] `[sam]` is in every default install. Checkpoints are fetched at first use.
- [ ] Smoke-test SAM2 escalation on CPU-Linux, CPU-mac, MPS, CUDA12, CUDA13 and Windows (CPU + CUDA).
- [ ] Smoke-test SAM3 escalation (base and published checkpoints) on the same matrix. It has never run on MPS or CPU. Decide whether `PYTORCH_ENABLE_MPS_FALLBACK` is needed.
- [ ] Add a device selector to SAM2/SAM3 escalation. The SAM2 job passes no device today.
- [ ] Make SAM2 `ensure_checkpoint` stream the file instead of loading it whole into RAM (sam2/checkpoints.py:77). Mirror the SAM3 fix.
- [ ] Add a headless CLI for escalation (e.g. `detectkit escalate --sam2|--sam3`). Today it is GUI-only.
- [ ] Gated SAM3 checkpoint: add a clear in-app prompt for HF licence + `hf auth login` when the download is refused.

## Phase 5 — SAM3 training: optional, CUDA only

- [ ] Make the installer `--with-sam3-train` / `make setup-sam3-train` refuse on non-CUDA hosts with an explicit message. Remove the `cpu|mps` options from `setup_sam3_train_env.sh`, or port the script into the installer.
- [ ] Use the same CUDA auto-detect / override for the sidecar. Today the script defaults to cuda12 independently.
- [ ] Support Windows + CUDA: triton on Windows (`triton-windows`) or the triton-free import path. Verify that `sam3` imports and trains.
- [ ] Remove the need for a repo checkout (`pip install --no-deps -e $REPO_ROOT`), so it works from a pip install of hydra-suite.
- [ ] GUI: when the sidecar isn't installed, the training panel says "SAM3 training not installed — run …". On non-CUDA it says "requires an NVIDIA GPU". Inference stays unaffected.
- [ ] Point the `availability.py:41-44` hint at the user guide, not an internal spec.
- [ ] Resolve `SUPPORTED_PRECISIONS=("bf16","fp32")` (preflight.py:140) vs the bf16-only rejection at cli.py:989.
- [ ] Fix the GUI "~32 GB" text (sam3_training_panel.py:366) to match the 12 GiB the code measures.

## Phase 6 — Documentation

- [ ] Rewrite `installation.md` as one decision tree. It covers:
  - the platform choice
  - the one installer command, or the `make` equivalent
  - optional add-ons: SAM3 training, SLEAP, dev
  - the `hydra doctor` verification step

  Add a Windows section.
- [ ] Add a new **SAM2 / SAM3** user page covering:
  - what's installed by default
  - checkpoints + HF licence/login
  - the device support table (inference everywhere, training CUDA only)
  - expected speed on MPS/CPU
  - GUI and CLI usage
  - the opt-in training install
- [ ] Add a SAM2 section to `detectkit.md`. SAM2 is currently only in the developer guide.
- [ ] Fix `detectkit-semantic-escalation.md`:
  - the "runtime probe / MPS would work with no change" claim
  - 32 GB vs 12 GiB
  - "sidecar never re-checks VRAM"
  - "public" vs gated checkpoint
  - "escalation-only machines need none of this"
- [ ] `platforms.md`: add a feature × device matrix, replacing "CPU supports all workflows".
- [ ] Fix `environments.md` / `installation.md` to describe the new SSOT model. Fix `mat` → `trackerkit` and the "fresh reinstall" step.
- [ ] Rewrite `compute-runtimes.md` for Runtime Gen-2 tiers. Reconcile the ORT-on-MPS claims.
- [ ] Update CLAUDE.md / README / AGENTS.md: the new install commands, quote `"hydra-suite[...]"`, and the CUDA auto-detect.
- [ ] Fix the `integrations.md` SLEAP troubleshooting section: add the cu128 / `nvidia-nccl-cu12` branch.

## Phase 7 — Prove it works

- [ ] GitHub Actions smoke matrix:
  - `ubuntu` / `macos-14` (arm64 MPS-capable) / `windows`
  - × Python 3.11 / 3.12 / 3.13
  - run the installer (`--cuda none`), import every kit, run `hydra doctor`, run the non-GPU pytest subset

  Include the dependency-declared tests.
- [ ] CI: a conda+make job on ubuntu and macos to confirm the wrappers still work.
- [ ] CI: installer `--dry-run` tests for the driver → CUDA mapping (mocked `nvidia-smi`, including driver 570 → cu128 and 595 → cu130).
- [ ] CI: a consistency test that every third-party import in `src/` is declared in a pyproject extra.
- [ ] Real-model SAM2/SAM3 inference smoke tests (marked, skip-if-no-weights). Run them locally on MPS, and on CPU in CI if the weights can be cached.
- [ ] Manual GPU gate (no GPU in CI):
  - fresh install on mehek (CUDA13, driver 595) and on diptera (CUDA12, driver 570, named idle GPU)
  - on each: `hydra doctor`, SAM2/SAM3 escalation, SAM3 training opt-in install + a short training run, and a fly_obb equivalence run
- [ ] Manual Windows + CUDA gate. A box is needed; identify one.
- [ ] Publish the final install-path × platform matrix (documented / CI / verified) in the docs.
