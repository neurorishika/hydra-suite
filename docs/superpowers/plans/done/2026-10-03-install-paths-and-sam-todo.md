# Install paths + SAM2/SAM3 availability — to-do checklist

**Status:** Shipped — merged to local main (feat/install-ssot) (design: `docs/superpowers/specs/2026-10-03-install-ssot-design.md`). Remaining deferrals are marked inline.
It comes from two read-only audits. Items marked (verified) were confirmed locally; the rest came from reading code and need a run to confirm.

## Decisions (agreed)

- **Single source of truth (SSOT):** `pyproject.toml` extras are the only place Python dependencies are declared.
  - A cross-platform Python installer (no bash) does the work.
  - `make` targets become thin wrappers around it, so existing users see no change.
  - Conda only provides what pip can't: Python, git, cmake, compilers, and the sidecar envs (conda ffmpeg later dropped: it broke Qt on Rocky 9; imageio-ffmpeg supplies the binary).
  - The `requirements*.txt` files are deleted or generated from pyproject.
- **Windows:** fully supported. Every installer path and runtime feature must work on it and be tested in CI.
- **CUDA version:** auto-detected from the driver (≥580 → cu130, otherwise cu128). The user can override with `--cuda 12|13|none` / `CUDA_MAJOR=`.
- **SAM inference** (SAM2 + SAM3 escalation, base and published checkpoints) installs by default on every device: CPU, MPS, CUDA, Windows.
- **SAM3 training:** a separate opt-in (`--with-sam3-train` / `make setup-sam3-train`), CUDA only. It refuses on non-CUDA hosts with a clear message, and never installs implicitly.

## Phase 1 — Installer + pyproject as SSOT

- [x] Write a design spec: installer CLI surface, extras layout, conda's reduced role, migration for existing `make` users.
- [x] Create a cross-platform installer, e.g. `python -m hydra_suite.install` or `tools/install.py`, with these flags:
  - platform detection
  - `--cuda auto|12|13|none`
  - `--with-sam3-train`
  - `--dev`
  - `--docs`
  - `--dry-run`, which prints the resolved plan
- [x] Driver detection uses `nvidia-smi` and reuses `_driver_max_cuda()` (verify_cuda_runtime.py:161). It picks the matching torch index URL (cu128 / cu130 / cpu).
- [x] Restructure the extras:
  - core
  - `[cpu]`
  - `[mps]`
  - `[cuda12]` / `[cuda13]`
  - `[sam]`, the inference extra included by default: SAM2, ultralytics SAM3, pinned CLIP
  - `[dev]`
  - `[docs]`

  The `[sam3-train]` extra stays out of the main env, because it is sidecar-only.
- [x] Rewire the `make setup*` / `install*` / `env-update*` targets to call the installer, keeping target names and `CUDA_MAJOR` passthrough.
- [x] Shrink the conda ymls to Python + ffmpeg + compilers. Drop the duplicate numpy/opencv/PySide6/numba, and drop `pyqtwebengine` (unused, pulls PyQt5) (verified).
- [x] Delete `requirements*.txt`, or generate them from pyproject and add a CI check that they match.
- [x] Add a packaged `hydra doctor` built from `verify_cuda_runtime.py`. It must work without `CONDA_PREFIX`, on Windows, and on aarch64.

## Phase 2 — Dependency correctness

- [x] Fix the opencv conflict: pyproject uses `opencv-python-headless`, ultralytics uses `opencv-python`, and conda adds its own `opencv`. All three write to `cv2/` (verified).
- [x] Fix the duplicate PySide6 (conda 6.10.2 + pip 6.11.0) (verified). Remove the `--upgrade` that overwrites conda-owned packages.
- [x] Add `optuna`, `optunahub`, `umap-learn` and `pyyaml` to pyproject. Pip users crash when opening the parameter helper (verified via grep).
- [x] Declare `torch`/`torchvision` explicitly. Fix the pyproject:107 comment.
- [x] CPU path: CPU torch on Linux and Windows (cpu index URL), plus `onnxruntime`.
- [x] MPS: `coremltools` is in the extra but missing from `make install-mps` (verified). Keep `torchvision>=0.16`.
- [x] Drop `torchaudio`. It is never imported and could pull a newer torch over cu128.
- [x] CUDA extras: add `PyNvVideoCodec`. Reconcile `onnxruntime-gpu` pins. Resolve the cupy-cuda13x `--pre` / custom-index handling.
- [x] Reconcile the duplicated pins: hnswlib/usearch/annoy appear in three places.

## Phase 3 — Windows + platform tooling

- [x] Replace the bash-only logic (`$prefix/bin/python` at makefile:71/101, `uname`, hard-coded `x86_64-linux`, Linux `.so` names) with installer code that handles Windows paths.
- [x] AprilTag fork: the installer builds it statically (all OSes, incl. Windows MSVC in CI) and fails loudly naming the missing compiler/cmake; `--skip-apriltag` escapes. (Prebuilt wheels not pursued.) Document the prerequisites (MSVC Build Tools on Windows; Xcode CLT plus licence on macOS).
- [x] MPS: the installer always applies the `configure-mps-libs` libomp fix. Today `env-update-mps` skips it.
- [x] Pin the CLIP git ref. Replace the silent `|| true` resets with loud failures.
- [x] SLEAP sidecar:
  - Add an installer step / `make setup-sleap` that uses the same CUDA auto-detect and pins `sleap==1.6.2`.
  - Add a runtime sleap-nn version check.
  - Make Windows work.
  - Fix the Python 3.11 vs 3.13 contradiction in the recipe.
- [x] Windows runtime audit: POSIX-only calls are guarded; fixed the batch fan-out (no SIGKILL on Windows, CTRL_BREAK hit the whole console group) and SAM3 trainer `resource` import. Host containment stays watchdog-only on Windows (accepted); orphaned grandchildren of an already-dead fan-out child cannot be reaped on Windows (documented limitation).

## Phase 4 — SAM inference on every device

- [x] `[sam]` is in every default install. Checkpoints are fetched at first use.
- [x] Smoke-test SAM2 escalation: MPS + CPU-mac (real frames, CLI), CUDA13 mehek. CPU-Linux/Windows CPU via CI imports only; CUDA12 (diptera) doctor-only; Windows CUDA deferred with the box.
- [x] Smoke-test SAM3 escalation (base checkpoint): MPS, CPU-mac, CUDA13 all ran real frames (3 flies/frame). `PYTORCH_ENABLE_MPS_FALLBACK` not needed. Published-checkpoint path unchanged and unexercised (no published checkpoint exists).
- [x] Add a device selector to SAM2/SAM3 escalation. The SAM2 job passes no device today.
- [x] Make SAM2 `ensure_checkpoint` stream the file instead of loading it whole into RAM (sam2/checkpoints.py:77). Mirror the SAM3 fix.
- [x] Add a headless CLI for escalation (e.g. `detectkit escalate --sam2|--sam3`). Today it is GUI-only.
- [x] Gated SAM3 checkpoint: add a clear in-app prompt for HF licence + `hf auth login` when the download is refused.

## Phase 5 — SAM3 training: optional, CUDA only

- [x] Make the installer `--with-sam3-train` / `make setup-sam3-train` refuse on non-CUDA hosts with an explicit message. Remove the `cpu|mps` options from `setup_sam3_train_env.sh`, or port the script into the installer.
- [x] Use the same CUDA auto-detect / override for the sidecar. Today the script defaults to cuda12 independently.
- [ ] **Deferred (no Windows GPU box yet):** Support Windows + CUDA SAM3 training. install.py adds `triton-windows` to the sidecar on Windows; that `sam3` imports and trains there is unverified.
- [x] Remove the need for a repo checkout: the sidecar installs the SAME hydra-suite source as the main env (from its direct_url.json: editable dir / git commit / wheel), and the availability probe flags source skew.
- [x] GUI: when the sidecar isn't installed, the training panel says "SAM3 training not installed — run …". On non-CUDA it says "requires an NVIDIA GPU". Inference stays unaffected.
- [x] Point the `availability.py:41-44` hint at the user guide, not an internal spec.
- [x] Resolve `SUPPORTED_PRECISIONS` vs the bf16-only runtime gate: narrowed preflight to bf16 (fp32 never verified end-to-end; ~58 GiB).
- [x] Fix the GUI "~32 GB" text (sam3_training_panel.py:366) to match the 12 GiB the code measures.

## Phase 6 — Documentation

- [x] Rewrite `installation.md` as one decision tree. It covers:
  - the platform choice
  - the one installer command, or the `make` equivalent
  - optional add-ons: SAM3 training, SLEAP, dev
  - the `hydra doctor` verification step

  Add a Windows section.
- [x] Add a new **SAM2 / SAM3** user page covering:
  - what's installed by default
  - checkpoints + HF licence/login
  - the device support table (inference everywhere, training CUDA only)
  - expected speed on MPS/CPU
  - GUI and CLI usage
  - the opt-in training install
- [x] Add a SAM2 section to `detectkit.md`. SAM2 is currently only in the developer guide.
- [x] Fix `detectkit-semantic-escalation.md`:
  - the "runtime probe / MPS would work with no change" claim
  - 32 GB vs 12 GiB
  - "sidecar never re-checks VRAM"
  - "public" vs gated checkpoint
  - "escalation-only machines need none of this"
- [x] `platforms.md`: add a feature × device matrix, replacing "CPU supports all workflows".
- [x] Fix `environments.md` / `installation.md` to describe the new SSOT model. Fix `mat` → `trackerkit` and the "fresh reinstall" step.
- [x] Rewrite `compute-runtimes.md` for Runtime Gen-2 tiers. Reconcile the ORT-on-MPS claims.
- [x] Update CLAUDE.md / README / AGENTS.md: the new install commands, quote `"hydra-suite[...]"`, and the CUDA auto-detect.
- [x] Fix the `integrations.md` SLEAP troubleshooting section: add the cu128 / `nvidia-nccl-cu12` branch.

## Phase 7 — Prove it works

- [x] GitHub Actions smoke matrix:
  - `ubuntu` / `macos-14` (arm64 MPS-capable) / `windows`
  - × Python 3.11 / 3.12 / 3.13
  - run the installer (`--cuda none`), import every kit, run `hydra doctor`, run the non-GPU pytest subset

  Include the dependency-declared tests.
- [x] CI: a conda+make job on ubuntu and macos to confirm the wrappers still work.
- [x] CI: installer `--dry-run` tests for the driver → CUDA mapping (mocked `nvidia-smi`, including driver 570 → cu128 and 595 → cu130).
- [x] CI: a consistency test that every third-party import in `src/` is declared in a pyproject extra.
- [x] Real-model SAM2/SAM3 inference smoke tests (`pytest -m slow tests/test_sam_real_models.py`, skip-if-no-weights): pass on MPS + CPU locally. Not in CI (weights not cached there; SAM3 is gated).
- [x] Manual GPU gate (no GPU in CI):
  - fresh install on mehek (CUDA13, driver 595) and on diptera (CUDA12, driver 570, named idle GPU)
  - on each: `hydra doctor`, SAM2/SAM3 escalation, SAM3 training opt-in install + a short training run, and a fly_obb equivalence run
- [ ] **Deferred (user will provide a box):** Manual Windows + CUDA gate.
- [x] Publish the final install-path × platform matrix (documented / CI / verified) in the docs.
