# SAM3 Training in a Dedicated Sidecar Environment — Design

**Status:** approved, extends `2026-08-31-detectkit-sam3-finetune-design.md` on the same branch.

## Why

DetectKit and the rest of HYDRA run in `hydra-mps` / `hydra-cuda`. Meta's `sam3`
cannot coexist there: it pins `numpy>=1.26,<2`, while those environments run
numpy 2.x (measured: `hydra-mps` 2.3.5, `hydra-cuda` 2.4.3). Installing the
training extra into either one silently downgrades numpy for every other
package in it.

This is not a hypothetical. The spike that produced this feature's evidence
never ran in `hydra-cuda` — it ran in a dedicated `sam3-lora` env on mehek
(python 3.12.14, numpy 1.26.4, `sam3`/`decord`/`iopath`/`timm`/`ftfy` present).
The sidecar split is therefore not a new architecture; it is making the
arrangement that already worked explicit and supported.

The repo has the precedent: `integrations/sleap/service.py` spawns a
user-selected conda env through `utils/conda_utils.py`
(`popen_conda`/`run_conda`, which handle conda's Windows batch entry point),
and preflights it with actionable repair commands. SAM3 follows that pattern
rather than inventing one.

## Scope

**In:** environment resolution and probing, the subprocess launcher, the
in-env CLI entry point, progress/cancellation across the process boundary,
GUI surface for choosing the env, and the documented env recipe.

**Out:** changing what training *does*. The trainer's algorithm, the
dataset builder, the publisher, and the load guard are unchanged. The
sidecar only moves where `train_sam3_lora`'s body executes.

## Platform decision (measured, not assumed)

Training runs wherever the sidecar env can import `sam3`. Concretely today:

- **CUDA (mehek): works.** `sam3-lora` env, RTX 6000 Ada 48 GB.
- **MPS: blocked by an upstream packaging bug, NOT by memory.**
  `sam3/__init__.py:5` → `model_builder.py:40` → `sam1_task_predictor.py:16`
  → `sam3_tracker_base.py:10` → `sam3_tracker_utils.py:9` → `edt.py:8`
  → `import triton`, at module scope with no `try/except`. `triton` ships no
  macOS wheel, so `import sam3` fails on any Mac. The kernel it guards is the
  **video tracker's** EDT; the image-training path never calls it.
  Unified memory is ample (128 GB on this box vs ~29 GB measured peak), so if
  that import becomes optional — upstream fix or a vendored patch — MPS works
  with no change to anything below.

Therefore: **the plumbing is platform-agnostic and the gate is a runtime
probe**, never a hardcoded `if platform == "Darwin"`. A stubbed `triton` was
tried and rejected: it needs an open-ended slice of a compiler's API surface
(`triton.language.dtype` and onward), and faking a module the vendor imports
is the same silent-substitution class this programme refused for the loss and
matcher.

## Architecture

### 1. Environment resolution

`training/sam3_lora/env.py` (Qt-free, no `sam3` import):

- `DEFAULT_SAM3_ENV = "hydra-sam3"`.
- `resolve_sam3_env(configured: str | None) -> str` — configured value, else
  `HYDRA_SAM3_ENV`, else the default.
- `sam3_env_command(env, module_args) -> list[str]` — builds
  `["conda", "run", "-n", env, "python", "-m", ...]`.
- `sam3_env_environ() -> dict[str, str]` — the environment overrides the
  child needs, including **`KMP_DUPLICATE_LIB_OK=TRUE`**. Without it a bare
  `import torch` aborts with `OMP Error #15` (observed while building the mac
  env; `tools/equivalence/run_matrix.sh` sets it for the same reason).

### 2. Probe inversion

Task 4's `probe_sam3_training_availability` currently asks "is `sam3`
importable **here**". It becomes "can the sidecar env import what training
needs", by running a short probe script in the child and parsing its JSON.

The probe must **report the child's real failure text**. On this Mac the user
should read `no module named 'triton'`, not a generic "unavailable" — the
whole value of the probe is that it names the thing to fix. `run_conda` gets a
timeout so a broken env cannot hang the GUI.

`TRAINING_PACKAGES` stays as the in-env checklist, now evaluated in the child.

### 3. Launcher and CLI

`train_sam3_lora(spec, run_dir, ...)` keeps its signature and return contract
(`success`, `artifact_path`, `metrics_path`, `canceled`) so Task 6's dispatch
and Task 9b's publish path are untouched. Its body becomes:

1. `preflight(spec)` — unchanged, still refuses without
   `label_quality_acknowledged`.
2. Serialise the spec to `run_dir/spec.json` (`TrainingRunSpec.to_dict`
   already recurses into `Sam3LoraParams` via `asdict`).
3. `popen_conda(sam3_env_command(env, ["hydra_suite.training.sam3_lora.cli",
   "--spec", ..., "--run-dir", ...]), env=..., stdout=PIPE, text=True)`.
4. Stream stdout; each line is either a JSON progress record (forwarded to
   `log_cb`/`progress_cb`) or plain log text.
5. `should_cancel()` → terminate the child, then kill on timeout; return
   `canceled: True`.
6. Child exit non-zero → `success: False` with the child's stderr tail as
   `error_message`. **Never** synthesise success from a failed child.

`training/sam3_lora/cli.py` runs *inside* the sidecar env: it loads the spec,
calls the existing training loop, writes `adapters.pt`, and emits progress
records. It imports `sam3` and torch; it imports no Qt and nothing from an app
layer. It needs `hydra_suite` importable in the child — the env recipe
installs the package (or `PYTHONPATH` covers a worktree).

### 4. GUI surface

The SAM3 panel gains an env row: a line edit defaulting to `hydra-sam3`, the
probe's status, and its reason text when unusable. Mirrors how SLEAP exposes
its env. `Sam3LoraParams` gains `env_name: str = ""` (empty = resolve the
default), so the choice travels with the spec and is recorded in the run.

### 5. Env recipe (documented, verified on macOS)

```
conda create -n hydra-sam3 python=3.12 'numpy<2'
conda run -n hydra-sam3 pip install torch torchvision
conda run -n hydra-sam3 pip install 'setuptools<81'
conda run -n hydra-sam3 pip install einops torchmetrics scipy decord iopath \
    opencv-python-headless pillow platformdirs
conda run -n hydra-sam3 pip install git+https://github.com/facebookresearch/sam3.git
conda run -n hydra-sam3 pip install -e /path/to/hydra-suite
```

Two pins are non-obvious and were found the hard way:

- **`setuptools<81`** — setuptools 81 removed `pkg_resources`, which
  `sam3/model_builder.py:8` imports at module scope.
- **`einops`** — imported by `sam3/sam/rope.py` but absent from sam3's
  declared dependencies.

## Testing

- `env.py` is pure string/dict construction — unit-tested with no subprocess.
- The probe is tested with a faked `run_conda` returning canned JSON for
  usable, unusable-with-reason, timeout, and malformed-output.
- The launcher is tested with a faked `popen_conda`: progress forwarding,
  cancellation terminating the child, non-zero exit producing
  `success: False`, and **a child that exits 0 without writing `adapters.pt`
  must still fail** — the fake-success discipline from Task 8 applies across
  the process boundary too.
- No test requires `sam3`, conda, or a GPU.

## Acceptance

1. On this Mac, the panel shows the role unavailable with the child's real
   reason naming `triton`.
2. On mehek with `-n sam3-lora`, the probe reports usable.
3. A real finetune runs end to end in the sidecar and publishes a checkpoint
   the guard accepts — the Task 12 gate, now run through the sidecar.
4. `hydra-mps` / `hydra-cuda` numpy versions are unchanged by any of it.

## Risks

- **conda absent from PATH** — the SLEAP path already fails loudly here; do
  the same, with the env name in the message.
- **Child stdout interleaving** — progress records are single-line JSON with a
  sentinel prefix so partial writes cannot be mistaken for progress.
- **Orphaned children** on a hard GUI kill: terminate in a `finally`.
