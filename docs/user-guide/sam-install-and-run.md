# SAM2 and SAM3: install and run

DetectKit uses two Segment Anything models to upgrade and extend your labels:

- **SAM2 — geometry escalation** ("Escalate to segment (SAM2)"): turns the
  boxes (AABB/OBB) you already have into polygon masks. It cannot add an
  animal that was never labelled.
- **SAM3 — semantic escalation** ("Semantic escalation (SAM3)"): takes a text
  prompt such as `ant` and finds instances of that concept across a source,
  including animals missing from your labels.

Both produce a **staged review**: nothing touches your labels until you
accept frames in DetectKit's review bar. This page covers what gets
installed, where the weights come from, which devices each model runs on, how
to run them from the GUI or the command line, and the optional SAM3
**training** sidecar. For SAM3 workflow details (prompting, tiling,
calibration, review) see
[DetectKit: Semantic Escalation (SAM3)](detectkit-semantic-escalation.md).

## What is installed by default

`python install.py` installs SAM2 and SAM3 **inference** on every tier
(CPU, Apple MPS, CUDA) — there is nothing extra to choose:

| Component | Comes from | Notes |
|---|---|---|
| `sam2` | core dependency in `pyproject.toml` | SAM2 inference |
| `ultralytics` (SAM3 predictor), `ftfy`, `regex` | core dependency + the `sam` extra | SAM3 inference; the `sam` extra is installed on every tier (`sam3` is an alias for it) |
| CLIP text encoder | installer step `clip` | The **ultralytics fork** of CLIP, pinned to a commit. It is git-only, so it cannot be a pyproject dependency; `install.py` installs it. |

SAM3 **training** is not installed by default — see
[SAM3 training](#sam3-training-optional-cuda-only).

Check the inference install with `hydra doctor`: the `sam2` and
`sam3 inference` lines should be `[ OK ]`.

## Model weights

| Model | Source | Size | Login needed? | Stored in |
|---|---|---|---|---|
| SAM2 (`sam2.1-hiera-tiny`, `-small`, `-base_plus` (default), `-large`) | `facebook/sam2.1-*` on Hugging Face | varies by variant | No — not gated | `<models dir>/sam2/` |
| SAM3 stock (`sam3`) | `facebook/sam3` on Hugging Face | ~3.45 GB | **Yes — licence-gated** | `<models dir>/sam3/` |
| SAM3 finetuned (your published checkpoints) | produced locally by SAM3 training | ~3.2 GB each | No | `<models dir>/sam3_finetuned/` |

`<models dir>` is `hydra_suite.paths.get_models_dir()` (see
[data directories](../getting-started/installation.md#data-directories)).
Weights download on first use. DetectKit never downloads the 3.45 GB SAM3
checkpoint without telling you: the escalation dialog warns and asks first.

### One-time Hugging Face access for SAM3

`facebook/sam3` is gated by Meta. Before the **first** SAM3 run on each
machine:

1. Open <https://huggingface.co/facebook/sam3> and accept the licence with
   your Hugging Face account.
2. Authenticate on that machine:

    ```bash
    hf auth login          # or: export HF_TOKEN=hf_...
    ```

Without this, the download fails with a 401 and DetectKit (or
`detectkit escalate sam3`) prints those two steps. Once the checkpoint is
downloaded, inference with it needs no further network access. Published
**finetuned** checkpoints are local files and never need a login. SAM2
weights are not gated.

## Device support

| | CPU (Linux) | CPU (Intel Mac) | Apple MPS | CUDA 12 | CUDA 13 | Windows CPU | Windows CUDA |
|---|---|---|---|---|---|---|---|
| SAM2 inference | Yes (slow) | Yes (slow) | Yes | Yes | Yes | Yes (slow) | Yes¹ |
| SAM3 inference | Yes (very slow) | Yes (very slow) | Yes (slow) | Yes | Yes | Yes (very slow) | Yes¹ |
| SAM3 training | No | No | No | Yes² | Yes² | No | Not yet verified³ |

1. Supported; not yet verified on Windows + NVIDIA hardware.
2. Needs compute capability ≥ 8.0 with bf16 and about 12 GiB of free VRAM.
   See [Hardware](#hardware).
3. `install.py --with-sam3-train` builds the sidecar on Windows and adds
   `triton-windows`, because Meta's `sam3` imports `triton` at module scope
   and official `triton` wheels are Linux-only. This has not been run on real
   hardware yet. Report results if you try it.

!!! note "Speed on CPU and Apple Silicon"

    Both models run everywhere, but they are large. SAM2 on CPU and SAM3 on
    Apple MPS are **slow**. Expect SAM3 on MPS to take tens of seconds per
    frame, and SAM3 on CPU to be slower still. That is fine for
    a few frames or for checking a prompt. For a full project, run the
    escalation on a CUDA machine. The "Test random image" button in the SAM3
    dialog measures the per-image time **on this machine** and extrapolates it
    to the selected sources, so you can decide before starting a long run.

## Running from the GUI

Open a project in `detectkit`. Both actions live in the **Escalation** group of
the Tools panel:

- **Escalate to segment (SAM2)** — pick one or more box-level sources (sources
  already at polygon level are shown disabled) and a SAM2 variant.
- **Semantic escalation (SAM3)** — pick sources, a prompt, and a model (the
  stock `sam3` or one of your published finetuned checkpoints). Calibrate and
  "Test random image" before a long run (see the
  [SAM3 guide](detectkit-semantic-escalation.md)).

Both dialogs have a **Run on** device picker: **Auto** (CUDA if available,
then Apple MPS, then CPU), plus only the accelerators this machine actually
has. A device saved with a project that is not available on the current
machine falls back to the best available device.

When the run finishes, the result is staged and a **review bar** appears above
the canvas. Accept, add or reject frame by frame. See
[Reviewing results](detectkit-semantic-escalation.md#reviewing-results-frame-by-frame-into-the-source-you-ran-on).

## Running from the command line

`detectkit escalate` runs the same jobs headless against a saved project and
leaves the result **staged for review** exactly as the GUI does. Open the
project in DetectKit afterwards to accept or reject the staged frames.

```text
detectkit escalate sam2 --project DIR [--source NAME]... [--variant VARIANT]
                        [--device auto|cuda|mps|cpu] [--overwrite]

detectkit escalate sam3 --project DIR --prompt TEXT [--class-name CLASS]
                        [--source NAME]... [--variant sam3|<finetuned key>]
                        [--confidence 0.35] [--max-instances 0]
                        [--tile-fraction F] [--reference-body-px PX]
                        [--device auto|cuda|mps|cpu] [--overwrite]
```

| Option | Meaning |
|---|---|
| `--project` | DetectKit project directory (required). |
| `--source` | Source to escalate; **repeat** the flag for several (`--source a --source b`). Default: every eligible source. SAM2 skips sources already at polygon level. |
| `--device` | `auto` (default) picks CUDA, then MPS, then CPU. |
| `--overwrite` | Replace an existing pending (unreviewed) escalation instead of skipping it. |
| `--variant` | SAM2: `sam2.1-hiera-tiny`, `sam2.1-hiera-small`, `sam2.1-hiera-base_plus` (default), `sam2.1-hiera-large`. SAM3: `sam3` (default) or a published finetuned model key. |
| `--prompt` | SAM3 only: the noun phrase to find, e.g. `"ant"` (required). |
| `--class-name` | SAM3 only: project class the staged instances are labelled as (default: the prompt). |
| `--confidence` | SAM3 only: score threshold (default `0.35`). It can be changed later in review without re-running. |
| `--max-instances` | SAM3 only: cap per image; `0` means unlimited. |
| `--tile-fraction` | Tile size = body size / this fraction; `0` means full frame. Default: what the DetectKit dialog would open with for this variant (saved settings, then the calibration, then the model's stamped scale, then the starting guess at the project's body size). |
| `--reference-body-px` | Typical longest animal side in pixels. Default: the dialog's value, then the project's sliced-training reference, then the median of your labels. |

Examples:

```bash
# Turn every box source into polygon masks on the best available device
detectkit escalate sam2 --project ~/projects/colony_A

# Two specific sources, the large SAM2 model, on Apple Silicon
detectkit escalate sam2 --project ~/projects/colony_A \
    --source cam1 --source cam2 --variant sam2.1-hiera-large --device mps

# Find every ant in one source with SAM3 on a CUDA box
detectkit escalate sam3 --project ~/projects/colony_A \
    --prompt "ant" --source cam1 --device cuda
```

Exit status: `0` on success, `2` if SAM3 is not usable in this environment (the
message names the missing piece), `3` if the gated SAM3 download is not
authorised on this machine (follow the
[Hugging Face steps](#one-time-hugging-face-access-for-sam3)).

## SAM3 training (optional, CUDA only)

DetectKit can finetune a SAM3 LoRA adapter on your polygon labels and publish a
merged checkpoint that then appears in the SAM3 escalation dialog's model list.
Training details (dataset, defaults, publishing) are in the
[SAM3 guide](detectkit-semantic-escalation.md#finetuning-your-own-sam3-checkpoint).

Meta's `sam3` training code pins `numpy<2`, while HYDRA needs numpy 2. The two
cannot share an environment, so training runs in a separate **sidecar conda
env**, `hydra-sam3`. DetectKit starts it as a subprocess and never imports
`sam3` itself.

### Install

On a CUDA machine (conda required):

```bash
python install.py --with-sam3-train      # or: make setup-sam3-train
```

This builds `hydra-sam3` (Python 3.12, `numpy<2`) with:

- the CUDA torch build matching the main env
- `sam3` pinned to commit `660a5e9e`
- its training dependencies, pinned so nothing pulls numpy 2 back in
- the **same hydra-suite source as your main env**: an editable checkout,
  the same git commit, or the same wheel, read from the main install's
  `direct_url.json`. The sidecar talks to DetectKit over a line protocol, so it
  must never run a different version. It is never installed from PyPI.

If `hydra-sam3` already exists it is reused, and its packages and
hydra-suite source are refreshed. Remove it
(`conda env remove -n hydra-sam3`) for a clean rebuild. On a machine without
CUDA the installer refuses `--with-sam3-train` and explains that inference
needs nothing extra. In the GUI, the SAM3 training tab says the same thing.

Training also needs the **Hugging Face login** from
[above](#one-time-hugging-face-access-for-sam3) on the training machine. The
model architecture config is fetched from the gated repo at build time, even
when the checkpoint is already on disk. Preflight refuses to start when no
credential is present.

Verify:

```bash
hydra doctor --require-sam3-train
```

### Hardware

| Requirement | Value |
|---|---|
| GPU | NVIDIA, compute capability ≥ 8.0 with bf16 support |
| Precision | bf16 only. fp32/fp16 are refused. |
| VRAM | Preflight admits a run against a 12 GiB measured bf16 device peak (batch 1). A 24 GB RTX 4090 trains comfortably. |
| Free disk | ~8 GB (3.45 GB base checkpoint + ~3.2 GB merged artifact) |

These checks run twice: once in the launching process before the sidecar
starts, and again inside the sidecar before `sam3` is imported.

### Headless training

`detectkit train --config run.json` runs SAM3 (and YOLO) training without the
GUI. See [DetectKit Headless Training](../runbooks/detectkit-headless-training.md).

## Smoke-testing on real models

The test suite includes a real-model smoke test. It runs SAM2 and SAM3 on
every device available on the machine and skips when the weights are not
present:

```bash
python -m pytest -m slow tests/test_sam_real_models.py
```

It reads `tools/equivalence/fixtures/clips/fly_obb.mp4`. From a worktree
without the fixtures, set `HYDRA_EQUIV_FIXTURES` to a fixtures directory
that contains `clips/fly_obb.mp4`.

## Troubleshooting

`hydra doctor` reports the state of every piece on this page. The messages
you are most likely to see:

| Symptom | Cause | Fix |
|---|---|---|
| doctor: `sam3 checkpoint` **WARN** "not downloaded (~3.45 GB, licence-gated)" | Normal before the first SAM3 run | Accept the licence and `hf auth login`; DetectKit downloads it on first use. |
| `401` / "The SAM3 weights are gated by Meta" | Licence not accepted, or no token on this machine | Both [Hugging Face steps](#one-time-hugging-face-access-for-sam3). A token from an account that has **not** accepted the licence still gets a 401. |
| "The installed 'clip' package is the wrong fork" | `openai/CLIP` is installed. SAM3 needs a tokenizer you can call, and only the ultralytics fork provides one. | `python install.py --only clip` |
| doctor: `sam3 inference` **FAIL** naming a missing package | `clip`, `ftfy` or `ultralytics` missing | `python install.py --only clip`, or re-run `python install.py` |
| doctor: `sam2` **FAIL** | `sam2` not importable | `python install.py --recreate` |
| SAM2 weight download fails with `401` (SAM2 is not gated) | A stale or invalid Hugging Face token is stored on this machine. Hugging Face rejects even public downloads that carry a bad token. | Re-run `hf auth login` with a valid token, or `hf auth logout` to download anonymously |
| SAM3 training unavailable: "requires an NVIDIA GPU" | Non-CUDA host | Expected. Train on a CUDA machine; inference works here. |
| "The SAM3 sidecar runs a different hydra-suite than this app" (or "hydra-suite is not installed in the SAM3 sidecar env") | The main env was reinstalled from a different source (another checkout, git commit or wheel) after `hydra-sam3` was built | `python install.py --with-sam3-train` — it reuses the env and reinstalls hydra-suite from the main env's source |
| SAM3 training refused: compute capability < 8.0 or no bf16 | GPU too old for bf16 | Use an Ampere-or-newer GPU. There is no fp32 fallback. |
| `OMP Error #15` when running sidecar commands by hand | Duplicate libomp | `export KMP_DUPLICATE_LIB_OK=TRUE` first (the GUI sets it for you) |
