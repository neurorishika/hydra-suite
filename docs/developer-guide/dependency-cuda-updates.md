# Dependency and CUDA Updates

How dependency versions are declared, pinned and bumped, and the CUDA-specific
constraints to re-check when you do.

## Where versions live

| What | File | Role |
|---|---|---|
| Allowed ranges | `pyproject.toml` | The **only** declaration of Python dependencies: core deps plus extras `cpu`, `mps`, `cuda12`, `cuda13` (`cuda` = alias for `cuda12`), `sam` (`sam3` alias), `sam3-train` (sidecar only), `dev`, `docs`, `onnx-tools`. |
| Tested pins | `constraints/base.txt`, `constraints/cuda12.txt`, `constraints/cuda13.txt` | Exact versions `install.py` applies by default. **Generated** — never edit by hand. |
| torch / torchvision | `TORCH_VERSION`, `TORCHVISION_VERSION` in `install.py` | A matched release pair, installed from the PyTorch index for the tier. |
| Driver → CUDA mapping | `CUDA13_MIN_DRIVER` in `install.py` (580) | Driver ≥ 580 gets CUDA 13 (`cu130`); older drivers get CUDA 12 (`cu128`). |
| Git-only pins | `CLIP_REF`, `APRILTAG_COMMIT`, `SAM3_REF` in `install.py` | ultralytics CLIP fork, Kronauer lab AprilTag fork, Meta `sam3` (training sidecar). |
| Sidecar pins | `SLEAP_PIN` (`1.6.2`) and the `sam3_train_steps` pin list in `install.py` | SLEAP and SAM3-training sidecar envs. |

`tools/lock_constraints.py` deliberately does **not** pin `torch`,
`torchvision`, `triton`, `nvidia-*`, `cuda-*` or `hydra-suite` itself. Torch
comes from the PyTorch wheel index, and the `nvidia-*` wheels must follow
whichever torch build is installed.

## Bumping the tested pins

```bash
python tools/lock_constraints.py                          # re-resolve, keep existing pins
python tools/lock_constraints.py --upgrade-package numpy  # bump one package
python tools/lock_constraints.py --upgrade                # bump everything
```

The tool resolves universally (one file covers Linux, macOS and Windows,
Python 3.11+). Then, **before committing new pins**:

1. Build a fresh env from the new pins (`python install.py --recreate`, or a
   throwaway `--env`).
2. Prove tracking output did not move with the env-swap gate. It keeps the
   `src/` tree fixed and swaps the interpreter (old env vs new env):

    ```bash
    OLD_PY=/path/to/old/env/bin/python NEW_PY=/path/to/new/env/bin/python \
    SRC=$PWD/src OUT=/tmp/envswap RUNTIME=mps \
      bash tools/equivalence/env_swap_gate.sh fly_obb worm_bgsub
    ```

    A clip passes as **BYTE-IDENTICAL**, or as **BLAS-FLOOR** when the only
    differences are BLAS float rounding in the Kalman diagnostic columns
    (`PositionUncertainty`, `AssignmentConfidence`). Anything else is a real
    change.
3. Run it on **MPS and CUDA** (CUDA 12 and CUDA 13 if `cuda12.txt` /
   `cuda13.txt` changed).

Users who want the newest releases regardless can install with
`python install.py --latest`.

## Bumping torch

1. Change `TORCH_VERSION` **and** `TORCHVISION_VERSION` in `install.py`
   together. They are a matched pair. Check the PyTorch release notes for
   the torchvision that goes with the new torch.
2. Confirm the `+cu128` and `+cu130` builds of that version exist on
   `https://download.pytorch.org/whl/cu128` and `.../cu130`. The local tag is
   what stops PyPI's default (CUDA 13) Linux wheel from being chosen on a
   CUDA 12 host.
3. If PyTorch drops or adds a CUDA line, update the tag map in
   `torch_requirements()` and `CUDA13_MIN_DRIVER`.
4. Rebuild and run `hydra doctor` on every tier, then the env-swap gate as
   above.

## CUDA-specific constraints to re-check

### ONNX Runtime GPU: one wheel family per CUDA major

| Extra | Range | Why |
|---|---|---|
| `cuda12` | `onnxruntime-gpu[cuda,cudnn]>=1.24,<1.27` | 1.26 is the last CUDA 12 build; its `[cuda,cudnn]` extra pulls the `nvidia-*-cu12` wheels that torch `+cu128` also uses. |
| `cuda13` | `onnxruntime-gpu[cuda,cudnn]>=1.28` | Built for CUDA 13; depends on the **unsuffixed** `nvidia-*` wheels, the same family as torch `+cu130`. 1.27 is skipped because it used `-cu13`-suffixed names. |

ONNX Runtime loads these libraries through `onnxruntime.preload_dlls()`
(`hydra_suite.runtime.onnx_providers.preload_ort_cuda_libraries`). No conda
CUDA packages or `LD_LIBRARY_PATH` hooks are involved. When bumping the
range, check that the new release's `nvidia-*` dependency family still matches
torch's.

### TensorRT and CuPy

`tensorrt-cu12*` / `cupy-cuda12x` go in `cuda12`, and `tensorrt-cu13*` /
`cupy-cuda13x` go in `cuda13`. The installer uninstalls the other family
before installing, because two TensorRT families in one env can leave
`import tensorrt` working while builder initialisation fails. CuPy CUDA 13
now has stable wheels, so it is pinned normally in `constraints/cuda13.txt`.

### Mixed wheel families

Both `nvidia-*-cu12` and unsuffixed `nvidia-*` wheels unpack into
`site-packages/nvidia/`. The CUDA checks in `hydra_suite.runtime.cuda_checks`
(run by `hydra doctor --tier cuda`) detect a mixed env and tell the user to
run `python install.py --recreate`.

### SLEAP sidecar

`sleap==1.6.2` pins `sleap-nn` 0.1.x. Newer `sleap-nn` breaks the
shared-memory transport, and the runtime preflight refuses anything but 0.1.x.
Re-verify on a fresh env before relaxing `SLEAP_PIN`.

### SAM3 training sidecar

Meta's `sam3` pins `numpy<2`, and the sidecar pin list (`setuptools<81`,
`scipy<1.14`, `opencv-python-headless<4.12`, …) exists to hold that pin. Keep
`pyproject.toml`'s `sam3-train` extra in sync with `TRAINING_PACKAGES` in
`hydra_suite/training/sam3_lora/availability.py`. When bumping `SAM3_REF`,
re-run a training smoke test on a CUDA box.

## Checklist

- [ ] Pins regenerated with `tools/lock_constraints.py` (no hand edits)
- [ ] `TORCH_VERSION`/`TORCHVISION_VERSION` bumped together, wheels exist for `cpu`/`cu128`/`cu130`
- [ ] Fresh env builds on MPS and CUDA; `hydra doctor` passes
- [ ] `tools/equivalence/env_swap_gate.sh` is BYTE-IDENTICAL or BLAS-FLOOR on MPS and CUDA
- [ ] `install-smoke` CI is green on Linux, macOS and Windows
- [ ] Installation docs updated if user-facing behaviour changed
