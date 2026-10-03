# Integrations

## SLEAP Integration (TrackerKit + PoseKit)

SLEAP inference is executed from the **SLEAP conda/mamba environment selected in the UI**.

### Why this env is separate

SLEAP runs in its own conda env, `sleap`. Its dependencies (`sleap-nn`, its
export extras, and its own torch) are not part of the main HYDRA env, and
HYDRA talks to it through a service subprocess (`conda run -n sleap ...`).

Select the env from:

- TrackerKit: `Analyze Individuals -> Pose Extraction -> SLEAP env`
- PoseKit: `Inference -> SLEAP -> Conda environment`

To use ONNX/TensorRT SLEAP prediction inside PoseKit, also enable:

- `Inference -> SLEAP -> Allow experimental SLEAP runtimes`

### Environment setup

The installer builds the env (conda required):

```bash
python install.py --with-sleap        # or: make setup-sleap
```

This creates `sleap` (Python 3.13) and installs:

- `sleap[nn,nn-export-gpu]==1.6.2` on CUDA hosts, `sleap[nn,nn-export]==1.6.2`
  elsewhere;
- then **force-reinstalls the sidecar's torch/torchvision to the same build the
  main env uses** (`cu128` on drivers older than 580, `cu130` on 580+, CPU or
  macOS wheels otherwise). Left alone, the `sleap` env would pick up whatever
  torch PyPI serves, which on a pre-580 driver is a CUDA 13 build that cannot
  see the GPU.

If the `sleap` env already exists it is reused and its packages are
refreshed. For a clean rebuild, run `conda env remove -n sleap` first.

**Why `sleap==1.6.2` is pinned.** An unpinned `pip install "sleap[nn,...]"`
currently resolves `sleap-nn` to 0.3.x, which breaks the pipeline's
shared-memory transport to the SLEAP service (it fails with an `AttributeError`
from `sleap_nn.data.utils.imread_`). `sleap==1.6.2` pulls in the working
`sleap-nn` 0.1.3. This was verified on a fresh Ubuntu/RTX 4090 box on
2026-09-07. HYDRA's SLEAP preflight also checks the env and refuses any
`sleap-nn` that is not 0.1.x with an "unsupported sleap-nn" error, instead of
failing mid-run. Do not remove the pin without re-verifying against a fresh
sleap-nn release.

**TensorRT export (optional, manual).** The installer does not install
SLEAP's `nn-tensorrt` extra. If you want SLEAP TensorRT export, add it to the
env by hand, keeping the pin:

```bash
conda run -n sleap python -m pip install "sleap[nn,nn-export-gpu,nn-tensorrt]==1.6.2"
```

A `torch-tensorrt` version warning after the torch re-pin is expected. It
does not affect the SLEAP *service* backend, only TensorRT export.

`hydra doctor` reports whether the `sleap` env exists.

### Compatibility Matrix

- **macOS (Apple Silicon)**:
  - Supported: SLEAP native (`mps`/`cpu`), ONNX CPU.
  - Not supported: TensorRT.
- **NVIDIA CUDA systems**:
  - Supported: SLEAP native CUDA, ONNX GPU, TensorRT (with the `nn-tensorrt` extra added manually).
- **CPU-only systems**:
  - Supported: SLEAP native CPU, ONNX CPU.

### Verify SLEAP Integration

```bash
conda run -n sleap python -c "import importlib.util as u; print('sleap_nn', bool(u.find_spec('sleap_nn'))); print('onnx', bool(u.find_spec('onnx'))); print('onnxruntime', bool(u.find_spec('onnxruntime')))"
conda run -n sleap python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('torchvision', torchvision.__version__)"
conda run -n sleap python -c "import importlib.metadata as m; print('sleap-nn', m.version('sleap-nn'))"   # must be 0.1.x
conda run -n sleap sleap-nn export --help
```

### Troubleshooting Install/Runtime Issues

The quickest fix for most sidecar problems is to rebuild it:

```bash
conda env remove -n sleap
python install.py --with-sleap
```

If you need to repair it by hand, first check which CUDA build the driver
supports: `nvidia-smi --query-gpu=driver_version --format=csv,noheader`.
Driver **580 or newer** uses the `cu130` wheels. **Older** drivers use `cu128`.

#### 1) `operator torchvision::nms does not exist`

`torch` and `torchvision` are incompatible builds. Reinstall one matching pair:

```bash
conda run -n sleap python -m pip uninstall -y torch torchvision torchaudio

# pick ONE:
conda run -n sleap python -m pip install --index-url https://download.pytorch.org/whl/cpu   torch torchvision   # CPU-only Linux/Windows
conda run -n sleap python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision   # driver < 580 (CUDA 12)
conda run -n sleap python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision   # driver >= 580 (CUDA 13)
conda run -n sleap python -m pip install torch torchvision                                                     # macOS

conda run -n sleap python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

#### 2) `libtorch_cuda.so: undefined symbol: ncclAlltoAll`

A CUDA/NCCL binary mismatch. The NCCL wheel must match the torch build:

=== "Driver < 580 (CUDA 12, cu128)"

    ```bash
    conda run -n sleap python -m pip uninstall -y torch torchvision torchaudio nvidia-nccl-cu12 nvidia-nccl-cu13
    conda run -n sleap python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
    conda run -n sleap python -m pip install --upgrade nvidia-nccl-cu12
    conda run -n sleap python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
    ```

=== "Driver >= 580 (CUDA 13, cu130)"

    ```bash
    conda run -n sleap python -m pip uninstall -y torch torchvision torchaudio nvidia-nccl-cu12 nvidia-nccl-cu13
    conda run -n sleap python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
    conda run -n sleap python -m pip install --upgrade nvidia-nccl-cu13
    conda run -n sleap python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
    ```

If this succeeds only when unsetting `LD_LIBRARY_PATH`, your shell is injecting incompatible CUDA/NCCL libraries:

```bash
env -u LD_LIBRARY_PATH conda run -n sleap python -c "import torch; print(torch.cuda.is_available())"
```

In that case, remove conflicting CUDA/NCCL paths from `LD_LIBRARY_PATH` for SLEAP runs.

#### 3) "unsupported sleap-nn"

The env has a `sleap-nn` other than 0.1.x, usually from an unpinned
`pip install sleap`. Reinstall with the pin
(`python install.py --with-sleap` after removing the env, or
`conda run -n sleap python -m pip install "sleap[nn,nn-export]==1.6.2"`).

For ONNX export smoke test:

```bash
conda run -n sleap sleap-nn export "/path/to/sleap_model_dir" --output "/tmp/sleap_export_test_onnx" --format onnx --device cpu --input-height 224 --input-width 224
```

For ONNX predictor smoke test:

```bash
conda run -n sleap python -c "from sleap_nn.export.predictors import load_exported_model as L; p=L('/tmp/sleap_export_test_onnx/model.onnx', runtime='onnx', providers=['CPUExecutionProvider']); print(type(p).__name__)"
```

See also:

- [Compute Runtimes (User Guide)](../user-guide/compute-runtimes.md)
- [Runtime Integration (Developer Guide)](../developer-guide/runtime-integration.md)

## X-AnyLabeling Integration

`X-AnyLabeling` is useful for adding labels to additional frames and correcting detection misses before rerunning TrackerKit pipelines.

### Install (git clone method)

```bash
git clone https://github.com/CVHub520/X-AnyLabeling.git
cd X-AnyLabeling
mamba create -n x-anylabeling python=3.11 -y
conda activate x-anylabeling
pip install -r requirements.txt
python app.py
```

If your local clone uses a different launch command, follow the repository instructions from the current branch README/get-started docs.

### Recommended labeling workflow for TrackerKit data

- Export or collect frames where TrackerKit missed animals or produced poor detections.
- Open those images in `X-AnyLabeling`.
- Use one of these tools to add/fix labels:

- `SAM2` for assisted segmentation
- `SAM3` for assisted segmentation
- Manual segmentation tool (recommended for usability)
- Manual OBB tool (use when oriented boxes are specifically needed)

- Prefer segmentation masks over OBB for day-to-day correction speed and annotation quality.
- Export labels in the format needed for your downstream TrackerKit training or validation workflow.

### Notes

- Segmentation editing is generally more user-friendly than manual OBB editing for correction tasks.
- OBB remains useful when your model or evaluation requires oriented boxes specifically.
- Keep class names and label conventions consistent with your TrackerKit dataset configuration.
