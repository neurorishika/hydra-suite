# Integrations

## SLEAP Integration (TrackerKit + PoseKit)

SLEAP inference is executed from the **SLEAP conda/mamba environment selected in the UI**.

### Why this env is separate

Do not rely on the main TrackerKit environment for SLEAP export/runtime dependencies.
In practice, the `uv`-based TrackerKit install path does not guarantee the SLEAP runtime/export modules needed for all SLEAP backends.

Create a dedicated SLEAP environment and select it from:

- TrackerKit: `Analyze Individuals -> Pose Extraction -> SLEAP env`
- PoseKit: `Inference -> SLEAP -> Conda environment`

To use ONNX/TensorRT SLEAP prediction inside PoseKit, also enable:

- `Inference -> SLEAP -> Allow experimental SLEAP runtimes`

### Environment Setup

```bash
mamba create -n sleap python=3.13 -y
conda activate sleap
```

Choose one install profile. **Pin `sleap==1.6.2`** — an unpinned
`pip install "sleap[nn,...]"` currently resolves `sleap-nn` to 0.3.3, which
breaks the pipeline's shared-memory transport to the SLEAP service (fails
with `AttributeError` from `sleap_nn.data.utils.imread_`). `sleap==1.6.2`
pulls in the working `sleap-nn 0.1.3`; verified on a fresh Ubuntu/RTX 4090
box (python 3.11.16, numpy 2.4.6, torch 2.14.0+cu130) on 2026-09-07. Do not
remove the pin without re-verifying against a fresh sleap-nn release.

```bash
# CPU + ONNX CPU export/runtime support
pip install "sleap[nn,nn-export]==1.6.2"

# GPU + ONNX GPU export/runtime support
pip install "sleap[nn,nn-export-gpu]==1.6.2"

# GPU + TensorRT export/runtime support
pip install "sleap[nn,nn-export-gpu,nn-tensorrt]==1.6.2"
```

!!! warning "The `sleap` env brings its own torch"

    This is a separate environment, so it resolves torch independently of
    `hydra-cuda` and lands on whatever PyPI currently serves — which on a
    pre-580 driver is a CUDA 13 build that cannot see the GPU. On CUDA 12
    hosts, pin it to match the driver right after installing SLEAP:

    ```bash
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        --force-reinstall torch torchvision
    ```

    Use the `cu130` index instead on driver 580+. A `torch-tensorrt` version
    warning after this is expected and harmless for the SLEAP *service*
    backend; it only affects TensorRT export.

### Compatibility Matrix

- **macOS (Apple Silicon)**:
  - Supported: SLEAP native (`mps`/`cpu`), ONNX CPU.
  - Not supported: TensorRT.
- **NVIDIA CUDA systems**:
  - Supported: SLEAP native CUDA, ONNX GPU, TensorRT (with `nn-tensorrt` extras and system CUDA/TensorRT compatibility).
- **CPU-only systems**:
  - Supported: SLEAP native CPU, ONNX CPU.

### Verify SLEAP Integration

```bash
conda run -n sleap python -c "import importlib.util as u; print('sleap_nn', bool(u.find_spec('sleap_nn'))); print('onnx', bool(u.find_spec('onnx'))); print('onnxruntime', bool(u.find_spec('onnxruntime')))"
conda run -n sleap python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
conda run -n sleap sleap-nn export --help
```

If `torchvision` import fails with `operator torchvision::nms does not exist`, reinstall matching `torch` + `torchvision` builds from the same channel (CPU or CUDA-specific), then rerun the verification commands above.

### Troubleshooting Install/Runtime Issues

If SLEAP preflight fails, use the commands below.

#### 1) `operator torchvision::nms does not exist`

This usually means `torch` and `torchvision` are incompatible builds.

```bash
conda run -n sleap python -m pip uninstall -y torch torchvision torchaudio

# Reinstall one matching stack (pick one):
# CPU-only
conda run -n sleap python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# CUDA 13.0
conda run -n sleap python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision

conda run -n sleap python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

#### 2) `libtorch_cuda.so: undefined symbol: ncclAlltoAll`

This indicates a CUDA/NCCL binary mismatch.

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
