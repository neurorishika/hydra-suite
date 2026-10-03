# Platform Notes

`python install.py` picks one of three install tiers: `cpu`, `mps` (Apple
Silicon) or `cuda` (NVIDIA; CUDA 12 or 13 from the driver). See
[Installation](installation.md). Inside the apps, the **runtime tier**
(`cpu` / `gpu` / `gpu_fast`) then chooses how each stage runs; see
[Compute Runtimes](../user-guide/compute-runtimes.md).

## Feature × device matrix

| Feature | CPU (Linux / Intel Mac) | Apple Silicon (MPS) | NVIDIA CUDA 12 / 13 (Linux) | Windows CPU | Windows CUDA |
|---|---|---|---|---|---|
| Tracking GUIs and CLI (all kits) | Yes | Yes | Yes | Yes | Yes¹ |
| YOLO detection / pose (`gpu` tier) | CPU only | Yes (MPS) | Yes | CPU only | Yes¹ |
| `gpu_fast` tier | — | CoreML | TensorRT | — | TensorRT¹ |
| Background subtraction acceleration | Numba (CPU) | PyTorch MPS | CuPy | Numba (CPU) | CuPy¹ |
| NVDEC video decode (PyNvVideoCodec) | — | — | Yes (`gpu_fast` tier) | — | Yes¹ |
| SLEAP pose (sidecar env) | CPU | MPS / CPU | CUDA | CPU | CUDA¹ |
| ViTPose inference | CPU | MPS | CUDA | CPU | CUDA¹ |
| SAM2 escalation (DetectKit) | Yes (slow) | Yes | Yes | Yes (slow) | Yes¹ |
| SAM3 semantic escalation (DetectKit) | Yes (very slow) | Yes (slow) | Yes | Yes (very slow) | Yes¹ |
| SAM3 LoRA training (DetectKit) | No | No | Yes (compute capability ≥ 8.0, bf16) | No | Not yet verified¹ |
| AprilTag identity | Yes² | Yes² | Yes² | Yes² | Yes² |
| Multi-GPU batch fan-out (`trackerkit track --gpus`) | — | — | Yes | — | Yes¹ |

1. Windows + NVIDIA is a supported install target, but it has not yet been
   verified on real hardware. CI covers Windows CPU installs only.
2. Needs the AprilTag fork, which is built from source by the installer: a C
   compiler is required (Xcode CLT on macOS, Visual Studio Build Tools on
   Windows). It is skipped with `--skip-apriltag`.

See [SAM2 and SAM3: install and run](../user-guide/sam-install-and-run.md) for
the SAM device table and speed expectations.

## NVIDIA CUDA

- Best throughput for YOLO-heavy workflows. `gpu_fast` exports TensorRT
  engines on first use.
- The CUDA build follows the driver: 580+ → CUDA 13, older → CUDA 12. The
  CUDA toolkit is not required.
- On shared multi-GPU machines, name GPUs explicitly rather than using
  `--gpus auto`.

## Apple Silicon (MPS)

- PyTorch MPS for detection, pose and SAM. `gpu_fast` uses CoreML exports.
- Good for local labeling and medium workloads. For full SAM3 escalation
  runs, use a CUDA machine.

## CPU-only

- Every workflow that does not explicitly need a GPU runs on CPU, but
  neural-network stages (YOLO, pose, SAM) are slow. SAM3 *training* is not
  available.
- Prefer background-subtraction detection mode and conservative preview
  settings.

## Portable defaults

- Start with the `gpu` runtime tier (or `cpu` on a machine with no
  accelerator). Switch to `gpu_fast` only after `gpu` works on your data,
  because `gpu_fast` trades a little accuracy for speed.
- Run `hydra doctor` after installing on a new machine.
