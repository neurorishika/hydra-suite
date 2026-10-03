# Compute Runtimes

HYDRA selects how every compute-heavy stage runs from **one setting: the
runtime tier**. There are no per-stage or per-backend runtime selectors.

## The three tiers

| Tier | UI label | What it means |
|---|---|---|
| `cpu` | **CPU** | Everything on the CPU with native PyTorch. |
| `gpu` | **GPU (CUDA)** / **GPU (Metal)** | Native PyTorch on the GPU. Results match the CPU path as closely as the hardware allows. |
| `gpu_fast` | **GPU-Fast (TensorRT)** / **GPU-Fast (CoreML)** | Exported, optimised artifacts: TensorRT engines on NVIDIA, CoreML packages on Apple Silicon. Fastest, at the cost of some accuracy. |

Only tiers this machine can run are offered. On a machine with neither CUDA
nor Apple MPS, only **CPU** appears.

Where to set it:

- **TrackerKit:** the **Compute tier** card in the Performance group. The
  same tier drives detection, head–tail, CNN identity, YOLO-pose, SLEAP and
  ViTPose pose, and background subtraction. Pose has no separate runtime
  control.
- **PoseKit:** `Inference → Runtime`.
- **Config files / CLI:** the `runtime_tier` key (`"cpu"`, `"gpu"`, `"gpu_fast"`).

## How a tier resolves per stage

The tier is resolved for each stage (OBB detection, head–tail, CNN identity,
YOLO-pose, SLEAP pose, ViTPose pose, background subtraction) into a concrete
backend and device:

| Tier | NVIDIA host | Apple Silicon host | CPU-only host |
|---|---|---|---|
| `cpu` | torch on CPU | torch on CPU | torch on CPU |
| `gpu` | torch on CUDA | torch on MPS | torch on CPU (fallback) |
| `gpu_fast` | TensorRT on CUDA | CoreML on MPS | torch on CPU (fallback) |

Fallbacks are explicit, never silent:

- On `gpu_fast`, a stage whose fast artifact is not available runs on the
  native GPU instead (torch on CUDA/MPS). TrackerKit shows a note under the
  tier selector when this happens.
- **Background subtraction** has no TensorRT/CoreML implementation. On
  `gpu_fast` it runs exactly as on `gpu`: CuPy on CUDA, PyTorch on MPS, Numba
  on CPU.

## ONNX Runtime

Stages that run on torch never use ONNX Runtime. Where an ONNX model is used,
the execution providers follow the resolved backend:

| Resolved backend | ONNX Runtime execution providers |
|---|---|
| `tensorrt` (NVIDIA `gpu_fast`) | TensorRT EP (with a persistent per-machine engine cache), then CUDA EP, then CPU |
| `coreml` (Apple `gpu_fast`) | CoreML EP, when ONNX Runtime's CoreML provider is available on an MPS host, then CPU |
| `torch` | CPU |

The Apple install includes an `onnxruntime` build with the CoreML provider.
`hydra doctor` warns if it is missing.

## Auto export and artifact location

Fast-tier artifacts are generated automatically the first time they are
needed, next to the source model:

- YOLO model `/path/model.pt` → TensorRT `/path/model.engine` (NVIDIA) or
  CoreML `/path/model.mlpackage` (Apple Silicon)
- SLEAP model directory `/path/sleap_model_dir` → `/path/sleap_model_dir.onnx`
  / `/path/sleap_model_dir.tensorrt`

No manual exported-model path is required. See
[Inference Fast Mode](../developer-guide/inference-fast-mode.md) for the
export details and determinism caveats.

## Behavior during runs

Runtime controls are locked during long-running prediction/tracking tasks to
prevent backend switching crashes.

## Older configs

Configs saved before runtime tiers existed used per-stage strings such as
`compute_runtime: "onnx_coreml"`. Loading a config without `runtime_tier` fails
with an error that points to the one-time migration:

```bash
python scripts/migrate_runtime_config.py <config.json>
```

## Troubleshooting

- **Only "CPU" is offered:** torch cannot see a GPU. Run `hydra doctor`. On
  NVIDIA this is usually a CUDA 13 build on a pre-580 driver; see
  [Installation](../getting-started/installation.md#troubleshooting).
- **`gpu_fast` is not faster, or a note says a stage fell back:** that stage
  has no fast artifact (for example, background subtraction) or its export
  failed. Check the log for the export error.
- **SLEAP fast export unavailable:** the SLEAP sidecar env needs its export
  extras. Rebuild it with `python install.py --with-sleap`.

See also: [Integrations](../getting-started/integrations.md),
[Platform Notes](../getting-started/platforms.md),
[Troubleshooting](troubleshooting.md)
