# Platform Notes

## NVIDIA CUDA

- Supports CUDA acceleration and TensorRT-oriented optimization paths.
- Best throughput for YOLO-heavy workflows.

## Apple Silicon (MPS)

- Uses PyTorch MPS backend where available.
- Good convenience/performance for local labeling and medium workloads.

## CPU-Only

- Supported for all workflows.
- Prefer background subtraction mode and conservative preview settings.

## Portable Defaults

- Start with `YOLO_DEVICE=auto`.
- Keep `ENABLE_TENSORRT=false` unless CUDA environment is verified.
- Use smaller resize factors for constrained devices.
