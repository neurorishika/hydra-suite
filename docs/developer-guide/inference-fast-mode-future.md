# Deferred: non-exact "fast mode" levers

Phase 1 shipped only the safe, determinism-preserving GPU cleanups
(`inference_mode` around crop extraction and the classifier forward paths;
pinned/`non_blocking` H2D for classifier crop uploads). The larger speedups
require changing numerics and are intentionally deferred to the `GPU-Fast` tier
(spec §3) or a future sub-option:

| Lever | Measured (RTX 6000 Ada, 2026-06-30) | Status / why deferred |
|---|---|---|
| `channels_last` (CUDA) | classifier EfficientNet-B0 @96×96 b64: **+0.15% (no-op)**; breaks OBB inference (ultralytics `fuse_conv_and_bn` `.view()` on channels_last → RuntimeError) | **NOT shipped.** The +11% seen in a synthetic 224×224 microbench did not generalize to the real 96×96 classifier, and it is unsafe for the OBB path. Reverted after CUDA verification. |
| TensorRT fp16 | classifier −4.5%; OBB fp16 larger | non bit-identical → `GPU-Fast` tier (Phase 2) |
| TF32 matmul | not benchmarked in isolation | breaks fp32 determinism; not in fast-mode export model |
| cudnn.benchmark | not benchmarked | run-to-run nondeterminism |

MPS note: `channels_last` also REGRESSES MPS by ~54% (b64) and was never applied there.

**Phase 1 net result:** the inference pipeline was already well-optimized for
exact inference (batching, GPU crops, NVDEC, warmup, `no_grad` already present).
The only remaining exact lever with apparent headroom — `channels_last` — proved
to be a no-op on the real classifier and unsafe for OBB on CUDA, so Phase 1 ships
as correctness-preserving `inference_mode`/pinned cleanups with no measurable
throughput change. The real speedups live in the deferred `GPU-Fast` tier.
