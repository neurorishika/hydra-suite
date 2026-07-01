# Deferred: non-exact "fast mode" levers

Phase 1 shipped the exact (determinism-preserving) GPU wins. The larger
speedups require changing numerics and are intentionally deferred to the
`GPU-Fast` tier (spec §3) or a future sub-option:

| Lever | Measured (RTX 6000 Ada, efficientnet_b0 b64, 2026-06-30) | Why deferred |
|---|---|---|
| `channels_last` (CUDA) | +11% CNN compute — SHIPPED in Phase 1 (exact-within-envelope) | n/a |
| TensorRT fp16 | classifier −4.5%; OBB fp16 larger | non bit-identical → GPU-Fast tier |
| TF32 matmul | not benchmarked in isolation | breaks fp32 determinism; not in fast-mode export model |
| cudnn.benchmark | not benchmarked | run-to-run nondeterminism |

MPS note: `channels_last` REGRESSES MPS by ~54% (b64) and is never applied there.
