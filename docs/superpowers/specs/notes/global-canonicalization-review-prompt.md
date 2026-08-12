# De novo review prompt — global crop canonicalization

Paste everything below the line into a fresh agent with no prior context.
It is deliberately written so the reviewer discovers problems itself rather than
confirming a summary: it states the REQUIREMENTS and the SUSPICIONS, and gives
no account of what was built or why any decision was made.

---

You are performing an adversarial, from-scratch review of a shipped refactor in
this repository. Assume it is wrong until the code shows you otherwise. Your
value here is finding what a confident implementer missed; a review that
confirms the design is worth less than one that names a concrete defect.

Repo: `/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker`
The work is already merged to `main` (merge commit subject begins
"Merge feat/global-canonicalization"). Read-only: do NOT modify files, do NOT
commit. Produce findings, not fixes.

## The domain, in one paragraph

This is a multi-animal tracking suite. A detector produces oriented bounding
boxes (OBBs) for animals in each video frame. Several models then consume a
per-animal image crop: a head/tail direction classifier, a CNN identity
classifier, and pose estimators (ViTPose, SLEAP, YOLO-pose). Crops are also
exported to disk to build training datasets for those same models (ClassKit for
classifiers, PoseKit for pose). The suite runs across compute tiers
(`cpu`, `gpu`, `gpu_fast`) on both CUDA and Apple MPS hosts, with different
execution backends per tier (torch, TensorRT, CoreML, ONNX).

## The requirements the refactor claims to satisfy

Judge the code against THESE, not against any comment or document in the repo:

1. **One canonicalization.** There is exactly one way an animal crop is
   produced, shared by every consumer and by dataset export. No second
   implementation, no per-consumer variant, no fallback that quietly differs.
2. **No distortion.** No step applies different scale factors to x and y. An
   animal must never be stretched or squashed, at any tier, on any device, in
   training or inference.
3. **Size preserving.** The crop must not normalise every animal to the same
   apparent size. A physically larger individual must render larger; a curled
   animal must render shorter. Body size must survive into the model input as
   signal.
4. **Uniform output.** Every crop has identical pixel dimensions and identical
   scale, determined by project configuration rather than by the individual
   detection.
5. **Clipping, not rescaling.** An animal exceeding the canvas is clipped at the
   edge and that fact is reported. It is never rescaled to fit, because that
   would reintroduce per-animal scale.
6. **Training and inference identical.** A given source image fed through the
   training path and through the inference path must produce the same model
   input tensor — same geometry, same dtype, same channel order, same
   resampling, same padding.
7. **No unnecessary round trips.** No crop should make an avoidable
   CPU↔GPU transfer or an avoidable trip through disk. Where frames are already
   resident on the GPU, the crop path should stay there.

## What to produce

**A. A coverage matrix.** Enumerate every path that produces or consumes a crop
and give a verdict per cell. Cover at minimum
`{cpu, gpu, gpu_fast} × {CUDA, MPS} × {head-tail, CNN identity, ViTPose, SLEAP,
YOLO-pose, YOLO-classify, crop-dataset export, oriented-video export,
interpolated-crop generation, ClassKit training, ClassKit inference, PoseKit
training}`. Some cells are unreachable — say so and prove it from the dispatch
logic rather than assuming. Do not sample; a gap in the matrix is the most
likely place a defect survives.

**B. A findings list**, ranked by consequence. For each: file:line for both
sides of the problem, what the requirement demands, what the code does, and the
concrete consequence for a user. Mark each VERIFIED (you read both sides) or
SUSPECTED.

**C. An explicit list of what you checked and found correct.** A clean bill on a
path is a useful result, but only if you name the path and the evidence.

## Specific things to attack

These are places where this class of refactor characteristically fails. Treat
each as a hypothesis to test, not a claim to trust:

- **The second resample.** Find every point where a crop is resized, warped, or
  interpolated between the frame and the model. Count them per path. Are any
  redundant? Does any path resize to one size and then immediately to another?
- **Anisotropy hiding in a pair of steps.** Two anisotropic transforms can
  compose to something that looks right on square inputs and is wrong on
  non-square ones. Check every path with a NON-SQUARE model input specifically.
  Ask what a `(H, W)` vs `(W, H)` confusion would look like and whether any test
  would catch it.
- **Fallback paths.** Search for every `except`, every `if ... is None`, every
  default-valued geometry or size parameter on a crop path. A fallback that
  differs from the main path is a second implementation. Does a real value
  always reach it in production, and what happens when it doesn't?
- **Device-dependent geometry.** The CPU and GPU branches of the same stage must
  produce the same geometry. Backends that resize internally are the risk: check
  what each `predict_batch_cuda` (or equivalent) does to a tensor it receives,
  and whether every caller has already brought it to the right size.
- **Parameters accepted and never supplied.** Grep for config keys that are READ
  but never WRITTEN, and for function parameters that callers never pass. This
  codebase has a documented history of this defect class; assume more instances
  exist.
- **Guards that cannot fire.** Find every accumulated statistic, warning, or
  validation on the crop path and confirm something actually consumes it.
  A counter nobody reads is not a guard.
- **Test adequacy.** For each requirement above, find the test that would fail
  if it were violated. If a requirement has no such test — or the test only
  exercises inputs where the bug is invisible (e.g. square-only) — say so.
  Then ask: has this test ever been *seen* to fail?
- **Round trips.** Trace the dtype and device of a crop from frame to model on
  the GPU paths. Count host transfers and dtype conversions. Are any avoidable?

## Ground rules

- Read the code. Do not trust docstrings, comments, commit messages, or any
  document under `docs/` — several were written by the implementer and may
  describe intent rather than behaviour.
- Where you assert a defect, show the evidence. Where you cannot determine
  something statically, say what you would run to settle it.
- Environment if you want to execute anything:
  `source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps`,
  `export KMP_DUPLICATE_LIB_OK=TRUE`, `export PYTHONPATH=$PWD/src`.
  Never run the whole test suite in one process — it contains modal-dialog hangs
  and a SIGABRT. Run per file.
- This machine is Apple Silicon; CUDA paths cannot be executed here. Reason
  about them from the code and say clearly which conclusions are static-only.
- A CUDA box is reachable at `rutalab@mehek.taild08eb9.ts.net` (`hydra-cuda`
  env, conda at `~/mambaforge`) if you need to settle a CUDA-only question.

## Starting points, not a reading list

`src/hydra_suite/core/canonicalization/`, `src/hydra_suite/core/inference/stages/`,
`src/hydra_suite/core/individual/`, `src/hydra_suite/core/post/interpolated_crops.py`,
`src/hydra_suite/training/`, `src/hydra_suite/classkit/`, `src/hydra_suite/posekit/`,
`src/hydra_suite/trackerkit/`. Follow the code from each model backwards to the
frame; that direction finds consumers the forward direction misses.
