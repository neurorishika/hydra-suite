# Known issues: sequential OBB + gpu_fast tier (2026-07-02)

Found while testing the `ant_obb_sequential` equivalence fixture (detect
stage → crop → OBB stage) across MPS (local), and CPU/GPU/GPU-fast tiers on
a remote CUDA box (`mehek`, RTX 6000 Ada). Five real bugs were found and
fixed on `feature/inference-pipeline-redesign`; this doc tracks everything
still open, split by where it lives.

Related work already merged/pushed:
- `feature/inference-pipeline-redesign`: `6692b51`, `9a6929f`, `1785cd0`,
  `bbd5960`, `22597f1` — sequential OBB crash + parity + gpu_fast fixes.
- `fix/nvdec-hang` (off `main`, PR not yet opened): `c02c0d8` — fixes the
  NVDEC hang described below.

---

## 1. On `main` (legacy) — not fixed here, needs separate work

### 1.1 Legacy sequential-OBB produces ~10x more detections on native CUDA than CPU

**Where:** `src/hydra_suite/core/detectors/yolo_detector.py`, legacy's
`_seq_*` sequential-OBB path (`_seq_stage1_predict`,
`_seq_run_stage2_obb_batched`, and friends).

**Repro:** run the `ant_obb_sequential` equivalence config (or any
`yolo_obb_mode=sequential` config) through legacy on `main` at
`RUNTIME=cpu` vs `RUNTIME=gpu` (native `cuda`, not `gpu_fast`/TensorRT) on
the same clip.

- CPU: `ant_obb_sleap_tracking_final.csv` → 1073 rows (~2.1 detections/frame
  average over 500 frames), matches the redesign's output on every tier.
- GPU (native CUDA): same config, same clip → **10,860 rows** (~21.7
  detections/frame) — verified not a duplicate-row/append artifact (500
  unique `FrameID`s, `df.duplicated().sum() == 0`, per-frame counts up to
  ~25–26, right at `MAX_TARGETS`).

The redesign's new pipeline stays at ~2.17 detections/frame on all of
CPU/MPS/GPU/GPU-fast for the same clip/config, so this looks like a
CUDA-specific defect in legacy's sequential-OBB code — most likely in
stage-1's NMS/filtering behavior only under the native-CUDA device path
(iou=1.0 disables NMS in both stages by design, but something downstream of
that — cap application, size/aspect filtering, or duplicate-suppression —
appears to behave differently on CUDA vs CPU). Not yet root-caused.

**Impact:** any user running sequential OBB mode on `main` with a CUDA GPU
selected (not just `gpu_fast`) may be getting far more spurious detections
than intended, silently degrading tracking quality.

**Suggested next step:** bisect within `yolo_detector.py`'s `_seq_*` methods
by comparing raw stage-1 box counts (before crop-building) between CPU and
CUDA on the same frame; if stage-1 counts already differ, the bug is in
detection/NMS, not downstream filtering.

### 1.2 NVDEC hard-fails instead of falling back to cv2 decode for high-macroblock-count clips

**Where:** `src/hydra_suite/core/tracking/detection_phase.py`,
`_try_open_nvdec` / `_read_nvdec_batch` / `run_batched_detection_phase`.

**What happens:** `PyNvVideoCodec` only validates the decoded stream's
macroblock count against the GPU's hardware limit *lazily*, on the first
real frame read (`sdec.get_batch_frames(1)`) — not when the decoder is
opened. So a clip whose resolution exceeds that limit (e.g. 4512×4512
against a 65536-macroblock cap on an RTX 6000 Ada) passes
`_try_open_nvdec`'s try/except cleanly, logs "NVDec GPU hardware decode
enabled", and only then raises
`_PyNvVideoCodec.PyNvVCExceptionUnsupported` deep inside the main detection
loop, uncaught.

**Status:** the *hang* this caused is fixed on `fix/nvdec-hang` (see below)
— it now fails fast with a clear error instead of hanging forever. The
underlying limitation is not: there's no retry/fallback to `cv2` decode when
this specific failure mode occurs, so any clip whose resolution exceeds this
GPU's macroblock limit still cannot be processed with NVDEC on this box at
all (must set `HYDRA_DISABLE_NVDEC=1` to force cv2 decode as a manual
workaround).

**Suggested next step:** in `_try_open_nvdec`, do a trial `get_batch_frames(1)`
call right after opening (mirroring the `HYDRA_DISABLE_NVDEC` kill-switch
docs already in the codebase — see
`fix(inference): NVDEC graceful fallback on unsupported clips (trial-decode)`
in the redesign branch's history, commit `a6709be`, which apparently already
implements exactly this pattern for the *new* pipeline's `sources.py`). The
fix likely already exists in the redesign branch and could be ported back to
legacy's `detection_phase.py`, or legacy could be pointed at the same
`FrameSource` abstraction instead of its own NVDEC code path.

### 1.3 `core/detectors/_direct_obb_runtime.py`'s TensorRT batch=1 chunking bug is presumably still present

**Where:** `src/hydra_suite/core/detectors/_direct_obb_runtime.py`,
`DirectTensorRTOBBExecutor.__init__`.

This file is **byte-identical** between `main@0a83c51` and the redesign
branch's starting commit (verified via `git diff` across both trees) — it's
shared/ported code, not something introduced by the redesign. The bug (see
section 2.5 below for the full writeup) means any TensorRT engine built
with a genuinely fixed batch dimension of exactly 1 will crash
(`IExecutionContext::setInputShape: ... Static dimension mismatch`) if
`predict()` is ever called with more than one frame/crop at once. This
affects:
- Sequential OBB's stage-2 (crop) batching (fixed on the redesign branch).
- Potentially **direct-mode OBB** too, whenever
  `_run_batched_detection_phase`/`InferenceRunner`'s batch pass feeds
  multiple frames at once to a `gpu_fast` direct executor — this was not
  specifically tested in this session for legacy's direct-mode path, since
  the equivalence fixtures used here were all sequential-mode.

**Suggested next step:** port the one-line fix
(`self._static_batch = batch_dim > 0` in `DirectTensorRTOBBExecutor.__init__`)
from the redesign branch (`22597f1`) to `main`'s copy of the same file, then
verify legacy's `gpu_fast` tier direct-mode fixtures (`emi_obb_identity`,
`fly_obb`, etc.) still pass with multi-frame batches on a real TensorRT box.

---

## 2. On `feature/inference-pipeline-redesign` — fixed, but with loose ends

The following five bugs were found and fixed this session (all committed
and pushed):

1. `imgsz=None` crash in `_run_sequential` (sequential OBB never ran at all
   with a default config) — `6692b51`.
2. Stage-2 crop-extraction floor/ceil rounding + pre-resize/scale-back
   parity with legacy (was causing ~40% detection mismatch vs legacy on
   CPU) — `6692b51`.
3. `run_matrix.sh`/`runner.py` harness bugs: `gpu`/`gpu_fast` tier names
   rejected by `run_matrix.sh`'s validation whitelist even though
   `runner.py` supported them; `runner.py` crashed on the legacy-side run
   for any non-tier runtime name (e.g. `mps`) because it unconditionally
   imported `hydra_suite.core.inference.config`, which only exists in the
   redesign tree — `9a6929f`.
4. Sequential stage-2 (crop) TensorRT/ONNX artifact exported at the
   checkpoint's own embedded imgsz (160) instead of the configured
   `stage2_image_size` (128), AND the computed imgsz was never actually
   passed to Ultralytics' `.export()` call for the TensorRT branch (only
   logged) — caused zero detections under `gpu_fast` — `1785cd0`.
5. Sequential stage-1 (detect) model routed through the OBB-output parser
   (`create_direct_obb_executor`) instead of the plain-detect parser
   (`create_direct_detect_executor`, which already existed in the ported
   code but was never wired into the new pipeline's `load_obb_executor`) —
   misread the class-score channel as an angle, always yielding
   `Results.boxes is None` — `bbd5960`.
6. `DirectTensorRTOBBExecutor` never set `_static_batch`, so its
   batch-chunking guard never fired for a fixed batch=1 TensorRT engine —
   crashed with a TensorRT shape-mismatch whenever fed more than one
   frame/crop at once (sequential stage-2's 16-crop batches) — `22597f1`.

All four/five of the `gpu_fast`-specific bugs were invisible on CPU/MPS/GPU
(native CUDA) tiers — only `gpu_fast` auto-exports/loads a direct TensorRT
executor, so none of the existing test suite or the CPU/MPS/GPU equivalence
runs could have caught them.

**Verified after all fixes**, on mehek (RTX 6000 Ada), `ant_obb_sequential`
fixture:
- Forward pass, legacy vs new (CPU, MPS, GPU): EQUIVALENT (matched
  593/593–688/687, position error ~1e-4–2px, floating-point-noise level).
- Determinism (new_a vs new_b): EQUIVALENT (bit-identical) on every tier
  including `gpu_fast`.
- `gpu_fast` (TensorRT FP16) now runs to completion with real detections
  (205 trajectories, consistent density with other tiers) instead of
  crashing or producing zero detections.
- GPU vs GPU-fast: **rough** equivalence only (expected — FP16 rounding
  near a confidence threshold that sits right at the model's natural output
  range causes many detections to flip sides of the 0.3 threshold; matched
  349/1087 with tight position error (max 2px) on matches, large
  unmatched counts). Not a bug — inherent to FP16 quantization here.
- Speed: CPU 6.8 fps → MPS 11.6 fps → GPU 19.1 fps → GPU-fast 19.2 fps (new
  pipeline only; GPU-fast isn't meaningfully faster than native GPU here
  because the TensorRT engines are forced to batch=1, so FP16/TensorRT's
  usual batching advantage doesn't apply).

### Loose ends still open on this branch

**2.1 — One residual tracking-stage discrepancy (not an OBB bug).**
Final-CSV comparison (CPU tier, legacy vs new_a) matches 940/941 rows; the
one outlier is a single detection whose trajectory is born one frame later
in the new pipeline (frame 165) than in legacy (frame 164), despite the raw
OBB detection at frame 164 being confirmed **bit-close identical** between
legacy and new (confidence 0.32317 vs 0.32317, centroid matching to 4+
decimal places — verified directly from both pipelines' detection caches).
So the OBB stage is not at fault; the discrepancy is in the
tracking/assignment (Kalman gating / trajectory-birth-confirmation) stage of
`core/tracking/worker.py`, and hasn't been root-caused beyond that
localization.

**2.2 — CoreML path never got the `imgsz_override`/`task` fix.**
`load_obb_executor`'s CoreML branch (`_load_coreml_executor`) still calls
`_resolve_imgsz(resolved)` unconditionally — the fix in
`runtime_artifacts.py` only threads `imgsz_override`/`task` through the
TensorRT direct-executor path (`_load_direct_executor`). If sequential OBB
is ever run on Apple Silicon's `gpu_fast` tier (CoreML) with a
`stage2_image_size` that differs from the crop model's own checkpoint
default, the same zero-detection bug from finding #4 above would likely
reproduce there. Untested (no CoreML-capable box was used this session)
and unfixed.

**2.3 — `YOLO_SEQ_STAGE2_POW2_PAD` has no equivalent in the new
`OBBSequentialConfig`.** Legacy's sequential OBB has a
`YOLO_SEQ_STAGE2_POW2_PAD` knob (pads the crop *list* — not crop pixel
content — to a power-of-two count for fixed-shape backends). Determined
this session to be irrelevant for correctness in the new pipeline's
direct-executor path (which already chunks properly once fix #6 above is
applied), but it was never explicitly ported/considered, so if a future
backend needs it, there's no config surface for it yet.

**2.4 — `ant_obb_sequential`'s model weights aren't in the release
archive.** The two-stage detect+crop-OBB models
(`detection/20260305-175022_26x_obiroi_v1.pt`,
`obb/cropped/20260305-175049_26s_obiroi_obbcrop.pt`) exist locally on this
machine and were manually copied to mehek for testing, but are not part of
`tools/equivalence/fixtures/manifest.json`'s `models_contained` list or the
actual `equiv-fixtures-v2` release tarball. A fresh clone running
`fetch_fixtures.sh` cannot run this fixture until someone runs
`make_manifest.py` and re-uploads the release archive with these two models
included. Documented in `tools/equivalence/README.md` but not yet done.

**2.5 — Direct-mode `gpu_fast` not re-verified after the batch-chunking
fix.** Fix #6 (`_static_batch`) should be strictly additive — it makes
chunking happen where it silently didn't before, which can only fix
previously-broken multi-frame-batch cases, not break already-working
single-frame ones — but no direct-mode `gpu_fast` fixture
(`emi_obb_identity`, `fly_obb`, etc.) was re-run on mehek after this change
to confirm there's no regression.

---

## 3. Process note (not a code issue)

Another session/agent was actively committing to
`feature/inference-pipeline-redesign` concurrently with this work
throughout — it clobbered uncommitted changes to `stages/obb.py` and
`tools/equivalence/README.md` twice before a commit-early workflow was
adopted. Not a code bug, but worth knowing if coordinating multiple
sessions against the same branch: commit early and often, and diff against
`HEAD` after any wakeup/resume to detect drift.

---

## 4. Suggested priority order

1. **1.3 / 2.5** — port the one-line `_static_batch` fix to `main`, then
   spot-check legacy's direct-mode `gpu_fast` fixtures on a real TensorRT
   box. Cheap, low-risk, plausibly fixes a live crash on `main`.
2. **1.1** — legacy CUDA over-detection bug. Silent correctness bug (not a
   crash) affecting anyone using sequential OBB + CUDA on `main` today;
   worth prioritizing over the CoreML/pow2-pad gaps below.
3. **1.2** — NVDEC fallback-instead-of-hard-fail. The redesign branch may
   already have the fix pattern (`sources.py`'s trial-decode / `FrameSource`
   abstraction, commit `a6709be`) — check whether it can be ported/reused
   rather than reimplemented.
4. **2.1** — tracking-stage trajectory-birth timing. Lower priority: tiny
   magnitude (1 row in ~1000), and the OBB layer is already confirmed
   correct.
5. **2.2, 2.3, 2.4** — CoreML imgsz-override, pow2-pad config surface,
   release-archive packaging. All lower urgency (no known live bug reports
   for CoreML sequential OBB; pow2-pad is an optimization knob; the
   release-archive gap only blocks *other machines* from running this one
   new fixture, not any existing one).
