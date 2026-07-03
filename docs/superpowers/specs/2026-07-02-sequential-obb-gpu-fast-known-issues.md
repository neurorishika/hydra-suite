# Known issues: sequential OBB + gpu_fast tier (2026-07-02)

Found while testing the `ant_obb_sequential` equivalence fixture (detect
stage → crop → OBB stage) across MPS (local), and CPU/GPU/GPU-fast tiers on
a remote CUDA box (`mehek`, RTX 6000 Ada). Five real bugs were found and
fixed on `feature/inference-pipeline-redesign`; this doc tracks everything
still open, split by where it lives.

Related work already merged/pushed:
- `feature/inference-pipeline-redesign`: `6692b51`, `9a6929f`, `1785cd0`,
  `bbd5960`, `22597f1` — sequential OBB crash + parity + gpu_fast fixes.
  Also `1b0f08b` (CoreML `imgsz_override` threading, closes §2.2) and
  `6cb2e49`/`3226db3` (fixtures manifest + release re-upload, closes §2.4),
  both landed shortly after this doc's initial commit.
- **Update (2026-07-03): all four `main`-side items in §1 are now merged**
  via PR #15 (`fix/seq-obb-cuda-overdetect`, closes 1.1), PR #16
  (`fix/nvdec-hang`, closes 1.2), PR #17 (`fix/direct-trt-static-batch`,
  closes 1.3/2.5), PR #18 (`fix/seq-detect-trt-imgsz`). Section 1 below is
  kept for historical context; see each subsection for the merge commit.

---

## 1. On `main` (legacy) — not fixed here, needs separate work

### 1.1 Legacy sequential-OBB produces ~10x more detections on native CUDA than CPU

**RESOLVED (2026-07-03), PR #15 (`fix/seq-obb-cuda-overdetect`, merged as
`08a3562`).** Root cause: the direct-CUDA OBB executor was being built at
the checkpoint's default embedded imgsz instead of the actual stage-2 crop
size (`YOLO_SEQ_STAGE2_IMGSZ`), causing crops to be internally re-resized to
the wrong square size and systematically inflating confidence scores on
native CUDA. Fixed via `_resolve_direct_cuda_obb_imgsz()` in
`yolo_detector.py`, which resolves the stage-2 crop imgsz when in sequential
mode (direct mode is unaffected and keeps its old resolution path).

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

**RESOLVED (2026-07-03), PR #16 (`fix/nvdec-hang`, merged as `fef5933`).**
Adds a trial-decode call right after opening the NVDEC decoder plus a cv2
fallback when the trial fails, mirroring the pattern already used by the
redesign branch's `sources.py`.

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

**RESOLVED (2026-07-03), PR #17 (`fix/direct-trt-static-batch`, merged as
`b750392`).** Ported the one-line fix from the redesign branch verbatim:
`self._static_batch = batch_dim > 0` in `DirectTensorRTOBBExecutor.__init__`.
(A related fix, PR #18 `fix/seq-detect-trt-imgsz`, merged as `aeedc15`,
addresses sequential stage-1 detect TRT imgsz resolution in the same area.)

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

**Update (2026-07-02, local re-run on the redesign branch's own worktree,
CPU tier):** reproduced the exact same 940-vs-941 discrepancy locally
(`RUNTIME=cpu ONLY=ant_obb_sequential bash tools/equivalence/run_matrix.sh`)
and localized it one level further. The missing row is entirely a
*backward-pass* artifact — the forward-pass CSVs (`..._forward.csv`) are
identical between legacy and new for this ant (trajectory born at frame 165
in both); the extra frame-164 row only appears in legacy's *backward*
replay CSV. Dumped the new pipeline's raw `InferenceRunner` detection cache
(`.inference_cache_<video>/detection.npz`) directly: the frame-164
detection at (1735.09, 1415.24) is present with confidence `0.32317126` —
bit-identical to legacy's value and comfortably clear of
`yolo_confidence_threshold=0.3` (not a threshold-boundary flip). So the
detection reaches `filter_with_indices` in `core/inference/stages/filtering.py`
and should survive its gates untouched (nothing else at that frame overlaps
it for NMS to contend with). This rules out both the OBB stage and the
CPU/MPS filtering stage as the cause. The remaining candidate is
slot-availability state during backward Kalman tracking in
`core/tracking/worker.py`'s free-slot loop (~line 3474 in the redesign
tree): if no track slot is in `"lost"` state at frame 164 in the new
pipeline's backward run (unlike legacy's run, where one apparently is), the
detection is silently dropped with no trajectory to attach to — that
free-slot loop has no diagnostic logging for the "no free slot" case, so
this could not be confirmed from the existing WARNING-level
`SLOT_REUSE`/`JUMP?` diagnostics alone. Confirming this would require
instrumenting `track_states` per-frame across both pipelines' full
backward runs (frames ~150–165) and diffing slot occupancy history —
not attempted here given the doc's own priority ranking (§4) already ranks
this below the CoreML/manifest/gpu_fast items, which were addressed instead
this session.

**Update (2026-07-03): root-caused. The free-slot hypothesis above was
wrong** — instrumented `track_states`/`tracking_continuity` per-frame plus
the full Hungarian cost matrix for both pipelines' backward runs on the
`ant_obb_sequential` CPU fixture (temporary logging in `worker.py` and
`hungarian.py`, reverted after). Findings, in order:

1. At frame 164, both pipelines see the **identical 3 measurements**
   (bit-identical `meas` arrays) and classify slots identically into
   `est=[0, 5, 6, 7]` / `lost=[1, 2, 3, 4, 8, ...]` / `unst=[]` — the
   free-slot loop (§`worker.py` free-slot loop) is not on the code path
   that produces this discrepancy at all; slot 7 (the track this
   detection belongs to) is `est` (established), not something waiting on
   a `"lost"` slot.
2. `raw_dist_mat[7, 1]` (slot 7 vs. the frame-164 detection) is
   **bit-identical** between legacy and new: `10.861329...`px, comfortably
   inside every gate.
3. But `cost[7, 1]` — the value actually fed to
   `linear_sum_assignment` — is **not** identical: legacy `15.803`,
   new `17.974`, a `~2.17` gap. Subtracting the identical raw distance
   isolates this to the *identity hint term* added by
   `Hungarian._apply_bayesian_identity_cost()` (gated on
   `ENABLE_IDENTITY_ONLINE_DECODER` + `ASSOCIATION_IDENTITY_HINT_SCALE`,
   default `0.3`): `alpha * -logsumexp(track_log_posterior[7] +
   det_log_likelihood[1])` evaluates to `4.94` in legacy vs. `7.11` in
   new — i.e. the online identity decoder's per-slot log-posterior (or
   the detection's log-likelihood) differs slightly between the two
   codebases' implementations of that decoder.
4. Phase 1 (`_assign_established_hungarian`) runs `linear_sum_assignment`
   on the full `4×3` (tracks × detections) cost submatrix — since there
   are more established tracks (4) than detections (3) that frame, the
   solver is *forced* to leave exactly one track unassigned every time,
   and which one is a genuinely global optimization over all four rows,
   not a per-row threshold check. The ~2-unit identity-cost gap for slot 7
   is enough to flip which of the 4 est tracks is the "odd one out":
   legacy's assignment keeps slot 7 in (→ matched to this detection,
   frame-164 row written), new's assignment drops slot 7 (→ detection
   ends up in `free_dets`, and separately, no slot happens to be `"lost"`
   for it to respawn into, so it's dropped with no trajectory).

**Conclusion:** this is not a logic bug in either pipeline's assignment
code — both correctly compute the (slightly different) global-optimal
assignment for their own cost matrix. The actual discrepancy traces to the
online identity decoder producing marginally different Bayesian
log-posteriors between legacy's inline implementation and the new
pipeline's `core/inference`-based one, most likely from ~190 backward-pass
frames of accumulated float32 evidence-update order/rounding differences
between the two decoder implementations. This is the tracking-stage analog
of the FP16-threshold-flip behavior already documented for the GPU/GPU-fast
tiers above (§ "GPU vs GPU-fast" note) — an inherently discrete
(combinatorial, over-subscribed Hungarian) decision amplifying a small,
expected floating-point-level difference into a binary present/absent
outcome for one row out of ~1000. Given the magnitude, root-causing the
*exact* source of the log-posterior discrepancy inside the identity decoder
refactor is a separate, much larger parity-audit task (out of scope here);
this item is closed as **understood and accepted, no code fix**, not "not
yet root-caused."

**2.2 — RESOLVED (2026-07-03, prior to this doc's initial commit — commit
`1b0f08b`).** `load_obb_executor`'s CoreML branch (`_load_coreml_executor`)
now accepts and applies `imgsz_override`, mirroring the TensorRT
direct-executor path (`_load_direct_executor`). Verified by reading
`runtime_artifacts.py`: `_load_coreml_executor(model_path, *, auto_export,
imgsz_override=None)` resolves `imgsz` from the override when given,
falling back to `_resolve_imgsz(resolved)` only when not. Still untested on
real Apple Silicon `gpu_fast` hardware (no CoreML-capable box was used this
session), but the code-level gap is closed.

**2.3 — RESOLVED (2026-07-03): `YOLO_SEQ_STAGE2_POW2_PAD` intentionally has
no equivalent in the new `OBBSequentialConfig` — confirmed not needed, not a
gap.** Legacy's sequential OBB has a `YOLO_SEQ_STAGE2_POW2_PAD` knob that
pads the crop *list* (not crop pixel content) to the next power-of-two count
before handing it to a fixed-shape backend, working around legacy's
executor requiring an exact match to one of its exported batch sizes.

Re-verified against the current code: the new pipeline's
`_BaseDirectOBBExecutor.predict()` (`core/detectors/_direct_obb_runtime.py`,
`predict()`/`_predict_chunk()`) already does *generic* batch chunking for
every direct-mode call (stage-1 detect and stage-2 crop-OBB alike) —
it chunks to the executor's own `_model_batch_size` and pads any
undersized final chunk by repeating the first frame, then truncates results
back to the real count. This runs independently of (and downstream of)
`_run_sequential`'s own `stage2_batch_size` loop (`core/inference/stages/obb.py`),
so crops are correctly batched regardless of what `stage2_batch_size` is set
to. This is strictly more precise than legacy's pow2 rounding — it pads to
the model's *actual* batch size, not merely the nearest power of two — so
there is no scenario where `YOLO_SEQ_STAGE2_POW2_PAD` would add correctness
or performance value that the new pipeline lacks. No config surface is
needed; closing this with no action.

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

**Resolved (2026-07-02, on mehek).** Re-ran direct-mode `gpu_fast` fixtures
and found two previously-undiscovered bugs — both specific to the NVDEC
CUDA-tensor frame path (never exercised by a direct-mode `gpu_fast` fixture
before this session) — fixed on `feature/inference-pipeline-redesign`:

1. `_resolve_imgsz()` in `core/inference/stages/obb.py` only knew how to
   read `imgsz` from an ultralytics-model-shaped object (`.overrides` /
   `.model.args`). `DirectExecutorAdapter` (the gpu_fast TensorRT/ONNX
   wrapper) has neither, so it always silently fell back to the hardcoded
   1024 default — harmless for `emi_obb_identity` (whose engine happens to
   be exported at 1024, masking the bug) but crashed `fly_obb`'s 640-imgsz
   engine with `TensorRT setInputShape: Static dimension mismatch` on every
   frame. Fixed by having `DirectExecutorAdapter` surface the underlying
   executor's own `.imgsz` — commit on `feature/inference-pipeline-redesign`.
2. Even with (1) fixed, `_run_direct`'s CUDA-tensor branch still crashed:
   it pre-letterboxes NVDEC frames into a single `(B,3,imgsz,imgsz)` tensor
   (necessary for a plain ultralytics model, which can't accept a list of
   CUDA tensors) and hands that to `model.predict()` unconditionally.
   `DirectExecutorAdapter.predict()` then did `list(batched_tensor)`,
   splitting it back into per-slice `(3,imgsz,imgsz)` "frames" and feeding
   them to the underlying executor's `_preprocess_cuda_batch`, which
   re-letterboxed each as if it were a raw `(H,W,3)` frame — corrupting the
   shape whenever `imgsz != 3`. Fixed by routing `DirectExecutorAdapter`
   through the plain frames-list path instead (its own executor already has
   correct native CUDA-list handling with correct original-frame coordinate
   output).

**Verified after both fixes**, on mehek: `fly_obb` (gpu_fast, direct mode)
now matches legacy on 1500/1500 rows (1 row differs at FP16-rounding level
— consistent with the already-documented FP16-quantization caveat), with
bit-identical new_a/new_b determinism. `emi_obb_identity` — whose NVDEC path
hits the 4512×4512 macroblock limit and falls back to CPU decode (the
redesign's `HYDRA_DISABLE_NVDEC`-pattern fallback working as intended,
unlike legacy's hang from finding 1.2) — produces **bit-identical** output
between the NVDEC-attempted run (exercises today's fixes) and the
CPU-decode-fallback run (doesn't), proving the fix produces correct results
end-to-end, not just avoids a crash.

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
   **DONE (2026-07-03) — PR #17, merged as `b750392`.**
2. **1.1** — legacy CUDA over-detection bug. Silent correctness bug (not a
   crash) affecting anyone using sequential OBB + CUDA on `main` today;
   worth prioritizing over the CoreML/pow2-pad gaps below.
   **DONE (2026-07-03) — PR #15, merged as `08a3562`.**
3. **1.2** — NVDEC fallback-instead-of-hard-fail. The redesign branch may
   already have the fix pattern (`sources.py`'s trial-decode / `FrameSource`
   abstraction, commit `a6709be`) — check whether it can be ported/reused
   rather than reimplemented.
   **DONE (2026-07-03) — PR #16, merged as `fef5933`.**
4. **2.1** — tracking-stage trajectory-birth timing. Lower priority: tiny
   magnitude (1 row in ~1000), and the OBB layer is already confirmed
   correct. **Root-caused and closed (2026-07-03) — no code fix.** Not the
   free-slot loop after all (that hypothesis is disproven — see updated
   write-up below); the actual cause is a small floating-point-level
   difference in the online identity decoder's Bayesian cost term between
   legacy and new, which flips the outcome of an over-subscribed
   (more-established-tracks-than-detections) Hungarian assignment at frame
   164. Inherent to the identity-decoder refactor, same class as the
   already-accepted FP16 threshold-flip behavior noted for GPU/GPU-fast
   above; not worth chasing for 1 row in ~1000.
5. **2.2, 2.3, 2.4** — CoreML imgsz-override, pow2-pad config surface,
   release-archive packaging.
   **2.2 and 2.4 already resolved** (commits `1b0f08b` and `6cb2e49`/`3226db3`,
   landed shortly after this doc's initial commit).
   **2.3 confirmed not needed — closed with no action (2026-07-03),** see
   updated write-up below: the new pipeline's generic per-chunk batch
   padding in `_BaseDirectOBBExecutor.predict()` already subsumes what
   pow2-pad was for.
