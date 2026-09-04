# DetectKit OOM Hardening — Agent Execution Instructions

Status: In progress — Sets 1–2 merged; Set 2B implemented and awaiting merge

## Goal

Make every high-memory DetectKit training, publishing, evaluation, and inference
path scale with an explicit batch or byte budget, and ensure that a bad estimate
can terminate only the worker process rather than destabilizing the computer or
the DetectKit GUI.

This document is intentionally divided into seven independently reviewable work
sets. Sets 1–3 form the minimum SAM3 safety milestone. Sets 4–6 harden inference
and all GUI-triggered model operations. Set 7 adds adaptive tuning and operational
evidence after the bounded algorithms and containment mechanisms exist.

## Required sequencing

| Set | Work | Depends on | Release gate |
|---|---|---|---|
| 1 | Bounded SAM3 data path | Current main | Required before SAM3 training is declared production-safe |
| 2 | Resource admission and process containment | Set 1 merged | Required before SAM3 training is declared production-safe |
| 2B | Bounded, process-isolated dataset preparation | Set 2 merged | Required before GUI/CLI prepare-and-train is machine-safe |
| 3 | Memory-safe SAM3 publishing | Sets 1–2 merged | Required before SAM3 training is declared production-safe |
| 4 | Bounded tile, proposal, crop, and result inference | Set 2 merged | Required for inference OOM guarantee |
| 5 | Chunked inference caches and bounded writer | Prefer Set 4 merged | Required for long-video OOM guarantee |
| 6 | Process-isolated DetectKit model operations | Sets 2 and 5 merged | Required for GUI-survival guarantee |
| 7 | Adaptive sizing, observability, and final gates | Sets 1–6 merged | Closes the hardening program |

Do not combine all seven sets into one branch. If an interface needed by a later
set changes, finish and merge the producing set first, then branch the consumer
from the new local HEAD. Sets 1–3 should be performed sequentially. Sets 4 and 5
may be prepared independently only if their agents coordinate the pipeline/cache
seam and do not edit the same files concurrently.

## Global instructions for every agent

1. Read the repository AGENTS.md and CLAUDE.md before acting. Read the active
   SAM3 plan and design when touching SAM3:
   docs/superpowers/plans/2026-08-31-detectkit-sam3-finetune.md and its linked
   design specification.
2. Inspect the primary checkout and preserve every existing user change. Create
   the dedicated worktree named in the brief from the current local HEAD, never
   from origin/main.
3. Reproduce or characterize the memory failure before editing. Add a regression
   test that fails for the right reason. Memory tests must run in an isolated
   child with deliberately small fixtures; never attempt to OOM the workstation.
4. Implement the smallest coherent change. Commit completed subsystems
   separately. Never mix formatting churn or unrelated cleanup into these
   branches.
5. Use the hydra-mps conda environment for local tests. Use hydra-cuda on
   rutalab@mehek.taild08eb9.ts.net when CUDA behavior or SAM3 training must be
   validated. Inspect processes before heavy runs and stop only confirmed stale
   Hydra/SLEAP processes.
6. Report both correctness and resource behavior. A test that proves output
   parity without bounding peak memory is insufficient; a memory test that does
   not prove semantic parity is also insufficient.
7. Review the full branch diff against its merge base. Resolve every actionable
   finding, rerun affected checks, merge into the primary checkout only after the
   branch is green, then rerun integration checks in the primary checkout.
8. Do not delete the worktree or branch until post-merge verification succeeds.
9. Do not claim a hard guarantee from a heuristic estimate. Estimates control
   admission and tuning; the worker boundary and OS enforcement provide
   containment.
10. Do not use garbage collection or empty-cache calls as the primary fix. They
    are acceptable only at explicit teardown boundaries after object lifetimes
    are corrected.

## Shared safety invariants

Every set must preserve or advance these invariants:

- Peak memory is bounded by configured in-flight work, not source length,
  dataset size, tile count, epoch count, or video duration.
- Any queue holding frames, tiles, masks, crops, or model results applies
  backpressure. Prefer byte-aware bounds when payload sizes vary materially.
- User-facing batch controls cannot bypass a hard safe maximum.
- Unlimited detections, masks, candidates, or worker prefetch are not normal
  production defaults.
- The GUI process never loads or executes a high-memory training, evaluation,
  publishing, or inference model after Set 6.
- Partial artifacts are written under temporary names and become visible only
  after validation and atomic promotion.
- Cancellation and OOM terminate the whole owned process group and never kill an
  unrelated process.
- An OOM result distinguishes host admission refusal, host soft-limit abort,
  host hard-limit kill, CUDA/MPS allocator OOM, and ordinary model failure.
- MPS unified memory is one shared host/device budget; it must not be counted as
  two independent pools.

---

## Agent brief 1 — Bounded SAM3 data path

### Dispatch prompt

Implement Set 1 of the DetectKit OOM hardening plan: make SAM3 LoRA train and
validation data loading memory-bounded and eliminate repeated image transforms
for negative prompts. Work only in a dedicated worktree created with:

    git worktree add .worktrees/detectkit-oom-sam3-streaming \
      -b codex/detectkit-oom-sam3-streaming HEAD

Do not add admission control, OS memory limits, checkpoint-publishing changes, or
general inference changes in this set.

### Primary files and seams

- src/hydra_suite/training/sam3_lora/dataloader.py
- src/hydra_suite/training/sam3_lora/datapoints.py
- src/hydra_suite/training/sam3_lora/cli.py
- src/hydra_suite/training/sam3_lora/dataset_build.py, only if lightweight
  descriptor metadata must be extended
- tests/test_sam3_dataloader.py
- New focused memory/lifetime tests under tests/

### Required investigation

1. Record the current retained bytes per 1008×1008 tile, including float image
   tensors, raw RGB images, positive object masks, and collated copies.
2. Inspect the installed Meta SAM3 Datapoint, Image, FindQueryLoaded, and
   collate_fn_api implementations in the actual SAM3 environment. Determine
   whether one Datapoint with one Image and multiple find queries is loss-
   equivalent to the current one-datapoint-per-query representation.
3. Characterize current transform call counts and the number of simultaneously
   live image tensors for a synthetic split with several tiles and negatives.
4. Confirm what raw_images is used for by Meta's collator/training path before
   removing or changing it.

### Required design

Introduce lightweight, serializable sample descriptors containing paths,
polygon/crowd metadata, positive prompt, and sampled negative prompts. Dataset
construction and epoch shuffling may materialize descriptors or integer indices,
but never decoded images, transformed tensors, dense masks, or collated batches.

Preferred representation:

- Decode a tile once when its work enters the active batch.
- Transform the image once.
- Attach the positive and sampled negative queries to the same image object.
- Rasterize positive masks only while that tile is active.
- Release tile pixels, masks, and collated output after the corresponding step.

Adopt the multi-query representation only after a parity test proves that Meta's
collator and loss produce the intended positive and negative targets. If it is
not equivalent, retain one logical datapoint per query but use a tile-scoped
shared image owner and process the tile's queries as a bounded group. Document
the semantic reason for the fallback.

Replace collate_batches and collate_epoch_batches list-returning behavior with
iterators. Preserve deterministic epoch variation by shuffling lightweight tile
or query identifiers using spec.seed + epoch. Do not shuffle tensor objects.

Training and validation must consume the same bounded implementation. Progress,
step counts, gradient accumulation, validation statistics, and empty-split error
behavior must remain correct without relying on len() of a materialized tensor
list.

### Required tests

- A synthetic N-tile test that proves no eager transform occurs while descriptors
  are built.
- Transform invocation count is once per tile, not 1 + num_negatives.
- Positive and negative queries reference the intended single image payload.
- Deterministic ordering for the same seed and different ordering for different
  epoch seeds.
- Every descriptor/query appears exactly once per epoch, including incomplete
  final batches.
- Empty and absent validation split behavior remains unchanged.
- Positive target masks, object IDs, crowd metadata, and negative empty targets
  match the previous path on a small fixture.
- An isolated-process peak-RSS test compares a small and much larger synthetic
  dataset at fixed batch/prefetch. Define a generous noise tolerance, but require
  growth to be O(batch), not O(number of tiles).
- If possible in the SAM3 environment, compare one forward/loss computation
  between old and new representations on a tiny fixed fixture.

Run at minimum:

    PYTHONPATH=src mamba run -n hydra-mps python -m pytest \
      tests/test_sam3_dataloader.py -v

Run the closest SAM3 training unit suites as well. Perform one bounded CUDA smoke
run with a tiny derived dataset before merge; report observed host RSS and CUDA
allocated/reserved peaks.

### Commit structure

1. Regression instrumentation and descriptor/ordering tests.
2. Lazy descriptor and transform-sharing implementation.
3. Lazy training/validation collation and CLI integration.
4. Any test-driven cleanup needed after branch review.

### Acceptance criteria

- No call in the train or validation path materializes all Datapoints or all
  collated batches.
- Dataset-size increases do not materially increase peak image/mask RSS at a
  fixed batch and prefetch configuration.
- A tile is transformed only once for its positive and negatives.
- Training semantics, counts, and deterministic seeds are preserved.
- Focused tests and CUDA smoke verification are green.

---

## Agent brief 2 — Resource admission and process containment

### Dispatch prompt

Implement Set 2 after Set 1 is merged. Add a shared lower-layer resource budget,
parameter-aware preflight, protected worker bootstrap, parent watchdog, and
cross-process heavy-job admission lease. Create:

    git worktree add .worktrees/detectkit-oom-containment \
      -b codex/detectkit-oom-containment HEAD

Integrate the first version with SAM3 training. Design the interfaces for reuse
by YOLO training and inference, but do not migrate every GUI operation in this
set.

### Primary files and seams

- New focused modules under src/hydra_suite/runtime/ or src/hydra_suite/utils/
  for resource budgets, leases, and sidecar supervision
- src/hydra_suite/training/sam3_lora/preflight.py
- src/hydra_suite/training/sam3_lora/train.py
- src/hydra_suite/training/sam3_lora/cli.py
- src/hydra_suite/utils/batch_optimizer.py, only to reuse or consolidate probes
- tests/test_sam3_preflight.py
- New resource-budget, supervisor, and process-limit tests

The shared modules must not import DetectKit or another app package.

### Required design

Create typed inputs and outputs rather than passing unstructured dicts. The
budget result must expose at least:

- estimated steady and peak host bytes;
- estimated accelerator steady and peak bytes;
- disk/transient artifact bytes where applicable;
- available host/device bytes at admission time;
- reserved host floor and accelerator safety fraction;
- effective batch, worker, prefetch, tile, and candidate limits when adjusted;
- refusal and warning reasons with the dominant allocations identified;
- an estimator/profile version for diagnostics and cache invalidation.

For post-Set-1 SAM3, estimate in-flight decoded pixels, one transformed image per
tile, active dense masks, collated copies, optimizer/model state, and validation
and publish phases separately. Do not multiply transformed image bytes by the
total dataset tile count after the streaming fix. Use COCO metadata to estimate
crowded-tile mask peaks without decoding the full dataset.

Use psutil.virtual_memory().available for the live host observation, but retain
an absolute and proportional reserve. The policy must be configurable and
reported. A reasonable default may reserve the larger of a fixed GiB amount and
a percentage of total RAM, but select final values from measurements and tests,
not intuition alone.

Create a child bootstrap that applies limits before importing torch, SAM3, or
Ultralytics. Do not use unsafe preexec_fn logic from a multithreaded Qt parent.
Limits must be inherited by the actual conda-launched workload and its
descendants.

Containment policy:

- On Linux, prefer cgroup v2/systemd scope enforcement where available. Support
  a soft threshold and a hard MemoryMax-style threshold, and record cgroup OOM
  evidence when mapping child exit to a diagnostic.
- Use RLIMIT_AS only as a documented POSIX fallback. Explicitly state that it
  counts virtual address space, may conflict with CUDA reservations, and does
  not constrain discrete GPU VRAM.
- On macOS/MPS, combine process isolation, supported framework allocator limits,
  conservative address-space containment, and a parent system-pressure/RSS
  watchdog. Treat MPS as unified host memory.
- On CUDA, combine host containment with device preflight, allocator telemetry,
  finite batches, and explicit handling of torch.cuda.OutOfMemoryError.

The parent watchdog must monitor the child process tree, not only the conda
wrapper PID. It must run independently of child log output and Python training
steps. At a soft limit it requests graceful termination; after a short bounded
grace it kills the owned process group. It must never target a PID or process
group it does not own.

Add a cross-process lease keyed by host and accelerator identity. Admission must
be atomic: two jobs cannot both observe the same free memory and start. Define
stale-lease recovery using PID/start-time validation rather than deleting locks
blindly.

Bound the sidecar stdout queue or retain only a fixed tail so a log flood cannot
become another memory leak.

### Required tests

- Boundary tests for host available memory, reserve floors, crowded-mask peaks,
  batch changes, precision changes, and no-CUDA behavior.
- Set 1's streaming estimate must remain independent of total tile count except
  for lightweight metadata/disk terms.
- An isolated child intentionally exceeds a tiny test cap; the parent survives,
  the complete child process tree is gone, and the result identifies a hard
  memory-limit failure.
- A cooperative child crosses the soft threshold and exits cleanly with the
  expected diagnostic.
- A blocked or silent child is still observed and terminated.
- A noisy child cannot grow the log queue without bound.
- Two concurrent admission attempts for the same resource yield one lease; a
  different resource key may proceed.
- Stale lease recovery never steals a live job's lease.
- Cancellation and ordinary nonzero exits retain their existing meanings.

Linux cgroup tests may be capability-gated, but the fallback path must be tested
locally. Perform a real CUDA SAM3 smoke run with telemetry before merge. Never
test a cap by consuming a dangerous amount of physical RAM.

### Commit structure

1. Typed resource budget and host/device probes.
2. Resource lease and isolated unit tests.
3. Sidecar bootstrap, limits, watchdog, and exit classification.
4. SAM3 preflight and launcher integration.

### Acceptance criteria

- SAM3 admission reflects host RAM and user parameters, not only a fixed VRAM
  threshold.
- A deliberately over-limit child is killed without killing or freezing the
  parent.
- Estimates and hard enforcement are clearly separated in code and messages.
- A second conflicting heavy job is refused before allocating its model.
- All focused tests and a real CUDA smoke run are green.

### Set 2 safety boundary

Set 2 contains the SAM3 training subprocess, but it is not an end-to-end OOM
guarantee. SAM3 checkpoint merge/publish remains a Set 3 dependency and must not
be described as production-safe until Sets 1–3 are complete. Dataset preparation
and inspection also still run in GUI worker threads and retain large frame/path
and COCO structures; their streaming/process-isolation root fix is a separate
follow-up set. Until that lands, prepare datasets separately with conservative
inputs and do not represent the GUI prepare-and-train workflow as machine-safe.

---

## Agent brief 2B — Bounded, process-isolated dataset preparation

Status: Implemented on `codex/detectkit-oom-dataset-preparation`; awaiting merge.

### Scope and sequencing

Set 2B branches from the merged Set 2 containment foundation and lands before
the GUI/CLI prepare-and-train workflow is described as machine-safe. It is
independent of Set 3 checkpoint publishing and must not broaden into inference,
active learning, or model execution. The implementation sequence is:

1. Add shared bounded filesystem/text primitives and explicit file, byte,
   directory-depth, pathname, line, point, metadata, and image-pixel caps.
2. Convert SAM3 COCO preparation to disk-backed ordering/splitting and
   one-frame-at-a-time decode/label/tile processing. Preserve the existing
   seeded split and output schema exactly.
3. Incrementally spool COCO arrays and manifests, fsync them, validate them,
   and atomically promote a private build directory.
4. Bound common DetectKit inspection and label/YAML/manifest reads, use fixed
   reservoir sampling for size inspection, and remove temporary eager recursive
   discovery collections in sibling builders where the shared primitive fits.
5. Run inspection plus all role derivation in a Set 2 CPU sidecar. Attach an
   immutable typed host/disk budget, canonical host lease, parent-death
   guardian, process-tree watchdog, bounded log/result protocol, and whole-tree
   cancellation. Repeat host, disk, and source-footprint checks while the lease
   is held immediately before launch.
6. Build the complete multi-role result under one private staging workspace and
   promote it only after successful validation. Cleanup may remove a private
   staging/final path only after quiescence is proved; retained-owner errors
   carry the exact recovery callback.
7. Route both the CLI and GUI worker through the sidecar. The GUI QThread may
   supervise and relay bounded events but must never inspect, decode, or derive
   the dataset itself.

### Acceptance gates

- Python heap growth is independent of source frame count for the SAM3 path;
  global deterministic order and reference-size median use SQLite rather than
  source-sized Python collections.
- No builder retains more than one decoded source image or one frame's parsed
  polygons at once; tile generation is lazy.
- File/count/depth/path/JSON/YAML/label/line/point/image-pixel caps reject
  adversarial inputs with a bounded diagnostic.
- COCO IDs, tile names, seeded frame membership/order, split semantics,
  reference size, categories, and manifest schema match pre-Set-2B fixtures.
- Disk/transient-space admission occurs before output construction and is
  repeated immediately before the child begins.
- Injection at discovery, decode, write, validation, promotion, cancellation,
  soft-limit, and hard-limit boundaries leaves no partially visible final
  dataset.
- GUI and CLI use only the contained entry point; cancellation remains
  responsive even when the child emits no progress.
- Focused preparation/SAM3/GUI/CLI tests, nearest training suites, formatting,
  lint, docs checks, a child-process RSS scaling probe, and full branch review
  are green before merge.

---

## Agent brief 3 — Memory-safe SAM3 publishing

### Dispatch prompt

Implement Set 3 after Sets 1 and 2 are merged. Reduce SAM3 publish peak memory to
approximately one base checkpoint plus one active tensor temporary, and execute
the high-memory merge inside the protected sidecar boundary. Create:

    git worktree add .worktrees/detectkit-oom-sam3-publish \
      -b codex/detectkit-oom-sam3-publish HEAD

Do not redesign the model registry or alter inference checkpoint semantics.

### Primary files and seams

- src/hydra_suite/training/sam3_lora/lora.py
- src/hydra_suite/training/sam3_lora/publish.py
- src/hydra_suite/training/sam3_lora/train.py and/or a focused publish sidecar
- src/hydra_suite/training/service.py
- Existing SAM3 publish/LoRA tests plus new peak-memory and failure-atomicity tests

### Required investigation

1. Measure current peak RSS for loading the base, cloning every tensor, forming
   LoRA deltas, hashing validation tensors, and torch.save.
2. Record the exact checkpoint wrapper/key layout, dtype behavior, metadata, and
   fingerprints expected by the current inference loader.
3. Determine whether the current serialization format permits true streaming.
   If not, implement the best bounded in-memory approach now and document a
   future sharded/safetensors migration rather than pretending torch.save is
   streaming.

### Required design

- Load one base state dictionary.
- Validate all adapter-to-base key mappings before mutating any tensor.
- Preserve any original base fingerprints needed for the unchanged/changed
  safety checks before applying an in-place update.
- For each adapter, compute and apply its delta one tensor at a time, releasing
  intermediates immediately. Do not clone the complete state dictionary.
- Preserve untouched stock-only keys, prefix behavior, dtype conversion,
  checkpoint-load guards, and published sidecar metadata exactly.
- Write the checkpoint and metadata to temporary paths. Validate the temporary
  artifact with the existing consumer-facing guards, fsync where appropriate,
  then atomically promote it.
- Run merge/serialization in a resource-capped sidecar. Keep the parent-side
  registry update small and atomic after the child reports a validated artifact.
- On cancellation, OOM, or validation failure, leave neither a registered model
  nor a partially named final checkpoint.

### Required tests

- Toy-state numerical equality between the old clone-based formula and the new
  in-place tensor-at-a-time implementation.
- Missing adapter key fails before any externally visible mutation or artifact.
- Dtype, prefix, untouched-key, and fingerprint behavior remains unchanged.
- An isolated peak-RSS test shows no second full checkpoint-sized clone.
- Inject failure during merge, save, validation, and atomic promotion; every
  case leaves the prior published model and registry intact.
- Sidecar OOM classification is surfaced through the Set 2 supervisor.
- Load the final test artifact through the same consumer guard used by semantic
  inference.

Perform a real base-checkpoint publish smoke test on an appropriate machine and
record peak RSS. Do not commit the checkpoint.

### Commit structure

1. Pre-mutation validation and in-place merge with numerical tests.
2. Atomic artifact writer and failure tests.
3. Protected publish sidecar and service integration.

### Acceptance criteria

- The complete base state dictionary is not cloned.
- Peak RSS measurement supports the claimed reduction.
- The published checkpoint remains behaviorally and structurally compatible.
- A failed publish cannot register or expose a partial artifact.
- Sets 1–3 pass together; SAM3 must not be declared production-safe until
  post-merge CUDA verification succeeds.

---

## Agent brief 4 — Bounded tile, proposal, crop, and result inference

### Dispatch prompt

Implement Set 4: make core DetectKit/TrackerKit inference bounded from frame
input through tiling, model execution, remapping, downstream crops, and result
delivery. Create:

    git worktree add .worktrees/detectkit-oom-inference-streaming \
      -b codex/detectkit-oom-inference-streaming HEAD

Set 2 must already be merged. Keep cache-format redesign out of this branch;
consume the existing CacheWriter interface so Set 5 can replace its storage.

### Primary files and seams

- src/hydra_suite/core/inference/pipeline.py
- src/hydra_suite/core/inference/stages/slicing.py
- src/hydra_suite/core/inference/stages/regions.py
- src/hydra_suite/core/inference/stages/crops.py
- src/hydra_suite/core/inference/stages/obb.py
- src/hydra_suite/core/inference/config.py
- src/hydra_suite/utils/slice_geometry.py
- src/hydra_suite/detectkit/gui/prediction_preview.py, only to reuse the shared
  bounded engine rather than maintain a second eager tiler
- Existing inference slicing, region, crop, depth, and CUDA tensor tests

### Required design

Introduce a bounded region/tile job iterator. It must:

- produce only the next permitted chunk of CPU/MPS tile copies;
- preserve zero-copy CUDA views where supported;
- predict the chunk, immediately extract compact frame-space arrays, then
  release vendor result objects and tile pixels;
- retain only the compact candidates needed for mathematically exact cross-tile
  merging;
- apply a finite resource-derived candidate cap before downstream crop or mask
  expansion;
- preserve deterministic tile order and current merge behavior on fixtures.

Do not equate the existing 4,096-tile geometric ceiling with a memory budget.
The effective plan must be admitted by estimated bytes and work. Refuse absurd
geometry early with a message containing frame size, tile size, tile count, and
estimated peak bytes.

Apply the same approach to sequential inference:

- bound stage-1 proposals before crop creation;
- extract and resize only one stage-2 crop chunk at a time;
- never interpret an unset stage2_batch_size as all crops;
- scatter compact stage-2 results before releasing the crop batch.

Downstream head-tail, CNN, pose, and AprilTag work must have a crop-batch budget
independent of detection frame batch. Avoid concatenating every detection crop
in a window. Foreign-mask handling must not clone an unbounded window-sized crop
tensor.

Change full-video Pipeline result retention to an explicit policy. The normal
runner path should count progress and stream results to consumers/caches without
retaining every FrameResult. Preserve an opt-in collection mode for callers and
tests that genuinely need an in-memory result list.

Set finite production defaults for raw detection and semantic candidate caps,
derived from expected animal counts where available. Retain an explicit expert
override only if it remains bounded by the resource budget.

### Required tests

- Tile input and merged-output parity for CPU NumPy, MPS where available, and
  CUDA tensor paths on small fixtures.
- A fake slow model records that no more than the configured tile chunk is live.
- Peak RSS remains bounded as tile count increases at fixed chunk/candidate cap.
- Sequential stage 2 never receives more than its effective crop batch.
- Downstream crop models receive all detections exactly once with correct frame
  and detection IDs.
- Foreign masking stays byte-identical on existing fixtures.
- Excessive tile geometry and candidate counts refuse before crop
  materialization.
- The normal pipeline does not retain full-video FrameResults; opt-in collection
  still works.
- Pipeline depth and cancellation tests remain green and no producer can remain
  blocked during teardown.

Run the nearest slicing, OBB, region, crop, pipeline-depth, MPS, and CUDA suites.
Because this is performance-sensitive tracking code, run the repository
equivalence harness as required by CLAUDE.md. Verify non-empty CSV outputs,
compare both forward and tracking-final CSVs, and enforce PERF_TOLERANCE.

### Commit structure

1. Bounded tile iterator and direct-sliced integration.
2. Bounded sequential proposals/crops.
3. Bounded downstream crop processing and candidate caps.
4. Optional Pipeline result collection.
5. Preview reuse of the shared bounded path.

### Acceptance criteria

- Tile and crop pixels scale with configured chunk size, not total tiles or
  detections.
- Full-video inference does not retain all FrameResults by default.
- Existing outputs remain equivalent within the repository's defined gates.
- MPS and CUDA verification is complete for the affected paths.

---

## Agent brief 5 — Chunked inference caches and bounded writer

### Dispatch prompt

Implement Set 5: replace whole-run in-memory cache accumulation with an
appendable, crash-resumable format and enforce bounded backpressure in
CacheWriter. Create:

    git worktree add .worktrees/detectkit-oom-chunked-cache \
      -b codex/detectkit-oom-chunked-cache HEAD

Prefer to start after Set 4 is merged. Preserve read compatibility with existing
NPZ caches. Do not silently rewrite or delete user caches.

### Primary files and seams

- src/hydra_suite/core/inference/cache/writer.py
- src/hydra_suite/core/inference/cache/store.py
- src/hydra_suite/core/inference/cache/reader.py
- src/hydra_suite/core/inference/cache/reuse.py
- src/hydra_suite/core/inference/cache/base.py
- src/hydra_suite/core/inference/runner.py
- tests/test_inference_cache_writer.py and cache/reuse/pipeline tests

### Required design investigation

Before implementation, write a short design note in the branch comparing:

- a directory of immutable NPZ chunks plus an atomic manifest;
- Zarr/HDF5 or another appendable dependency;
- a compact SQLite/Arrow representation where appropriate.

Prefer the least complex format that supports bounded writes, indexed frame
reads, atomic completion, and no whole-file load. Avoid adding a heavy dependency
unless it materially improves correctness and packaging remains reliable.

The format must represent explicitly processed empty frames, detection IDs,
class IDs, downstream phase data, pose validity, AprilTags, cache key/version,
and the covered frame range. It must support interrupted-run resume without
mistaking an incomplete chunk for complete data.

Write one bounded chunk at a time. A safe pattern is write temporary chunk,
flush/close, atomic rename, then atomically update the manifest. Never concatenate
the entire run on close. Reads must load only the requested chunk or memory-map
an indexed backing store.

CacheWriter must enforce backpressure. An item-count limit is acceptable only if
every item is tightly bounded; otherwise add byte accounting and block producers
when queued payload bytes exceed the configured budget. Worker failure must wake
blocked producers and surface exactly once without deadlock.

Keep legacy NPZ files readable. New writes should use the new format. If an
explicit migration helper is added, it must write beside the old cache and
atomically promote only after validation; automatic destructive migration is
forbidden.

### Required tests

- Detection and every downstream cache type round-trip across multiple chunks.
- Processed empty frames remain distinguishable from missing frames.
- Random indexed reads do not load unrelated chunks.
- Peak RSS remains bounded as frame count grows at fixed chunk size.
- A deliberately slow writer causes producer backpressure at the configured
  queue/byte bound.
- Worker exceptions wake blocked producers; flush/close never deadlocks.
- Crash before chunk rename, after chunk rename, and before manifest promotion
  produces a recoverable and honest cache state.
- Resume processes only missing/incomplete frames.
- Legacy NPZ fixtures retain read parity.
- Cache-key mismatch and truncated data are rejected, not treated as valid empty
  results.
- Forward/backward tracking cache coverage behavior remains correct.

Run cache, pipeline-depth, runner, AL reuse, and relevant equivalence tests. Use
long synthetic metadata/results, not huge decoded frames, for the memory test.

### Commit structure

1. Format design note, schema, and low-level chunk store.
2. Detection and downstream handle implementations.
3. Bounded CacheWriter and failure/backpressure handling.
4. Runner/reuse integration and legacy compatibility.

### Acceptance criteria

- No cache handle retains a whole video's results.
- Closing a cache does not allocate a whole-run concatenation.
- Queue memory is bounded and backpressure is proven.
- Interrupted writes are recoverable and legacy caches remain readable.

---

## Agent brief 6 — Process-isolated DetectKit model operations

### Dispatch prompt

Implement Set 6 after Sets 2 and 5 are merged. Move every high-memory
DetectKit-triggered model operation out of the GUI process and route it through
the shared protected sidecar supervisor. Create:

    git worktree add .worktrees/detectkit-oom-gui-sidecars \
      -b codex/detectkit-oom-gui-sidecars HEAD

Do not add model/business logic to MainWindow. Keep workers and protocol code in
focused modules.

### Operations in scope

- Dataset-wide direct, sliced, sequential OBB/detect/segment inference
- DetectKit model evaluation
- SAM3 semantic escalation
- Semantic calibration and previews that load SAM3
- Any SAM3 publishing still reachable from the GUI process after Set 3
- Generic Ultralytics training launch integration if it does not yet use the Set
  2 supervisor and memory policy

### Primary files and seams

- src/hydra_suite/detectkit/gui/main_window.py
- src/hydra_suite/detectkit/jobs/evaluation.py
- src/hydra_suite/detectkit/jobs/semantic_escalation.py
- src/hydra_suite/detectkit/gui/prediction_preview.py
- src/hydra_suite/detectkit/jobs/inference_stager.py
- src/hydra_suite/training/runner.py
- New focused Qt-free sidecar protocol/entry-point modules
- Existing BaseWorker remains a thin GUI coordinator, not the model executor

### Required design

Define a versioned, typed request/result protocol. Requests may use JSON for
small metadata, but frame arrays, polygons, masks, and complete result sets must
travel through bounded files/chunks or an explicitly bounded transport—not an
unbounded stdout message or one giant returned dictionary.

For dataset inference:

- The sidecar processes one admitted frame/tile/crop window at a time.
- Results are committed incrementally through the Set 5 cache/staging format.
- The GUI retains only aggregate counts, the current/visible image's candidates,
  and a small bounded LRU for navigation.
- Slider filtering reads indexed results; it must not duplicate every detection
  into multiple whole-source dicts/lists.

For semantic escalation and calibration:

- Store candidates per frame/chunk instead of loading and rewriting one complete
  JSON document after each frame.
- Use a finite, stratified calibration sample budget and report which frames were
  sampled.
- Enforce max instances inside the predictor if the backend supports it, before
  full mask materialization. If not, document the limitation and retain a strict
  admitted tile/candidate budget.
- Keep candidate collection and merge resumable without accepting a half-tiled
  frame as complete.

For evaluation and generic training, launch the tool in a protected sidecar and
stream bounded progress. Avoid importing Ultralytics or torch model weights in
the GUI process.

Cancellation must terminate the owned process group even when the child is
silent or blocked in native code. Child OOM must yield a specific, actionable UI
message with the configured limit, observed peak if available, and suggested
safe adjustment. It must not be collapsed into a generic exception.

### Required tests

- Assert GUI worker modules do not load model weights or call model.predict/val
  in-process; use import seams or fake executors.
- Protocol round-trip, version mismatch, malformed messages, and bounded log
  handling.
- Dataset inference writes incrementally and the GUI's retained result cache has
  a fixed bound independent of source length.
- Slider/staging semantics remain unchanged on existing fixtures.
- Evaluation results and persisted metrics match the old path.
- Semantic cancellation never commits a half-frame.
- Calibration respects its sample budget and remains deterministic.
- Simulated child host OOM, accelerator OOM, cancellation, crash, and success all
  produce distinct results while leaving the GUI process alive.
- Process groups and temporary artifacts are cleaned without touching unrelated
  processes/files.

Run focused Qt tests offscreen, job tests, inference staging tests, semantic
tests, evaluation tests, and one end-to-end GUI-launched smoke operation on MPS
and CUDA where applicable.

### Commit structure

1. Versioned sidecar protocol and Qt-free entry points.
2. Dataset inference isolation and indexed GUI result access.
3. Evaluation/training supervisor migration.
4. Semantic escalation/calibration/preview isolation.
5. UI diagnostics and teardown hardening.

### Acceptance criteria

- The GUI process does not construct a high-memory model for any in-scope
  operation.
- Dataset-wide results are not retained in one GUI dictionary.
- OOM or native hangs kill only the sidecar and leave the GUI responsive.
- Existing inference, staging, evaluation, and semantic behavior remains
  equivalent on fixtures.

---

## Agent brief 7 — Adaptive sizing, observability, and final safety gates

### Dispatch prompt

Implement Set 7 only after Sets 1–6 are merged and verified. Add empirical memory
profiles, bounded adaptive batch selection, structured peak reporting, and final
cross-platform safety/equivalence gates. Create:

    git worktree add .worktrees/detectkit-oom-adaptive \
      -b codex/detectkit-oom-adaptive HEAD

Do not weaken hard limits or reintroduce eager algorithms in pursuit of higher
throughput.

### Primary files and seams

- Shared resource budget/supervisor modules from Set 2
- src/hydra_suite/utils/batch_optimizer.py
- Training and inference configuration schemas
- Training history/run manifests and inference profiling outputs
- Documentation for memory policy and OOM diagnostics
- tools/equivalence/ only where reusable measurement support belongs there

### Required design

Replace the current input-bytes-times-constant estimator with empirical profiles
keyed by at least:

- operation type;
- model/checkpoint identity or architecture family;
- backend and device identity;
- precision;
- task type;
- input resolution and tiling mode;
- batch, pipeline depth, worker/prefetch, and crop batch;
- relevant adapter scope/rank for SAM3 training.

Profiles must be versioned and invalidated when the estimator/schema or model
identity changes. Never transfer a CUDA profile to MPS or treat MPS device memory
as independent of host memory.

For an unknown profile, perform a bounded probe at the smallest viable batch.
Use measured allocated/reserved accelerator memory and process-tree RSS to fit a
conservative batch recommendation. Increase via bounded search only while the
reserve remains intact. The probe itself must run inside the hard-contained
sidecar.

Adaptive retry rules:

- Retry only recognized accelerator OOMs or resource soft-limit failures.
- Reduce the specific pressure source: frame batch, tile chunk, crop batch,
  workers/prefetch, or cache chunk—not an unrelated parameter.
- Clear failed child state by starting a fresh sidecar; do not rely on a damaged
  allocator context.
- Use a small finite retry count and never retry a host hard-limit kill with the
  same configuration.
- Preserve deterministic seeds and record every adjustment in the run manifest.
- Do not mask shape errors, corrupted checkpoints, data errors, or programmer
  exceptions as OOM retries.

Emit structured telemetry for each phase: admission estimate, applied limits,
effective parameters, peak parent/child RSS, system minimum available RAM,
accelerator allocated/reserved peak, queue high-water marks, cache chunk size,
exit classification, and retry history. Keep filesystem paths and sensitive
environment values out of general telemetry unless already part of the local run
manifest.

Review temporary conservative clamps introduced by earlier sets. Relax them only
when empirical profiles and containment tests prove the new range safe.

### Required tests and final verification

- Profile-key separation and invalidation tests.
- Unknown-device probe begins at the minimum and cannot exceed the hard budget.
- Monotonic recommendations: increasing available memory cannot recommend a
  smaller batch absent another changed constraint; larger inputs cannot
  recommend a larger batch.
- Recognized OOM reduces the correct pressure parameter and succeeds on a fresh
  child; unrelated exceptions do not retry.
- Retry count is finite and every adjustment is recorded.
- MPS unified-memory accounting is not fractioned twice.
- Concurrent jobs honor Set 2's leases and reservations.
- Structured telemetry accurately reports simulated peaks/high-water marks.
- Complete Sets 1–6 regression suites pass.
- Full appropriate MPS and CUDA equivalence runs produce non-empty CSVs, match
  required forward/tracking-final outputs, and remain within PERF_TOLERANCE.
- Long synthetic train, tile, cache, and dataset-inference soak tests show no
  source-length-correlated RSS growth.

### Commit structure

1. Versioned profile schema and probe measurements.
2. Adaptive batch/chunk selection and bounded retry.
3. Structured telemetry and user-facing reporting.
4. Cross-platform soak/equivalence gates and documentation.

### Acceptance criteria

- Safe defaults are measurement-backed and device-specific.
- Every automatic adjustment is bounded, visible, and reproducible.
- Host and accelerator peaks remain below enforced budgets during soak tests.
- The final cross-platform and equivalence gates are green.

---

## Program completion checklist

The hardening program is complete only when all of the following are true:

- [ ] SAM3 train and validation peak image/mask memory is independent of dataset
      size at fixed in-flight settings.
- [ ] Positive and negative SAM3 queries share one tile transform without loss
      or target-semantic drift.
- [ ] Host and accelerator admission reflects real parameters and preserves a
      system reserve.
- [ ] A test child that exceeds its limit is killed while its parent and the GUI
      survive.
- [ ] SAM3 publishing does not clone a complete checkpoint and is atomic.
- [ ] Tile, proposal, and downstream crop memory is bounded end-to-end.
- [ ] Full-video FrameResults are not retained unless explicitly requested.
- [ ] Cache queues and storage are bounded; long-video cache close does not
      concatenate the complete run.
- [ ] Dataset inference, evaluation, semantic work, publishing, and training all
      execute outside the GUI process.
- [ ] A global lease prevents conflicting heavy jobs from overcommitting the same
      host/device.
- [ ] MPS and CUDA tests, equivalence comparisons, and soak tests are complete
      with non-empty outputs and recorded peak metrics.
- [ ] All completed implementation plans/designs are moved to their done/
      directories during final merge according to AGENTS.md.

## Required final handoff from every agent

Each agent's final report must include:

1. Branch and commit list.
2. Files and public interfaces changed.
3. Reproduction or baseline measurements.
4. Exact focused, broader, formatting, lint, documentation, equivalence, and
   platform commands run, with pass/fail/skip status.
5. Peak RSS and accelerator-memory measurements relevant to the set.
6. Full-diff review findings and how each was resolved.
7. Remaining limitations or follow-up work, without calling the set verified if
   a required gate did not run or failed.
