# Chunked inference cache format decision

**Status:** Accepted for implementation

## Context

The inference cache handles currently retain every frame result in Python lists
and concatenate the complete run when `close()` writes one NPZ file. Reads use
`dict(np.load(...))`, which likewise materializes the whole cache. Both paths
therefore grow with video length and can exhaust host memory.

The replacement must bound resident data, distinguish a processed empty frame
from a missing frame, support indexed reads and interrupted-run resume, and keep
existing NPZ caches readable without destructively migrating them.

## Options considered

### Immutable NPZ chunks plus an atomic manifest — selected

Each logical cache generation contains a small `*.npz` manifest and immutable
payload chunks in a sibling directory. Chunks are created with exclusive-create
semantics, flushed, and only then referenced by an atomically replaced member
manifest. One bounded `cache_set.json` indirection selects a shared generation
of detection, head/tail, CNN, pose, and AprilTag members with a single atomic
rename. Resumes hard-link the selected immutable chunks into a staging revision
of the same run generation, append only missing frames there, and atomically
select that revision after all members validate. An unchanged resume discards
its staging revision. Canonical `*.npz` compatibility links remain available for existing
artifact discovery, but cache-aware readers treat the set manifest as the
commit point. Unreferenced generations and chunks from an interruption are
never considered complete.

Advantages:

- NumPy is already required, so packaging gains no new dependency.
- A read opens only the manifest and the chunk containing the requested frame.
- Chunk size directly bounds write buffering and close-time allocation.
- Canonical filenames remain compatible with artifact discovery code.
- Legacy monolithic NPZ files can be detected and read through the same handle.

Tradeoffs are additional small files and lower compression/throughput than a
specialized array store. Chunk sizes can be tuned without changing the schema.

### Zarr or HDF5

Both provide mature chunked arrays and indexed reads. HDF5 adds a native binary
dependency and has awkward concurrent/crash semantics around a single mutable
file. Zarr would add a new dependency and still requires careful atomic metadata
publication. Neither materially improves correctness for the current
single-writer, per-frame access pattern enough to justify its packaging cost.

### SQLite or Arrow

SQLite has excellent transactions and indexing, but multidimensional numeric
arrays would need a custom blob schema and copies during encode/decode. Arrow is
well suited to tabular detections but less natural for ragged CNN factors,
keypoints, and tag corners, and it adds a substantial dependency. A future
cross-run analytics store may justify Arrow; this cache does not.

## On-disk contract

Each generation's `*.npz` member is an atomic manifest containing:

- chunked-format version and cache kind;
- serialized `CacheKey`;
- shared generation and member-session identifiers;
- ordered chunk metadata, including compressed exact processed-frame ranges;
- payload byte size and SHA-256 for each referenced immutable chunk.

Payload chunks are ordinary, non-pickled NPZ files. Every chunk includes its
expected cache kind, key, generation, session, manifest position, and exact
processed-frame array even when all corresponding results are empty. Detection
IDs, class IDs, head/tail arrays, CNN factor vocabulary and probabilities, pose
validity, and AprilTag arrays remain typed NumPy arrays.

A frame is complete only when it appears in a chunk referenced by the current
manifest. Temporary files and renamed-but-unpublished chunks are ignored. A
valid existing manifest is appendable, so a resumed run can query missing frames
and commit only those results. Cache-key mismatch, missing/truncated referenced
chunks, and malformed metadata invalidate the cache rather than becoming empty
results.

A deliberate replacement uses a new shared generation directory and keeps the
prior cache-set manifest visible until every enabled member closes and passes
deep validation with equal coverage. Stops, exceptions, per-member crashes, and
stuck cache-writer teardown leave replacement generations unselected rather
than promoting partial coverage over the last reusable generation.

Before `np.load`, ZIP member names/counts, compressed and declared sizes,
aggregate bytes, expansion ratios, and NPY headers are bounded. Reusable reads
then enforce kind-specific fields, dtypes, ranks, shapes, row alignment, frame
ordering, and per-frame identifier uniqueness.

Legacy numeric NPZ files remain read-only compatible. Object-dtype legacy NPY
members are retired because reading them requires executable pickle
deserialization; metadata preflight rejects them before NumPy loads any array.
New writes use the chunked format; there is no automatic migration or deletion
of a legacy cache.

## Memory and concurrency contract

Each handle has a byte-triggered buffer limit in addition to its frame-count
limit and rejects a single frame larger than that budget. Indexed readers retain
at most one decoded chunk. `CacheWriter` treats its configured limit as an
aggregate retained-memory budget: it partitions bytes across handle buffers and,
in asynchronous mode, the queued/current payload. Both synchronous and
asynchronous writers reject an oversized payload. A worker failure wakes blocked
producers, while flush and close use bounded deadlines so callers never close a
handle that a stuck worker may still be using.

The inference pipeline itself is deliberately fail-closed around a backend call:
Python cannot safely kill a thread inside model/device inference, so `run()` does
not return or release model ownership until its producer exits. Set 5 therefore
does not claim a bounded wall-clock teardown for a hung backend. The process
isolation in Set 6 and the Set 2 watchdog provide that hard bound by terminating
the owning worker process.
