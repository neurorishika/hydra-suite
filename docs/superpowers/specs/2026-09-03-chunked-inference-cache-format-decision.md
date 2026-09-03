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

Each logical cache keeps its existing canonical `*.npz` path, but new files at
that path contain only a small manifest. Immutable payload chunks live in a
sibling directory. A chunk is written to a temporary file, flushed, atomically
renamed, and only then referenced by an atomically replaced manifest. The
manifest records the cache key, cache kind, format version, covered frame IDs,
chunk byte sizes, and checksums. Unreferenced chunks from an interruption are
never considered complete.

Advantages:

- NumPy is already required, so packaging gains no new dependency.
- A read opens only the manifest and the chunk containing the requested frame.
- Chunk size directly bounds write buffering and close-time allocation.
- The canonical filename remains compatible with artifact discovery code.
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

The canonical `*.npz` is an atomic manifest containing:

- chunked-format version and cache kind;
- serialized `CacheKey`;
- session identifier;
- ordered chunk metadata, including compressed exact processed-frame ranges;
- payload byte size and SHA-256 for each referenced immutable chunk.

Payload chunks are ordinary, non-pickled NPZ files. Every chunk includes its
exact processed-frame array even when all corresponding results are empty. Detection
IDs, class IDs, head/tail arrays, CNN factor vocabulary and probabilities, pose
validity, and AprilTag arrays remain typed NumPy arrays.

A frame is complete only when it appears in a chunk referenced by the current
manifest. Temporary files and renamed-but-unpublished chunks are ignored. A
valid existing manifest is appendable, so a resumed run can query missing frames
and commit only those results. Cache-key mismatch, missing/truncated referenced
chunks, and malformed metadata invalidate the cache rather than becoming empty
results.

Legacy NPZ files remain read-only compatible. New writes use the chunked format;
there is no automatic migration or deletion of a legacy cache.

## Memory and concurrency contract

Each handle buffers at most one configured chunk and releases it immediately
after publication. Indexed readers retain at most one decoded chunk. The async
`CacheWriter` additionally accounts for payload bytes and blocks producers once
its configured queue-byte budget is occupied. A worker failure wakes blocked
producers and is surfaced once without waiting for an undrainable queue.
