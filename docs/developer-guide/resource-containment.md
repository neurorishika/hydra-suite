# Resource admission and process containment

High-memory jobs need two separate safety mechanisms:

1. Admission estimates decide whether a proposed job is reasonable.
2. Process containment protects the computer when an estimate is wrong.

An admitted estimate is never a hard guarantee. Heavy model work must still run
in a supervised child process.

## Foundation modules

`hydra_suite.runtime.resource_budget` defines typed, phase-based estimates. A
request reports host, accelerator, and transient disk costs separately, along
with the effective batch, worker, prefetch, tile, and candidate limits. Training,
validation, and publishing are different phases, so their peaks are compared
rather than incorrectly summed.

MPS uses unified memory. For an MPS observation, host and accelerator estimates
are added and evaluated against one host pool. A separate MPS device-memory pool
is rejected. CPU observations likewise cannot claim an accelerator pool, and a
request with accelerator allocations fails closed when the resolved observation
is CPU-only. CUDA remains a discrete pool and must have a measured free-memory
value before admission. Diagnostics retain distinct host-dominant and
accelerator-dominant phases and allocation lists because those peaks need not
occur together.

`hydra_suite.runtime.resource_lease` provides a non-blocking, host-local lease.
Acquire this lease before the final resource probe so two jobs cannot both
observe the same free memory and begin allocating. The OS file lock is
authoritative. Owner metadata includes both PID and process creation time to
guard against PID reuse; stale metadata is overwritten only after the OS lock
has been acquired. Use `canonical_heavy_job_lease`: it stores locks below the
shared Hydra data directory, maps every MPS job to the single unified-memory
key, and prefers a CUDA device UUID over a logical device index.

`hydra_suite.runtime.resource_limits` builds a protected launch command. The
minimal child bootstrap applies limits and MPS environment configuration before
it replaces itself with the conda launcher or workload. Its import path must
remain free of torch, SAM3, Ultralytics, ONNX Runtime, and Qt imports.

`hydra_suite.runtime.process_supervisor` owns the dedicated child process group,
monitors tree RSS and system available memory independently of log or training
progress, and retains only a bounded output tail. Exit classification keeps
admission refusal, soft host limit, hard host limit, accelerator OOM, user
cancellation, signal termination, and ordinary failure distinct.

`LimitedLaunch` retains its immutable `ProcessMemoryLimits`; a
`ContainmentPlan` derives watchdog thresholds from those same values. Callers
must not pass a second, independently calculated soft or hard boundary. The
supervisor acquires the heavy-job lease itself and keeps it until every owned
process has been terminated and reaped. A timed-out `wait()` performs that
teardown before raising; the exceptional `WorkloadStillOwnedError` explicitly
returns the still-owning sidecar if the operating system cannot confirm exit.

## Platform enforcement

### Linux

When a delegated user cgroup v2 manager is available, use a transient systemd
scope with `MemoryHigh` and `MemoryMax`. The parent watchdog requests graceful
termination at the same soft boundary; `MemoryMax` is the kernel backstop.
`MemorySwapMax=0` prevents the scope from exhausting swap to evade the resident
limit. Cgroup result properties are collected with a bounded timeout and retain
an explicit unavailable/error state when the transient unit has disappeared.
Systemd remains the authoritative tree-wide signal mechanism for a scope.

If a user scope is unavailable, Linux falls back to `RLIMIT_AS`. This limits
virtual address space rather than resident memory. CUDA commonly reserves large
virtual mappings, and `RLIMIT_AS` does not constrain discrete GPU VRAM. Record
these limitations in the run manifest and continue using device preflight,
finite batches, accelerator OOM handling, and the parent watchdog.
The bootstrap installs Linux `PR_SET_PDEATHSIG` and closes the parent-exit race
before importing the workload, so a dead supervisor cannot orphan the direct
child.

### macOS and MPS

macOS exposes an `RLIMIT_AS` constant but rejects attempts to set it, so the
runtime must not claim that boundary. Configure PyTorch's MPS high-watermark
ratio deliberately before importing torch—an MPS launch without an explicit
ratio is rejected—and enforce both the child-tree RSS limit and the system
available-memory reserve from the parent. A minimal inherited-pipe guardian
kills the watchdog-only child group if the supervisor disappears. MPS
allocations consume the same physical pool as ordinary host allocations.

### Windows

The current foundation provides watchdog enforcement but no Job Object memory
adapter. Until a Job Object implementation is added, diagnostics must disclose
that there is no kernel hard cap.

## Integration order

For each high-memory operation:

1. Derive lightweight, phase-based estimates from the job configuration.
2. Construct the canonical accelerator or unified-memory lease.
3. Pass final resource probing/evaluation as the supervisor's pre-launch check;
   it runs after the supervisor acquires the lease and before it creates a child.
4. Refuse with the budget's dominant phase and allocations, or calculate soft
   and hard child limits while preserving the configured system reserve.
5. Build one `ContainmentPlan` and start `SupervisedSidecar` in a fresh process
   group; any setup failure after spawn kills and reaps that child.
6. Consume output through `BoundedLineBuffer`, forward structured progress, and
   preserve only the bounded tail.
7. Classify completion from watchdog, cgroup, allocator, cancellation, and exit
   evidence before accepting any artifact.
8. Release the lease only after the process group, captured escaped descendants,
   and systemd scope (when used) have been torn down and result evidence has been
   collected.

Never apply these limits to the GUI process. Partial artifacts must remain under
temporary names until a successful child result has been validated.
