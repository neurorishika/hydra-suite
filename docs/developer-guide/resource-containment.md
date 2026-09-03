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
is rejected. CUDA remains a discrete pool and must have a measured free-memory
value before admission.

`hydra_suite.runtime.resource_lease` provides a non-blocking, host-local lease.
Acquire this lease before the final resource probe so two jobs cannot both
observe the same free memory and begin allocating. The OS file lock is
authoritative. Owner metadata includes both PID and process creation time to
guard against PID reuse; stale metadata is overwritten only after the OS lock
has been acquired.

`hydra_suite.runtime.resource_limits` builds a protected launch command. The
minimal child bootstrap applies limits and MPS environment configuration before
it replaces itself with the conda launcher or workload. Its import path must
remain free of torch, SAM3, Ultralytics, ONNX Runtime, and Qt imports.

`hydra_suite.runtime.process_supervisor` owns the dedicated child process group,
monitors tree RSS and system available memory independently of log or training
progress, and retains only a bounded output tail. Exit classification keeps
admission refusal, soft host limit, hard host limit, accelerator OOM, user
cancellation, signal termination, and ordinary failure distinct.

## Platform enforcement

### Linux

When a delegated user cgroup v2 manager is available, use a transient systemd
scope with `MemoryHigh` and `MemoryMax`. The parent watchdog requests graceful
termination at the same soft boundary; `MemoryMax` is the kernel backstop.
Cgroup result properties are evidence for classifying a kernel OOM kill.

If a user scope is unavailable, Linux falls back to `RLIMIT_AS`. This limits
virtual address space rather than resident memory. CUDA commonly reserves large
virtual mappings, and `RLIMIT_AS` does not constrain discrete GPU VRAM. Record
these limitations in the run manifest and continue using device preflight,
finite batches, accelerator OOM handling, and the parent watchdog.

### macOS and MPS

macOS exposes an `RLIMIT_AS` constant but rejects attempts to set it, so the
runtime must not claim that boundary. Configure PyTorch's MPS high-watermark
ratio before importing torch and enforce both the child-tree RSS limit and the
system available-memory reserve from the parent. MPS allocations consume the
same physical pool as ordinary host allocations.

### Windows

The current foundation provides watchdog enforcement but no Job Object memory
adapter. Until a Job Object implementation is added, diagnostics must disclose
that there is no kernel hard cap.

## Integration order

For each high-memory operation:

1. Derive lightweight, phase-based estimates from the job configuration.
2. Acquire the accelerator or unified-memory lease.
3. Probe live resources and evaluate the request while holding the lease.
4. Refuse with the budget's dominant phase and allocations, or calculate soft
   and hard child limits while preserving the configured system reserve.
5. Build the protected child command and start `SupervisedSidecar` in a fresh
   process group.
6. Consume output through `BoundedLineBuffer`, forward structured progress, and
   preserve only the bounded tail.
7. Classify completion from watchdog, cgroup, allocator, cancellation, and exit
   evidence before accepting any artifact.
8. Release the lease only after the complete child process group has exited and
   artifacts have been validated.

Never apply these limits to the GUI process. Partial artifacts must remain under
temporary names until a successful child result has been validated.
