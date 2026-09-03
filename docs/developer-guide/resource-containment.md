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

`hydra_suite.runtime.resource_lease` provides non-blocking, host-local lease
sets. The supervisor constructs this set internally before the final resource
probe, so callers cannot substitute raw keys or an alternate lock directory.
CUDA jobs lease both the shared host-RAM pool and their physical accelerator.
MPS deduplicates those aliases into its one unified-memory key. A logical CUDA
index is never accepted: `LimitedLaunch` must retain the physical UUID or PCI
identity supplied by the runtime resolver. The OS file locks are authoritative.
Owner metadata includes both PID and process creation time to guard against PID
reuse; stale metadata is overwritten only after the OS lock has been acquired.
Locks live below the shared Hydra data directory.

`hydra_suite.runtime.resource_limits` builds a protected launch command. The
minimal child bootstrap applies limits and MPS environment configuration before
it replaces itself with the conda launcher or workload. Its import path must
remain free of torch, SAM3, Ultralytics, ONNX Runtime, and Qt imports.
`hydra_suite.runtime.process_guardian` is a separate, out-of-scope process. The
bootstrap waits on a start gate until that guardian has proved it can scan
launch-scoped process identities.

`hydra_suite.runtime.process_supervisor` owns the dedicated child process group,
monitors tree RSS and system available memory independently of log or training
progress, and reads output in fixed-size chunks before incrementally splitting
records into a bounded tail. Exit classification keeps
admission refusal, soft host limit, hard host limit, accelerator OOM, user
cancellation, signal termination, and ordinary failure distinct.

`LimitedLaunch` retains its immutable `ProcessMemoryLimits`; a
`ContainmentPlan` derives watchdog thresholds from those same values. Callers
must not pass a second, independently calculated soft or hard boundary. The plan
derives its canonical host/device keys from the local host and resolved physical
device; raw caller keys are not accepted. The supervisor acquires the complete
set itself and keeps it until every owned process has been terminated and
reaped. It also requires a quiescence acknowledgement from the guardian before
reporting success or releasing locks. A timed-out `wait()` performs teardown
before raising; the exceptional `WorkloadStillOwnedError` explicitly returns
the still-owning sidecar if the operating system cannot confirm exit.

## Platform enforcement

### Linux

When a delegated user cgroup v2 manager is available, use a transient systemd
scope with `MemoryHigh` and `MemoryMax`. The parent watchdog requests graceful
termination at the same soft boundary; `MemoryMax` is the kernel backstop.
`MemorySwapMax=0` prevents the scope from exhausting swap to evade the resident
limit, and `TasksMax` provides a kernel backstop for process-tree fan-out.
Cgroup result properties are collected with a bounded timeout and retain
an explicit unavailable/error state when the transient unit has disappeared.
Systemd remains the authoritative tree-wide signal mechanism for a scope. A
successful `systemctl kill` request is not sufficient: the exact unit must also
be inactive with `cgroup.events` proving its recursive population is zero; only
an unloaded unit is clear when that cgroup evidence is unavailable. Unit
membership compares exact
cgroup path components, never substrings. Only token-captured processes proven
outside the scope may be signalled directly. The guardian is launched by the
supervisor outside the limited cgroup, so `MemoryMax` and `TasksMax` cannot kill
the process responsible for cleanup and lock retention. Scope names are random
and internal, and guardian readiness binds the known launcher identity to an
exact cgroup plus systemd `InvocationID`. The same invocation is revalidated
before signalling, so a colliding or reused unit is never treated as owned. An
already-unloaded unit is quiescent; transport and malformed-evidence failures
remain unknown and retain ownership.

If a user scope is unavailable, Linux falls back to `RLIMIT_AS`. This limits
virtual address space rather than resident memory. CUDA commonly reserves large
virtual mappings, and `RLIMIT_AS` does not constrain discrete GPU VRAM. Record
these limitations in the run manifest and continue using device preflight,
finite batches, accelerator OOM handling, and the parent watchdog.
The bootstrap installs Linux `PR_SET_PDEATHSIG` and closes the parent-exit race
before importing the workload, so a dead supervisor cannot orphan the direct
child. Every POSIX backend also starts a separate-session parent-liveness
guardian before releasing the workload's start gate. A random launch token is
inherited by descendants, allowing the guardian to find a short-lived parent's
daemonized `setsid` child after ancestry has disappeared. On normal completion
the supervisor requests teardown and waits for an acknowledgement after two
stable empty token scans. On abrupt parent death, pipe EOF or loss of the
supervisor's PID-plus-creation-time identity triggers the same cleanup, so an
unrelated post-launch fork cannot conceal parent death by retaining the pipe.
Supervisor-only liveness, acknowledgement, and lease descriptors are closed in
such fork children; the guardian remains the sole independent lease holder.
Fallback token scanning records the launch start identity and does not inspect
process environments that provably predate it. Any inaccessible concurrent
process is classified while the workload is still gated; a later new or owned
inaccessible identity fails closed. Systemd uses its exact cgroup boundary and
never scans process environments.
Identity registries are locked, use a bounded streaming parent-PID scan, retain
a first overflow identity, and never treat access denial as process death. Failed
direct signals permanently retain ownership. If authoritative cleanup cannot be
proved, the guardian retains inherited locks instead of declaring resources
free.

### macOS and MPS

macOS exposes an `RLIMIT_AS` constant but rejects attempts to set it, so the
runtime must not claim that boundary. Configure PyTorch's MPS high-watermark
ratio deliberately before importing torch—an MPS launch without an explicit
ratio is rejected—and enforce both the child-tree RSS limit and the system
available-memory reserve from the parent. The token-scanning guardian protects
watchdog-only launches and session escapees if the supervisor disappears. A
host that forbids process-table enumeration is refused because it cannot
provide that fallback guarantee; unrelated processes with protected
environments do not block launch merely because their environments are private.
MPS allocations consume the same physical pool as ordinary host allocations.

### Windows

The current foundation provides watchdog enforcement but no Job Object memory
adapter. Until a Job Object implementation is added, diagnostics must disclose
that there is no kernel hard cap. Leased heavy-job launches fail closed because
the POSIX parent-death guardian is unavailable.

## Integration order

For each high-memory operation:

1. Derive lightweight, phase-based estimates from the job configuration.
2. Put the resolver-supplied CUDA UUID or PCI identity in `LimitedLaunch`; the
   immutable plan derives the canonical local ownership keys.
3. Pass final resource probing/evaluation as the supervisor's pre-launch check;
   it runs after the supervisor acquires the lease and before it creates a child.
4. Refuse with the budget's dominant phase and allocations, or calculate soft
   and hard child limits while preserving the configured system reserve.
5. Build one `ContainmentPlan`; `SupervisedSidecar` internally acquires the
   canonical leases, starts the out-of-scope guardian, then releases the gated
   workload in a fresh process group.
6. Consume output through `BoundedLineBuffer`, forward structured progress, and
   preserve only the bounded tail.
7. Classify completion from watchdog, cgroup, allocator, cancellation, and exit
   evidence before accepting any artifact.
8. Release the lease set only after the process group, token-captured escaped
   descendants, and exact systemd scope (when used) are quiescent and the
   guardian has acknowledged teardown.

Never apply these limits to the GUI process. Partial artifacts must remain under
temporary names until a successful child result has been validated.
