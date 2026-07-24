# Cloud GPU Inference for TrackerKit

**Date:** 2026-07-23
**Status:** Design (approved) — pending implementation plan
**Author:** Rishika Mohanta
**Related:** `project_pose_runtime_golden_rule`, `project_bgsub_inference_unification_done`,
`project_runtime_gen2_core_done` (memory), `docs/developer-guide/runtime-integration.md`
**Scope:** TrackerKit only, all `InferenceRunner`-backed stages (bgsub, detector,
pose/SLEAP, classifier/identity). Training and other kits are explicitly out of scope
(see Non-Goals).

## Problem

TrackerKit's inference (background subtraction, detection, pose/SLEAP, identity
classification) currently only runs against local compute (`RuntimeResolver` resolves to
CPU/MPS/CUDA on the user's own machine, per `runtime_tier`). Users without a capable local
GPU cannot get GPU-tier speed or accuracy without buying hardware. We want to let a user
rent a GPU instance (Vast.ai) and run the *exact same* interactive TrackerKit
workflow — setup, live preview/tuning, full run, results — against that instance's GPU,
with no behavior change from the user's perspective beyond "which GPU is doing the work."

## Goal

A **Cloud panel** in TrackerKit lets a user rent/manage a Vast.ai GPU instance. Once an
instance is ready, the user can put the current session into **remote mode**: the local
app keeps its UI and video decoder, but every `InferenceRunner` call (bgsub, detector,
pose/SLEAP, classifier/identity) — both interactive preview calls and the full pipeline
run — executes on the remote instance instead of locally. Final CSVs and other outputs are
downloaded back automatically on completion.

Non-goals (explicitly deferred):
- Auto-provisioning (instances are only started/stopped by explicit user action).
- Cloud GPU support for PoseKit/ClassKit *training* jobs.
- Multi-tenant job queues / multiple concurrent users per instance.
- Resumable/chunked video upload, or job migration across instances on preemption.
- Cloud support for kits other than TrackerKit.

## Design

### 1. Architecture overview

```
┌─────────────────────────┐        SSH tunnel        ┌──────────────────────────┐
│  TrackerKit (local)     │  (local port-forward)     │  Vast.ai instance         │
│                          │◄──────────────────────────►│  (Docker image)          │
│  - UI, video decode      │   control API (HTTP)      │  - inference server       │
│  - session/config state  │   preview stream (WS)      │  - hydra_suite core/*    │
│  - Cloud panel           │                            │    (same code as local)  │
└─────────────────────────┘                            └──────────────────────────┘
```

The remote **inference server** is a thin ASGI wrapper around the *existing*
`hydra_suite.core` / `hydra_suite.runtime` / `hydra_suite.data` layers — the same
Qt-free pipeline code that already runs locally via `InferenceRunner`. No separate
"remote pipeline" implementation is written or maintained; the server imports and calls
the same modules the local app does, so pipeline behavior (and its equivalence
guarantees, per `tools/equivalence/`) is identical whether local or remote — the only
difference is *which machine's GPU* runs the calls. `InferenceRunner` is the single
remoting boundary: any call it currently dispatches to a local `ResolvedBackend` is
dispatched instead to the remote server's `ResolvedBackend` when the session is in
remote mode, for all four stages (bgsub, detector, pose/SLEAP, classifier/identity).

### 2. Cloud panel & instance lifecycle

- New `hydra_suite.integrations.vastai` module: thin client over the Vast.ai API
  (list offers, rent, start, stop, destroy, status poll). Lives under `integrations/`
  alongside the SLEAP/X-AnyLabeling bridges — it talks to an external service, not core
  pipeline logic, and may import from Core/Runtime/Data/Utils per the existing dependency
  rules, never the reverse.
- Cloud panel widget (TrackerKit-first, built as a reusable shared widget so other kits
  can adopt it later without rework): API key entry (stored via the existing
  `HYDRA_CONFIG_DIR` mechanism, never inside project files), instance browse/rent,
  status display (`offline → provisioning → booting → ready → running job → idle`),
  start/stop/destroy controls, and running cost-so-far display.
- **Manual lifecycle only.** The app never rents or terminates an instance on its own
  initiative — cost stakes are real. The panel does surface an idle-time warning (e.g.
  "instance has been idle 30+ min") so users don't forget a running meter, but this is a
  UI nudge, not an automatic action.
- The instance boots our **prebuilt Docker image** — `hydra-suite[cuda]` plus the
  inference server, plus (see §5) a `conda`+`sleap` environment for the pose backend —
  pre-installed and version-pinned to a specific hydra-suite release. "Ready" means the
  server is already listening; the user never sees an install step.

### 3. Connection: SSH tunnel

- On "start," TrackerKit opens an SSH connection to the instance using the SSH key Vast.ai
  already associates with the rental, and local-port-forwards a single control port to the
  inference server process. All traffic — control-plane calls and the interactive preview
  stream — rides that one tunnel. No other ports are exposed; the instance never has a
  public HTTP/WS port to defend.
- The inference server holds exactly one active session at a time, matching the
  "one user rents one box" model. No session multiplexing/queueing.

### 4. Inference server API

Two channels over the tunneled port:

- **Control-plane (HTTP)**: upload video (see §5), set/update session config, trigger a
  full pipeline run, fetch job status, fetch/download result artifacts (CSVs, caches),
  fetch server version (see §6).
- **Interactive preview (WebSocket)**: for every UI action that currently triggers a
  local `InferenceRunner` call for live preview/tuning (bgsub threshold, detector
  confidence, pose overlay, classifier/identity preview) — the client sends the frame
  index + current config, the server runs that stage against its own copy of the video
  and returns just the result (detections/keypoints/labels/overlay coordinates — not
  pixels, since the local decoder already has the frame to draw on). This keeps
  round-trips small so interactive tuning feels close to local-GPU speed.

### 5. Data flow within a session

1. **Open video in remote mode** → local app uploads the video file once over the SSH
   connection (SFTP-style) to a working directory on the instance. One-time, shown with a
   progress bar. Not resumable in v1 (see Non-Goals) — the clips this targets are short
   per-trial recordings, not hours-long footage, so a from-scratch retry on failure is an
   acceptable cost.
2. **Interactive setup** → WebSocket round-trips per §4, for all four `InferenceRunner`
   stages, matching whatever the local UI already offers for live preview.
3. **Press "Run"** → control-plane call triggers the full session pipeline server-side:
   bgsub → detection → pose/SLEAP → classifier/identity → Kalman → assignment →
   post-processing → CSV export, using the same session/headless-execution path
   TrackerKit's CLI already uses locally (`run_headless_tracking_session` for
   detection-only sessions; the hidden-`MainWindow`-under-`QApplication` fallback for
   sessions needing pose/identity — both run under `QT_QPA_PLATFORM=offscreen` inside the
   container, since neither actually needs a display). Local UI polls the control-plane
   status endpoint the same way `BaseWorker` reports local job progress.
4. **Completion** → final CSVs and other outputs are pulled back over the same SSH
   connection; from that point the local app treats them exactly like local-run outputs
   (same post-processing UI, same export dialogs — no remote-specific result handling).

### 6. Runtime-tier integration

Remote mode is a session-level **execution location** flag (`local` | `remote`),
orthogonal to the existing `runtime_tier` knob. Tier selection still happens — just on
whichever machine is executing: locally when `execution location = local` (unchanged
today), or inside the server when `execution location = remote` (server always resolves
against its own `gpu`/`gpu_fast` tier, since the whole point of renting the instance is
its GPU). No new tier value is introduced; `RuntimeResolver`, `ResolvedBackend`, and
`execution_providers_for` are unchanged. `InferenceRunner`'s call sites gain a check: if
the active session is in remote mode, forward the call over the connection instead of
resolving/dispatching a local backend.

### 7. Error handling & edge cases

- **Connection drop mid-session** (tunnel dies): client detects it, shows a reconnect
  banner, and re-establishes the SSH tunnel to the same instance. Because the server owns
  session state (video already uploaded, config already set), reconnecting resumes rather
  than restarting — client re-syncs current status only.
- **Full run in progress, connection drops**: the server-side job is driven by the
  server's own event loop, not the client's — it keeps running. On reconnect the client
  polls status and picks up progress/completion normally, so a dropped local connection
  doesn't kill an in-progress run.
- **Instance stopped/preempted by Vast.ai mid-job**: surfaced as a hard failure in the
  Cloud panel, run marked failed. No automatic retry or migration — that's job-queue-grade
  resilience, explicitly out of scope for this pass.
- **Upload interrupted**: retried from scratch (not resumable — see §5).
- **Version mismatch**: server reports its hydra-suite version on connect; client warns
  (does not block) if it differs from the local version, since silent pipeline drift
  between environments is exactly what the equivalence-harness work in this repo exists to
  prevent.

### 8. Docker image requirements

- Base: CUDA-capable image with `hydra-suite[cuda]` installed at a pinned version.
- Inference server process (ASGI app) installed and set to auto-start.
- A `conda` install with a `sleap` environment on `PATH`, matching the existing local
  requirement that `SleapServiceBackend` shells out via `conda run -n sleap`
  (`project_pose_runtime_golden_rule`) — easy to miss since the base CUDA/torch stack
  alone is not sufficient for the pose stage.
- Image is rebuilt and republished on each hydra-suite release that touches inference
  code, version-tagged so the Cloud panel can pick a matching image for the user's local
  version.

## Testing

- Inference-server unit tests: control-plane and WebSocket handlers, run against a local
  process (no real Vast.ai instance needed) to verify request/response contracts.
- Reuse `tools/equivalence/` methodology: run the same fixture clips through local
  `execution location=local` GPU tier and through a real (or containerized-locally)
  remote server, and diff outputs the same way legacy-vs-current is diffed today — remote
  mode should hit the same equivalence/determinism floor as local GPU.
- Manual test plan: full round trip against a real Vast.ai instance for at least one clip
  per stage combination (detection-only, +pose, +identity) before calling this shippable.
