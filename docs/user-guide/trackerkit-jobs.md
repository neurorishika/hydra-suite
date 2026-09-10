# Portable tracking jobs (`trackerkit job …`)

A tracking experiment is usually staged on one machine (a laptop, in the
TrackerKit GUI or CLI) and run on another (a CUDA box). `trackerkit job`
packages everything a run needs — models, config, videos — into a
self-contained **job directory** that runs on any compute box with **zero
model registration**, and whose outputs (including reusable detection
caches) come home beside the original videos.

See `docs/superpowers/specs/2026-09-09-portable-tracking-jobs-design.md` for
the full design; this page is the day-to-day walkthrough.

## Lifecycle walkthrough

```bash
# 1. Pack — turn a config (single video or batch) into a job directory
trackerkit job pack /tmp/jobs/colony_A a.mp4 b.mp4 c.mp4
trackerkit job pack /tmp/jobs/colony_A --video-list batch.txt

# 2. Push — rsync the job to a compute box over ssh
trackerkit job push /tmp/jobs/colony_A rutalab@firebrat:/home/rutalab/jobs/colony_A

# 3. Preflight — everything the run needs is on the box, loudly, before anything runs
ssh rutalab@firebrat \
  'source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda && \
   cd /home/rutalab/jobs/colony_A && trackerkit job preflight .'

# 3b. Calibrate (optional) — one-click calibration is host-fingerprinted, so a
#     fresh compute box needs its own profile before --apply-tuned-inference works
trackerkit job calibrate rutalab@firebrat:/home/rutalab/jobs/colony_A \
  --remote-bootstrap 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda'

# 4. Run — zero registration: the job dir supplies the models root and config root
trackerkit job run rutalab@firebrat:/home/rutalab/jobs/colony_A \
  --remote-bootstrap 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda' \
  --gpus auto

# 4b. Or check on a run in progress / after it finishes
trackerkit job status rutalab@firebrat:/home/rutalab/jobs/colony_A

# 5. Pull — outputs (CSVs, videos, detection caches) come home beside the
#    original videos, not into the job directory
trackerkit job pull rutalab@firebrat:/home/rutalab/jobs/colony_A /tmp/jobs/colony_A
```

Every subcommand that names a remote target accepts `--remote-bootstrap`
(see below); `pack`, `verify` and `shared-root` are always local.

The full subcommand list: `pack`, `verify`, `shared-root`, `push`,
`preflight`, `run`, `calibrate`, `status`, `pull`. Run `trackerkit job
<subcommand> --help` for the exact flags — `pack` deliberately has no
`--gpus`/`--jobs`/`--threads-per-job` (those describe the **compute box**,
not the experiment, and belong to `run`), and `run`/`calibrate` deliberately
reject `--sahi-profile`/`--inference-autotune-manual` (those are fixed at
pack time and always forwarded automatically, because the calibration
lookup key depends on them matching exactly).

## What travels, what doesn't

| Travels in the job directory | Stays on the host, never travels |
|---|---|
| Referenced models (with sidecars, registry entries, multi-head bundles) | TensorRT/ONNX/CoreML compiled engines (`<data>/runtime-artifacts`) — host-specific, content-addressed, rebuilt automatically on the compute box |
| The config snapshot (`advanced_config.json`, skeletons, presets) | Calibration profiles (`<data>/inference_tuning_profiles`) — host-fingerprinted, would never match; re-derive per box with `job calibrate` |
| The keystone + per-video sidecar configs | Conda environments (e.g. `sleap`) — the job records what it needs and `preflight` fails loudly naming the missing env, it does not install anything |
| Videos (copied by default, or referenced via a shared-root symlink — see below) | |
| A generated, executable `run.sh` | |

The job directory supplies `HYDRA_MODELS_DIR=<job>/models` and
`HYDRA_CONFIG_DIR=<job>/config` for the duration of the run; the compute
box's own `HYDRA_DATA_DIR` (engines, calibration profiles, training runs)
is untouched. This is why there is no registration step: nothing is ever
imported into the host's own model registry.

Which models are shipped is derived from
`hydra_suite.trackerkit.engine_params.iter_model_references(params)`, not
from a hand-maintained list — see the developer-guide note below.

## Shared-root video references (no copy on a mounted share)

If videos already live on a share mounted on both the staging machine and
the compute box (e.g. a lab NAS), `pack` does not copy them — it records a
`shared: {alias, relpath}` reference in the manifest, and the compute box
materializes a symlink into `videos/` during `preflight`. Each host keeps
its own flat alias → mount-path map, managed with `job shared-root`:

```bash
# On the laptop
trackerkit job shared-root add labnas /Volumes/lab

# On firebrat (same alias name, different local mount path)
ssh rutalab@firebrat trackerkit job shared-root add labnas /mnt/lab
```

which persists to `<config>/shared_roots.json`:

| Host | `shared_roots.json` |
|---|---|
| laptop | `{"labnas": "/Volumes/lab"}` |
| firebrat | `{"labnas": "/mnt/lab"}` |

Only the **alias name** has to agree across machines — the underlying mount
path can differ per host. `pack` matches a video's real path against every
known alias (longest match wins) and excludes a matched video from the push
input set entirely. `--no-shared` disables the match for one `pack` call
(always copy); `--shared-only` fails `pack` if any video is *not* under a
known alias, for labs that never want a silent fallback to a copy.

A one-off alias can also be supplied directly to `preflight`/`run` without
persisting it: `--shared-root labnas=/mnt/lab`.

`preflight` resolves the alias, checks the target exists and its size/content
signature match what `pack` recorded (catches a stale mirror or a
re-encoded file), then replaces (never overwrites a regular file) the
`videos/<basename>` symlink. Outputs are still written into the job tree
beside the symlink and `pull` brings them home exactly as it does for a
copied video — the mount is never written to directly by the remote.

## The conda-env requirement

`trackerkit` and the packages it depends on must be importable in whatever
environment the compute box uses to run the job — `job preflight` checks
this explicitly (`hydra_suite` importable, version floor met, and every
`requirements.conda_envs` entry from the packed job — e.g. `sleap` for a
pose backend — present in `conda env list`). None of this is installed by
`pack`, `push`, or `run`; a missing environment fails preflight loudly,
naming the exact environment, rather than failing partway through a run.

## `--remote-bootstrap`

Every subcommand that dispatches over `ssh` (`push`'s post-push verify,
`preflight`, `run`, `calibrate`, `status`) accepts `--remote-bootstrap`, a
shell snippet run before the remote command. It defaults to `""` —
deliberately: an empty bootstrap fails loudly on a box where `trackerkit`
is not on a bare, non-interactive `ssh` `PATH`, rather than silently
mis-scheduling.

**Verified on `firebrat`:** a bare `ssh host 'trackerkit ...'` — and even
`ssh host 'bash -lc "trackerkit ..."'` — cannot find `trackerkit` on that
box. The value that works there:

```bash
--remote-bootstrap 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-cuda'
```

Use the box's own conda install path and environment name; there is no
universal default.

## The dataset/media-export limitation (§6.6)

The headless CLI path (which `job run` uses, via `trackerkit track`) does
not derive `DATASET_OUTPUT_DIR`, `FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR`, or
`INDIVIDUAL_DATASET_OUTPUT_DIR` — those are computed only in the GUI's
config orchestrator today. A packed job whose config enables active-learning
export, oriented-video export, or individual-crop export will run to
completion but **produce none of those exports** on the compute box; `pack`
warns at pack time when it detects an export stage enabled, pointing back
at this limitation. Closing this gap (deriving the same `<stem>_datasets/`
paths headlessly that the GUI already derives) is tracked as a follow-up
and is gated by the byte-identity harness, since it changes CLI engine
parameters.
