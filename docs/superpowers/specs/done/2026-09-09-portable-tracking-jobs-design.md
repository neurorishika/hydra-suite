# Portable tracking jobs: stage anywhere, run anywhere, sync back

**Status:** Shipped — merged to `main`. Implemented via `docs/superpowers/plans/done/2026-09-09-portable-tracking-jobs.md`. Verified: MPS equivalence 42 EQUIVALENT/0 divergent, CUDA (firebrat) 54/0, full Mac→firebrat→Mac round trip with zero registration, and Goal 4 proven (a cache computed on firebrat stores byte-identically the key a local Mac run computes). Seven spec defects were found during planning and corrected in the plan — see its "Spec corrections established before planning" section; most consequentially §7b was under-scoped (`_model_signature` folded a raw path+mtime into `config_hash`, so OBB caches would have stayed machine-local even after the documented fix).

> ## AMENDED 2026-09-09 — path-independent cache keys scoped IN (user decision)
> The first draft left pulled `.inference_cache_<stem>/` as a record only because the cache key embeds the absolute model path. The user wants remote caches to be usable locally without regeneration, so §7b makes model identity and video identity content-based and bumps the cache schema. Goal 4 and §17 updated accordingly. Headless dataset/media export stays a follow-up (§17 item 1). Same day: §6.7 adds shared-root video references so videos on a lab share are never copied (user request).
**Repo:** `/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker` @ `main` (`8678c5b6`).

## 1. Problem

A tracking experiment is staged on one machine (usually a laptop, in the TrackerKit GUI) and run on another (a CUDA box). Today that requires re-registering every referenced model on the compute box by hand, copying videos and configs by hand, and copying outputs back by hand. Each step is a place where the run silently diverges from what was staged (wrong model, missing SAHI sidecar, stale config).

The lab wants a durable mechanism, not a one-off script: a **job** is packaged once on the staging machine with everything needed to run it, pushed to a compute box that has never seen those models, run there without any registration step, and its outputs (including caches) pulled back beside the original videos.

Machines share nothing except `ssh`/`rsync` in the general case. When a lab share is mounted on both sides, videos must not be copied (§6.7).

## 2. Goals and non-goals

**Goals**

1. `trackerkit job pack` turns a tracking config (single video or `--video-list` batch) into a self-contained job directory: manifest, referenced models with sidecars and registry entries, config snapshot, videos, and a generated runner.
2. `trackerkit job push` / `pull` move the job to and from a remote over `rsync` over `ssh`, incrementally.
3. `trackerkit job run` executes the job on the compute box with **zero registration**: the job directory supplies the models root and the config root; the host keeps its own data dir.
4. `pull` places every artifact the run produced back beside the original video, including `.inference_cache_<stem>/`, **and those caches are reusable locally**: a backward pass, a rerun, a parameter-optimizer session or a replay session on the staging machine hits the cache produced on the compute box. This requires the cache key to identify models and videos by content rather than by absolute path and mtime (§7b).
5. The set of files a job needs is **derived from the engine parameter builder**, so new model roles cannot be forgotten silently.
6. Configs become portable by construction: the two save-side leaks (`color_tag_model_path`, `cnn_classifiers[].model_path`) are fixed at the source.
7. Videos that live on a share mounted on both machines are referenced, not copied (§6.7).

**Non-goals**

- Byte-identical results across CUDA and MPS. The guarantee is "same models, same config, runs to completion". Cross-device equivalence remains the job of `tools/equivalence/`.
- Shipping TensorRT/ONNX/CoreML engines. These are host-specific, content-addressed under `<data>/runtime-artifacts`, and are rebuilt on the compute box.
- Shipping calibration profiles (`<data>/inference_tuning_profiles`). They are host-fingerprinted and would never match; they are re-derived per compute box with `job calibrate` (§9.3).
- Installing conda environments (e.g. the `sleap` env). The job records what it needs and preflight fails loudly.
- A GUI for the job lifecycle. The Data-layer core is Qt-free so a GUI button can be added later; this spec ships the CLI only.
- Job scheduling, queues, or multi-remote fan-out beyond what `trackerkit track --gpus` already does on one host.

## 3. Why approach A (job supplies models + config, host keeps data dir)

| Approach | Registration step? | Engines rebuilt per job? | Pull drags host state home? | `paths.py` change |
|---|---|---|---|---|
| **A. `HYDRA_MODELS_DIR=<job>/models`, `HYDRA_CONFIG_DIR=<job>/config`, `HYDRA_DATA_DIR` untouched** | No | No (cache keyed by model sha256 + toolchain, host-wide) | No | one new env var |
| B. `HYDRA_DATA_DIR=<job>` | No | Yes, every job | Yes unless excluded | none |
| C. Import into remote registry | Yes (the thing being removed) | No | No | none |

A is chosen. The one required change is that `get_models_dir()` gains an independent override so the models root can be relocated without relocating the whole data dir.

## 4. Job directory layout

```
<job>/
  hydra_job.json                 # manifest (§5)
  run.sh                         # generated runner (§9), executable
  config/                        # becomes HYDRA_CONFIG_DIR on the remote
    advanced_config.json         # snapshot of the staging machine's advanced config
    skeletons/                   # only the skeleton file(s) the job references
    presets/.seeded              # marker so get_presets_dir() does not reseed bundled presets
    skeletons/.seeded
  models/                        # becomes HYDRA_MODELS_DIR on the remote
    model_registry.json          # v2 registry containing ONLY the shipped entries
    obb/<file>.pt                # each referenced model at its models-root-relative key
    obb/<file>.pt.slice_meta.json
    obb/<file>.pt.canonical_meta.json
    pose/SLEAP/<dir>/...         # pose models are directories, copied whole
    classification/identity/...  # ClassKit bundles: manifest + every listed artifact
  videos/
    <stem>.mp4                   # symlink on the staging machine, real file after push
    <stem>_config.json           # per-video sidecar, job-relative (§6)
    ... run outputs land here on the remote ...
  videos.txt                     # --video-list input, one job-relative path per line
  logs/
    run.log                      # stdout+stderr of `job run`
    preflight.json               # result of the last preflight
```

Every path inside the job is relative to the job root. The job is valid at any absolute location on any machine.

## 5. Manifest: `hydra_job.json`

Follows the `project_bundle.py` conventions (`bundle_version` integer, `to_dict`/`from_dict` dataclass, `ValueError` on version mismatch, path-traversal validation).

```json
{
  "job_version": 1,
  "job_id": "2026-09-09T14-03-12_ant_colony_A",
  "created_at": "2026-09-09T14:03:12Z",
  "created_on": {"hostname": "rishika-mbp", "platform": "darwin", "hydra_suite_version": "x.y.z", "git_sha": "8678c5b6"},
  "keystone": {"video": "videos/colony_A_cam1.mp4", "config": "videos/colony_A_cam1_config.json"},
  "videos": [
    {
      "job_path": "videos/colony_A_cam1.mp4",
      "origin_path": "/Volumes/lab/2026-09/colony_A_cam1.mp4",
      "size_bytes": 8123456789,
      "config_job_path": "videos/colony_A_cam1_config.json",
      "config_provenance": "own-sidecar",
      "pushed_siblings": ["videos/colony_A_cam1_config.json"],
      "redirected_outputs": {"videos/colony_A_cam1_tracking.mp4": "/Volumes/renders/colony_A_cam1.mp4"}
    },
    {
      "job_path": "videos/colony_A_cam2.mp4",
      "origin_path": "/Volumes/lab/2026-09/colony_A_cam2.mp4",
      "size_bytes": 8123456789,
      "signature": "8123456789:3f9a…",
      "shared": {"alias": "labnas", "relpath": "2026-09/colony_A_cam2.mp4"},
      "config_job_path": "videos/colony_A_cam2_config.json",
      "config_provenance": "keystone-baseline",
      "pushed_siblings": ["videos/colony_A_cam2_config.json"]
    }
  ],
  "models": [
    {
      "key": "obb/2026-08-01_ant_obb.pt",
      "roles": ["YOLO_OBB_DIRECT_MODEL_PATH"],
      "origin_path": "/Users/rishika/Library/Application Support/hydra-suite/models/obb/2026-08-01_ant_obb.pt",
      "kind": "file",
      "sha256": "…",
      "size_bytes": 123456,
      "sidecars": ["obb/2026-08-01_ant_obb.pt.slice_meta.json"],
      "registry_entry_present": true
    },
    {
      "key": "pose/SLEAP/ant_pose_v3",
      "_note": "key is whatever make_pose_model_path_relative returns; copy target is the directory resolve_pose_model_path(key, backend) yields under the job models root",
      "roles": ["POSE_MODEL_DIR"],
      "kind": "directory",
      "files": ["pose/SLEAP/ant_pose_v3/best.ckpt", "pose/SLEAP/ant_pose_v3/training_config.json"],
      "registry_entry_present": false
    }
  ],
  "config_snapshot": {
    "advanced_config": "config/advanced_config.json",
    "skeletons": ["config/skeletons/ant_10kp.json"]
  },
  "requirements": {
    "conda_envs": ["sleap"],
    "runtime_tier": "gpu",
    "min_hydra_suite_version": "x.y.z"
  },
  "track_args": {
    "video_list": "videos.txt",
    "keystone_override": false,
    "sahi_profile": null,
    "apply_tuned_inference": null,
    "inference_autotune_manual": []
  },
  "pull_history": [
    {"pulled_at": "…", "files": 14, "bytes": 123456789}
  ]
}
```

Field rules:

- `videos[].origin_path` is the **only** place absolute staging-machine paths live. It exists purely so `pull` can put outputs back. It is never read by `run`.
- `videos[].pushed_siblings` is the list of files that existed beside the video inside the job **before** the run. `pull` defines "output" as anything under `videos/` that is not the video itself and not in `pushed_siblings` (§10).
- `models[].roles` lists the engine-parameter keys that resolved to this model, for diagnostics only.
- `videos[].shared`, when present, means the video is **not** in the job tree and not pushed: each host resolves `alias` through its own mount table (§6.7). `videos[].signature` is the content signature of §7b.2 and is what proves the remote resolved the same file.
- `videos[].redirected_outputs` maps a job-relative output path to the absolute path the user originally chose (§6.4), so `pull` can restore it.
- **The remote never mutates `hydra_job.json`.** `run` appends one JSON line per run to `logs/runs.jsonl` (`started_at, finished_at, hostname, exit_code, hydra_suite_version, git_sha, argv`); `pull` fetches that file and merges it into the local manifest's `pull_history` entry. This is what lets `push` be a pure input sync (§8.1).
- Manifest writes use `write_json_atomic` from `project_bundle.py`.

## 6. Pack: building the job

Entry point: `hydra_suite.data.tracking_job.pack_job(...)` (Qt-free), wrapped by `trackerkit job pack`.

### 6.1 Inputs

```
trackerkit job pack <job_dir> (VIDEO... | --video-list FILE) [--config FILE] [--keystone-override]
                    [--sahi-profile NAME] [--apply-tuned-inference | --no-apply-tuned-inference]
                    [--inference-autotune-manual FIELD]... [--name NAME] [--copy-videos] [--no-shared | --shared-only]
```

The video/config arguments are **exactly** those of `trackerkit track`, resolved through the same functions: `resolve_track_video_inputs()` (`app.py:352`) for the video list and `plan_batch_jobs()` → `build_batch_video_plan()` for per-video config resolution with the existing `own-sidecar | explicit | keystone-baseline` provenance. Pack does not reimplement config precedence; it consumes the plan.

`--gpus/--jobs/--threads-per-job` are **not** accepted by `pack`. They describe the compute box, not the experiment, and are given to `job run` (§9). Mirrors the `calibrate` precedent (`app.py:196-200`): argparse rejects them loudly.

### 6.2 Steps

1. **Plan.** Build the batch plan as `track` would. Abort on the same errors `track` would raise (missing keystone, collisions).
2. **Resolve engine params per video.** For each planned video call `build_engine_params(cfg, ctx)` with a `RuntimeContext` identical to the CLI's (`cli_config.py`). This is the authoritative resolution of every model path.
3. **Collect model references.** Walk the returned params dict with `iter_model_references(params)` (§6.3). Each reference is `(role_key, resolved_path, kind)` where `kind ∈ {file, directory}`.
4. **Compute the job key** for each reference: `make_model_path_relative(path)` (or the pose variant). If the result is still absolute (the model lives outside the models root), the key becomes `external/<sha256[:12]>/<basename>` and the config is rewritten to that key (§6.4).
5. **Copy models** into `<job>/models/<key>`:
   - files: `shutil.copy2`, then `copy_model_metadata_sidecars(src, dst)` (`model_paths.py:25`), then ClassKit bundle expansion: `discover_multihead_model_bundle(src)`; if a manifest is found, copy the manifest and every listed artifact with their own sidecars.
   - directories (pose): copy the whole directory tree. Pose backends read `best.ckpt`, `training_config.{json,yaml}`, `export_metadata.json`, `metadata.json` and any `.pt/.ckpt/.onnx/.engine/.trt` sibling (`core/individual/pose/artifacts.py:14-31`); shipping the whole directory is the only future-proof choice. `.hydra-runtime-artifacts/` subtrees are excluded (they are the local-fallback engine cache).
   - Record `sha256` of every file for `verify`.
6. **Registry subset.** Load the staging machine's `model_registry.json` via `model_publish.iter_registry_entries`; write a new v2 registry into `<job>/models/model_registry.json` containing only entries whose key is in the shipped set. Entries' `source_path` is rewritten to `null` (it is a staging-machine absolute path and meaningless elsewhere). Models with no registry entry are shipped anyway and flagged `registry_entry_present: false`; the run does not need registry entries, the GUI on the remote does.
7. **Config snapshot.** Copy `get_advanced_config_path()` if it exists. Copy every `pose_skeleton_file` referenced (resolved via the same fallback as the engine) into `config/skeletons/` and rewrite the config key to the job-relative path. Write `.seeded` markers into `config/presets/` and `config/skeletons/` so `get_presets_dir()`/`get_skeleton_dir()` do not seed bundled defaults on top of the snapshot.
8. **Videos.** For each planned video, create `<job>/videos/<basename>` as a **symlink** to the origin (default) or a copy (`--copy-videos`). Basename collisions across different directories are rejected (the batch planner already rejects duplicate videos; this adds the basename check with a clear message listing both origins).
9. **Per-video config sidecar.** Write `<job>/videos/<stem>_config.json` for **every** video regardless of provenance, containing the fully resolved config dict for that video after the rewrites of §6.4. Because every video now has its own sidecar, the remote run needs no keystone logic at all; `track_args.keystone_override` is recorded for provenance only.
10. **`videos.txt`.** One job-relative path per line, keystone first.
11. **Requirements.** `conda_envs`: `[cfg["pose_sleap_env"]]` if any video uses the SLEAP backend with the service path; `runtime_tier` from the keystone config.
12. **`run.sh`** (§9) and manifest.
13. **Self-verify.** Run `verify_job()` (§11) on the freshly packed directory and fail pack if it fails.

### 6.3 `iter_model_references(params)`: the derivation rule

Lives in `hydra_suite/trackerkit/engine_params.py` next to `build_engine_params`, because that file owns the key vocabulary.

```python
MODEL_FILE_PARAM_KEYS = (
    "YOLO_OBB_DIRECT_MODEL_PATH", "YOLO_DETECT_MODEL_PATH", "YOLO_CROP_OBB_MODEL_PATH",
    "YOLO_HEADTAIL_MODEL_PATH", "COLOR_TAG_MODEL_PATH", "CNN_CLASSIFIER_MODEL_PATH",
)
MODEL_DIR_PARAM_KEYS = ("POSE_MODEL_DIR",)
MODEL_LIST_PARAM_KEYS = {"CNN_CLASSIFIERS": "model_path"}
NON_MODEL_PATH_PARAM_KEYS = (
    "YOLO_MODEL_PATH",             # alias of one of the two OBB keys, never distinct
    "POSE_EXPORTED_MODEL_PATH",    # always "" today
    "POSE_SKELETON_FILE",          # config asset, handled by the config snapshot
    "DATASET_OUTPUT_DIR", "FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR", "INDIVIDUAL_DATASET_OUTPUT_DIR",
)
```

The generator yields every non-empty value under those keys. Empty strings are skipped: `build_engine_params` already emits `""` for disabled roles (e.g. head-tail with the group off, `engine_params.py:826-843`), so pack ships exactly what the run will load.

**Contract guard test** (`tests/test_engine_params_model_reference_contract.py`): build params from a config that enables every role, then assert that every key whose name ends in `_MODEL_PATH` or `_MODEL_DIR`, or whose value is a list of dicts containing `model_path`, is in exactly one of the tuples above, **and** that every key ending in `_PATH` or `_DIR` is in the union of the four tuples plus an explicit `NON_MODEL_PATH_PARAM_KEYS` list (CSV/video/cache/output paths). The plan enumerates that list from real params once; afterwards any new path-ish key must be classified. A new role added to `build_engine_params` without being classified fails this test. This is the mechanism that makes the feature long-term rather than a patch, and it is the kind of reflective guard that memory `feedback_run_contract_guards_after_field_additions` says must run on every field addition.

### 6.4 Config rewrites performed by pack

Per-video config dicts are rewritten before being written as sidecars:

| Key | Rewrite |
|---|---|
| `file_path` | `videos/<basename>` (job-relative) |
| `csv_path`, `video_output_path` | If they were the defaults derived from the video (`<stem>_tracking.csv/.mp4` in the video's directory), rewrite to the job-relative equivalent. If the user pointed them elsewhere, rewrite to `videos/<stem>_<original basename>` and record the original absolute path in `videos[].redirected_outputs` so `pull` can restore it. Never leave a staging-machine absolute path in a sidecar. |
| `yolo_*_model_path`, `pose_*_model_dir`, `pose_model_dir`, `color_tag_model_path`, `cnn_classifiers[].model_path` | The job key from §6.2 step 4 (models-root-relative, or `external/...`). |
| `pose_skeleton_file` | `config/skeletons/<basename>` (job-relative). Consumers read `POSE_SKELETON_FILE` verbatim with no models-root resolution (`core/inference/config.py:1233`, `core/post/pose_merge.py:292`, `core/post/media_export.py:401`, `core/individual/properties/cache.py:221`), so a job-relative path works only because `run.sh` does `cd "$JOB"` first. `verify` asserts the file exists at that relative path. |
| `pose_sleap_env` | Unchanged; recorded in `requirements.conda_envs`. |
| Provenance copies (`yolo_model_size`, `*_model_info`, …) | Unchanged. |

### 6.5 Save-side leak fixes (source fixes, not packer workarounds)

In `trackerkit/gui/orchestrators/config.py`:

- `color_tag_model_path` (`:2096`) goes through `make_model_path_relative` like the other model keys at `:1763-1783`.
- `cnn_classifiers[].model_path` entries are relativized on save at the point the identity config is serialized.

Load side is already symmetric (`resolve_model_path`). A characterization test asserts that a config saved from a GUI state whose classifiers live under the models root contains no absolute paths in those two keys. The GUI-vs-CLI parity tests (`test_gui_cli_param_equivalence.py`) must stay green: relativizing on save and resolving on load is a round trip that leaves `build_engine_params` output unchanged.

### 6.6 Headless output-dir gap

The CLI leaves `DATASET_OUTPUT_DIR`, `FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR`, `INDIVIDUAL_DATASET_OUTPUT_DIR` at `None` (`cli_config.py:279-301`), so a job that enables dataset or media export produces nothing on the remote. This is an existing CLI gap, not a job-specific one. Pack **warns** when any export stage is enabled in a config and points at this limitation. Closing the gap (deriving `<stem>_datasets/<subfolder>` in `cli_config.py` exactly as the GUI does at `orchestrators/config.py:2241-2252`) is listed as a follow-up, gated by the byte-identity harness because it changes CLI engine params.

### 6.7 Shared-root video references (no copy when a share is mounted on both sides)

**Mount table.** Each host keeps `<config>/shared_roots.json`, a flat `{alias: absolute_mount_path}` map:

```json
{"labnas": "/Volumes/lab"}     // laptop
{"labnas": "/mnt/lab"}         // mehek
```

Only the alias name has to agree across machines. Managed by `trackerkit job shared-root add|remove|list` or by hand; read through a new `paths.get_shared_roots_path()` so `HYDRA_CONFIG_DIR` relocation applies. Note that on the remote `HYDRA_CONFIG_DIR` points at `<job>/config` during `run` (§9.1), so the mount table is read from the **host's** config dir explicitly, never from the job snapshot: `run.sh` captures `HYDRA_HOST_CONFIG_DIR` before overriding.

**Pack.** For each planned video, `pack` resolves the origin with `os.path.realpath` and tests it against every local alias root (longest match wins). If it matches, the manifest entry gets `shared: {alias, relpath}` and `signature`; no symlink is created under `videos/`, and the video is excluded from the push input set. Otherwise the video is handled as in §6.2 step 8. `--no-shared` disables matching for one pack; `--shared-only` fails pack if any video is not under an alias (for labs that never want copies).

**Preflight / materialize on the remote.** For each `shared` entry, in order:

1. Resolve `alias` through the host mount table, or through a one-off `--shared-root ALIAS=PATH` given to `run`/`preflight` (which also offers to persist it). Unknown alias fails with the alias name and the list of known aliases.
2. Check `<root>/<relpath>` exists and is readable.
3. Compare `size_bytes` and the content `signature` (§7b.2) with the manifest. Mismatch fails with both paths and both signatures; this catches a stale mirror or a re-encoded file.
4. Create `videos/<basename>` as a symlink to the resolved path (replace an existing symlink, never a regular file).

Because the sidecar `file_path` is still `videos/<basename>`, every output is written into the job tree beside the symlink, exactly as for a copied video. `pull` and the origin mapping of §8.2 are unchanged: the local `origin_path` is the laptop's mount of the same file, so outputs land beside the original on the share.

**What is deliberately not done.**

- Outputs are never written to the share directly by the remote; the job tree remains the single record and `pull` the single return path. Writing to the share from two hosts would also reintroduce the collision cases `_reject_collisions` guards against.
- Models are always copied. They are small next to videos, and copying is what makes the job self-contained. The alias mechanism is reusable for a shared models root later without changing the manifest format.
- No attempt is made to detect a share without the mount table (e.g. by comparing inodes or file signatures across hosts). An explicit alias is one line of config and fails in one obvious way.

## 7. `paths.py` change: `HYDRA_MODELS_DIR`

```python
def get_models_dir() -> Path:
    override = os.environ.get("HYDRA_MODELS_DIR")
    if override:
        path = Path(override).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return _user_data_dir() / "models"
```

Read per call like the other overrides (no caching), so tests monkeypatch `os.environ`. Documented in `paths.py:1-16`, `installation.md:242-258`, and `print_paths()`.

Everything that already routes through `get_models_dir()` (registry path, `get_models_root_directory()`, SAM3 checkpoints) follows automatically. `get_data_dir()`-rooted stores (`runtime-artifacts`, `inference_tuning_profiles`, `training/runs`, `vitpose-assets`) stay on the host. The plan includes a grep gate: no module may compute `get_data_dir() / "models"` directly.

## 7b. Path-independent cache keys

### 7b.1 What is wrong today (verified)

`CacheKey` (`core/inference/cache/base.py:22-48`) is `(schema_version, model_path, model_mtime, config_hash)`. Every cache on disk stores `key.as_string()` and is accepted only on exact string equality (`cache/store.py:67`, `cache/chunked.py:540,580,1124`). The ingredients that break across machines:

| Ingredient | Where | Why it breaks |
|---|---|---|
| `model_path` = absolute resolved path (`"detect|obb"` joined for sequential) | `keys.py:110-131,355,364,394` | `<job>/models/obb/x.pt` on the remote vs `~/Library/.../models/obb/x.pt` locally |
| `model_mtime` = `os.path.getmtime` | `keys.py:419-423` | survives `copy2`/`rsync -a` by accident, not by contract |
| `video_signature` = `f"{size}:{st_mtime_ns}"` | `keys.py:36-49`, folded via `with_video_signature` | nanosecond mtime survives only with rsync protocol ≥31 on both ends (this Mac 3.4.3, mehek 3.2.7: OK today; USB, `cp`, SMB, older rsync: broken) |

Everything else in the keys (canonical geometry, ROI mask sha, batch terms, stage params) is already content-based. Consumers that fold the video signature (`runner.py:543,975`, `cache/reuse.py:63-67`, density regions `worker.py:1434`, optimizer `optimizer.py:826`/`optimizer_workers.py:343`, `production_replay.py:185`, identity evidence `base_signature`, `parameter_helper.py:1818`) all call the one `video_signature()` function, so fixing it there fixes them uniformly.

### 7b.2 New definition

```python
@dataclass(frozen=True)
class CacheKey:
    schema_version: int
    model_id: str      # content identity, see below; sentinel strings unchanged
    config_hash: str
    def as_string(self): return f"v{self.schema_version}|{self.model_id}|{self.config_hash}"
    def matches(self, other): return self.as_string() == other.as_string()
```

`model_mtime` is removed. `model_path` becomes `model_id`:

- **File model** (`.pt`, `.pth`, `.onnx`, ClassKit checkpoint): `sha256:<hex>` of the file bytes. Sidecars are **not** hashed: the SAHI/canonical sidecars' content already flows into `config_hash` through the stage config (slice profile, geometry), so hashing them again would only add spurious misses when a sidecar is re-stamped.
- **Directory model** (pose SLEAP/ViTPose/YOLO dirs): `dirsha256:<hex>` over the sorted list of `(relative_path, sha256(file))` for every file the artifact fingerprint already considers (`core/individual/pose/artifacts.py:14-31`), excluding `.hydra-runtime-artifacts/`.
- **Sequential OBB**: `sha256:<detect>|sha256:<obb>` (same join as today).
- **Sentinels** `"background_subtraction"` and `""` (AprilTag) unchanged.
- ClassKit multi-head bundles: `model_id` of the selected checkpoint only, as today; sibling heads travel with it and are covered by `discover_multihead_model_bundle` at pack time.

`video_signature(path)` becomes `f"{size}:{sha256(head 8 MiB ‖ tail 8 MiB)[:32]}"`. Size plus the first and last 8 MiB catches every realistic replacement (re-encode, trim, different clip with the same name) without reading a 50 GB file; hashing 16 MiB costs ~30 ms. Symlinks are followed as today.

### 7b.3 Cost control

Hashing a 100 MB `.pt` costs ~0.3 s. `model_id` is computed once per process per `(realpath, size, mtime_ns)` through an in-process LRU (`functools.lru_cache` keyed on that triple), so mtime still serves as a **local** fast-path hint but never reaches the key. The optimizer and preview paths that rebuild keys many times per session therefore pay once.

### 7b.4 Schema bump and migration

`CACHE_SCHEMA_VERSION` 4 → 5. Every existing cache is invalidated once. No converter: caches are derived data and regenerate. The `v5` comment in `base.py` records the reason (path-independent identity for portable jobs).

### 7b.5 What does not change

- Which frames, arrays and dtypes are cached. This is a key change only.
- `config_hash` composition. ROI, geometry, batch terms, stage params are untouched.
- The read-only-handle rule from the detection-cache-wipe fix (read the stored key first when a cache is "ignored") still applies; the stored key is just a different string.
- Tracking output. The equivalence gate must show byte-identical CSVs before and after, on MPS and CUDA, with caches cleared on both sides (a stale `__pycache__` or an old `v4` cache is the known trap; see memory on poisoned numba caches).

### 7b.6 Tests specific to this change

1. Key invariance: the same model copied to a different absolute path with a different mtime yields an identical `as_string()`; a one-byte change yields a different one. Same for pose directories and for a video with touched mtime vs re-encoded content.
2. Round trip: pack a fixture job, run it into the job tree, `pull`, then run the same config locally in resume mode and assert the runner reports a cache hit for every stage (detection, head/tail, pose, CNN) with zero recomputed frames.
3. Schema bump: a `v4` cache is rejected and rebuilt, and the rebuilt key starts with `v5|sha256:`.
4. Memoization: hashing a model twice in one process reads the file once (count `open` calls via a fake).

## 8. Push and pull: transport

Both are thin wrappers over `rsync` over `ssh`; the tool never implements file transfer itself. Remote target syntax is `[user@]host:/abs/path/to/jobs/<job_id>`. The tool checks `rsync` is on PATH locally and remotely, and fails with the install hint otherwise.

### 8.1 `trackerkit job push <job_dir> <remote>`

Push is a **pure input sync**. The file list is built from the manifest, never from an exclude list:

```
rsync -a --copy-links --partial --info=progress2 --files-from=<inputs.txt> <job_dir>/ <remote>/
```

where `inputs.txt` = `hydra_job.json`, `run.sh`, `videos.txt`, `config/**`, every `models[]` file, sidecar and directory, every `videos[].job_path` **without** a `shared` entry, and every `pushed_siblings`. Nothing else travels, so:

- a re-push after a local `pull` cannot overwrite remote outputs with the stale copies now sitting in the local job tree;
- `logs/runs.jsonl` and every output the remote produced are untouched;
- `--copy-links` (`-L`) dereferences the video symlinks so real files land on the remote;
- never `--delete`.

After transfer, run `job verify` remotely over `ssh` (§11) and report. Push writes nothing to the manifest; it is idempotent and stateless.

### 8.2 `trackerkit job pull <remote> <job_dir> [--dry-run] [--no-caches]`

1. `rsync` the remote `logs/` first (including `logs/runs.jsonl`); the remote manifest is never fetched because it is never modified there.
2. Compute the output set on the remote: everything under `videos/` minus the video files minus each video's `pushed_siblings`. Done with one `ssh find` rather than enumerating names, so new artifact types (a future `<stem>_something/`) are pulled without a code change.
3. `rsync -a --partial --info=progress2 --files-from=<list> <remote>/ <job_dir>/` for that set, minus `.inference_cache_*` when `--no-caches`.
4. **Place back beside the originals.** For each video, for each pulled output relative to `videos/`, compute the destination by replacing the `videos/` prefix with `dirname(origin_path)`. Outputs that were redirected at pack time (`redirected_outputs`) go back to the recorded absolute path. Copy (hardlink when same filesystem) from the job tree to the destination; the job tree keeps its copy so the job remains a complete record.
5. Collision policy: if a destination exists and differs (sha256), refuse unless `--overwrite`; list every collision before touching anything.
6. Append a `pull_history` entry.

`--dry-run` prints the full destination map and stops.

### 8.3 Why both directions keep a job-tree copy

The job directory is the unit of reproducibility: after `pull` it contains the inputs, the exact config that ran, the models, the run log, and the outputs. Deleting it is the user's decision (`job clean` is deliberately not in scope).

## 9. Run: executing on the compute box

### 9.1 `run.sh` (generated by pack, executable)

```bash
#!/usr/bin/env bash
set -euo pipefail
JOB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HYDRA_HOST_CONFIG_DIR="${HYDRA_CONFIG_DIR:-}"   # host mount table lives here (§6.7)
export HYDRA_MODELS_DIR="$JOB/models"
export HYDRA_CONFIG_DIR="$JOB/config"
# HYDRA_DATA_DIR intentionally NOT set: engines + calibration stay host-scoped.
export KMP_DUPLICATE_LIB_OK=TRUE
cd "$JOB"
START="$(date -u +%FT%TZ)"
set +e
trackerkit track --video-list videos.txt "$@" 2>&1 | tee -a logs/run.log
CODE=${PIPESTATUS[0]}
set -e
trackerkit job _record-run --started "$START" --exit-code "$CODE" -- "$@"   # appends to logs/runs.jsonl
exit "$CODE"
```

`HYDRA_CONFIG_DIR=<job>/config` **shadows the compute box's own `advanced_config.json`**; the run uses the staging machine's snapshot. That is the intent (the experiment travels whole). `run.sh` honours `HYDRA_JOB_HOST_ADVANCED_CONFIG=1` as the escape hatch: it copies the host's `advanced_config.json` over the snapshot before launching and logs that it did.

`cd "$JOB"` is what makes the job-relative `videos.txt` and sidecar `file_path` values work with `load_video_list()`'s CWD-relative semantics (`app.py:314-349`) without changing that function. Extra arguments (`--gpus auto`, `--jobs 2`) pass straight through to `track`.

`run.sh` is the executable contract. Anyone can `ssh` in and run it by hand with `nohup`; `job run` is sugar.

### 9.2 `trackerkit job run <job_dir_or_remote> [--gpus …] [--jobs …] [--threads-per-job …] [--detach]`

- Local job dir: runs `preflight` (§9.4), then `run.sh` in the foreground; `run.sh` appends the run record to `logs/runs.jsonl`.
- Remote target: `ssh <host> 'cd <job> && ./run.sh …'`; with `--detach`, wraps in `nohup … &` and prints the log path. `job status <remote>` tails `logs/run.log` and reports the last `runs` entry.
- `track_args` recorded at pack time (`--sahi-profile`, `--apply-tuned-inference`, `--inference-autotune-manual`) are **always** forwarded by `run`, because the calibration lookup key includes the manual-field baseline digest (`calibrate_cli.py:112-118`); dropping them makes the profile lookup silently miss. `run` refuses `--sahi-profile` etc. on its own command line: those belong to the experiment and are fixed at pack time.

### 9.3 `trackerkit job calibrate <job_dir|remote> [--budget-seconds N]`

One-click calibration is lookup-only at track time (`trackerkit calibrate` writes the profile, `track` only reads it), and the profile key is host-fingerprinted. So on a fresh compute box `--apply-tuned-inference` always misses unless calibration runs there first. `job calibrate` runs `trackerkit calibrate` inside the job environment (same `HYDRA_MODELS_DIR`/`HYDRA_CONFIG_DIR`, same `cd`), against the keystone video, forwarding `track_args.inference_autotune_manual` and `sahi_profile` verbatim so the baseline digest matches what `run` will look up (`calibrate_cli.py:112-118`). `run --calibrate` chains the two. The profile stays on the host under `<data>/inference_tuning_profiles`; it is never pulled.

### 9.4 Preflight (`job preflight <job_dir>`, also run automatically by `run`)

Fails loudly, all checks reported before exit:

1. Manifest version supported; `verify` (§11) passes.
2. `hydra_suite` importable and `>= requirements.min_hydra_suite_version`.
3. Every `requirements.conda_envs` entry exists (`conda env list`), and `conda` is on PATH. This is the `pose_sleap_env` problem: not fixable by packing, so it fails here with the exact env name and a pointer to the SLEAP setup docs.
4. Requested `runtime_tier` is in `available_tiers()` for this host; if not, print the fallback the resolver would take and require `--allow-tier-fallback`.
5. Every `models/` entry hashes to the manifest sha256 (skippable with `--fast` after a verified push).
6. Shared-root references resolved, verified and materialized as symlinks (§6.7).
7. Free disk under `videos/` ≥ 1.5× total video bytes (caches and outputs), counting shared videos' sizes for caches but not for the videos themselves.

Result is written to `logs/preflight.json`.

## 10. Output discovery contract

"Output" is defined structurally, not by name: any path under `<job>/videos/` after the run that is not a video listed in the manifest and not in that video's `pushed_siblings`. For documentation and tests, the artifacts known today are:

| Artifact | Source |
|---|---|
| `<stem>_tracking.csv`, `<stem>_tracking.mp4` | `cli_config.py:270-276` |
| `<stem>_tracking_forward_processed.csv`, `<stem>_tracks.csv`, `*_with_individual.csv` (Debug) | `trajectory_writer.py` |
| `<stem>_logs/` | `video_artifacts.py:112-128` |
| `.inference_cache_<stem>/` (`detection.npz`, `opt/`) | `video_artifacts.py:95-109` |
| `<stem>_datasets/{active_learning,oriented_videos,individual_crops}/` | GUI-only today (§6.6) |
| `<stem>_config.json` | in `pushed_siblings`, **not** pulled back (the local original is authoritative) |

A test packs a fixture job, fabricates each of these on the "remote" side, and asserts `pull --dry-run` maps every one of them beside the origin video and nothing else.

## 11. Verify: `job verify <job_dir>`

Pure local check, no network:

- manifest parses, `job_version == 1`;
- every `models[]` file exists with matching sha256 and size; every `sidecars[]` exists;
- every `videos[].job_path` without `shared` exists (symlink target exists on the staging machine; regular file on the remote); `shared` entries are checked by preflight, not verify, because verify is offline and mount-agnostic; `videos[].config_job_path` exists;
- no sidecar config contains an absolute path in any key listed in §6.4 (this is the "portable by construction" assertion, run against real configs);
- `videos.txt` lines all exist and the first equals the keystone;
- every relative path in the manifest passes the traversal check (`_validated_archive_relpath` pattern from `project_bundle.py:264-274`, generalized to a `validate_job_relpath`).

`verify` runs at the end of `pack`, at the end of `push` (remotely), and at the start of `run`.

## 12. Module layout and dependency direction

```
src/hydra_suite/data/tracking_job/
    __init__.py          # public API: pack_job, verify_job, plan_pull, preflight_job, JobManifest
    manifest.py          # JobManifest dataclass, to_dict/from_dict, atomic write, validate_job_relpath
    pack.py              # pack_job: plan → resolve → collect → copy → rewrite → runner → verify
    references.py        # copy_model_reference (file/dir/bundle/sidecars), registry subset
    outputs.py           # output-set discovery + origin mapping (used by pull)
    runner.py            # run.sh template
    preflight.py         # host checks, shared-root resolution + materialization
    shared_roots.py      # mount table read/write, alias matching (longest root wins)
    transport.py         # rsync/ssh command construction + execution (subprocess), no policy
src/hydra_suite/trackerkit/job_cli.py   # argparse wiring for `trackerkit job …`, thin
src/hydra_suite/trackerkit/engine_params.py  # + iter_model_references + the four key tuples
src/hydra_suite/paths.py                     # + HYDRA_MODELS_DIR
src/hydra_suite/trackerkit/gui/orchestrators/config.py  # leak fixes (§6.5)
```

Dependency direction (Core/Data must never import an app layer):

- `discover_multihead_model_bundle` lives in `classkit/model_bundle.py` (app layer). It is **not** moved: `trackerkit/job_cli.py` performs bundle discovery and passes the expanded file list down to `pack.py` as part of each model reference, the same pattern used for engine params below.
- `build_engine_params` and the batch planner live in `trackerkit/` (app layer). `pack.py` therefore takes the planned per-video `(cfg, params)` pairs as **input**; `trackerkit/job_cli.py` does the planning and parameter building and hands the results down. `data/tracking_job` imports only from `data`, `core`, `training.model_publish`, `paths`, and the standard library.

## 13. CLI surface (final)

```
trackerkit job pack     <job_dir> (VIDEO... | --video-list FILE) [--config] [--keystone-override] [--sahi-profile] [--apply-tuned-inference|--no-apply-tuned-inference] [--inference-autotune-manual]... [--name] [--copy-videos] [--no-shared | --shared-only]
trackerkit job verify   <job_dir>
trackerkit job shared-root add <alias> <path> | remove <alias> | list
trackerkit job push     <job_dir> <remote>
trackerkit job preflight <job_dir> [--fast] [--allow-tier-fallback] [--shared-root ALIAS=PATH]...
trackerkit job run      <job_dir|remote> [--gpus] [--jobs] [--threads-per-job] [--detach] [--calibrate] [--allow-tier-fallback] [--shared-root ALIAS=PATH]...
trackerkit job calibrate <job_dir|remote> [--budget-seconds]
trackerkit job status   <remote>
trackerkit job pull     <remote> <job_dir> [--dry-run] [--no-caches] [--overwrite]
```

Registered in `build_parser()` (`app.py:57`) as `job` with nested subparsers, `allow_abbrev=False`; validated in `parse_arguments`; dispatched in `main` by string comparison like `track` and `calibrate`. Exit codes: 0 success, 2 argument/validation error, 3 preflight failure, 4 transport failure, 5 pull collision.

## 14. Error handling

- All job errors are `TrackingJobError(ValueError)` with a `code` attribute matching the exit codes; the CLI prints `error: <message>` and exits with the code. Follows `project_bundle.py`'s plain-`ValueError` style but adds the code for scripting.
- `rsync`/`ssh` failures surface the subprocess's stderr verbatim and the exact command line, so the user can rerun by hand.
- `pull` is all-or-nothing at the planning stage (collisions listed before any copy) but resumable at the transfer stage (`rsync --partial`).
- `pack` on a config that references a model that does not resolve on the staging machine fails immediately with the role key and the raw config value; it does not ship a broken job.

## 15. Testing

Unit (no network, tmp dirs, `HYDRA_*` monkeypatched):

1. `iter_model_references` contract guard (§6.3).
2. Pack rewrites: every §6.4 key ends job-relative; out-of-root model gets an `external/` key; ClassKit bundle siblings and pose directories copied whole; `.hydra-runtime-artifacts` excluded; registry subset contains exactly the shipped keys with `source_path=null`.
3. `HYDRA_MODELS_DIR` override: `get_models_dir()`, registry path, `get_models_root_directory()`, SAM3 checkpoint root all follow it while `get_data_dir()` does not.
4. `run.sh` + sidecars resolve: with `HYDRA_MODELS_DIR`/`HYDRA_CONFIG_DIR` pointed at a packed job and CWD at the job root, `load_tracker_cli_session` on each sidecar builds engine params whose model paths all lie inside `<job>/models` and equal, key for key, the params built on the staging side (the byte-identity of **params**, which is what the GUI/CLI parity tests already check).
5. Output discovery and origin mapping (§10), including redirected outputs and the collision policy.
6. Transport: `transport.py` builds the exact `rsync` argv (asserted as a list); the push `--files-from` list equals the manifest input set and contains no output path even when the local job tree holds pulled outputs and is exercised end-to-end against `localhost` only when `HYDRA_TEST_SSH_LOCALHOST=1`.
7. Leak fixes: GUI save of a config with identity classifiers under the models root has no absolute paths in `color_tag_model_path` / `cnn_classifiers[].model_path`; `test_gui_cli_param_equivalence.py` unchanged and green.
8. Preflight: missing conda env, unavailable tier, sha mismatch each fail with the expected code.
9. Shared roots: a video under a local alias is packed as a reference, absent from the push list and from `videos/`; on a fake remote with a different mount path the same alias resolves, the signature check passes, a symlink is materialized, and outputs written beside it are mapped back to the origin; a wrong alias, a missing file and a re-encoded file each fail preflight with the expected message; `--shared-only` fails pack for an off-share video.

Integration (manual, in the plan's acceptance section): pack `fly_obb` and `ant_pose_headtail` fixtures on this Mac, push to mehek, run, pull, and diff the pulled `_tracking.csv` against a native run on mehek with the same config: identical rows (same host, same models, same config).

Equivalence gate: two changes touch the tracking path, the save-side relativization (a load-side no-op) and the cache-key redefinition (§7b, key-only). Run the **full** matrix on MPS and CUDA with `.inference_cache_*` and `__pycache__` cleared on both sides; acceptance is byte-identical CSVs at the determinism floor. Then run the §7b.6 round-trip test against mehek as the acceptance proof for Goal 4.

## 16. Documentation

- New `docs/user-guide/trackerkit-jobs.md`: lifecycle walkthrough (pack → push → run → pull), the "what travels, what doesn't" table, the shared-root mount table with a two-host example, the conda-env requirement, and the nine-GPU example adapted to `job run --gpus auto`.
- `docs/user-guide/trackerkit-cli.md`: cross-link from `## A batch`.
- `docs/getting-started/installation.md:242-258`: add `HYDRA_MODELS_DIR`.
- `docs/developer-guide/`: a short note on `iter_model_references` as the contract every new model role must join.

## 17. Follow-ups (explicitly out of scope)

1. **Headless CLI dataset/media export** (user-requested follow-up). Close the gap in §6.6: derive `DATASET_OUTPUT_DIR`, `FINAL_MEDIA_EXPORT_VIDEO_OUTPUT_DIR`, `INDIVIDUAL_DATASET_OUTPUT_DIR` in `cli_config.py` exactly as the GUI does (`orchestrators/config.py:2241-2252`, `<video_dir>/<stem>_datasets/<subfolder>`), so a job run on a compute box emits active-learning, oriented-video and individual-crop exports beside the video and `pull` brings them home through the structural output discovery of §10 without further change. Gated by the byte-identity harness because it changes CLI engine params; until then `pack` keeps warning when an export stage is enabled.
2. GUI "Package job…" action in TrackerKit calling `pack_job` through a `BaseWorker`.
3. `job clean` and retention policy.
4. Unify the duplicated sidecar-path formula (`session_plan.py:20-26` vs `orchestrators/config.py:97-103`) into `session_plan.get_video_config_path`.
5. Registry-name references in configs (instead of relative paths) belong to the deferred model-registry unification spec and are not needed for portability.
