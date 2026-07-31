# Model Registry Unification — Design Spec

> **Status:** APPROVED, ready for planning. Prerequisite (ViTPose completion) merged to
> main 2026-07-30 (f8e8ff3b). Supersedes the earlier deferred stub of the same name.
> **Decided:** 2026-07-31.

## Goal

Make `model_registry.json` the single, authoritative inventory of every model across all
kits, with one owning module, rich per-family metadata that the model pickers actually
consume, all backends (including the three pose backends) registered on import, and an
interactive CLI to backfill pre-existing models so the registry is complete and usable.

## Motivation (verified on main)

- **Two parallel writer surfaces on one file.** `trackerkit/gui/model_utils.py`
  (`register_yolo_model`, `load/save_yolo_model_registry`, an **app layer**) and
  `training/model_publish.py` (`load/save_model_registry`, **Training layer**) both
  read/write `models/model_registry.json` with divergent schemas. Drift-prone.
- **Registry is near-write-only.** Nothing reads it to populate the model pickers — every
  kit globs the models directory and parses metadata out of the **filename**
  (`{ts}_{size}_{species}_{info}.pt`). Brittle; no structured metadata.
- **No pose backend is registered.** YOLO-pose, SLEAP, ViTPose imports copy into
  `models/pose/<Backend>/` but write no registry entry. `TrainingRole`
  (`training/contracts.py`) has OBB/detect/segment/classify roles but **no pose role**, so
  `publish_trained_model` cannot publish any pose model.

## Decisions (locked during brainstorming)

1. **Scope = three outcomes** (cross-kit browser explicitly deferred):
   (a) unify the two writer APIs; (b) registry-driven picker metadata; (c) pose in the
   registry + a pose publish role.
2. **Registry is the SOURCE OF TRUTH for listing.** Pickers list only registered models
   (not a directory glob). Nothing vanishes silently — see the upgrade-cliff notice.
3. **Interactive CLI migration** backfills metadata for pre-existing/unregistered models.
4. **Retire the old APIs outright** — no shims. Repoint every call site to the new module,
   delete the old functions, cover each moved caller with tests.

## Architecture

New **`hydra_suite/data/model_registry.py`** — the single owner of `model_registry.json`.
It lives in the **Data layer** (CLAUDE.md: "Data layer must be reusable from both GUI and
scripts"), so app kits, the Training layer, and the standalone CLI can all import it
without violating dependency direction (Core/Data/Training never import app layers).

```
hydra_suite/data/model_registry.py
    ModelRegistryEntry            (typed dataclass; see Schema)
    load_registry() / save_registry()          # atomic write; schema-versioned
    register_model(path, entry)                 # the one writer
    unregister_model(path) / get_entry(path)
    list_entries(task_family=None, backend=None)# the reader pickers use
    read_legacy_entry(raw) -> ModelRegistryEntry# back-compat: BOTH old formats
    find_unregistered(models_dirs) -> [Path]    # drives the migration CLI + notices
    entry_is_stale(entry) -> bool               # registered but file missing

consumers (old APIs DELETED, not shimmed):
    trackerkit/gui/orchestrators/config.py       -> register_model (all 5 families)
    trackerkit/gui/model_utils.py                -> remove_model_from_repository via new API
    training/model_publish.py::publish_trained_model -> register_model (incl. pose roles)
    <all kit pickers>                            -> list_entries()  (source of truth)
    scripts/migrate_model_registry.py (CLI)      -> find_unregistered + register_model
```

Call sites to repoint then delete-old (enumerated on main):
- `trackerkit/gui/model_utils.py` — `get_yolo_model_registry_path`, `load/save/register/
  unregister_yolo_model`, `_extract_registry_entries`, and the registry part of
  `remove_model_from_repository`.
- `trackerkit/gui/main_window.py:76-79` — the 4 re-exports.
- `trackerkit/gui/orchestrators/config.py:3991` — `register_yolo_model(...)` (extend to all
  families incl. the 3 pose backends).
- `training/model_publish.py` — `load/save_model_registry` (6 uses) + `publish_trained_model`.
- Tests to migrate/replace: `test_model_registry_helpers.py`,
  `test_model_publish_slice_geometry.py`, and `test_vitpose_registration_parity.py`
  (the "pose parity = no registry" lock is REVERSED here → replace with "pose registers").

## Schema — `ModelRegistryEntry`

Flat dataclass; `path` (repo-relative to models root) is the dict key; family-specific
fields optional. Root file: `{schema_version, entries: {path: entry}}`.

| Field | Applies to | Notes |
|---|---|---|
| `task_family` | all | `obb` / `detect` / `segment` / `classify` / `pose` |
| `backend` | all | `yolo` / `sleap` / `vitpose` (format/loader) |
| `usage_role` | all | existing concept; a `TrainingRole` value where applicable |
| `species` | all | user metadata |
| `notes` | all | free text (was `model_info`) |
| `added_at` | all | ISO timestamp |
| `source_path` | all | original path imported from |
| `stored_filename` | all | filename inside the repo |
| `size` | yolo | s/m/l/x (optional) |
| `num_keypoints` | pose | optional |
| `skeleton_name` | pose | optional |
| `num_classes` | classify | optional |
| `class_names` | classify | optional |
| `needs_review` | all | set by `--auto` backfill; UI can flag |
| `extra` | all | free dict — carries `.slice_meta` / `.multihead` refs; forward-compat |

`read_legacy_entry` maps both old formats into this: the GUI format
(`{size, species, model_info, added_at, source_path, stored_filename, task_family,
usage_role}`) and the `model_publish` format. Unknown legacy keys land in `extra`.

## Registration wiring

- **One writer.** Delete `register_yolo_model`, `load/save_yolo_model_registry`,
  `unregister_yolo_model`, and `model_publish`'s `load/save_model_registry`. Repoint
  `config.py` GUI imports so **all five families** register — including the three pose
  backends (`task_family="pose"`, `backend∈{yolo,sleap,vitpose}`, `num_keypoints` filled).
- **Pose publish role.** Add pose members to `TrainingRole` (e.g. `POSE_YOLO`, `POSE_SLEAP`,
  `POSE_VITPOSE`) so `publish_trained_model` can publish trained pose models like
  OBB/classify. (ViTPose training already produces a registerable `best.pt` — merged.)

## Pickers as source of truth

Each kit's picker lists **registry entries** filtered by `task_family`/`backend`, rendering
structured metadata (species, date, role, keypoint-count) instead of parsing filenames.

- **Upgrade-cliff notice (required).** On picker load, if `find_unregistered(models_dirs)`
  is non-empty, show a **non-blocking** banner/notice: "N models aren't in the registry yet
  — run `migrate-registry` to add them." The disappearance of unregistered/pose models is
  thus explained and actionable, never silent.
- **Stale entry.** `entry_is_stale` (registered but file missing) → picker greys/hides it;
  the CLI offers to prune. No crash on a dangling path.

## Interactive CLI migration — `scripts/migrate_model_registry.py` (also `python -m`)

- **Scan** all model dirs (obb / detection / classification / pose/{YOLO,SLEAP,ViTPose})
  via `find_unregistered()`.
- **Default interactive mode:** prompt per unregistered model for species / role / etc.,
  pre-filling guesses parsed from the filename so the user confirms/edits.
- **`--auto` mode:** non-interactive best-effort backfill from filename; entries flagged
  `needs_review`. One-shot so nothing vanishes; refine later.
- **Legacy upgrade:** also rewrites any legacy on-disk entries (both old formats) to the
  unified schema. **Idempotent**; safe to re-run. Offers to prune stale entries.

## Testing

- **Data module (heaviest):** schema round-trip; reading BOTH legacy formats → unified
  entry; `register`/`unregister`/`get`/`list_entries` (family/backend filters);
  `find_unregistered`; `entry_is_stale`; atomic write.
- **Each moved caller:** proves it now goes through `data.model_registry` — GUI import for
  all 5 families (incl. 3 pose backends); `publish_trained_model` with a pose role.
- **CLI:** `--auto` end-to-end + interactive via fed stdin; entries created; idempotent
  re-run; legacy upgrade; stale prune.
- **Picker source-of-truth:** registered models listed with metadata; unregistered → not
  listed + notice triggered.
- Replace `test_vitpose_registration_parity.py` with a "pose IS registered" test.

## Out of scope

Cross-kit "what models do I have" browser (deferred; depends on this foundation existing).
No changes to how models are loaded/run — this is inventory/metadata only.
