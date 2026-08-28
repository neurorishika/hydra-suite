# DetectKit Source Unification (Part A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** One registered `OBBSource` per logical dataset in a DetectKit project — collapse
AL-round import to the authoritative root only, and make SAM2 escalation upgrade a source's
canonical labels in place behind a staged accept/reject review step, instead of either path
spawning a new sibling `OBBSource`.

**Architecture:** This session's uncommitted multi-root AL-round registration (`inspect_al_round`,
`materialize_al_round`, `_add_al_round_sources`) is reverted — `inspect_detectkit_source`'s
existing single-redirect-to-authoritative-root behavior (with its stale-manifest-path fallback and
refuse-if-authoritative-missing guard) is kept and becomes the only AL-round import path.
`al_worker.py` registers exactly one source per round. `sam2_escalation.py::run_escalation` writes
its SAM2-primed result to a per-source staging directory under `artifacts/pending_escalations/`
and records that on the source's new `pending_escalation` field, instead of writing a `<name>_seg`
sibling and appending a new `OBBSource`. A new `ReviewEscalationsDialog` drives
`accept_pending_escalation`/`reject_pending_escalation` (new pure functions in
`sam2_escalation.py`) to promote or discard the staged result.

**Tech Stack:** Python 3.11+, PySide6 (Qt), pytest. `hydra-mps` conda env for all test runs in this
repo (see CLAUDE.md).

**Spec:** `docs/superpowers/specs/2026-08-27-detectkit-source-unification-design.md`

## Global Constraints

- No new module for Part A: only files already named in the spec's Architecture section change,
  plus one new dialog file (`review_escalations_dialog.py`) which the spec explicitly allows
  ("or a new sibling dialog").
- `data/al/export.py` (the manifest/multi-root writer) is **unchanged** — it still writes `obb/` +
  `aabb/` sibling folders to disk; only DetectKit's *registration* of those roots as project
  sources changes.
- No automatic migration of a project's existing flat sibling sources (manual cleanup only).
- `reviewed`'s lifecycle is unchanged beyond "escalation-accept resets it to `False`" — there is no
  "mark reviewed" action to add; the existing `_on_mark_reviewed` in `main_window.py` stays as-is.
- Naming: a registered source's display name is the bare originally-selected folder's name (no
  level suffix) — `Path(selected_path).name` for manual import, `al_round_<timestamp>` for
  AL-worker-generated rounds (both already timestamp-unique, no collision handling needed).
- All test runs use `conda activate hydra-mps` first (per CLAUDE.md). No CUDA/MPS equivalence gate
  needed for this work — it does not touch the tracking/inference pipeline.
- Commit as the configured git user — do not add a `Co-Authored-By: Claude` trailer or
  `Claude-Session:` line to any commit message in this plan.

---

## Task 1: `PendingEscalation` model + `OBBSource.pending_escalation` field

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/models.py`
- Test: `tests/test_obbsource_reviewed.py`

**Interfaces:**
- Produces: `PendingEscalation` dataclass (`staged_path: str`, `target_level: str`,
  `sam2_variant: str`, `created_at: str`) with `to_dict()`/`from_dict()`; `OBBSource.pending_escalation:
  PendingEscalation | None = None` (new field, appended after `sam2_variant` in the dataclass and in
  `to_dict`/`from_dict`). Later tasks (`sam2_escalation.py`, `review_escalations_dialog.py`) read
  and write `source.pending_escalation` and construct `PendingEscalation(...)` directly.

- [ ] **Step 1: Write the failing round-trip test**

Add to `tests/test_obbsource_reviewed.py`:

```python
def test_pending_escalation_roundtrip():
    from hydra_suite.detectkit.gui.models import PendingEscalation

    pending = PendingEscalation(
        staged_path="/tmp/proj/artifacts/pending_escalations/orig-sam2.1-abc123",
        target_level="polygon",
        sam2_variant="sam2.1-hiera-base_plus",
        created_at="2026-08-27T12:00:00",
    )
    back = PendingEscalation.from_dict(pending.to_dict())
    assert back == pending


def test_obbsource_pending_escalation_roundtrip():
    from hydra_suite.detectkit.gui.models import PendingEscalation

    pending = PendingEscalation(
        staged_path="/tmp/staged",
        target_level="polygon",
        sam2_variant="sam2.1-hiera-base_plus",
        created_at="2026-08-27T12:00:00",
    )
    s = OBBSource(name="orig", pending_escalation=pending)
    back = OBBSource.from_dict(s.to_dict())
    assert back.pending_escalation == pending


def test_obbsource_pending_escalation_defaults_none():
    back = OBBSource.from_dict({"name": "legacy", "level": "obb"})
    assert back.pending_escalation is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_obbsource_reviewed.py -v`
Expected: the three new tests FAIL with `ImportError: cannot import name 'PendingEscalation'` (or
`AttributeError`/`TypeError: unexpected keyword argument 'pending_escalation'`).

- [ ] **Step 3: Implement `PendingEscalation` and the new field**

In `src/hydra_suite/detectkit/gui/models.py`, add right after the module docstring/imports (needs
`@dataclass` already imported), immediately before the `OBBSource` class:

```python
@dataclass
class PendingEscalation:
    """A staged (not-yet-reviewed) SAM2 escalation result awaiting accept/reject."""

    staged_path: str = ""
    target_level: str = "polygon"
    sam2_variant: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "staged_path": self.staged_path,
            "target_level": self.target_level,
            "sam2_variant": self.sam2_variant,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "PendingEscalation":
        """Restore a PendingEscalation from a dictionary."""
        return PendingEscalation(
            staged_path=str(d.get("staged_path", "")),
            target_level=str(d.get("target_level", "polygon") or "polygon"),
            sam2_variant=str(d.get("sam2_variant", "")),
            created_at=str(d.get("created_at", "")),
        )
```

Then in `OBBSource`, add the field after `sam2_variant`:

```python
    sam2_variant: str | None = None  # SAM2 version that primed a derived source
    pending_escalation: PendingEscalation | None = None  # staged, unreviewed escalation
```

And in `OBBSource.to_dict`, after `"sam2_variant": self.sam2_variant,`:

```python
            "pending_escalation": (
                self.pending_escalation.to_dict()
                if self.pending_escalation is not None
                else None
            ),
```

And in `OBBSource.from_dict`, after `sam2_variant=(d.get("sam2_variant") or None),`:

```python
            pending_escalation=(
                PendingEscalation.from_dict(d["pending_escalation"])
                if d.get("pending_escalation")
                else None
            ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_obbsource_reviewed.py -v`
Expected: all PASS (6 tests total: the 3 pre-existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/models.py tests/test_obbsource_reviewed.py
git commit -m "feat(detectkit): add PendingEscalation model for staged SAM2 review"
```

---

## Task 2: Revert `source_import.py` to single-redirect AL-round handling

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/source_import.py`
- Test: `tests/test_detectkit_source_import.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `inspect_detectkit_source(source_root)` unchanged in behavior (redirects an AL-round
  container to its authoritative root's inspection, `source_kind="detectkit_al"`, raises
  `ValueError` if the authoritative root can't be resolved). `materialize_detectkit_source` unchanged.
  `inspect_al_round`, `materialize_al_round`, `ALRoundRoot`, `MaterializedALRoundRoot` **no longer
  exist** — Task 3 (`source_manager.py`) must not import them.

- [ ] **Step 1: Remove the multi-root functions/dataclasses from `source_import.py`**

In `src/hydra_suite/detectkit/gui/source_import.py`, delete these blocks entirely (they were all
added this session and are consumed only by the multi-root registration path being reverted):

- The `ALRoundRoot` dataclass (currently lines 272–280).
- The `inspect_al_round` function (currently lines 283–329).
- The `MaterializedALRoundRoot` dataclass (currently lines 765–772, near the bottom of the file).
- The `materialize_al_round` function (currently lines 775–811, the rest of the file).

Keep everything else as-is, in particular: `_load_al_round_roots`, `_resolve_al_round_entry_path`,
`_select_al_round_authoritative_root`, and `inspect_detectkit_source`'s AL-round redirect block
(the `al_roots = _load_al_round_roots(root)` section near the end of `inspect_detectkit_source`) —
these are still needed by the single-redirect path and by `materialize_detectkit_source`.

After deletion, the file should end at `materialize_detectkit_source` (currently ending around line
762 with the final `return MaterializedDetectKitSource(...)` for the imported case) — nothing after
it.

- [ ] **Step 2: Update `tests/test_detectkit_source_import.py` to drop removed-function tests**

Remove `inspect_al_round` and `materialize_al_round` from the import block at the top of the file
(lines 11–17):

```python
from hydra_suite.detectkit.gui.source_import import (
    IMPORT_MODE_LINKED,
    inspect_detectkit_source,
    materialize_detectkit_source,
)
```

Delete these test functions entirely (they test functions that no longer exist):
- `test_inspect_al_round_returns_every_sibling_authoritative_first`
- `test_inspect_al_round_returns_none_for_non_al_round_folder`
- `test_materialize_al_round_imports_every_sibling`
- `test_inspect_al_round_falls_back_when_manifest_paths_are_stale`

Replace `test_inspect_al_round_returns_none_when_authoritative_root_missing` with a version that
drops the `inspect_al_round` assertion and keeps only the `inspect_detectkit_source` refusal (this
behavior — refusing when the authoritative root is missing — is being kept, just accessed through
one function now instead of two):

```python
def test_inspect_detectkit_source_raises_when_authoritative_root_missing(
    tmp_path: Path,
):
    """If the authoritative root is gone (deleted, and its manifest path is
    also stale/unresolvable) but a derived sibling survives, the single-root
    redirect must refuse rather than silently presenting the unreviewed
    derived sibling as if it were the whole round."""
    round_dir = tmp_path / "active_learning" / "20260827_172624"
    _write_al_round(round_dir)
    shutil.rmtree(round_dir / "obb")

    with pytest.raises(ValueError):
        inspect_detectkit_source(round_dir)
```

Keep all other tests unchanged: `test_inspect_detectkit_source_accepts_existing_canonical_root`,
`test_materialize_detectkit_source_converts_yolo_detect_boxes`,
`test_materialize_detectkit_source_converts_coco_bbox_annotations`, `_write_al_round` (the helper),
`test_inspect_detectkit_source_resolves_al_round_to_authoritative_root`,
`test_materialize_detectkit_source_imports_al_round_authoritative_root`,
`test_inspect_detectkit_source_falls_back_when_manifest_paths_are_stale`,
`test_materialize_detectkit_source_can_link_and_normalize_in_place`.

- [ ] **Step 3: Run the test file**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_source_import.py -v`
Expected: all PASS, no `ImportError`.

- [ ] **Step 4: Run flake8/black/isort on the two changed files**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/source_import.py tests/test_detectkit_source_import.py && isort src/hydra_suite/detectkit/gui/source_import.py tests/test_detectkit_source_import.py`
Expected: no errors; files reformatted if needed.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/source_import.py tests/test_detectkit_source_import.py
git commit -m "refactor(detectkit): revert AL-round import to single authoritative-root redirect"
```

---

## Task 3: Revert `source_manager.py::_add_source` to single-source registration

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/source_manager.py`
- Test: `tests/test_detectkit_source_manager_dialog.py`

**Interfaces:**
- Consumes: `inspect_detectkit_source`, `materialize_detectkit_source` from `source_import.py`
  (Task 2's surviving functions — `inspect_al_round`/`materialize_al_round` imports must be removed).
- Produces: `SourceManagerDialog._add_source()` registers exactly one `OBBSource` per pick, named
  `Path(selected_path).name` — for an AL-round pick this is the round folder's own name (e.g.
  `20260827_172624`), never `obb` or a level-suffixed name. `_add_al_round_sources` **no longer
  exists**.

- [ ] **Step 1: Update the test file first (TDD — these tests currently describe the old behavior)**

In `tests/test_detectkit_source_manager_dialog.py`, replace
`test_source_manager_adds_al_round_registers_every_sibling` and
`test_source_manager_al_round_links_to_already_registered_authoritative_root` with a single test
asserting the collapsed behavior:

```python
def test_source_manager_add_source_collapses_al_round_to_one_source(
    qapp, tmp_path, monkeypatch
):
    """Picking an AL round container registers exactly ONE source -- the
    authoritative root -- named after the round folder itself, not the
    resolved level subfolder ("obb") and not one entry per sibling level."""
    import json

    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.dialogs.source_validation import (
        SOURCE_ADD_MODE_PORTABLE,
        DetectKitSourceAdditionChoice,
    )

    round_dir = tmp_path / "active_learning" / "20260827_172624"
    for level in ("obb", "aabb"):
        level_dir = round_dir / level
        (level_dir / "images").mkdir(parents=True)
        (level_dir / "labels").mkdir(parents=True)
        (level_dir / "images" / "f001.jpg").write_bytes(b"fake-image")
        (level_dir / "labels" / "f001.txt").write_text(
            "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
        )
        (level_dir / "classes.txt").write_text("ant\n", encoding="utf-8")

    (round_dir / "manifest.json").write_text(
        json.dumps(
            {
                "roots": [
                    {
                        "level": "obb",
                        "authoritative": True,
                        "reviewed": True,
                        "path": str(round_dir / "obb"),
                    },
                    {
                        "level": "aabb",
                        "authoritative": False,
                        "reviewed": False,
                        "path": str(round_dir / "aabb"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(round_dir),
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.confirm_detectkit_source_addition",
        lambda *args, **kwargs: DetectKitSourceAdditionChoice(
            mode=SOURCE_ADD_MODE_PORTABLE
        ),
    )

    proj = _make_proj(tmp_path)
    dlg = SourceManagerDialog(proj)
    dlg._add_source()

    assert len(proj.sources) == 1
    added = proj.sources[0]
    assert added.name == "20260827_172624"
    assert added.level == "obb"
    assert added.source_kind == "detectkit_al"
    assert added.reviewed is True
    assert added.derived_from is None
    assert Path(added.path).is_dir()
    assert dlg._source_list.count() == 1
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_source_manager_dialog.py::test_source_manager_add_source_collapses_al_round_to_one_source -v`
Expected: FAIL — either an assertion mismatch (`len(proj.sources) == 2` currently) or an error, since
`_add_source` still calls the multi-root path.

- [ ] **Step 3: Revert `_add_source` and remove `_add_al_round_sources`**

In `src/hydra_suite/detectkit/gui/dialogs/source_manager.py`, change the import block:

```python
from ..source_import import (
    IMPORT_MODE_LINKED,
    IMPORT_MODE_PORTABLE,
    compute_positional_class_remap,
    inspect_detectkit_source,
    materialize_detectkit_source,
    remap_materialized_source_classes,
)
```

(drop `inspect_al_round`, `materialize_al_round`).

Replace the entire `_add_source` method body from `al_roots = inspect_al_round(selected_path)`
through the end of the method (the `self._project.sources.append(...)` / `self._refresh_list()` at
its end) with:

```python
        try:
            inspection = inspect_detectkit_source(selected_path)
        except Exception as exc:
            QMessageBox.warning(self, "Add Source", str(exc))
            return

        # inspection.dataset_root may differ from selected_path when the user
        # picked an active-learning round container -- review the resolved
        # (authoritative) dataset root, not the container.
        selection = confirm_detectkit_source_addition(
            self, str(inspection.dataset_root), inspection
        )
        if selection in {None, False}:
            return
        selection_mode = getattr(
            selection,
            "mode",
            (
                SOURCE_ADD_MODE_LINKED
                if selection == SOURCE_ADD_MODE_LINKED
                else SOURCE_ADD_MODE_PORTABLE
            ),
        )
        import_mode = (
            IMPORT_MODE_LINKED
            if selection_mode == SOURCE_ADD_MODE_LINKED
            else IMPORT_MODE_PORTABLE
        )

        force_remap = False
        project_classes = list(self._project.class_names)
        source_classes = list(inspection.discovered_labels)
        if source_classes != project_classes:
            remap_preview = compute_positional_class_remap(
                source_classes, project_classes
            )
            mapping_lines: list[str] = []
            for source_idx, target_idx in sorted(remap_preview.items()):
                source_name = (
                    source_classes[source_idx]
                    if 0 <= source_idx < len(source_classes)
                    else f"class {source_idx}"
                )
                target_name = (
                    project_classes[target_idx]
                    if 0 <= target_idx < len(project_classes)
                    else f"class {target_idx}"
                )
                mapping_lines.append(
                    f"  source[{source_idx}] {source_name!r} → "
                    f"project[{target_idx}] {target_name!r}"
                )
            dropped = sorted(
                {
                    source_idx
                    for source_idx in range(len(source_classes))
                    if source_idx not in remap_preview
                }
            )
            preview_text = (
                "Source classes do not match the project class scheme.\n\n"
                f"Project classes: {project_classes}\n"
                f"Source classes:  {source_classes}\n\n"
                "Force the source labels to match the project classes by mapping "
                "by position?\n" + "\n".join(mapping_lines)
            )
            if dropped:
                dropped_names = ", ".join(
                    f"{i}:{source_classes[i]!r}"
                    for i in dropped
                    if 0 <= i < len(source_classes)
                )
                preview_text += (
                    "\n\nThese source classes will be dropped: " + dropped_names
                )
            answer = QMessageBox.question(
                self,
                "Class Mismatch",
                preview_text,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            force_remap = True

        try:
            materialized = materialize_detectkit_source(
                selected_path,
                self._project.project_dir,
                import_mode=import_mode,
                force_import=force_remap,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Add Source", str(exc))
            return

        if force_remap:
            try:
                remap = compute_positional_class_remap(source_classes, project_classes)
                remap_materialized_source_classes(
                    Path(materialized.canonical_path),
                    project_classes,
                    remap,
                )
            except Exception as exc:
                QMessageBox.warning(self, "Add Source", str(exc))
                return

        canonical_path = str(materialized.canonical_path)
        original_path = str(materialized.source_root)
        if canonical_path in existing_paths or original_path in existing_paths:
            QMessageBox.information(self, "Add Source", "Source already added.")
            return

        self._project.sources.append(
            OBBSource(
                path=canonical_path,
                name=Path(selected_path).name,
                original_path=original_path,
                source_kind=materialized.source_kind,
                imported=materialized.imported,
                level=(
                    selection.level
                    if getattr(selection, "level", None)
                    else materialized.level
                ),
            )
        )
        self._refresh_list()
```

Note the one behavior change from the pre-session original: `name=Path(selected_path).name` (was
`materialized.display_name`, which is `root.name` where `root` is the *resolved* dataset root —
wrong for an AL-round pick, since that resolves to the level subfolder, e.g. `obb`).

Then delete the entire `_add_al_round_sources` method (everything from `def _add_al_round_sources(`
through its closing `self._refresh_list()`, right before `_remove_selected`).

- [ ] **Step 4: Run the new test to verify it passes**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_source_manager_dialog.py -v`
Expected: all PASS, including the new collapsed-source test and every pre-existing test in the file
(`test_source_manager_adds_imported_yolo_detect_source`,
`test_source_manager_does_not_add_source_when_validation_cancelled`,
`test_source_manager_adds_linked_source_in_place`, etc.).

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/dialogs/source_manager.py tests/test_detectkit_source_manager_dialog.py && isort src/hydra_suite/detectkit/gui/dialogs/source_manager.py tests/test_detectkit_source_manager_dialog.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/dialogs/source_manager.py tests/test_detectkit_source_manager_dialog.py
git commit -m "refactor(detectkit): collapse AL-round source-manager import to one source"
```

---

## Task 4: `al_worker.py` registers one source per round

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/al_worker.py`
- Test: `tests/test_detectkit_al_worker.py`

**Interfaces:**
- Consumes: `manifest = export_al_dataset(...)` (unchanged — `export.py` is out of scope). Each
  `manifest["roots"]` entry has `level`, `derived_from` (`None` for the authoritative root, else the
  native level's label), `reviewed`, `path`.
- Produces: `run_active_learning` appends exactly one `OBBSource` to `req.project.sources`, for the
  root where `derived_from is None`, named `al_round_<timestamp>` (was `al_round_<timestamp>_<level>`
  per root). `ALResult.source_path` is that root's path (unchanged shape, just resolved differently).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_detectkit_al_worker.py` (after `test_al_worker_writes_seeded_labels_and_registers_source`):

```python
def test_al_worker_registers_only_authoritative_source_for_multi_level_export(tmp_path):
    """A round exported at multiple levels (obb authoritative + aabb derived)
    must register exactly ONE project source -- the authoritative root -- not
    one sibling per level. The derived level's folder still gets written to
    disk by export_al_dataset (unchanged), it's just not registered."""
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])

    folder = _seed_image_folder(tmp_path, n=3)

    def fake_detector(frame, conf, iou):
        return [(10, 10, 8, 4, 0.0, 0.95)]

    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=3,
        preset="balanced",
        expected_count=1,
        detector_fn=fake_detector,
        diversity_window=0,
        probabilistic=False,
        export_levels=["obb", "aabb"],
        native_level="obb",
    )

    result = run_active_learning(request)

    assert len(project.sources) == 1
    registered = project.sources[0]
    assert registered.level == "obb"
    assert registered.derived_from is None
    assert registered.name.startswith("al_round_")
    assert "_obb" not in registered.name
    assert "_aabb" not in registered.name
    assert registered.path == result.source_path

    # The derived aabb sibling still exists on disk (export.py is unchanged)
    # even though it was not registered as a project source.
    aabb_root = Path(result.source_path).parent / "aabb"
    assert aabb_root.is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_al_worker.py::test_al_worker_registers_only_authoritative_source_for_multi_level_export -v`
Expected: FAIL with `assert 2 == 1` (current code registers one source per level).

- [ ] **Step 3: Implement the single-registration change**

In `src/hydra_suite/detectkit/jobs/al_worker.py`, replace the block from the comment
`# One OBBSource per written level. ...` through `source_path = manifest["roots"][0]["path"]`
(currently lines 305–335) with:

```python
    # ONE OBBSource for the round's authoritative (native-level) root only.
    # The exporter still writes every requested level's sibling folder to
    # disk (data/al/export.py is unchanged) -- those siblings are simply not
    # registered as separate project sources; training derives lower levels
    # from the registered source on demand, same as any other source.
    authoritative_root = next(
        root_meta for root_meta in manifest["roots"] if root_meta["derived_from"] is None
    )
    req.project.sources.append(
        OBBSource(
            path=authoritative_root["path"],
            name=f"al_round_{timestamp}",
            validated=False,
            original_path=req.input_path,
            source_kind="detectkit_al",
            imported=True,
            level=authoritative_root["level"],
            reviewed=bool(authoritative_root["reviewed"]),
            derived_from=None,
        )
    )

    source_path = authoritative_root["path"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_al_worker.py -v`
Expected: all PASS, including the 4 pre-existing tests in the file (they only ever exercised
single-level export, so this change is behavior-preserving for them).

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/jobs/al_worker.py tests/test_detectkit_al_worker.py && isort src/hydra_suite/detectkit/jobs/al_worker.py tests/test_detectkit_al_worker.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/jobs/al_worker.py tests/test_detectkit_al_worker.py
git commit -m "refactor(detectkit): al_worker registers one source per round, not one per level"
```

---

## Task 5: Staged accept/reject SAM2 escalation

**Files:**
- Modify: `src/hydra_suite/detectkit/jobs/sam2_escalation.py`
- Test: `tests/test_sam2_escalation.py`

**Interfaces:**
- Consumes: `PendingEscalation` from `hydra_suite.detectkit.gui.models` (Task 1);
  `ensure_bundle_subdirectory` from `hydra_suite.data.project_bundle` (existing).
- Produces: `EscalationResult.staged: list[str]` (replaces `derived`) — names of sources that got a
  fresh `pending_escalation`. `run_escalation(req, executor, *, overwrite=False, progress=None)` no
  longer appends any `OBBSource`; it only mutates `source.pending_escalation` on existing sources.
  Two new pure functions: `accept_pending_escalation(source: OBBSource) -> None` and
  `reject_pending_escalation(source: OBBSource) -> None`, both raising `ValueError` if
  `source.pending_escalation is None`. Task 6 (review dialog) and Task 7 (main_window wiring) call
  `result.staged`, `accept_pending_escalation`, `reject_pending_escalation`.

- [ ] **Step 1: Replace `tests/test_sam2_escalation.py` with tests for the new behavior**

This is a full rewrite of the file (the whole `<name>_seg`-sibling behavior it tested is gone).
Write the new file content:

```python
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.detectkit.gui.models import OBBSource, PendingEscalation
from hydra_suite.detectkit.jobs.sam2_escalation import (
    EscalationRequest,
    accept_pending_escalation,
    reject_pending_escalation,
    run_escalation,
)


class _FakeExec:
    """Returns a full-object mask for detection 0, empty mask for others."""

    def __init__(self):
        self.calls = 0

    def set_image(self, img):
        pass

    def segment(self, box, pos, neg):
        self.calls += 1
        if self.calls == 1:
            m = np.zeros((100, 100), bool)
            m[10:40, 10:40] = True
            return m, 0.9
        return np.zeros((100, 100), bool), 0.0  # -> fallback


def _make_source(tmp_path):
    root = tmp_path / "sources" / "orig"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "a.jpg"), np.zeros((100, 100, 3), np.uint8))
    # two OBB detections
    (root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n" "0 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n"
    )
    (root / "classes.txt").write_text("ant\n")
    return OBBSource(path=str(root), name="orig", level="obb")


def test_escalation_stages_without_touching_canonical_labels(tmp_path):
    src = _make_source(tmp_path)
    original_label_text = (Path(src.path) / "labels" / "a.txt").read_text()
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )

    result = run_escalation(req, _FakeExec())

    assert result.staged == ["orig"]
    assert result.primed == 1 and result.fell_back == 1
    assert result.skipped == []

    # Canonical source untouched.
    assert (Path(src.path) / "labels" / "a.txt").read_text() == original_label_text
    assert src.level == "obb"
    assert src.reviewed is True

    # No new OBBSource registered.
    assert [s.name for s in project.sources] == ["orig"]

    pending = src.pending_escalation
    assert pending is not None
    assert pending.target_level == "polygon"
    assert pending.sam2_variant == "sam2.1-hiera-base_plus"
    staged_label = Path(pending.staged_path) / "labels" / "a.txt"
    assert staged_label.exists() and len(staged_label.read_text().splitlines()) == 2
    assert (Path(pending.staged_path) / "classes.txt").read_text() == "ant\n"


def test_rerun_without_overwrite_skips_existing_pending(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    first_staged_path = src.pending_escalation.staged_path

    result2 = run_escalation(req, _FakeExec(), overwrite=False)

    assert result2.staged == []
    assert len(result2.skipped) == 1 and result2.skipped[0][0] == "orig"
    assert src.pending_escalation.staged_path == first_staged_path  # untouched


def test_rerun_with_overwrite_restages(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    first_staged_path = src.pending_escalation.staged_path

    result2 = run_escalation(req, _FakeExec(), overwrite=True)

    assert result2.staged == ["orig"]
    assert result2.skipped == []
    # Same content-hashed staging dir reused, not accumulated.
    assert src.pending_escalation.staged_path == first_staged_path
    assert Path(first_staged_path).is_dir()


def test_accept_pending_escalation_promotes_labels_and_resets_reviewed(tmp_path):
    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    staged_label_text = (Path(src.pending_escalation.staged_path) / "labels" / "a.txt").read_text()
    staged_path = src.pending_escalation.staged_path

    accept_pending_escalation(src)

    assert src.pending_escalation is None
    assert src.level == "polygon"
    assert src.reviewed is False
    assert src.sam2_variant == "sam2.1-hiera-base_plus"
    assert (Path(src.path) / "labels" / "a.txt").read_text() == staged_label_text
    assert not Path(staged_path).exists()


def test_reject_pending_escalation_discards_staging_leaves_source_untouched(tmp_path):
    src = _make_source(tmp_path)
    original_label_text = (Path(src.path) / "labels" / "a.txt").read_text()
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())
    staged_path = src.pending_escalation.staged_path

    reject_pending_escalation(src)

    assert src.pending_escalation is None
    assert src.level == "obb"
    assert src.reviewed is True
    assert (Path(src.path) / "labels" / "a.txt").read_text() == original_label_text
    assert not Path(staged_path).exists()


def test_accept_without_pending_raises():
    src = OBBSource(name="orig", level="obb")
    with pytest.raises(ValueError):
        accept_pending_escalation(src)


def test_reject_without_pending_raises():
    src = OBBSource(name="orig", level="obb")
    with pytest.raises(ValueError):
        reject_pending_escalation(src)


def test_worker_runs_with_injected_executor(tmp_path):
    from hydra_suite.detectkit.jobs.sam2_escalation import Sam2EscalationWorker

    src = _make_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project,
        source_names=["orig"],
        variant="sam2.1-hiera-base_plus",
    )
    worker = Sam2EscalationWorker(req, executor=_FakeExec())
    captured = {}
    worker.result_ready.connect(lambda r: captured.update(staged=r.staged))
    worker.execute()  # call directly (no thread) — BaseWorker pattern
    assert captured["staged"] == ["orig"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_sam2_escalation.py -v`
Expected: FAIL — `ImportError: cannot import name 'accept_pending_escalation'` (and similar).

- [ ] **Step 3: Rewrite `sam2_escalation.py`**

Replace the full content of `src/hydra_suite/detectkit/jobs/sam2_escalation.py` with:

```python
"""SAM2 escalation orchestrator: existing OBB/box labels -> staged polygon review."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.core.inference.sam2.masks import mask_to_contour
from hydra_suite.data.al.escalation import LabelRecord
from hydra_suite.data.al.labels import write_label_file
from hydra_suite.data.project_bundle import ensure_bundle_subdirectory
from hydra_suite.detectkit.gui.models import OBBSource, PendingEscalation
from hydra_suite.utils.geometry_levels import GeometryLevel
from hydra_suite.widgets.workers import BaseWorker

from .sam2_prompts import build_prompts, read_boxes_from_label


@dataclass
class EscalationRequest:
    project: object  # has .project_dir and .sources (list[OBBSource])
    source_names: list[str]
    variant: str
    overwrite: bool = False


@dataclass
class EscalationResult:
    # Names of sources that received a fresh `pending_escalation` this run.
    staged: list[str] = field(default_factory=list)
    primed: int = 0
    fell_back: int = 0
    # (source_name, reason) pairs for sources skipped because they already
    # have a pending escalation and overwrite was not requested.
    skipped: list[tuple[str, str]] = field(default_factory=list)


class Sam2EscalationWorker(BaseWorker):
    """QThread wrapper around run_escalation (BaseWorker signals + result_ready)."""

    result_ready = Signal(object)  # EscalationResult

    def __init__(self, request: EscalationRequest, executor=None, parent=None) -> None:
        super().__init__(parent)
        self._request = request
        self._executor = executor

    def execute(self) -> None:
        from hydra_suite.core.inference.sam2.executor import Sam2SegmentExecutor

        executor = self._executor or Sam2SegmentExecutor.from_variant(
            self._request.variant
        )
        self.status.emit(f"Escalating {len(self._request.source_names)} source(s)...")
        result = run_escalation(
            self._request,
            executor,
            overwrite=self._request.overwrite,
            progress=lambda pct, msg: (
                self.progress.emit(pct),
                self.status.emit(msg),
            ),
        )
        self.status.emit(
            f"Done: {len(result.staged)} staged, {result.primed} primed, "
            f"{result.fell_back} fell back (review these first)."
        )
        self.result_ready.emit(result)


def _sources_by_name(project) -> dict[str, OBBSource]:
    return {s.name: s for s in project.sources}


def run_escalation(
    req: EscalationRequest,
    executor,
    *,
    overwrite: bool = False,
    progress: Callable[[int, str], None] | None = None,
) -> EscalationResult:
    """Stage each named source's SAM2-primed polygon labels for review.

    Writes the primed result to a per-source staging directory under
    ``artifacts/pending_escalations/`` and records it on the source's
    ``pending_escalation`` field. It does NOT touch the source's own
    canonical labels and does NOT register any new source -- a caller (the
    escalation review dialog) must call ``accept_pending_escalation`` or
    ``reject_pending_escalation`` to promote or discard the staged result.

    Re-running escalation over a source that already has a pending
    escalation is guarded: by default it's skipped (recorded in
    ``result.skipped``) rather than silently clobbering a staged result the
    user hasn't reviewed yet. Pass ``overwrite=True`` to re-stage (replaces
    the staging directory in place).
    """
    result = EscalationResult()
    by_name = _sources_by_name(req.project)
    todo = [
        by_name[n]
        for n in req.source_names
        if n in by_name and by_name[n].level != "polygon"
    ]
    project_root = Path(req.project.project_dir)
    for si, src in enumerate(todo):
        if src.pending_escalation is not None and not overwrite:
            result.skipped.append(
                (
                    src.name,
                    f"'{src.name}' already has a pending escalation; review it, "
                    "or re-run with overwrite to replace it.",
                )
            )
            continue

        src_root = Path(src.path)
        images_dir = src_root / "images"
        labels_dir = src_root / "labels"

        content_hash = sha1(
            (str(src_root.resolve()) + req.variant).encode("utf-8")
        ).hexdigest()[:10]
        staged_dirname = f"{src.name}-{req.variant}-{content_hash}"
        staged_root = ensure_bundle_subdirectory(
            project_root, f"artifacts/pending_escalations/{staged_dirname}"
        )
        shutil.rmtree(staged_root, ignore_errors=True)
        (staged_root / "labels").mkdir(parents=True, exist_ok=True)

        images = sorted(
            p
            for p in images_dir.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for ii, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            label_path = labels_dir / f"{img_path.stem}.txt"
            boxes = read_boxes_from_label(label_path, w, h)
            records: list[LabelRecord] = []
            if boxes:
                prompts = build_prompts(boxes)
                executor.set_image(img)
                for box, prompt in zip(boxes, prompts):
                    mask, _iou = executor.segment(
                        prompt.box_xyxy, prompt.positive_points, prompt.negative_points
                    )
                    contour = mask_to_contour(mask)
                    if contour is not None:
                        result.primed += 1
                        poly = contour
                    else:  # fallback: original OBB corners as the polygon
                        result.fell_back += 1
                        poly = box.polygon_px
                    records.append(
                        LabelRecord(
                            class_id=0,
                            confidence=1.0,
                            points=np.asarray(poly, dtype=np.float32).reshape(-1, 2),
                            level=GeometryLevel.POLYGON,
                        )
                    )
            write_label_file(
                staged_root / "labels" / f"{img_path.stem}.txt",
                records,
                frame_size=(h, w),
                level=GeometryLevel.POLYGON,
            )
            if progress:
                progress(
                    int(100 * (si + (ii + 1) / max(len(images), 1)) / len(todo)),
                    f"{src.name}: {ii + 1}/{len(images)}",
                )

        (staged_root / "classes.txt").write_text(
            (src_root / "classes.txt").read_text()
            if (src_root / "classes.txt").exists()
            else "object\n"
        )
        src.pending_escalation = PendingEscalation(
            staged_path=str(staged_root),
            target_level=GeometryLevel.POLYGON.label,
            sam2_variant=req.variant,
            created_at=datetime.now().isoformat(),
        )
        result.staged.append(src.name)
    return result


def accept_pending_escalation(source: OBBSource) -> None:
    """Promote *source*'s staged escalation result to its canonical labels.

    Overwrites the source's ``labels/`` + ``classes.txt`` from the staged
    copy, sets ``level``/``sam2_variant`` from the pending record, resets
    ``reviewed`` to ``False`` (same meaning as any other machine-derived,
    not-yet-human-confirmed result -- just attached to the existing source
    instead of a new sibling), removes the staging directory, and clears
    ``pending_escalation``.

    Raises ValueError if the source has no pending escalation.
    """
    pending = source.pending_escalation
    if pending is None:
        raise ValueError(f"Source '{source.name}' has no pending escalation.")

    staged_root = Path(pending.staged_path)
    source_root = Path(source.path)

    labels_dst = source_root / "labels"
    shutil.rmtree(labels_dst, ignore_errors=True)
    shutil.copytree(staged_root / "labels", labels_dst)
    classes_src = staged_root / "classes.txt"
    if classes_src.exists():
        shutil.copyfile(classes_src, source_root / "classes.txt")

    source.level = pending.target_level
    source.reviewed = False
    source.sam2_variant = pending.sam2_variant

    shutil.rmtree(staged_root, ignore_errors=True)
    source.pending_escalation = None


def reject_pending_escalation(source: OBBSource) -> None:
    """Discard *source*'s staged escalation result, leaving it untouched.

    Raises ValueError if the source has no pending escalation.
    """
    pending = source.pending_escalation
    if pending is None:
        raise ValueError(f"Source '{source.name}' has no pending escalation.")
    shutil.rmtree(Path(pending.staged_path), ignore_errors=True)
    source.pending_escalation = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_sam2_escalation.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/jobs/sam2_escalation.py tests/test_sam2_escalation.py && isort src/hydra_suite/detectkit/jobs/sam2_escalation.py tests/test_sam2_escalation.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/jobs/sam2_escalation.py tests/test_sam2_escalation.py
git commit -m "feat(detectkit): stage SAM2 escalation results for accept/reject instead of new sibling source"
```

---

## Task 6: `ReviewEscalationsDialog`

**Files:**
- Create: `src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py`
- Test: Create `tests/test_detectkit_review_escalations_dialog.py`

**Interfaces:**
- Consumes: `accept_pending_escalation`, `reject_pending_escalation` from `jobs/sam2_escalation.py`
  (Task 5); `OBBSource`/`PendingEscalation` from `gui/models.py` (Task 1); `BaseDialog` from
  `widgets/dialogs.py` (existing).
- Produces: `ReviewEscalationsDialog(pending_sources: list[OBBSource], parent=None)` — a `BaseDialog`
  subclass. `.accepted_names: list[str]` and `.rejected_names: list[str]` record what was actioned
  during the dialog's lifetime, for the caller (Task 7's `main_window.py`) to log/report. Every row
  starts checked; "Accept Checked" / "Reject Checked" apply immediately (not deferred to dialog
  close) and remove actioned rows from the list.

- [ ] **Step 1: Write the failing test**

Create `tests/test_detectkit_review_escalations_dialog.py`:

```python
"""Tests for DetectKit ReviewEscalationsDialog."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _make_pending_source(tmp_path, name="orig"):
    from hydra_suite.detectkit.gui.models import OBBSource, PendingEscalation

    source_root = tmp_path / name
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images").mkdir(parents=True)
    (source_root / "classes.txt").write_text("ant\n", encoding="utf-8")

    staged_root = tmp_path / f"{name}-staged"
    (staged_root / "labels").mkdir(parents=True)
    (staged_root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2 0.1 0.1\n", encoding="utf-8"
    )
    (staged_root / "classes.txt").write_text("ant\n", encoding="utf-8")

    return OBBSource(
        path=str(source_root),
        name=name,
        level="obb",
        pending_escalation=PendingEscalation(
            staged_path=str(staged_root),
            target_level="polygon",
            sam2_variant="sam2.1-hiera-base_plus",
            created_at="2026-08-27T00:00:00",
        ),
    )


def test_review_escalations_dialog_is_base_dialog(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.review_escalations_dialog import (
        ReviewEscalationsDialog,
    )
    from hydra_suite.widgets.dialogs import BaseDialog

    src = _make_pending_source(tmp_path)
    dlg = ReviewEscalationsDialog([src])
    assert isinstance(dlg, BaseDialog)
    assert dlg._list.count() == 1


def test_review_escalations_dialog_accept_checked_promotes_source(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.review_escalations_dialog import (
        ReviewEscalationsDialog,
    )

    src = _make_pending_source(tmp_path)
    dlg = ReviewEscalationsDialog([src])
    dlg._list.item(0).setCheckState(Qt.Checked)

    dlg._apply_checked(accept=True)

    assert dlg.accepted_names == ["orig"]
    assert dlg._list.count() == 0
    assert src.pending_escalation is None
    assert src.level == "polygon"
    assert src.reviewed is False
    assert (Path(src.path) / "labels" / "a.txt").exists()


def test_review_escalations_dialog_reject_checked_discards_staging(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.review_escalations_dialog import (
        ReviewEscalationsDialog,
    )

    src = _make_pending_source(tmp_path)
    staged_path = src.pending_escalation.staged_path
    dlg = ReviewEscalationsDialog([src])
    dlg._list.item(0).setCheckState(Qt.Checked)

    dlg._apply_checked(accept=False)

    assert dlg.rejected_names == ["orig"]
    assert dlg._list.count() == 0
    assert src.pending_escalation is None
    assert src.level == "obb"
    assert not Path(staged_path).exists()


def test_review_escalations_dialog_skips_unchecked_rows(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.review_escalations_dialog import (
        ReviewEscalationsDialog,
    )

    src = _make_pending_source(tmp_path)
    dlg = ReviewEscalationsDialog([src])
    dlg._list.item(0).setCheckState(Qt.Unchecked)

    dlg._apply_checked(accept=True)

    assert dlg.accepted_names == []
    assert dlg._list.count() == 1
    assert src.pending_escalation is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_review_escalations_dialog.py -v`
Expected: FAIL with `ModuleNotFoundError` (the dialog module doesn't exist yet).

- [ ] **Step 3: Create the dialog**

Create `src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py`:

```python
"""ReviewEscalationsDialog — accept/reject staged SAM2 escalation results."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog

from ...jobs.sam2_escalation import accept_pending_escalation, reject_pending_escalation


class ReviewEscalationsDialog(BaseDialog):
    """Review sources with a pending SAM2 escalation: accept or reject each.

    Each checked row is actioned immediately when its button is clicked (not
    deferred to dialog close) and removed from the list on success -- this is
    a working queue, not a form.
    """

    def __init__(self, pending_sources: list, parent=None) -> None:
        super().__init__(
            "Review Escalations",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self.accepted_names: list[str] = []
        self.rejected_names: list[str] = []

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(
            QLabel(
                "These sources have a staged SAM2 segmentation result awaiting "
                "review. Accept to replace the source's labels with the staged "
                "result; reject to discard it."
            )
        )

        self._list = QListWidget()
        for src in pending_sources:
            pending = src.pending_escalation
            if pending is None:
                continue
            item = QListWidgetItem(
                f"{src.name}  ->  {pending.target_level} "
                f"({pending.sam2_variant}, staged {pending.created_at})"
            )
            item.setData(Qt.UserRole, src)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._list.addItem(item)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._btn_accept = QPushButton("Accept Checked")
        self._btn_accept.clicked.connect(lambda: self._apply_checked(accept=True))
        self._btn_reject = QPushButton("Reject Checked")
        self._btn_reject.clicked.connect(lambda: self._apply_checked(accept=False))
        btn_row.addWidget(self._btn_accept)
        btn_row.addWidget(self._btn_reject)
        layout.addLayout(btn_row)

        self.add_content(container)

    def _checked_rows(self) -> list[int]:
        return [
            i
            for i in range(self._list.count())
            if self._list.item(i).checkState() == Qt.Checked
        ]

    def _apply_checked(self, *, accept: bool) -> None:
        rows = self._checked_rows()
        if not rows:
            QMessageBox.information(self, "Review Escalations", "No sources checked.")
            return
        for row in sorted(rows, reverse=True):
            item = self._list.item(row)
            src = item.data(Qt.UserRole)
            try:
                if accept:
                    accept_pending_escalation(src)
                    self.accepted_names.append(src.name)
                else:
                    reject_pending_escalation(src)
                    self.rejected_names.append(src.name)
            except Exception as exc:
                QMessageBox.warning(self, "Review Escalations", str(exc))
                continue
            self._list.takeItem(row)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_review_escalations_dialog.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py tests/test_detectkit_review_escalations_dialog.py && isort src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py tests/test_detectkit_review_escalations_dialog.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/dialogs/review_escalations_dialog.py tests/test_detectkit_review_escalations_dialog.py
git commit -m "feat(detectkit): add ReviewEscalationsDialog for staged SAM2 accept/reject"
```

---

## Task 7: Wire the review dialog into `main_window.py`

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/main_window.py`
- Test: `tests/test_detectkit_sam2_escalation_wiring.py`

**Interfaces:**
- Consumes: `EscalationResult.staged` (Task 5), `ReviewEscalationsDialog` (Task 6).
- Produces: `_on_escalate_to_segment_sam2` opens `ReviewEscalationsDialog` for any newly staged
  sources as soon as the escalation worker finishes, then saves + refreshes.

- [ ] **Step 1: Write the failing wiring test**

Add to `tests/test_detectkit_sam2_escalation_wiring.py`:

```python
def test_main_window_imports_review_escalations_dialog():
    """The escalation handler must reference ReviewEscalationsDialog, not the
    old 'created a new sibling source' messaging."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._on_escalate_to_segment_sam2)
    assert "ReviewEscalationsDialog" in source
    assert "result.staged" in source or "getattr(result, \"staged\"" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_sam2_escalation_wiring.py::test_main_window_imports_review_escalations_dialog -v`
Expected: FAIL — `ReviewEscalationsDialog` not yet referenced in `_on_escalate_to_segment_sam2`.

- [ ] **Step 3: Update `_on_escalate_to_segment_sam2` in `main_window.py`**

Replace the `would_overwrite` block (currently the section starting `existing_names = {s.name for s
in self._project.sources}` through `overwrite = True` before `request = EscalationRequest(...)`)
with:

```python
        existing_by_name = {s.name: s for s in self._project.sources}
        would_conflict = [
            n
            for n in source_names
            if existing_by_name.get(n) is not None
            and existing_by_name[n].pending_escalation is not None
        ]
        overwrite = False
        if would_conflict:
            reply = QMessageBox.question(
                self,
                "Escalate to segment (SAM2)",
                (
                    "The following source(s) already have a pending escalation "
                    "awaiting review, which will be replaced:\n\n"
                    f"{', '.join(would_conflict)}\n\n"
                    "Continue and replace the staged result?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                source_names = [n for n in source_names if n not in would_conflict]
                if not source_names:
                    return
            else:
                overwrite = True
```

Then replace the `_handle_result` function body (currently from `derived = list(getattr(result,
"derived", []) or [])` through the final `QMessageBox.information(...)` call) with:

```python
        def _handle_result(result: object) -> None:
            staged = list(getattr(result, "staged", []) or [])
            primed = int(getattr(result, "primed", 0))
            fell_back = int(getattr(result, "fell_back", 0))
            skipped = list(getattr(result, "skipped", []) or [])
            # The worker set pending_escalation on existing sources; persist + refresh.
            self._save_current_project()
            self._dataset_panel.refresh_sources(self._project)
            self._tools_panel.refresh_overview()

            if staged:
                pending_sources = [
                    s
                    for s in self._project.sources
                    if s.name in staged and s.pending_escalation is not None
                ]
                from .dialogs.review_escalations_dialog import ReviewEscalationsDialog

                review_dlg = ReviewEscalationsDialog(pending_sources, parent=self)
                review_dlg.exec()
                self._save_current_project()
                self._dataset_panel.refresh_sources(self._project)
                self._tools_panel.refresh_overview()

            skipped_note = (
                (
                    "\n\nSkipped (already has a pending escalation, not "
                    "overwritten): "
                    + ", ".join(f"{name} ({reason})" for name, reason in skipped)
                )
                if skipped
                else ""
            )
            if staged:
                if skipped:
                    QMessageBox.information(
                        self,
                        "Escalate to segment (SAM2)",
                        (
                            f"{primed} instance(s) primed, {fell_back} fell back "
                            f"to the original box.{skipped_note}"
                        ),
                    )
            else:
                QMessageBox.information(
                    self,
                    "Escalate to segment (SAM2)",
                    f"No sources were staged for escalation.{skipped_note}",
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_sam2_escalation_wiring.py -v`
Expected: all PASS (the new test plus the 3 pre-existing ones).

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_sam2_escalation_wiring.py && isort src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_sam2_escalation_wiring.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_sam2_escalation_wiring.py
git commit -m "feat(detectkit): open ReviewEscalationsDialog after staged SAM2 escalation"
```

---

## Task 8: Full DetectKit test sweep + lint

**Files:** none new — this is a verification-only task across everything Tasks 1–7 touched, plus a
check that nothing else in the codebase still references the removed multi-root functions.

- [ ] **Step 1: Confirm no remaining references to removed symbols**

Run: `grep -rn "inspect_al_round\|materialize_al_round\|ALRoundRoot\|MaterializedALRoundRoot" src/ tests/ docs/superpowers/plans docs/superpowers/specs`

Expected: no matches in `src/` or `tests/` (only historical mentions inside the design spec's
"Relationship to prior work" section are acceptable, since that section documents what was reverted
— leave the spec file untouched).

- [ ] **Step 2: Run the full DetectKit test slice**

Run:
```bash
conda activate hydra-mps
python -m pytest tests/test_detectkit_source_import.py tests/test_detectkit_source_manager_dialog.py \
  tests/test_detectkit_al_worker.py tests/test_sam2_escalation.py \
  tests/test_detectkit_review_escalations_dialog.py tests/test_detectkit_sam2_escalation_wiring.py \
  tests/test_obbsource_reviewed.py tests/test_training_gating_reviewed.py \
  tests/test_escalate_sam2_dialog.py -v
```
Expected: all PASS.

- [ ] **Step 3: Run `make format-check` and `make lint`**

Run: `conda activate hydra-mps && make format-check && make lint`
Expected: no formatting diffs, no new lint findings introduced by this plan's files (pre-existing
findings elsewhere in the repo are out of scope).

- [ ] **Step 4: Manual smoke test (GUI)**

Per CLAUDE.md: for GUI changes, start the app and exercise the golden path before calling this done.

```bash
conda activate hydra-mps
detectkit
```

In the running app: open (or create) a DetectKit project, use Manage Sources → Add Source on a
folder containing an AL-round `manifest.json` (or generate one via an AL round if a model is
available), and confirm exactly one source appears, named after the round folder. Then, if a SAM2
checkpoint is available, escalate that source and confirm the Review Escalations dialog appears
with Accept/Reject options, and that accepting swaps the source's level to `polygon` in place
(still one entry in the source list, not two).

If no SAM2 checkpoint is available in this environment, skip the escalation half of the smoke test
and say so explicitly rather than claiming it was verified.

- [ ] **Step 5: Commit (only if Steps 1-4 required any fixes)**

If any fixes were needed:
```bash
git add -A
git commit -m "fix(detectkit): address lint/test findings from source-unification sweep"
```

If no fixes were needed, skip this step — there is nothing to commit.

---

## After this plan ships

Per CLAUDE.md's "Docs lifecycle" convention: once this plan's branch/work is merged to `main`,
`git mv` this plan file and `docs/superpowers/specs/2026-08-27-detectkit-source-unification-design.md`
into their respective `done/` subfolders in the same commit.

Then, per the spec's own sequencing ("As soon as A is implemented B+C will be implemented"), return
to `superpowers:brainstorming` to turn the spec's Part B (multi-level canvas visualization) and Part
C (clear-labels actions) design notes into their own committed specs — starting with Part B's
required investigation task (root-causing the "colored dots, no box outlines" rendering issue from
the user's screenshots against this plan's actual output, not a stale flat-sibling source from
before this fix).
