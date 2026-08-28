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
- Per CLAUDE.md's Isolation rule: all implementation happens in a git worktree branched from
  local HEAD, never committed directly onto `main`. Before Task 1, run:
  `git worktree add .worktrees/detectkit-source-unification -b feat/detectkit-source-unification HEAD`.
  Every `git add`/`git commit` step in this plan runs inside that worktree
  (`.worktrees/detectkit-source-unification/`), and every `pytest`/`black`/`isort`/`make` command
  runs with that worktree as the working directory.

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
  exist** — Task 3 (`source_manager.py`) must not import them. New:
  `resolve_al_round_authoritative_level(source_root) -> str | None` — returns the AL round's
  manifest-declared authoritative-root `level`, or `None` if *source_root* is not an AL round.
  Task 3 uses this so a manually-added external round trusts the manifest's declared level instead
  of re-scanning label geometry (label files in an AL export are always 9-field quads regardless of
  level — `AABB` and `OBB` are visually indistinguishable by re-scanning; only the manifest knows
  which is which).

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

- [ ] **Step 2: Add `resolve_al_round_authoritative_level`**

Add this function to `source_import.py`, directly below `_select_al_round_authoritative_root`
(which it reuses the manifest-loading half of):

```python
def resolve_al_round_authoritative_level(source_root: str | Path) -> str | None:
    """Return an AL round's manifest-declared authoritative-root level.

    An AL-export root's labels are always stored as 9-field quads regardless
    of level (see `_detect_source_level`'s `intended_level=OBB` re-scan,
    which cannot distinguish a genuine OBB from an axis-aligned-quad-encoded
    AABB by re-scanning). Only the manifest recorded which is which at
    export time -- callers that need an AL round's true level (rather than a
    re-scanned guess) must go through this function instead of
    `_detect_source_level`.

    Returns ``None`` if *source_root* is not an AL round container (no
    ``manifest.json`` with a ``roots`` list) or has no authoritative entry.
    """
    root = Path(source_root).expanduser().resolve()
    al_roots = _load_al_round_roots(root)
    if al_roots is None:
        return None
    for entry in al_roots:
        if entry.get("authoritative"):
            level = entry.get("level")
            return str(level) if level else None
    return None
```

- [ ] **Step 3: Add a failing test for `resolve_al_round_authoritative_level`, then verify it passes**

Add to `tests/test_detectkit_source_import.py` (after `_write_al_round`):

```python
def test_resolve_al_round_authoritative_level_reads_manifest(tmp_path: Path):
    round_dir = tmp_path / "active_learning" / "20260827_172624"
    _write_al_round(
        round_dir,
        levels=(("aabb", True), ("obb", False)),
    )

    assert resolve_al_round_authoritative_level(round_dir) == "aabb"


def test_resolve_al_round_authoritative_level_none_for_non_al_round(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "classes.txt").write_text("ant\n", encoding="utf-8")

    assert resolve_al_round_authoritative_level(tmp_path) is None
```

Add `resolve_al_round_authoritative_level` to the import block at the top of the test file (see
Step 4 below — do both import edits together). Run:
`conda activate hydra-mps && python -m pytest tests/test_detectkit_source_import.py -k resolve_al_round_authoritative_level -v`
Expected: FAIL first (function doesn't exist until Step 2 above is applied — if doing these steps
in order, Step 2 already landed, so this should PASS immediately; if TDD-ing strictly, comment out
Step 2's function body temporarily to see the test fail, then restore it).

- [ ] **Step 4: Update `tests/test_detectkit_source_import.py` to drop removed-function tests**

Remove `inspect_al_round` and `materialize_al_round` from the import block at the top of the file
(lines 11–17), and add `resolve_al_round_authoritative_level` (needed by Step 3 above):

```python
from hydra_suite.detectkit.gui.source_import import (
    IMPORT_MODE_LINKED,
    inspect_detectkit_source,
    materialize_detectkit_source,
    resolve_al_round_authoritative_level,
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

- [ ] **Step 5: Run the test file**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_source_import.py -v`
Expected: all PASS, no `ImportError`.

- [ ] **Step 6: Run flake8/black/isort on the two changed files**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/source_import.py tests/test_detectkit_source_import.py && isort src/hydra_suite/detectkit/gui/source_import.py tests/test_detectkit_source_import.py`
Expected: no errors; files reformatted if needed.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/detectkit/gui/source_import.py tests/test_detectkit_source_import.py
git commit -m "refactor(detectkit): revert AL-round import to single authoritative-root redirect"
```

(This commit includes `resolve_al_round_authoritative_level` from Steps 2–3, since it lives in the
same file and is part of the same revert-and-clean-up unit of work.)

---

## Task 3: Revert `source_manager.py::_add_source` to single-source registration

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/dialogs/source_manager.py`
- Test: `tests/test_detectkit_source_manager_dialog.py`

**Interfaces:**
- Consumes: `inspect_detectkit_source`, `materialize_detectkit_source`,
  `resolve_al_round_authoritative_level` from `source_import.py` (Task 2's surviving/new functions —
  `inspect_al_round`/`materialize_al_round` imports must be removed).
- Produces: `SourceManagerDialog._add_source()` registers exactly one `OBBSource` per pick, named
  `Path(selected_path).name` — for an AL-round pick this is the round folder's own name (e.g.
  `20260827_172624`), never `obb` or a level-suffixed name, with `level` taken from the manifest's
  declared authoritative level when the pick is an AL round (not re-scanned — see Task 2's docstring
  for why re-scanning can't tell OBB from AABB for this format). `_add_al_round_sources` **no longer
  exists**. `_remove_selected` now also deletes any pending-escalation staging directory belonging
  to the removed source, so escalating a source and then removing it doesn't leak an orphaned
  `artifacts/pending_escalations/` directory.

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

Also add a second test proving the manifest's declared level is trusted over a re-scan, for a round
whose authoritative level is `aabb` (not `obb`) — this is the case where re-scanning the 9-field quad
labels would silently over-claim `obb`:

```python
def test_source_manager_add_source_trusts_manifest_level_for_aabb_round(
    qapp, tmp_path, monkeypatch
):
    """An AL round whose AUTHORITATIVE level is aabb must be registered as
    level='aabb', not 'obb' -- re-scanning the label files can't tell them
    apart (both are 9-field quads), so the manifest's declared level must be
    trusted, not the geometry-scan result."""
    import json

    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.dialogs.source_validation import (
        SOURCE_ADD_MODE_PORTABLE,
        DetectKitSourceAdditionChoice,
    )

    round_dir = tmp_path / "active_learning" / "20260827_180000"
    level_dir = round_dir / "aabb"
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
                        "level": "aabb",
                        "authoritative": True,
                        "reviewed": True,
                        "path": str(round_dir / "aabb"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(round_dir),
    )
    # DetectKitSourceAdditionChoice defaults to level="obb" -- this is the
    # dialog's own re-scanned guess, which is exactly the wrong value this
    # test must NOT see land on the registered source.
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
    assert proj.sources[0].level == "aabb"


def test_remove_selected_deletes_pending_escalation_staging_dir(qapp, tmp_path):
    """Removing a source with an unreviewed pending escalation must not leak
    its staging directory under artifacts/pending_escalations/."""
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.models import OBBSource, PendingEscalation

    staged_dir = tmp_path / "artifacts" / "pending_escalations" / "orig-variant-abc123"
    staged_dir.mkdir(parents=True)
    (staged_dir / "labels").mkdir()

    proj = _make_proj(tmp_path)
    proj.sources = [
        OBBSource(
            path=str(tmp_path / "orig"),
            name="orig",
            pending_escalation=PendingEscalation(
                staged_path=str(staged_dir),
                target_level="polygon",
                sam2_variant="sam2.1-hiera-base_plus",
                created_at="2026-08-27T00:00:00",
            ),
        )
    ]
    dlg = SourceManagerDialog(proj)
    dlg._source_list.setCurrentRow(0)
    dlg._remove_selected()

    assert proj.sources == []
    assert not staged_dir.exists()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_source_manager_dialog.py -k "collapses_al_round_to_one_source or trusts_manifest_level or deletes_pending_escalation_staging_dir" -v`
Expected: FAIL — either an assertion mismatch (`len(proj.sources) == 2` currently, or `level == "obb"`
for the aabb-authoritative case) or an `AttributeError` (`_remove_selected` doesn't yet clean up
staging dirs), since `_add_source`/`_remove_selected` still have the old behavior.

- [ ] **Step 3: Revert `_add_source` and remove `_add_al_round_sources`**

In `src/hydra_suite/detectkit/gui/dialogs/source_manager.py`, add `import shutil` to the top-level
imports (needed by the `_remove_selected` change in Step 3 below):

```python
import logging
import shutil
from pathlib import Path
```

Change the `..source_import` import block:

```python
from ..source_import import (
    IMPORT_MODE_LINKED,
    IMPORT_MODE_PORTABLE,
    compute_positional_class_remap,
    inspect_detectkit_source,
    materialize_detectkit_source,
    remap_materialized_source_classes,
    resolve_al_round_authoritative_level,
)
```

(drop `inspect_al_round`, `materialize_al_round`; add `resolve_al_round_authoritative_level`).

Replace the 8-line comment block above the old multi-root dispatch (starting `# An active-learning
export round (manifest.json + one sibling dataset` at line 117) through the end of the `_add_source`
method (the `self._project.sources.append(...)` / `self._refresh_list()` at its end, currently line
268) with:

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

        # An AL round's manifest declares which level is authoritative --
        # trust that over the validation dialog's re-scanned `selection.level`
        # (label files in an AL export are 9-field quads for every level, so
        # a re-scan cannot tell OBB apart from an axis-aligned-quad AABB;
        # see resolve_al_round_authoritative_level's docstring).
        manifest_level = resolve_al_round_authoritative_level(selected_path)
        level = (
            manifest_level
            if manifest_level is not None
            else (
                selection.level
                if getattr(selection, "level", None)
                else materialized.level
            )
        )

        self._project.sources.append(
            OBBSource(
                path=canonical_path,
                name=Path(selected_path).name,
                original_path=original_path,
                source_kind=materialized.source_kind,
                imported=materialized.imported,
                level=level,
            )
        )
        self._refresh_list()
```

Note two behavior changes from the pre-session original: `name=Path(selected_path).name` (was
`materialized.display_name`, which is `root.name` where `root` is the *resolved* dataset root —
wrong for an AL-round pick, since that resolves to the level subfolder, e.g. `obb`); and the
manifest-level override above (the pre-session original didn't have AL-round handling at all, so
this is new behavior, not a revert).

Then delete the entire `_add_al_round_sources` method (everything from `def _add_al_round_sources(`
through its closing `self._refresh_list()`, right before `_remove_selected`).

Finally, update `_remove_selected` to clean up a removed source's pending-escalation staging
directory (this closes a leak that Task 5 introduces — without it, escalating a source then removing
it via "Remove Selected" would strand its `artifacts/pending_escalations/` directory forever, since
nothing else ever visits a removed `OBBSource`):

```python
    def _remove_selected(self) -> None:
        row = self._source_list.currentRow()
        if row < 0 or row >= len(self._project.sources):
            return
        removed = self._project.sources.pop(row)
        pending = removed.pending_escalation
        if pending is not None and pending.staged_path:
            shutil.rmtree(Path(pending.staged_path), ignore_errors=True)
        self._refresh_list()
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_source_manager_dialog.py -v`
Expected: all PASS, including the three new tests and every pre-existing test in the file
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
    # ONE OBBSource for the round's authoritative root only -- the root
    # export_al_dataset marks derived_from=None (the highest level actually
    # requested, which equals native_level whenever native_level itself was
    # among the requested levels -- see data/al/export.py's _write_root).
    # The exporter still writes every requested level's sibling folder to
    # disk (data/al/export.py is unchanged) -- those siblings are simply not
    # registered as separate project sources; training derives lower levels
    # from the registered source on demand, same as any other source.
    authoritative_root = next(
        (root_meta for root_meta in manifest["roots"] if root_meta["derived_from"] is None),
        None,
    )
    if authoritative_root is None:
        raise RuntimeError(
            "AL round manifest has no authoritative root (derived_from=None "
            "entry) -- this indicates a corrupt or incompatible manifest."
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

from hydra_suite.detectkit.gui.models import OBBSource
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


def _make_nested_source(tmp_path):
    """A source whose images/labels use a nested split layout (images/train/...),
    as dataset_inspector.py's directory-layout scan supports and as
    source_import.py's materializer can produce."""
    root = tmp_path / "sources" / "nested"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    cv2.imwrite(
        str(root / "images" / "train" / "a.jpg"), np.zeros((100, 100, 3), np.uint8)
    )
    (root / "labels" / "train" / "a.txt").write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"
    )
    (root / "classes.txt").write_text("ant\n")
    return OBBSource(path=str(root), name="nested", level="obb")


def test_escalation_stages_nested_image_layout_correctly(tmp_path):
    """Regression: staging must mirror the source's directory structure, not
    flatten to top-level images/*.* -- a split layout (images/train/...) has
    zero images at the top level, which used to silently stage nothing and
    made accept() delete every label with no staged replacement."""
    src = _make_nested_source(tmp_path)
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["nested"], variant="sam2.1-hiera-base_plus"
    )

    result = run_escalation(req, _FakeExec())

    assert result.staged == ["nested"]
    staged_label = (
        Path(src.pending_escalation.staged_path) / "labels" / "train" / "a.txt"
    )
    assert staged_label.exists()
    assert len(staged_label.read_text().splitlines()) == 1


def test_accept_refuses_when_staged_labels_missing_files(tmp_path):
    """If staging skipped a label (e.g. an unreadable image during escalation),
    accept must refuse rather than deleting that image's original label with
    nothing to replace it."""
    src = _make_source(tmp_path)
    # Add a second image/label pair.
    cv2.imwrite(str(Path(src.path) / "images" / "b.jpg"), np.zeros((100, 100, 3), np.uint8))
    (Path(src.path) / "labels" / "b.txt").write_text(
        "0 0.2 0.2 0.5 0.2 0.5 0.5 0.2 0.5\n"
    )
    project = types.SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    req = EscalationRequest(
        project=project, source_names=["orig"], variant="sam2.1-hiera-base_plus"
    )
    run_escalation(req, _FakeExec())

    # Simulate an image that failed to stage (e.g. cv2.imread returned None
    # during escalation): remove its staged label after the fact.
    staged_b = Path(src.pending_escalation.staged_path) / "labels" / "b.txt"
    staged_b.unlink()

    original_a = (Path(src.path) / "labels" / "a.txt").read_text()
    original_b = (Path(src.path) / "labels" / "b.txt").read_text()

    with pytest.raises(RuntimeError):
        accept_pending_escalation(src)

    # Nothing was touched -- refusal happens before any deletion.
    assert src.pending_escalation is not None
    assert (Path(src.path) / "labels" / "a.txt").read_text() == original_a
    assert (Path(src.path) / "labels" / "b.txt").read_text() == original_b


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

        # A source with an existing pending escalation under a DIFFERENT
        # staging path (e.g. this is a re-escalation with a different SAM2
        # variant, which hashes to a different directory) must have its old
        # staging dir cleaned up -- otherwise it's orphaned forever, since
        # nothing else ever revisits a replaced pending_escalation.
        old_pending = src.pending_escalation
        if old_pending is not None and old_pending.staged_path != str(staged_root):
            shutil.rmtree(Path(old_pending.staged_path), ignore_errors=True)

        shutil.rmtree(staged_root, ignore_errors=True)
        (staged_root / "labels").mkdir(parents=True, exist_ok=True)

        # Recursive + path-mirroring: a source's images/labels can be nested
        # (e.g. images/train/..., images/val/...) -- source_import.py's
        # materializer can produce this layout. A flat top-level glob would
        # silently stage ZERO labels for such a source, and accept() would
        # then delete every real label with nothing to replace it.
        images = sorted(
            p
            for p in images_dir.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for ii, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            relative_label = img_path.relative_to(images_dir).with_suffix(".txt")
            label_path = labels_dir / relative_label
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
            staged_label_path = staged_root / "labels" / relative_label
            staged_label_path.parent.mkdir(parents=True, exist_ok=True)
            write_label_file(
                staged_label_path,
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

    Validates BEFORE deleting anything: refuses (raising ``RuntimeError``,
    source left untouched) if the staging directory is missing on disk, or
    if it is missing a label file for an image the source currently has a
    label for (e.g. an image that failed to decode during escalation and was
    silently skipped by ``run_escalation``) -- accepting such a staged result
    would otherwise delete real labels with nothing staged to replace them.

    Raises ValueError if the source has no pending escalation.
    """
    pending = source.pending_escalation
    if pending is None:
        raise ValueError(f"Source '{source.name}' has no pending escalation.")

    staged_root = Path(pending.staged_path)
    staged_labels = staged_root / "labels"
    if not staged_labels.is_dir():
        raise RuntimeError(
            f"Staged escalation for '{source.name}' is missing on disk "
            f"({staged_labels}); nothing was changed. Reject this escalation "
            "and re-run it."
        )

    source_root = Path(source.path)
    source_labels = source_root / "labels"
    existing_rel = (
        {p.relative_to(source_labels) for p in source_labels.rglob("*.txt")}
        if source_labels.is_dir()
        else set()
    )
    staged_rel = {p.relative_to(staged_labels) for p in staged_labels.rglob("*.txt")}
    missing = sorted(str(p) for p in existing_rel - staged_rel)
    if missing:
        raise RuntimeError(
            f"Staged escalation for '{source.name}' is missing "
            f"{len(missing)} label file(s) that exist in the source (likely "
            "an unreadable image during escalation) -- refusing to accept, "
            f"as this would delete those labels: {missing[:5]}"
        )

    shutil.rmtree(source_labels, ignore_errors=True)
    shutil.copytree(staged_labels, source_labels)
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
Expected: all 10 tests PASS.

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


def test_review_escalations_dialog_skips_unchecked_rows(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.dialogs.review_escalations_dialog import (
        ReviewEscalationsDialog,
    )

    # With no rows checked, _apply_checked shows a blocking QMessageBox.information
    # to tell the user nothing was selected -- under QT_QPA_PLATFORM=offscreen that
    # still opens a real (invisible) event loop and hangs the test process waiting
    # for a click that will never come, so it must be monkeypatched out here (this
    # repo has hit exactly this class of hang before -- see CLAUDE.md's "main
    # whole-suite blockers" note on modal-dialog hangs).
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.review_escalations_dialog.QMessageBox.information",
        lambda *args, **kwargs: None,
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
        intro = QLabel(
            "These sources have a staged SAM2 segmentation result awaiting "
            "review. Accept to replace the source's labels with the staged "
            "result; reject to discard it.\n\n"
            "Accepted sources are marked unreviewed and are excluded from "
            "training until you use \"Mark reviewed…\" for them."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

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

## Task 7: Wire the review dialog into `main_window.py` (with a standalone entry point)

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/tools_panel.py`
- Modify: `src/hydra_suite/detectkit/gui/main_window.py`
- Test: `tests/test_detectkit_sam2_escalation_wiring.py`

**Interfaces:**
- Consumes: `EscalationResult.staged`/`.skipped`/`.primed`/`.fell_back` (Task 5),
  `ReviewEscalationsDialog` (Task 6).
- Produces: `ToolsPanel.review_escalations_requested: Signal()` + `ToolsPanel._btn_review_escalations`
  (new button, next to "Mark reviewed…"). `MainWindow._on_review_escalations()` — opens
  `ReviewEscalationsDialog` for **every** source in the current project with
  `pending_escalation is not None` (not just ones from the run that just finished), so a pending
  escalation the user closed without acting on is always reachable again later, per the spec's
  "survives project close/reopen ... so the user can come back to it." `_on_escalate_to_segment_sam2`
  no longer opens any dialog from `_handle_result` (which can run on the worker thread, and would
  otherwise stack a second modal dialog underneath the still-open application-modal progress
  dialog) — it stashes the result and defers all UI (message boxes, `_on_review_escalations`) to
  `_finish`, which runs after `progress.close()`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_detectkit_sam2_escalation_wiring.py`:

```python
def test_tools_panel_exposes_review_escalations_button(qapp):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = ToolsPanel()
    assert hasattr(panel, "_btn_review_escalations")
    assert hasattr(panel, "review_escalations_requested")


def test_main_window_has_review_escalations_handler():
    from hydra_suite.detectkit.gui.main_window import MainWindow

    assert callable(getattr(MainWindow, "_on_review_escalations", None))


def test_main_window_escalation_finish_defers_dialog_past_progress_close():
    """The review dialog (and any post-run message box) must NOT be opened
    from _handle_result, which BaseWorker's result_ready signal can deliver
    on the worker thread, and which fires while the application-modal
    progress dialog is still open -- both would make the dialog undismissable
    or crash Qt. It must be deferred to _finish, which runs after
    progress.close()."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._on_escalate_to_segment_sam2)
    assert "_on_review_escalations" in source
    assert "getattr(result, \"staged\"" in source

    handle_result_start = source.index("def _handle_result")
    finish_start = source.index("def _finish")
    handle_result_body = source[handle_result_start:finish_start]
    finish_body = source[finish_start:]

    assert "ReviewEscalationsDialog" not in handle_result_body
    assert "QMessageBox" not in handle_result_body
    assert "ReviewEscalationsDialog" in finish_body or "_on_review_escalations" in finish_body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_sam2_escalation_wiring.py -k "review_escalations or finish_defers" -v`
Expected: FAIL — `_btn_review_escalations`/`review_escalations_requested` don't exist yet,
`_on_review_escalations` doesn't exist yet, and `_handle_result` still builds `QMessageBox`/dialog
content directly.

- [ ] **Step 3: Add the "Review escalations…" button to `ToolsPanel`**

In `src/hydra_suite/detectkit/gui/panels/tools_panel.py`, add a new signal next to
`mark_reviewed_requested`:

```python
    escalate_sam2_requested = Signal()
    mark_reviewed_requested = Signal()
    review_escalations_requested = Signal()
```

Then add the button in `_build_escalation_group`, right after the existing
`self._btn_mark_reviewed` block:

```python
        self._btn_mark_reviewed = QPushButton("Mark reviewed…")
        self._btn_mark_reviewed.clicked.connect(self.mark_reviewed_requested)
        v.addWidget(self._btn_mark_reviewed)

        self._btn_review_escalations = QPushButton("Review escalations…")
        self._btn_review_escalations.clicked.connect(self.review_escalations_requested)
        v.addWidget(self._btn_review_escalations)
```

- [ ] **Step 4: Wire the button and add `_on_review_escalations` in `main_window.py`**

Add the connection next to the existing `mark_reviewed_requested` connection (currently
`self._tools_panel.mark_reviewed_requested.connect(self._on_mark_reviewed)`):

```python
        self._tools_panel.mark_reviewed_requested.connect(self._on_mark_reviewed)
        self._tools_panel.review_escalations_requested.connect(
            self._on_review_escalations
        )
```

Add `self._last_escalation_result: object | None = None` next to the existing
`self._escalation_worker = None` / `self._escalation_progress_dialog = None` initialization (used by
Step 5's `_finish` below to carry the result across the thread/modal boundary).

Add a new method near `_on_mark_reviewed`:

```python
    def _on_review_escalations(self) -> None:
        """Open the review dialog for every source with a pending escalation,
        regardless of when it was staged. Reachable independent of an
        escalation run having just finished, so closing the dialog without
        acting never strands a pending escalation."""
        if self._project is None:
            QMessageBox.information(self, "Review Escalations", "Open a project first.")
            return

        pending_sources = [
            s for s in self._project.sources if s.pending_escalation is not None
        ]
        if not pending_sources:
            QMessageBox.information(
                self,
                "Review Escalations",
                "There are no pending escalations to review.",
            )
            return

        from .dialogs.review_escalations_dialog import ReviewEscalationsDialog

        review_dlg = ReviewEscalationsDialog(pending_sources, parent=self)
        review_dlg.exec()
        self._save_current_project()
        self._dataset_panel.refresh_sources(self._project)
        self._tools_panel.refresh_overview()
```

- [ ] **Step 5: Update `_on_escalate_to_segment_sam2`: conflict check, and defer all UI to `_finish`**

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
            # The worker set pending_escalation on existing sources on a
            # background thread; persist + refresh immediately. Everything
            # UI-facing (message boxes, the review dialog) is deferred to
            # _finish -- result_ready can be delivered on the worker thread,
            # and the application-modal progress dialog is still open here.
            self._save_current_project()
            self._dataset_panel.refresh_sources(self._project)
            self._tools_panel.refresh_overview()
            self._last_escalation_result = result
```

And replace the existing `_finish` function body (currently just `progress.close()` +
`self._escalation_worker = None` + `self._escalation_progress_dialog = None`) with:

```python
        def _finish() -> None:
            progress.close()
            self._escalation_worker = None
            self._escalation_progress_dialog = None

            result = self._last_escalation_result
            self._last_escalation_result = None
            if result is None:
                return

            staged = list(getattr(result, "staged", []) or [])
            primed = int(getattr(result, "primed", 0))
            fell_back = int(getattr(result, "fell_back", 0))
            skipped = list(getattr(result, "skipped", []) or [])
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
                QMessageBox.information(
                    self,
                    "Escalate to segment (SAM2)",
                    (
                        f"Staged {len(staged)} source(s) for review: "
                        f"{', '.join(staged)}.\n\n"
                        f"{primed} instance(s) primed, {fell_back} fell back "
                        f"to the original box.{skipped_note}"
                    ),
                )
                self._on_review_escalations()
            else:
                QMessageBox.information(
                    self,
                    "Escalate to segment (SAM2)",
                    f"No sources were staged for escalation.{skipped_note}",
                )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_sam2_escalation_wiring.py -v`
Expected: all PASS (the 3 new tests plus the 3 pre-existing ones).

- [ ] **Step 7: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/panels/tools_panel.py src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_sam2_escalation_wiring.py && isort src/hydra_suite/detectkit/gui/panels/tools_panel.py src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_sam2_escalation_wiring.py`

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/tools_panel.py src/hydra_suite/detectkit/gui/main_window.py tests/test_detectkit_sam2_escalation_wiring.py
git commit -m "feat(detectkit): add standalone Review Escalations entry point, defer dialog past progress close"
```

---

## Task 8: Stale docs, full DetectKit test sweep + lint

**Files:**
- Modify: `docs/developer-guide/runtime-integration.md` (its "SAM2 Escalation" section documents the
  removed `<name>_seg`-sibling behavior in detail and must be corrected)
- Verification-only otherwise, across everything Tasks 1–7 touched, plus a check that nothing else
  in the codebase still references the removed multi-root functions.

- [ ] **Step 1: Confirm no remaining references to removed symbols**

Run: `grep -rln "inspect_al_round\|materialize_al_round\|ALRoundRoot\|MaterializedALRoundRoot" src/ tests/`

Expected: no output (empty). (Scope this grep to `src/` and `tests/` only — the design spec's
"Relationship to prior work" section legitimately mentions these names to document what was
reverted, and this plan file itself names them throughout; neither should be grepped here.)

- [ ] **Step 2: Update the stale SAM2 escalation docs**

`docs/developer-guide/runtime-integration.md`'s "SAM2 Escalation" section currently documents the
removed behavior verbatim (*"Results are written to a new derived source named `<source>_seg`...
the original source's images/labels are never touched... Re-running escalation over a source whose
`<name>_seg` already exists is guarded..."*). Rewrite that section to describe the staged
accept/reject flow instead: `run_escalation` writes to a per-source staging directory under
`artifacts/pending_escalations/` and sets `OBBSource.pending_escalation`, without touching the
source's canonical labels or registering a new source; a re-run over a source with an existing
pending escalation is skipped by default (`overwrite=True` replaces it); `accept_pending_escalation`
promotes the staged result into the source's canonical `labels/`/`classes.txt` in place (setting
`level`, `sam2_variant`, resetting `reviewed=False`), `reject_pending_escalation` discards it; both
are driven from the "Review escalations…" button (`ToolsPanel.review_escalations_requested` →
`MainWindow._on_review_escalations`), which lists every source with a pending escalation project-wide,
not just ones from the run that just finished.

- [ ] **Step 3: Run the full DetectKit test slice**

Run:
```bash
conda activate hydra-mps
python -m pytest tests/test_detectkit_source_import.py tests/test_detectkit_source_manager_dialog.py \
  tests/test_detectkit_al_worker.py tests/test_sam2_escalation.py \
  tests/test_detectkit_review_escalations_dialog.py tests/test_detectkit_sam2_escalation_wiring.py \
  tests/test_detectkit_tools_panel.py tests/test_obbsource_reviewed.py \
  tests/test_training_gating_reviewed.py tests/test_escalate_sam2_dialog.py -v
```
Expected: all PASS.

- [ ] **Step 4: Run `make format-check`, `make lint`, and `make docs-check`**

Run: `conda activate hydra-mps && make format-check && make lint && make docs-check`
Expected: no formatting diffs, no new lint findings introduced by this plan's files (pre-existing
findings elsewhere in the repo are out of scope), and `make docs-check` passes against the updated
`runtime-integration.md`.

- [ ] **Step 5: Manual smoke test (GUI)**

Per CLAUDE.md: for GUI changes, start the app and exercise the golden path before calling this done.

```bash
conda activate hydra-mps
detectkit
```

In the running app: open (or create) a DetectKit project, use Manage Sources → Add Source on a
folder containing an AL-round `manifest.json` (or generate one via an AL round if a model is
available), and confirm exactly one source appears, named after the round folder. Then, if a SAM2
checkpoint is available, escalate that source and confirm: the escalation finishes, a summary
message box appears, and the Review Escalations dialog opens automatically with Accept/Reject
options. Reject it, then click the new "Review escalations…" button in the Tools panel and confirm
it reports no pending escalations (proving reject actually cleared the field). Escalate again and
this time Accept — confirm the source's level swaps to `polygon` in place (still one entry in the
source list, not two) and that it now shows as unreviewed. Finally, close the project and reopen it
without touching escalation, escalate a different source, close the Review Escalations dialog
without acting on it, and confirm the "Review escalations…" button still finds and reopens it for
that source (proving a pending escalation survives being left unattended, not just a fresh run).

If no SAM2 checkpoint is available in this environment, skip the escalation half of the smoke test
and say so explicitly rather than claiming it was verified.

- [ ] **Step 6: Commit (only if Steps 1-5 required any fixes)**

If any fixes were needed:
```bash
git add -A
git commit -m "fix(detectkit): address lint/test/docs findings from source-unification sweep"
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
