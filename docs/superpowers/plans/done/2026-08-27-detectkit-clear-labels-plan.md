# DetectKit Clear-Labels Actions (Part C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task. **Tasks 1-4 MUST run strictly serially, in this session, in order — do NOT
> fan them out to parallel subagents.** All four edit the same two files (`gui/utils.py`,
> `gui/panels/dataset_panel.py`), Task 3's tests depend on a helper Task 2 defines, and Task 4
> replaces a stub Task 3 creates. Task 5 is verification-only and may run after Tasks 1-4 land.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three destructive "clear labels" actions in DetectKit's dataset panel (frame, source,
project scope), each behind a confirmation matching its blast radius, backed by one shared
Qt-free helper.

**Architecture:** One new function `clear_labels_for_source` in `gui/utils.py` (genuinely
Qt-free, alongside the existing `find_label_for_image` it's a narrower sibling of) truncates
label files to empty (never deletes files/images). Three new UI entry points in
`dataset_panel.py` call it: a right-click context-menu item (frame scope), and two new buttons
under the Images section (source scope: Yes/Cancel; project scope: type-the-project-folder-name).

**Tech Stack:** Python 3.11+, PySide6, pytest, `hydra-mps` conda env for all test runs.

**Spec:** `docs/superpowers/specs/2026-08-27-detectkit-clear-labels-design.md`

## Global Constraints

- "Clear" means truncate to empty (`write_text("")`), never delete the label file or the image.
- One shared helper backs all three scopes — no per-scope duplicated file-walking logic.
- The helper's filtered (frame-scope) resolution deliberately does **not** reuse the existing
  `find_label_for_image`/`_label_path_for_image` functions' third strategy (an unanchored
  recursive `labels_dir.rglob(f"{stem}.txt")` search). Verified this session: with a split
  layout like `images/train/f001.jpg` (unlabeled) and `images/val/f001.jpg` ->
  `labels/val/f001.txt`, that unanchored search resolves the *train* image to the *val* image's
  label file — a real, silent wrong-file clear. This is a pre-existing bug in those two
  functions (not introduced here, and NOT in this plan's scope to fix them), but this plan's new
  helper must not inherit it into a brand-new destructive action. `clear_labels_for_source` uses
  only the mirrored-path (`images/<rel>` -> `labels/<rel>.txt`) and flat-stem-at-labels-root
  strategies — never the recursive fallback.
- Confirmation strength: frame/source = single Yes/Cancel (default Cancel); project = type the
  project folder name (`Path(project.project_dir).name`) exactly, via `QInputDialog.getText`,
  case-sensitive (deliberate — matches real on-disk casing on every platform this app targets,
  and loosening a confirmation gate whose entire purpose is friction would be a regression, not
  an improvement).
- No `_build_ui` method exists on `DatasetPanel` — its entire UI is built inline in `__init__`
  (`src/hydra_suite/detectkit/gui/panels/dataset_panel.py:82-231`). Every step below references
  `__init__`, not `_build_ui` — do not invent a `_build_ui` method or refactor `__init__` into
  one; that would be an unrequested restructuring of a ~700-line class.
- After clearing labels, re-render the currently displayed image in place — do NOT rebuild the
  image list and jump to row 0. `_on_source_combo_changed` (the pattern `_delete_selected_images`
  uses) rebuilds the list and unconditionally resets to row 0, which is correct when images were
  actually removed but wrong here (a labels-only edit changes nothing about which images exist).
- Commit as the configured git user — no `Co-Authored-By: Claude` trailer, no
  `Claude-Session:` line.
- All test runs: `conda activate hydra-mps` first, from this plan's dedicated worktree.
- Per CLAUDE.md's docs lifecycle: once this plan's branch merges to `main`, `git mv` this plan
  and its spec into `docs/superpowers/plans/done/`/`specs/done/` in the same commit (Task 5
  Step 5 is a reminder, not a substitute for doing this at actual merge time).

---

## Task 1: `clear_labels_for_source` — Qt-free helper in `gui/utils.py`

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/utils.py`
- Test: `tests/test_detectkit_dataset_panel.py` (this file is genuinely Qt-free today — it tests
  `gui.utils` functions like `find_label_for_image` with no `qapp` fixture and no
  `pytest.importorskip("PySide6")` — the new tests below follow that same convention, not the
  `dataset_panel_widget.py` file's Qt-heavy one)

**Interfaces:**
- Produces: `clear_labels_for_source(source_path: str | Path, image_paths: list[Path] | None =
  None) -> int` — module-level function in `hydra_suite.detectkit.gui.utils`. Tasks 2/3/4 import
  it from there (`from ..utils import clear_labels_for_source` in `dataset_panel.py`) with this
  exact signature.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_detectkit_dataset_panel.py`:

```python
def test_clear_labels_for_source_unfiltered_clears_every_label_file(tmp_path: Path):
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels" / "train").mkdir(parents=True)
    (source_root / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    (source_root / "labels" / "train" / "b.txt").write_text("0 0.5 0.5 0.4 0.2\n")
    (source_root / "classes.txt").write_text("ant\n")

    count = clear_labels_for_source(source_root)

    assert count == 2
    assert (source_root / "labels" / "a.txt").read_text() == ""
    assert (source_root / "labels" / "train" / "b.txt").read_text() == ""
    # Untouched: images dir and classes.txt (at the source root) survive.
    assert (source_root / "classes.txt").read_text() == "ant\n"


def test_clear_labels_for_source_unfiltered_skips_a_stray_classes_txt_under_labels(
    tmp_path: Path,
):
    """Defensive: classes.txt belongs at the source root by convention, but
    if one is ever found under labels/ (e.g. from a manual copy mistake),
    the unfiltered clear must not wipe it -- it isn't a label file."""
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "labels").mkdir(parents=True)
    (source_root / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    (source_root / "labels" / "classes.txt").write_text("ant\n")

    count = clear_labels_for_source(source_root)

    assert count == 1
    assert (source_root / "labels" / "a.txt").read_text() == ""
    assert (source_root / "labels" / "classes.txt").read_text() == "ant\n"


def test_clear_labels_for_source_filtered_clears_only_matching_images(tmp_path: Path):
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images" / "a.jpg").write_bytes(b"fake")
    (source_root / "images" / "b.jpg").write_bytes(b"fake")
    (source_root / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    (source_root / "labels" / "b.txt").write_text("0 0.5 0.5 0.4 0.2\n")

    count = clear_labels_for_source(source_root, [source_root / "images" / "a.jpg"])

    assert count == 1
    assert (source_root / "labels" / "a.txt").read_text() == ""
    assert (source_root / "labels" / "b.txt").read_text() != ""  # untouched


def test_clear_labels_for_source_filtered_skips_image_with_no_label_file(tmp_path: Path):
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images" / "unlabeled.jpg").write_bytes(b"fake")

    count = clear_labels_for_source(source_root, [source_root / "images" / "unlabeled.jpg"])

    assert count == 0  # no error, just nothing to clear


def test_clear_labels_for_source_unfiltered_on_empty_labels_dir(tmp_path: Path):
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "labels").mkdir(parents=True)

    assert clear_labels_for_source(source_root) == 0


def test_clear_labels_for_source_filtered_does_not_cross_split_stem_collision(
    tmp_path: Path,
):
    """Regression for a real bug found in this session's adversarial review:
    find_label_for_image's unanchored recursive-search fallback can resolve
    an unlabeled image in one split to a DIFFERENT split's same-stem label
    file. clear_labels_for_source must never do this -- it must skip an
    image it can't resolve via the mirrored-path or flat-stem strategies,
    not fall back to a wrong file found elsewhere in the tree."""
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images" / "train").mkdir(parents=True)
    (source_root / "images" / "val").mkdir(parents=True)
    (source_root / "labels" / "val").mkdir(parents=True)
    (source_root / "images" / "train" / "f001.jpg").write_bytes(b"fake")  # no label
    (source_root / "images" / "val" / "f001.jpg").write_bytes(b"fake")
    (source_root / "labels" / "val" / "f001.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
    )

    count = clear_labels_for_source(
        source_root, [source_root / "images" / "train" / "f001.jpg"]
    )

    assert count == 0  # nothing resolved -- NOT the val split's label file
    assert (source_root / "labels" / "val" / "f001.txt").read_text() != ""  # untouched


def test_clear_labels_for_source_filtered_dedupes_when_two_images_resolve_same_label(
    tmp_path: Path,
):
    from hydra_suite.detectkit.gui.utils import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images" / "a.jpg").write_bytes(b"fake")
    (source_root / "images" / "a.png").write_bytes(b"fake")  # same stem, different ext
    (source_root / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")

    count = clear_labels_for_source(
        source_root,
        [source_root / "images" / "a.jpg", source_root / "images" / "a.png"],
    )

    assert count == 1  # counted once, not twice, for the one file actually cleared
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel.py -k clear_labels_for_source -v`
Expected: FAIL with `ImportError: cannot import name 'clear_labels_for_source'`.

- [ ] **Step 3: Implement `clear_labels_for_source`**

Add to `src/hydra_suite/detectkit/gui/utils.py`, right after `find_label_for_image` (they're
siblings — this is a narrower, write-capable variant of the same resolution idea, deliberately
excluding `find_label_for_image`'s unanchored recursive "Strategy 3", per the Global Constraints
note above):

```python
def clear_labels_for_source(
    source_path: str | Path, image_paths: list[Path] | None = None
) -> int:
    """Truncate label files to empty for a source. Returns the count cleared.

    Unfiltered (image_paths=None): every "*.txt" under source_path/labels/,
    recursively, except a stray classes.txt (which belongs at the source
    root, not under labels/, but is skipped defensively if found there).

    Filtered: only the label files matching the given image paths, resolved
    via (1) mirroring images/<rel> -> labels/<rel>.txt, then (2) a direct
    stem match at the labels/ root -- deliberately NOT an unanchored
    recursive search (see find_label_for_image's Strategy 3, which this
    function does not use): with a split layout, two images in different
    splits can share a stem, and an unanchored search would silently
    resolve to the WRONG image's label file. An image that doesn't resolve
    via (1) or (2) is skipped, not an error -- "no label file for this
    image" is a legitimate, common state.

    Never deletes a file or touches images/classes.txt -- "clear" means
    truncate to empty content, matching the "Clear labels from frame" name
    (the image's own row in the browser persists).
    """
    source_root = Path(source_path)
    labels_dir = source_root / "labels"
    if not labels_dir.is_dir():
        return 0

    if image_paths is None:
        label_paths = [p for p in labels_dir.rglob("*.txt") if p.name != "classes.txt"]
    else:
        images_dir = source_root / "images"
        seen: set[Path] = set()
        label_paths = []
        for raw_image_path in image_paths:
            image_path = Path(raw_image_path)
            candidate: Path | None = None
            if images_dir.is_dir():
                try:
                    rel = image_path.relative_to(images_dir)
                    mirrored = labels_dir / rel.with_suffix(".txt")
                    if mirrored.exists():
                        candidate = mirrored
                except ValueError:
                    pass
            if candidate is None:
                flat = labels_dir / f"{image_path.stem}.txt"
                if flat.exists():
                    candidate = flat
            if candidate is not None and candidate not in seen:
                seen.add(candidate)
                label_paths.append(candidate)

    cleared = 0
    for label_path in label_paths:
        try:
            label_path.write_text("", encoding="utf-8")
            cleared += 1
        except Exception:
            logger.warning("Failed to clear labels at %s", label_path)

    return cleared
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel.py -k clear_labels_for_source -v`
Expected: all 7 PASS.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/utils.py tests/test_detectkit_dataset_panel.py && isort src/hydra_suite/detectkit/gui/utils.py tests/test_detectkit_dataset_panel.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/utils.py tests/test_detectkit_dataset_panel.py
git commit -m "feat(detectkit): add clear_labels_for_source helper"
```

---

## Task 2: "Clear labels from frame" — right-click menu action

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`
- Test: `tests/test_detectkit_dataset_panel_widget.py`

**Interfaces:**
- Consumes: `clear_labels_for_source` from `hydra_suite.detectkit.gui.utils` (Task 1).
- Produces: `DatasetPanel._clear_labels_from_frame()` — new method, wired from a new context-menu
  action alongside the existing "Delete image..." action. `_make_panel_with_source(qapp,
  tmp_path)` — a new test helper in this test file, reused by Tasks 2, 3, and 4's tests (all
  three are appended to the same file in order — this only works if Tasks 2-4 run serially, per
  this plan's header).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_detectkit_dataset_panel_widget.py`:

```python
def _make_panel_with_source(qapp, tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images" / "a.jpg").write_bytes(b"fake")
    (source_root / "labels" / "a.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    (source_root / "classes.txt").write_text("ant\n")

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.sources = [OBBSource(path=str(source_root), name="src", level="obb")]

    panel = DatasetPanel()
    panel.set_project(proj, main_window=None)
    return panel, source_root


def test_clear_labels_from_frame_requires_confirmation(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    panel.image_list.setCurrentRow(0)
    panel.image_list.item(0).setSelected(True)

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Cancel,
    )

    panel._clear_labels_from_frame()

    assert (source_root / "labels" / "a.txt").read_text() != ""  # untouched, confirm declined


def test_clear_labels_from_frame_clears_on_confirm(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    panel.image_list.setCurrentRow(0)
    panel.image_list.item(0).setSelected(True)

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )

    panel._clear_labels_from_frame()

    assert (source_root / "labels" / "a.txt").read_text() == ""


def test_clear_labels_from_frame_confirmation_names_frame_count(qapp, tmp_path, monkeypatch):
    """The confirmation dialog text must actually name what's about to be
    cleared -- not just exist. Captures the call instead of asserting
    nothing about its content."""
    from PySide6.QtWidgets import QMessageBox

    panel, _source_root = _make_panel_with_source(qapp, tmp_path)
    panel.image_list.setCurrentRow(0)
    panel.image_list.item(0).setSelected(True)

    captured = {}

    def _capture_warning(self, title, text, *a, **k):
        captured["title"] = title
        captured["text"] = text
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        _capture_warning,
    )

    panel._clear_labels_from_frame()

    assert "1 frame" in captured["text"] or "a.jpg" in captured["text"]


def test_clear_labels_from_frame_reselects_same_row_not_row_zero(qapp, tmp_path, monkeypatch):
    """Regression: clearing a frame's labels must re-render that frame in
    place, not rebuild the image list and jump back to row 0 -- nothing
    about which images exist has changed."""
    from hydra_suite.detectkit.gui.models import OBBSource
    from PySide6.QtWidgets import QMessageBox

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    # Add a second image so row 0 vs row 1 is a meaningful distinction.
    (source_root / "images" / "b.jpg").write_bytes(b"fake")
    (source_root / "labels" / "b.txt").write_text("0 0.5 0.5 0.4 0.2\n")
    panel._project.sources = [OBBSource(path=str(source_root), name="src", level="obb")]
    panel.refresh_sources(panel._project)

    # Select whichever row corresponds to b.jpg.
    b_row = next(
        i
        for i in range(panel.image_list.count())
        if panel.image_list.item(i).data(Qt.UserRole)
        and Path(str(panel.image_list.item(i).data(Qt.UserRole))).name == "b.jpg"
    )
    panel.image_list.setCurrentRow(b_row)
    panel.image_list.item(b_row).setSelected(True)

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )

    panel._clear_labels_from_frame()

    assert panel.image_list.currentRow() == b_row
```

(`Qt`/`Path` are already imported at the top of this test file's module scope or via the panel's
own imports — check the file's existing import block; add `from PySide6.QtCore import Qt` and
`from pathlib import Path` at module level if they aren't already there, following whatever this
file's existing convention is for tests that need them.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k clear_labels_from_frame -v`
Expected: FAIL — `AttributeError: 'DatasetPanel' object has no attribute '_clear_labels_from_frame'`.

- [ ] **Step 3: Implement the context-menu action and handler**

In `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`, add `clear_labels_for_source` to the
existing `from ..utils import (...)` import block. Modify `_on_image_list_context_menu`
(currently ends with `menu.addAction(delete_action)` then `menu.exec(...)`, at
`dataset_panel.py:318-334`) to insert a new action before the exec call:

```python
    def _on_image_list_context_menu(self, pos) -> None:
        """Show right-click menu on the image list."""
        items = self.image_list.selectedItems()
        if not items:
            item = self.image_list.itemAt(pos)
            if item is not None:
                items = [item]
        if not items:
            return
        menu = QMenu(self.image_list)
        label_text = (
            f"Delete {len(items)} images..." if len(items) > 1 else "Delete image..."
        )
        delete_action = QAction(label_text, menu)
        delete_action.triggered.connect(self._delete_selected_images)
        menu.addAction(delete_action)

        clear_label_text = (
            f"Clear labels from {len(items)} frames..."
            if len(items) > 1
            else "Clear labels from frame..."
        )
        clear_action = QAction(clear_label_text, menu)
        clear_action.triggered.connect(self._clear_labels_from_frame)
        menu.addAction(clear_action)

        menu.exec(self.image_list.viewport().mapToGlobal(pos))
```

Add the new handler right after `_delete_selected_images` (before `_label_path_for_image`):

```python
    def _clear_labels_from_frame(self) -> None:
        """Empty the label file(s) for the currently selected image(s),
        leaving the images and every other frame's labels untouched. The
        currently displayed image is re-rendered in place -- nothing about
        which images exist has changed, so the list itself is not rebuilt."""
        items = self.image_list.selectedItems()
        if not items:
            return

        source_path = self._selected_source_path()
        if source_path is None:
            return

        image_paths: list[Path] = []
        for item in items:
            data = item.data(Qt.UserRole)
            if data:
                image_paths.append(Path(str(data)))
        if not image_paths:
            return

        sample_names = ", ".join(p.name for p in image_paths[:3])
        if len(image_paths) > 3:
            sample_names += f", ... (+{len(image_paths) - 3} more)"
        confirm = QMessageBox.warning(
            self,
            "Clear Labels",
            (
                f"Clear all labels for {len(image_paths)} frame(s)? The "
                f"image(s) stay, only their annotations are removed.\n\n"
                f"{sample_names}\n\nThis cannot be undone."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        cleared = clear_labels_for_source(source_path, image_paths)
        if cleared < len(image_paths):
            QMessageBox.warning(
                self,
                "Clear Labels",
                f"Cleared {cleared} of {len(image_paths)} label file(s); "
                "some could not be written.",
            )
        self._on_image_changed(self.image_list.currentRow())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k clear_labels_from_frame -v`
Expected: all 4 PASS.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel_widget.py && isort src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel_widget.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel_widget.py
git commit -m "feat(detectkit): add Clear labels from frame context-menu action"
```

---

## Task 3: "Remove all labels from source" button

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`
- Test: `tests/test_detectkit_dataset_panel_widget.py`

**Interfaces:**
- Consumes: `clear_labels_for_source` (Task 1), `_selected_source_obj`/`_selected_source_path`
  (existing), `_make_panel_with_source` (Task 2's test helper — this task's tests call it, so
  Task 2 must have already landed in this same test file).
- Produces: `self.btn_clear_source_labels` (new button, added in `__init__`),
  `DatasetPanel._clear_labels_from_source()` (new handler).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_detectkit_dataset_panel_widget.py`:

```python
def test_dataset_panel_has_clear_source_labels_button(qapp):
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    panel = DatasetPanel()
    assert hasattr(panel, "btn_clear_source_labels")


def test_clear_labels_from_source_requires_confirmation(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Cancel,
    )

    panel._clear_labels_from_source()

    assert (source_root / "labels" / "a.txt").read_text() != ""


def test_clear_labels_from_source_confirmation_names_source_and_count(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    panel, _source_root = _make_panel_with_source(qapp, tmp_path)
    captured = {}

    def _capture_warning(self, title, text, *a, **k):
        captured["text"] = text
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        _capture_warning,
    )

    panel._clear_labels_from_source()

    assert "src" in captured["text"]
    assert "1" in captured["text"]  # 1 label file


def test_clear_labels_from_source_clears_all_on_confirm(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )

    panel._clear_labels_from_source()

    assert (source_root / "labels" / "a.txt").read_text() == ""


def test_clear_labels_from_source_no_op_when_no_labels(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel
    from PySide6.QtWidgets import QMessageBox

    source_root = tmp_path / "empty_src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "classes.txt").write_text("ant\n")
    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.sources = [OBBSource(path=str(source_root), name="empty", level="obb")]
    panel = DatasetPanel()
    panel.set_project(proj, main_window=None)

    warn_called = []
    info_called = []
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: warn_called.append(True) or QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.information",
        lambda *a, **k: info_called.append(True),
    )

    panel._clear_labels_from_source()

    assert not warn_called  # no pointless confirm
    assert info_called  # info dialog shown instead
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k clear_labels_from_source -v`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Add the button and handler**

In `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`'s `__init__`, right after the line
`images_layout.addWidget(self.image_list)` (currently line 154) and before the
`self._delete_shortcut = QShortcut(...)` setup that follows it, add both new buttons **stacked
vertically** (matching this file's existing paired-action precedent, `btn_train`/`btn_history` at
lines 221-227, which stack rather than sit side-by-side in this narrow left-dock panel — a
`QHBoxLayout` row for two long-labeled buttons would elide badly here). Implement both buttons in
this task to avoid a second layout-edit pass (Task 4 only adds the second button's real handler
body):

```python
        self.btn_clear_source_labels = QPushButton("Remove all labels from source")
        self.btn_clear_source_labels.setProperty("detectkitVariant", "danger")
        self.btn_clear_source_labels.clicked.connect(self._clear_labels_from_source)
        images_layout.addWidget(self.btn_clear_source_labels)

        self.btn_clear_all_labels = QPushButton("Remove ALL labels from all sources")
        self.btn_clear_all_labels.setProperty("detectkitVariant", "danger")
        self.btn_clear_all_labels.clicked.connect(self._clear_labels_from_all_sources)
        images_layout.addWidget(self.btn_clear_all_labels)
```

Add `QInputDialog` to the `PySide6.QtWidgets` import block now (alphabetical, next to
`QHBoxLayout`/`QGroupBox`) — Task 4 needs it, and adding it once here avoids a second import edit.

Add the source-scope handler after `_clear_labels_from_frame`:

```python
    def _clear_labels_from_source(self) -> None:
        """Empty every label file for the currently selected source."""
        source_path = self._selected_source_path()
        if source_path is None:
            return
        src_obj = self._selected_source_obj()
        name = src_obj.name if src_obj else Path(source_path).name

        count = len(list((Path(source_path) / "labels").rglob("*.txt")))
        if count == 0:
            QMessageBox.information(
                self, "Remove Labels", f"'{name}' has no label files to clear."
            )
            return

        confirm = QMessageBox.warning(
            self,
            "Remove Labels",
            (
                f"Clear ALL labels for source '{name}' ({count} label "
                "file(s))? Images are not affected.\n\nThis cannot be "
                "undone."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        cleared = clear_labels_for_source(source_path)
        if cleared < count:
            QMessageBox.warning(
                self,
                "Remove Labels",
                f"Cleared {cleared} of {count} label file(s); some could "
                "not be written.",
            )
        row = self.image_list.currentRow()
        self._on_source_combo_changed(self.source_combo.currentIndex())
        if 0 <= row < self.image_list.count():
            self.image_list.setCurrentRow(row)
```

Add a stub for `_clear_labels_from_all_sources` now (this task's own tests never exercise it —
they only call `_clear_labels_from_source` — so a stub is safe here; Task 4 replaces it in full):

```python
    def _clear_labels_from_all_sources(self) -> None:
        """Placeholder -- replaced in full by Task 4 of this plan."""
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k "clear_labels_from_source or clear_source_labels_button" -v`
Expected: all 5 PASS.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel_widget.py && isort src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel_widget.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel_widget.py
git commit -m "feat(detectkit): add Remove all labels from source button"
```

---

## Task 4: "Remove ALL labels from all sources" button (type-to-confirm)

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`
- Test: `tests/test_detectkit_dataset_panel_widget.py`

**Interfaces:**
- Consumes: `clear_labels_for_source` (Task 1), `self.btn_clear_all_labels` (Task 3),
  `_make_panel_with_source`/`_make_panel_with_two_sources` (test helpers).
- Produces: `DatasetPanel._clear_labels_from_all_sources()` (real implementation, replacing Task
  3's stub).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_detectkit_dataset_panel_widget.py`:

```python
def _make_panel_with_two_sources(qapp, tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    proj_dir = tmp_path / "myproject"
    proj_dir.mkdir()
    sources = []
    for name, is_linked in (("a", False), ("b", True)):
        # One source lives inside the project dir, one outside it (a
        # "linked" source, per DetectKitProject's portability concept) --
        # exercises the project-scope warning about out-of-project blast
        # radius.
        root = (proj_dir if not is_linked else tmp_path / "elsewhere") / f"src_{name}"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir(parents=True)
        (root / "images" / "f.jpg").write_bytes(b"fake")
        (root / "labels" / "f.txt").write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
        (root / "classes.txt").write_text("ant\n")
        sources.append(OBBSource(path=str(root), name=name, level="obb"))

    proj = DetectKitProject(project_dir=proj_dir, class_names=["ant"])
    proj.sources = sources
    panel = DatasetPanel()
    panel.set_project(proj, main_window=None)
    return panel, sources


def test_clear_all_labels_rejects_wrong_typed_name(qapp, tmp_path, monkeypatch):
    panel, sources = _make_panel_with_two_sources(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        lambda *a, **k: ("not-the-project-name", True),
    )

    panel._clear_labels_from_all_sources()

    for src in sources:
        assert (Path(src.path) / "labels" / "f.txt").read_text() != ""


def test_clear_all_labels_rejects_empty_typed_name(qapp, tmp_path, monkeypatch):
    """Regression: an empty typed value must never satisfy the confirmation,
    even in the (non-live-reachable-today, but cheap-to-guard) edge case of
    an empty project folder name."""
    panel, sources = _make_panel_with_two_sources(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        lambda *a, **k: ("", True),
    )

    panel._clear_labels_from_all_sources()

    for src in sources:
        assert (Path(src.path) / "labels" / "f.txt").read_text() != ""


def test_clear_all_labels_accepts_correct_typed_name(qapp, tmp_path, monkeypatch):
    panel, sources = _make_panel_with_two_sources(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        lambda *a, **k: ("myproject", True),
    )

    panel._clear_labels_from_all_sources()

    for src in sources:
        assert (Path(src.path) / "labels" / "f.txt").read_text() == ""


def test_clear_all_labels_prompt_warns_about_linked_sources(qapp, tmp_path, monkeypatch):
    """The confirmation prompt must call out that some sources live outside
    the project folder -- "type the project name" implies project-bounded
    blast radius, but a linked source can be anywhere on disk."""
    panel, _sources = _make_panel_with_two_sources(qapp, tmp_path)
    captured = {}

    def _capture_get_text(cls, parent, title, text, *a, **k):
        captured["text"] = text
        return "", False

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        _capture_get_text,
    )

    panel._clear_labels_from_all_sources()

    assert "outside" in captured["text"].lower()


def test_clear_all_labels_no_op_when_project_empty(qapp, tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui.models import DetectKitProject
    from hydra_suite.detectkit.gui.panels.dataset_panel import DatasetPanel

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    panel = DatasetPanel()
    panel.set_project(proj, main_window=None)

    prompted = []
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        lambda *a, **k: (prompted.append(True), ("", False))[1],
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.information",
        lambda *a, **k: None,
    )

    panel._clear_labels_from_all_sources()

    assert not prompted  # info dialog shown instead of a pointless type-to-confirm prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k clear_all_labels -v`
Expected: FAIL — the stub raises `NotImplementedError`.

- [ ] **Step 3: Replace the stub with the real implementation**

Replace the `_clear_labels_from_all_sources` stub from Task 3 with:

```python
    def _clear_labels_from_all_sources(self) -> None:
        """Empty every label file for every source in the project, behind a
        type-the-project-name confirmation -- the strongest gate in this
        panel, for the strongest blast radius."""
        if self._project is None or not self._project.sources:
            QMessageBox.information(self, "Remove Labels", "No sources in this project.")
            return

        total = sum(
            sum(
                1
                for p in (Path(src.path) / "labels").rglob("*.txt")
                if p.name != "classes.txt"
            )
            for src in self._project.sources
        )
        if total == 0:
            QMessageBox.information(
                self, "Remove Labels", "No label files exist in this project."
            )
            return

        project_dir = Path(self._project.project_dir).resolve()
        project_name = project_dir.name or str(project_dir)
        linked_count = sum(
            1
            for src in self._project.sources
            if project_dir not in Path(src.path).resolve().parents
            and Path(src.path).resolve() != project_dir
        )
        linked_note = (
            f"\n\n{linked_count} source(s) live outside this project folder and "
            "will also be cleared."
            if linked_count
            else ""
        )

        typed, ok = QInputDialog.getText(
            self,
            "Remove ALL Labels",
            (
                f"This clears ALL labels across {len(self._project.sources)} "
                f"source(s) ({total} label file(s) total). Images are not "
                f"affected. This cannot be undone.{linked_note}\n\nType the "
                f"project name to confirm: {project_name}"
            ),
        )
        typed = typed.strip() if ok else ""
        if not typed or typed != project_name:
            return

        cleared_total = 0
        for src in self._project.sources:
            cleared_total += clear_labels_for_source(src.path)
        if cleared_total < total:
            QMessageBox.warning(
                self,
                "Remove Labels",
                f"Cleared {cleared_total} of {total} label file(s) across "
                "the project; some could not be written.",
            )
        row = self.image_list.currentRow()
        self._on_source_combo_changed(self.source_combo.currentIndex())
        if 0 <= row < self.image_list.count():
            self.image_list.setCurrentRow(row)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k clear_all_labels -v`
Expected: all 5 PASS.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel_widget.py && isort src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel_widget.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel_widget.py
git commit -m "feat(detectkit): add Remove ALL labels from all sources, type-to-confirm"
```

---

## Task 5: "danger" button style + full sweep

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/main_window.py` (the shared stylesheet block)
- Verification-only otherwise.

- [ ] **Step 1: Add the `danger` button variant**

In `src/hydra_suite/detectkit/gui/main_window.py`, find `QPushButton[detectkitVariant="secondary"]`
(around line 387) and `QPushButton[detectkitVariant="quiet"]` (around line 394), inside the
`_DARK_STYLESHEET` block. The base `QPushButton` rule already sets `border: none;` and other
shared properties — `secondary`/`quiet` override only `background-color`/`color` on top of the
base rule; match that shape rather than repeating every property. Add a third rule immediately
after `quiet`'s:

```css
QPushButton[detectkitVariant="danger"] {
    background-color: #c0392b;
    color: white;
}
QPushButton[detectkitVariant="danger"]:hover {
    background-color: #a93226;
}
QPushButton[detectkitVariant="danger"]:pressed {
    background-color: #922b21;
}
```

- [ ] **Step 2: Run the full Part C test slice**

Run:
```bash
conda activate hydra-mps
python -m pytest tests/test_detectkit_dataset_panel.py tests/test_detectkit_dataset_panel_widget.py -v
```
Expected: all PASS, including every pre-existing test in both files (no regressions from the new
button rows or menu action).

- [ ] **Step 3: Run `make format-check` and `make lint`**

Run: `conda activate hydra-mps && make format-check && make lint`
Expected: no formatting diffs; no new lint findings in files this plan touched — in particular,
confirm no unused imports were left behind in any of this plan's new test functions (Task 1-4's
test snippets were written to import only what they use; double-check the actual committed code
matches that, since `flake8`'s `F401` is NOT ignored for test files in this repo's
`.flake8.moderate` config).

- [ ] **Step 4: Manual smoke test (GUI)**

Per CLAUDE.md: start `detectkit`, open a project with at least one source that has labeled
frames, and verify: right-click on the image list shows "Clear labels from frame..."; clicking
it and confirming empties that frame's label file (image stays, and the canvas/image list stays
on the same frame, not jumping to frame 0); "Remove all labels from source" clears every label in
the current source after confirming; "Remove ALL labels from all sources" requires typing the
exact project folder name and clears every source's labels on match (warning if any sources are
linked/outside the project folder), no-ops on a wrong/empty typed name. If no interactive display
is available in this environment, say so explicitly rather than claiming this step was completed.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/detectkit/gui/main_window.py
git commit -m "style(detectkit): add danger button variant for destructive label actions"
```

If Steps 2-4 also required fixes beyond the danger-style addition, include them in this same
commit (`git add -A` instead) rather than leaving them uncommitted.

- [ ] **Step 6: Reminder — docs lifecycle at actual merge time**

Not a step to execute now: per CLAUDE.md, when this plan's branch is finally merged to `main`,
`git mv` this plan file and its spec
(`docs/superpowers/specs/2026-08-27-detectkit-clear-labels-design.md`) into their `done/`
subfolders in the same merge commit.
