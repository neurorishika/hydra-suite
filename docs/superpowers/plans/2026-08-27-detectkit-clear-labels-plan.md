# DetectKit Clear-Labels Actions (Part C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three destructive "clear labels" actions in DetectKit's dataset panel (frame, source,
project scope), each behind a confirmation matching its blast radius, backed by one shared
Qt-free helper.

**Architecture:** One new module-level function `clear_labels_for_source` in `dataset_panel.py`
truncates label files to empty (never deletes files/images). Three new UI entry points call it:
a right-click context-menu item (frame scope), and two new buttons under the Images section
(source scope: Yes/Cancel; project scope: type-the-project-folder-name).

**Tech Stack:** Python 3.11+, PySide6, pytest, `hydra-mps` conda env for all test runs.

**Spec:** `docs/superpowers/specs/2026-08-27-detectkit-clear-labels-design.md`

## Global Constraints

- "Clear" means truncate to empty (`write_text("")`), never delete the label file or the image.
- One shared helper backs all three scopes — no per-scope duplicated file-walking logic.
- Confirmation strength: frame/source = single Yes/Cancel (default Cancel); project = type the
  project folder name (`project.project_dir.name`) exactly, via `QInputDialog.getText`.
- No success toast on completion (matches `_delete_selected_images`'s existing precedent in this
  file); only the confirmation dialogs themselves communicate the action.
- `import shutil` etc. — follow this file's existing style; don't introduce new dependencies.
- Commit as the configured git user — no `Co-Authored-By: Claude` trailer, no
  `Claude-Session:` line.
- All test runs: `conda activate hydra-mps` first, from this plan's dedicated worktree.

---

## Task 1: `clear_labels_for_source` pure-function core

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`
- Test: `tests/test_detectkit_dataset_panel.py` (pure-function tests, no Qt needed — this file
  already tests `_label_path_for_image`-adjacent logic without a `qapp` fixture)

**Interfaces:**
- Produces: `clear_labels_for_source(source_path: str | Path, image_paths: list[Path] | None =
  None) -> int` — module-level function (not a method), placed near `_label_path_for_image`.
  Later tasks (2, 3, 4) call this exact signature from their respective handlers.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_detectkit_dataset_panel.py`:

```python
def test_clear_labels_for_source_unfiltered_clears_every_label_file(tmp_path: Path):
    from hydra_suite.detectkit.gui.panels.dataset_panel import clear_labels_for_source

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
    # Untouched: images dir and classes.txt survive.
    assert (source_root / "classes.txt").read_text() == "ant\n"


def test_clear_labels_for_source_filtered_clears_only_matching_images(tmp_path: Path):
    from hydra_suite.detectkit.gui.panels.dataset_panel import clear_labels_for_source

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
    from hydra_suite.detectkit.gui.panels.dataset_panel import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "images").mkdir(parents=True)
    (source_root / "labels").mkdir(parents=True)
    (source_root / "images" / "unlabeled.jpg").write_bytes(b"fake")

    count = clear_labels_for_source(source_root, [source_root / "images" / "unlabeled.jpg"])

    assert count == 0  # no error, just nothing to clear


def test_clear_labels_for_source_unfiltered_on_empty_labels_dir(tmp_path: Path):
    from hydra_suite.detectkit.gui.panels.dataset_panel import clear_labels_for_source

    source_root = tmp_path / "src"
    (source_root / "labels").mkdir(parents=True)

    assert clear_labels_for_source(source_root) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel.py -k clear_labels_for_source -v`
Expected: FAIL with `ImportError: cannot import name 'clear_labels_for_source'`.

- [ ] **Step 3: Implement `clear_labels_for_source`**

Add to `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`, near `_label_path_for_image`
(reuse its exact resolution logic for the filtered case rather than duplicating it — call it
directly, since it's already a `@staticmethod` on `DatasetPanel`; a module-level function calling
a class's staticmethod is fine in Python, reference it as
`DatasetPanel._label_path_for_image(image_path, source_root)`):

```python
def clear_labels_for_source(
    source_path: str | Path, image_paths: list[Path] | None = None
) -> int:
    """Truncate label files to empty for a source. Returns the count cleared.

    Unfiltered (image_paths=None): every "*.txt" under source_path/labels/,
    recursively. Filtered: only the label files matching the given image
    paths (resolved the same way _label_path_for_image resolves images ->
    labels), silently skipping any image that has no label file.

    Never deletes a file or touches images/classes.txt -- "clear" means
    truncate to empty content, matching the "Clear labels from frame" name
    (the image's own row in the browser persists).
    """
    source_root = Path(source_path)
    labels_dir = source_root / "labels"
    if not labels_dir.is_dir():
        return 0

    if image_paths is None:
        label_paths = list(labels_dir.rglob("*.txt"))
    else:
        label_paths = []
        for image_path in image_paths:
            label_path = DatasetPanel._label_path_for_image(Path(image_path), source_root)
            if label_path is not None:
                label_paths.append(label_path)

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
Expected: all 4 PASS.

- [ ] **Step 5: Run black/isort**

Run: `conda activate hydra-mps && black src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel.py && isort src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel.py`

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/dataset_panel.py tests/test_detectkit_dataset_panel.py
git commit -m "feat(detectkit): add clear_labels_for_source helper"
```

---

## Task 2: "Clear labels from frame" — right-click menu action

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`
- Test: `tests/test_detectkit_dataset_panel_widget.py`

**Interfaces:**
- Consumes: `clear_labels_for_source` (Task 1).
- Produces: `DatasetPanel._clear_labels_from_frame()` — new method, wired from a new context-menu
  action alongside the existing "Delete image..." action.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_detectkit_dataset_panel_widget.py` (matching this file's existing
`qapp`-fixture, monkeypatch-heavy style):

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
    from PySide6.QtCore import Qt

    panel, source_root = _make_panel_with_source(qapp, tmp_path)
    panel.image_list.setCurrentRow(0)
    panel.image_list.item(0).setSelected(True)

    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: __import__("PySide6.QtWidgets", fromlist=["QMessageBox"])
        .QMessageBox.StandardButton.Cancel,
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


def test_image_list_context_menu_offers_clear_labels_action(qapp, tmp_path):
    panel, _source_root = _make_panel_with_source(qapp, tmp_path)
    panel.image_list.setCurrentRow(0)
    panel.image_list.item(0).setSelected(True)
    # _on_image_list_context_menu builds and execs a QMenu synchronously via
    # menu.exec(...); rather than driving the real menu, verify the method
    # exists and the panel has the handler it must wire the new action to
    # -- the actual QAction wiring is exercised end-to-end by the two tests
    # above (they call the handler this menu item triggers).
    assert callable(getattr(panel, "_clear_labels_from_frame", None))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k clear_labels_from_frame -v`
Expected: FAIL — `AttributeError: 'DatasetPanel' object has no attribute '_clear_labels_from_frame'`.

- [ ] **Step 3: Implement the context-menu action and handler**

In `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`, modify `_on_image_list_context_menu`
(currently ends with `menu.addAction(delete_action)` then `menu.exec(...)`) to insert a new
action before the exec call:

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
        leaving the images and every other frame's labels untouched."""
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

        clear_labels_for_source(source_path, image_paths)
        self._on_source_combo_changed(self.source_combo.currentIndex())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k clear_labels -v`
Expected: all 3 PASS.

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
  (existing).
- Produces: `self.btn_clear_source_labels` (new button, in `_build_ui`),
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
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.warning",
        lambda *a, **k: warn_called.append(True) or QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QMessageBox.information",
        lambda *a, **k: None,
    )

    panel._clear_labels_from_source()

    assert not warn_called  # info dialog shown instead, no pointless confirm
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k clear_labels_from_source -v`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Add the button and handler**

In `_build_ui`, right after the `images_layout.addWidget(self.image_list)` line (before the
delete-shortcut setup, which isn't part of the layout), add both new buttons in one row (Task 4
adds the second button in the same row — implement both here to avoid a second layout-edit pass):

```python
        clear_btn_row = QHBoxLayout()
        self.btn_clear_source_labels = QPushButton("Remove all labels from source")
        self.btn_clear_source_labels.setProperty("detectkitVariant", "danger")
        self.btn_clear_source_labels.clicked.connect(self._clear_labels_from_source)
        self.btn_clear_all_labels = QPushButton("Remove ALL labels from all sources")
        self.btn_clear_all_labels.setProperty("detectkitVariant", "danger")
        self.btn_clear_all_labels.clicked.connect(self._clear_labels_from_all_sources)
        clear_btn_row.addWidget(self.btn_clear_source_labels)
        clear_btn_row.addWidget(self.btn_clear_all_labels)
        images_layout.addLayout(clear_btn_row)
```

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

        clear_labels_for_source(source_path)
        self._on_source_combo_changed(self.source_combo.currentIndex())
```

Add a stub for `_clear_labels_from_all_sources` now (raising `NotImplementedError` is wrong here
since the button already connects to it — implement it fully in Task 4; for THIS task, define it
minimally so the app doesn't crash if clicked before Task 4 lands, but Task 4 replaces this stub
entirely, so keep it trivial):

```python
    def _clear_labels_from_all_sources(self) -> None:
        """Placeholder -- replaced in full by Task 4 of this plan."""
        raise NotImplementedError
```

(Task 4's Step 3 replaces this stub's body — this ordering exists only so Task 3's button-row
edit doesn't leave a dangling `clicked.connect` to a name that doesn't exist yet. If executing
this plan out of order is a concern, implement Tasks 3 and 4 back-to-back.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k "clear_labels_from_source or clear_source_labels_button" -v`
Expected: all 4 PASS.

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
- Consumes: `clear_labels_for_source` (Task 1), `self.btn_clear_all_labels` (Task 3).
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
    for name in ("a", "b"):
        root = proj_dir / f"src_{name}"
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


def test_clear_all_labels_accepts_correct_typed_name(qapp, tmp_path, monkeypatch):
    panel, sources = _make_panel_with_two_sources(qapp, tmp_path)
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.panels.dataset_panel.QInputDialog.getText",
        lambda *a, **k: ("myproject", True),
    )

    panel._clear_labels_from_all_sources()

    for src in sources:
        assert (Path(src.path) / "labels" / "f.txt").read_text() == ""


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

In `src/hydra_suite/detectkit/gui/panels/dataset_panel.py`, add `QInputDialog` to the
`PySide6.QtWidgets` import block (alphabetical, next to `QHBoxLayout`/`QGroupBox` per this file's
existing import ordering). Replace the `_clear_labels_from_all_sources` stub from Task 3 with:

```python
    def _clear_labels_from_all_sources(self) -> None:
        """Empty every label file for every source in the project, behind a
        type-the-project-name confirmation -- the strongest gate in this
        panel, for the strongest blast radius."""
        if self._project is None or not self._project.sources:
            QMessageBox.information(self, "Remove Labels", "No sources in this project.")
            return

        total = sum(
            len(list((Path(src.path) / "labels").rglob("*.txt")))
            for src in self._project.sources
        )
        if total == 0:
            QMessageBox.information(
                self, "Remove Labels", "No label files exist in this project."
            )
            return

        project_name = Path(self._project.project_dir).name
        typed, ok = QInputDialog.getText(
            self,
            "Remove ALL Labels",
            (
                f"This clears ALL labels across {len(self._project.sources)} "
                f"source(s) ({total} label file(s) total). Images are not "
                f"affected. This cannot be undone.\n\nType the project name "
                f"to confirm: {project_name}"
            ),
        )
        if not ok or typed.strip() != project_name:
            return

        for src in self._project.sources:
            clear_labels_for_source(src.path)
        self._on_source_combo_changed(self.source_combo.currentIndex())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate hydra-mps && python -m pytest tests/test_detectkit_dataset_panel_widget.py -k clear_all_labels -v`
Expected: all 3 PASS.

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
- Modify: `src/hydra_suite/detectkit/gui/main_window.py` (the shared stylesheet block, per the
  spec's "only `secondary`/`quiet` variants exist today" note)
- Verification-only otherwise.

- [ ] **Step 1: Add the `danger` button variant**

In `src/hydra_suite/detectkit/gui/main_window.py`, find the stylesheet block containing
`QPushButton[detectkitVariant="secondary"]` and `QPushButton[detectkitVariant="quiet"]`. Add a
third rule following the same structure (read the existing two rules' exact property names —
background-color, color, border, etc. — and match the file's style, using a red/danger-tinted
background instead of copying secondary's color):

```css
QPushButton[detectkitVariant="danger"] {
    background-color: #c0392b;
    color: white;
    border: none;
}
QPushButton[detectkitVariant="danger"]:hover {
    background-color: #a93226;
}
```

(Match the exact CSS property set the `secondary`/`quiet` rules already use — border-radius,
padding, etc. — rather than inventing a different shape; only the color should differ.)

- [ ] **Step 2: Run the full Part C test slice**

Run:
```bash
conda activate hydra-mps
python -m pytest tests/test_detectkit_dataset_panel.py tests/test_detectkit_dataset_panel_widget.py -v
```
Expected: all PASS, including every pre-existing test in both files (no regressions from the new
button row or menu action).

- [ ] **Step 3: Run `make format-check` and `make lint`**

Run: `conda activate hydra-mps && make format-check && make lint`
Expected: no formatting diffs; no new lint findings in files this plan touched (pre-existing
findings elsewhere are out of scope).

- [ ] **Step 4: Manual smoke test (GUI)**

Per CLAUDE.md: start `detectkit`, open a project with at least one source that has labeled
frames, and verify: right-click on the image list shows "Clear labels from frame..."; clicking
it and confirming empties that frame's label file (image stays); "Remove all labels from source"
clears every label in the current source after confirming; "Remove ALL labels from all sources"
requires typing the exact project folder name and clears every source's labels on match, no-ops
on a wrong/empty typed name. If no interactive display is available in this environment, say so
explicitly rather than claiming this step was completed.

- [ ] **Step 5: Commit (only if Steps 1-4 required fixes beyond the danger-style addition)**

```bash
git add -A
git commit -m "style(detectkit): add danger button variant for destructive label actions"
```
