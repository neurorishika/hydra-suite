# DetectKit Clear-Labels Actions — Design Spec (Part C)

> **Status:** APPROVED, ready for planning.
> **Decided:** 2026-08-27.
> **Scope:** Part C of the DetectKit source-unification effort (see
> `docs/superpowers/specs/done/2026-08-27-detectkit-source-unification-design.md`'s "Design
> note: Part C" for the original lighter-weight sketch this spec formalizes). Runs in parallel
> with Part B (`docs/superpowers/specs/2026-08-27-detectkit-canvas-multi-level-design.md`).

## Goal

Three destructive "clear labels" actions in DetectKit's dataset panel, each behind a
confirmation whose strength matches its blast radius:

1. **Clear labels from frame** — right-click, current image-list selection (single or multi).
2. **Remove all labels from source** — new button, the currently selected source only.
3. **Remove ALL labels from all sources** — new button, every registered source in the project.

All three empty label files; none ever delete an image or a source registration.

## Motivation

The dataset panel already has one destructive action (`_delete_selected_images`) with no
"labels only" equivalent — today, fixing a mislabeled frame or resetting a source's labels for
re-annotation requires deleting and re-adding images, or manually editing files outside
DetectKit. This is friction users hit routinely during iterative labeling.

## Decisions

1. **"Clear" means truncate to empty, never delete the file.** A label file's row disappears
   from that image's annotations; the file itself (and the image) stays in place. This matches
   the "Clear labels from frame" name (image-level state persists) and keeps every scope's
   semantics identical — scope only changes which files get truncated.
2. **One shared, Qt-free helper backs all three actions.** `clear_labels_for_source(source_path,
   image_paths=None)` — unfiltered clears every `*.txt` under the source's `labels/`; filtered
   (`image_paths` given) clears only the label files matching those images. One tested
   implementation, three UI entry points. **Corrected during adversarial review:** it does
   deliberately NOT reuse the existing `_label_path_for_image`/`find_label_for_image`
   resolvers' full 3-strategy chain — their third strategy (an unanchored recursive
   `labels_dir.rglob(f"{stem}.txt")` search) can resolve an unlabeled image in one split to a
   *different* split's same-stem label file, silently clearing the wrong frame. Verified this
   session with a concrete `images/train/f001.jpg` (unlabeled) vs. `images/val/f001.jpg` ->
   `labels/val/f001.txt` reproduction. `clear_labels_for_source` uses only the two safe
   strategies (mirrored path, flat stem match at the `labels/` root) and skips an image it can't
   resolve that way — silently missing a legitimately-differently-organized label is an
   acceptable "skip", silently clearing the wrong frame's labels is not.
3. **Confirmation strength scales with scope**, per the locked decision from brainstorming:
   - Frame: single Yes/Cancel (matches `_delete_selected_images`'s existing pattern exactly).
   - Source: single Yes/Cancel, naming the source and the count of label files that will be
     cleared (same strength as frame, since the pattern is already used for a per-source-scale
     destructive action — image deletion — in this same panel).
   - Project: **type the project's folder name to confirm**, since `DetectKitProject` has no
     separate display name — `project.project_dir.name` is the closest thing to it, and it's
     what appears in "Manage Sources" and the window title today.
4. **No backup, no undo.** Consistent with `_delete_selected_images`'s existing behavior in this
   file — the confirmation dialog is the safety mechanism, not a recovery path.

## Architecture

**Corrected during adversarial review:** the pure-function core lives in
`src/hydra_suite/detectkit/gui/utils.py` (not `dataset_panel.py`) — that module already holds
`find_label_for_image` (the near-duplicate resolver `clear_labels_for_source` is a narrower
sibling of) and is genuinely Qt-free, unlike `dataset_panel.py` (which imports `PySide6` at
module scope). This also keeps `DatasetPanel` — already ~700 lines, over CLAUDE.md's ~500-line
guideline — from growing further with logic that doesn't need `self`.

Also corrected: `DatasetPanel` has no `_build_ui` method — its entire UI is built inline in
`__init__`. Every reference below is to `__init__`, not a nonexistent method.

```
src/hydra_suite/detectkit/gui/utils.py
    clear_labels_for_source(source_path, image_paths=None) -> int   # NEW, module-level,
                                                                      # Qt-free, returns count
                                                                      # of files cleared

src/hydra_suite/detectkit/gui/panels/dataset_panel.py
    _on_image_list_context_menu()        # MODIFIED: add "Clear labels from frame..." entry
    _clear_labels_from_frame()           # NEW: frame-scope handler, re-renders the current
                                          # image in place (does NOT rebuild the image list --
                                          # nothing about which images exist has changed)
    _clear_labels_from_source()          # NEW: source-scope handler, new button
    _clear_labels_from_all_sources()     # NEW: project-scope handler, new button,
                                          # type-to-confirm dialog
    __init__()                           # MODIFIED: add two new buttons under Images section,
                                          # stacked vertically (matching this panel's existing
                                          # paired-button precedent) rather than side-by-side
```

## `clear_labels_for_source`

```python
def clear_labels_for_source(
    source_path: str | Path, image_paths: list[Path] | None = None
) -> int:
    """Truncate label files to empty for a source. Returns the count cleared.

    Unfiltered (image_paths=None): every "*.txt" under source_path/labels/.
    Filtered: only the label files matching the given image paths (resolved
    the same way _label_path_for_image resolves images -> labels), skipping
    any image that has no label file (nothing to clear).
    """
```

Implementation notes:
- Unfiltered case: `(Path(source_path) / "labels").rglob("*.txt")`, write `""` to each existing
  file, skipping a stray `classes.txt` if one is ever found under `labels/` by mistake (it
  belongs at the source root by convention). Recursive, so it clears nested split layouts
  (`labels/train/...`) the same way the materializer can produce them.
- Filtered case: resolve each `image_path` via only the mirrored-path and flat-stem-at-root
  strategies (see Decision 2's correction above) — deliberately NOT the unanchored recursive
  fallback `_label_path_for_image`/`find_label_for_image` use. A resolution miss (no label file
  for that image) is silently skipped, not an error. Results are deduplicated (two images
  resolving to the same label file count once).
- Returns the number of files actually truncated. Callers use this to detect partial failure
  (a per-file write error is logged and skipped, not raised) and surface a follow-up warning —
  matching `_delete_selected_images`'s existing failure-reporting pattern in this panel.

## UI wiring

**Note (added during adversarial review):** the code sketches below are illustrative of the
intended shape; the implementation plan
(`docs/superpowers/plans/2026-08-27-detectkit-clear-labels-plan.md`) is authoritative for the
exact, corrected code — it fixes several things this section's original sketches got wrong: (1)
`_build_ui` doesn't exist, the buttons are added inline in `__init__`; (2) after clearing labels,
the handlers must re-render the current image/row in place, not call
`_on_source_combo_changed` unconditionally (that rebuilds the image list and resets to row 0,
which is correct for actual image deletion but wrong here — nothing about which images exist
changed); (3) `clear_labels_for_source`'s return value IS used, to surface a partial-failure
warning (see Decision 2's note above — the "doesn't need failure-reporting UI" paragraph
originally in this section was wrong; matching `_delete_selected_images`'s existing pattern means
matching its failure reporting too, not just its confirmation dialog); (4) the two new buttons
stack vertically, not in an `QHBoxLayout` row (see Architecture section above).

### 1. Clear labels from frame

```python
def _on_image_list_context_menu(self, pos) -> None:
    ...  # existing selection-resolution logic, unchanged
    menu = QMenu(self.image_list)
    ...  # existing "Delete image(s)..." action, unchanged
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

```python
def _clear_labels_from_frame(self) -> None:
    """Empty the label file(s) for the currently selected image(s), leaving
    the images and every other frame's labels untouched."""
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
            f"Clear all labels for {len(image_paths)} frame(s)? The image(s) "
            f"stay, only their annotations are removed.\n\n{sample_names}\n\n"
            "This cannot be undone."
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    if confirm != QMessageBox.StandardButton.Yes:
        return

    clear_labels_for_source(source_path, image_paths)
    self._on_source_combo_changed(self.source_combo.currentIndex())
```

No success toast — matches `_delete_selected_images`'s existing precedent in this same file
(silent on success). A per-file `try/except` inside the helper skips and logs a failed write
rather than raising; the caller checks the returned count against how many files it expected to
clear and shows a `QMessageBox.warning` only when they differ — matching
`_delete_selected_images`'s existing failure-reporting pattern, not skipping it.

### 2 & 3. Source-scope and project-scope buttons

Added inline in `__init__` (there is no `_build_ui` method), inside `images_layout`, directly
after `self.image_list` (and its shortcuts, which aren't part of the layout) — visually grouped
under the Images section per the original request ("two new buttons needs to be added under the
images section"), stacked vertically rather than side-by-side (see Architecture section above):

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

Only `"secondary"` and `"quiet"` button variants exist today (`main_window.py`'s stylesheet
block, `QPushButton[detectkitVariant="secondary"]`/`"quiet"`). Add a third,
`detectkitVariant="danger"` (red-tinted), to the same stylesheet block, for these two buttons —
matching the escalated visual weight the destructive-actions requirement calls for, alongside
the escalated dialog strength.

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
            f"Clear ALL labels for source '{name}' ({count} label file(s))? "
            "Images are not affected.\n\nThis cannot be undone."
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    if confirm != QMessageBox.StandardButton.Yes:
        return

    clear_labels_for_source(source_path)
    self._on_source_combo_changed(self.source_combo.currentIndex())
```

```python
def _clear_labels_from_all_sources(self) -> None:
    """Empty every label file for every source in the project, behind a
    type-the-project-name confirmation (the strongest gate in this panel,
    for the strongest blast radius)."""
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
            f"This clears ALL labels across {len(self._project.sources)} source(s) "
            f"({total} label file(s) total). Images are not affected. This cannot "
            f"be undone.\n\nType the project name to confirm: {project_name}"
        ),
    )
    if not ok or typed.strip() != project_name:
        return

    for src in self._project.sources:
        clear_labels_for_source(src.path)
    self._on_source_combo_changed(self.source_combo.currentIndex())
```

## Testing

- Pure-function: `clear_labels_for_source` unfiltered (clears every label file, leaves images
  and non-label files alone, nested-layout coverage), filtered (clears only matching, skips
  images with no label file, returns the correct count).
- Panel-level: right-click menu shows the correct singular/plural "Clear labels from frame(s)"
  text and wires to the handler; the source-scope button's confirm dialog names the right
  source and count; the project-scope button rejects a wrong/empty typed name (no files
  touched) and accepts the exact project folder name; both zero-label-files early-exits show an
  info dialog instead of a pointless confirm.
- Follows the existing `_delete_selected_images`/`test_detectkit_dataset_panel_widget.py`
  patterns (QT_QPA_PLATFORM=offscreen, monkeypatched `QMessageBox`/`QInputDialog` to avoid the
  documented modal-hang risk in this repo's test suite).

## Out of scope

- Undo/recovery for a cleared label (decision: no backup, matches existing delete-images
  behavior in this panel).
- Clearing a single class's labels selectively (all three scopes clear everything for their
  scope; no per-class filter).
- Any change to `pending_escalation`/staged SAM2 results — clearing a source's canonical labels
  does not touch its staging directory if one exists (Part A's `accept`/`reject` own that
  lifecycle independently).
