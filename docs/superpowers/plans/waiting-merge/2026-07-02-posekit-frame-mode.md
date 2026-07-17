# PoseKit Frame Mode Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Frame Mode" toggle to PoseKit that makes every image-selection flow (manual labeling, the two bulk-move buttons, Random Selection, Smart Select, and Smart Select's Embedding Explorer) operate on whole source frames — pulling in every detected individual on a frame, not just one — so users can build labeling sets suited for bottom-up multi-animal pose training.

**Architecture:** A new frame-grouping helper (reusing MAT/FilterKit's `did<detection_id>` filename convention) groups PoseKit's flat image list into per-source frames. A single `PoseKitConfig.frame_mode` flag gates all behavior changes. A shared method on `MainWindow`, `_add_indices_to_labeling`, becomes the one place that expands a set of indices to their frame companions and asks for confirmation before mutating `self.labeling_frames`; every entry point routes through it (or, for Smart Select's Embedding Explorer sub-dialog, through an equivalent local expansion since that nested dialog has no `MainWindow` reference). Smart Select's clustering algorithm is extended with a new frame-coverage greedy selector that operates on individual embeddings directly (no averaging).

**Tech Stack:** Python, PySide6 (Qt), pytest + pytest-qt-style `qapp` fixture (offscreen platform) for GUI tests, existing `hydra_suite.core.identity.dataset.naming.parse_identity_image_filename`.

## Global Constraints

- Reuse `parse_identity_image_filename` from `hydra_suite.core.identity.dataset.naming` verbatim — do not reimplement filename parsing. It returns `dict[str, Any] | None`; `None` means the filename doesn't match the identity-crop convention (treat as a singleton frame).
- `PoseKitConfig.frame_mode: bool = False` is the single global toggle. It is unrelated to the existing `mode: str = "frame"` field (pose canvas frame/keypoint editing mode) — do not conflate the two.
- No back-solving of frame counts anywhere. In Frame Mode, every spinbox that specifies "how many to add" means "how many frames" directly — Random Selection's `spin_random_count` and Smart Select's `n_spin` both keep their literal values as frame counts.
- Confirmation dialog wording (used verbatim everywhere a confirmation is shown): `"This will add {frame_count} frame(s) comprising {total_count} total instance(s), including {companion_count} companion instance(s), to the labeling set. Continue?"` shown via `QMessageBox.question(self, title, message, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)`.
- Frame Mode checkbox tooltip text (used verbatim): `"Frame Mode: sampling and labeling operations act on entire frames (all detected individuals together), not single crops. Required if you're building a dataset for bottom-up multi-animal pose models."`
- GUI tests follow the two existing conventions in `tests/`: (a) pure-logic `MainWindow` methods are tested by constructing a `types.SimpleNamespace` standing in for `self` and calling `MainWindow.method_name(namespace, ...)` unbound (see `tests/test_posekit_main_window.py`); (b) real dialog widget behavior is tested by instantiating the actual `QDialog` subclass under a `qapp` fixture with `QT_QPA_PLATFORM=offscreen` (see `tests/test_posekit_new_project_dialog.py`).
- Run tests with: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/<file> -v` from the repo/worktree root (the `hydra-mps` conda env's editable install resolves to the wrong `src/`, so `PYTHONPATH=src` is mandatory).
- `make format` may reformat unrelated pre-existing-drift files; if so, `git checkout --` those files before committing, per prior FilterKit-feature experience.

---

### Task 1: Frame-grouping helper module

**Files:**
- Create: `src/hydra_suite/posekit/core/frame_grouping.py`
- Test: `tests/test_posekit_frame_grouping.py`

**Interfaces:**
- Produces: `group_indices_by_frame(filenames: Sequence[str], source_ids: Sequence[Any]) -> dict[tuple[Any, int], list[int]]` — every later task that needs frame grouping calls this exact function with the *full* project image list (not a filtered subset), so frame keys stay globally consistent across callers.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for PoseKit's frame-grouping helper."""

from __future__ import annotations


def test_group_indices_by_frame_groups_matching_identity_filenames():
    from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame

    filenames = ["did10000.jpg", "did10001.jpg", "did20000.jpg", "plain.png"]
    source_ids = ["src_a", "src_a", "src_a", "src_a"]

    groups = group_indices_by_frame(filenames, source_ids)

    assert groups[("src_a", 1)] == [0, 1]
    assert groups[("src_a", 2)] == [2]
    # Non-matching filenames get their own singleton key.
    assert groups[("src_a", -4)] == [3]


def test_group_indices_by_frame_scopes_by_source():
    from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame

    filenames = ["did10000.jpg", "did10000.jpg"]
    source_ids = ["src_a", "src_b"]

    groups = group_indices_by_frame(filenames, source_ids)

    # Same frame_idx (1) but different sources must not collide.
    assert groups[("src_a", 1)] == [0]
    assert groups[("src_b", 1)] == [1]


def test_group_indices_by_frame_singleton_keys_never_collide_with_real_frames():
    from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame

    # frame_idx=0 is a real, valid frame index (did0.jpg -> detection_id=0 -> frame_idx=0).
    filenames = ["did0.jpg", "plain_0.png", "plain_1.png"]
    source_ids = ["src_a", "src_a", "src_a"]

    groups = group_indices_by_frame(filenames, source_ids)

    assert groups[("src_a", 0)] == [0]
    # Singleton keys use negative frame components, so they can never
    # collide with a real (always >= 0) frame_idx.
    assert groups[("src_a", -2)] == [1]
    assert groups[("src_a", -3)] == [2]


def test_group_indices_by_frame_empty_input_returns_empty_dict():
    from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame

    assert group_indices_by_frame([], []) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_frame_grouping.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.posekit.core.frame_grouping'`

- [ ] **Step 3: Write the implementation**

```python
"""Group PoseKit image indices into per-source frames.

Reuses the MAT/FilterKit identity-crop filename convention
(``did<detection_id>.<ext>``, where ``frame_idx = detection_id // 10000``)
so images produced by a FilterKit export can be regrouped by their
original source frame.
"""

from __future__ import annotations

from typing import Any, Sequence

from hydra_suite.core.identity.dataset.naming import parse_identity_image_filename


def group_indices_by_frame(
    filenames: Sequence[str], source_ids: Sequence[Any]
) -> dict[tuple[Any, int], list[int]]:
    """Group image indices by ``(source_id, frame_idx)``.

    ``filenames[i]``/``source_ids[i]`` describe the image at global index
    ``i``. Filenames that don't match the identity-crop convention each
    get a unique singleton key ``(source_id, -(i + 1))`` — real
    ``frame_idx`` values are always >= 0, so singleton keys never collide
    with a genuine frame.
    """
    groups: dict[tuple[Any, int], list[int]] = {}
    for idx, (filename, source_id) in enumerate(zip(filenames, source_ids)):
        parsed = parse_identity_image_filename(filename)
        frame_component = parsed["frame_idx"] if parsed is not None else -(idx + 1)
        key = (source_id, frame_component)
        groups.setdefault(key, []).append(idx)
    return groups
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_frame_grouping.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/posekit/core/frame_grouping.py tests/test_posekit_frame_grouping.py
git commit -m "feat: add posekit frame-grouping helper"
```

---

### Task 2: Config schema field

**Files:**
- Modify: `src/hydra_suite/posekit/config/schemas.py`
- Test: `tests/test_posekit_config.py`

**Interfaces:**
- Consumes: none.
- Produces: `PoseKitConfig.frame_mode: bool` (default `False`), round-tripped through `to_dict`/`from_dict`. All later GUI tasks read/write `self.config.frame_mode` on `MainWindow`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_posekit_config.py`:

```python
def test_posekit_config_defaults_frame_mode_off():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    cfg = PoseKitConfig()
    assert cfg.frame_mode is False


def test_posekit_config_round_trip_frame_mode():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    cfg = PoseKitConfig(frame_mode=True)
    restored = PoseKitConfig.from_dict(cfg.to_dict())
    assert restored.frame_mode is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_config.py -v`
Expected: FAIL with `TypeError: PoseKitConfig() got an unexpected keyword argument 'frame_mode'` / `AttributeError`

- [ ] **Step 3: Write the implementation**

Modify `src/hydra_suite/posekit/config/schemas.py` to:

```python
"""Runtime configuration schema for PoseKit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PoseKitConfig:
    """User-configurable preferences for the PoseKit labeling application.

    Project data (project object, image_paths) is passed at construction
    time and does not belong here.
    """

    mode: str = "frame"  # 'frame' or 'keypoint' progression
    show_predictions: bool = True
    show_pred_conf: bool = False
    sleap_env_path: str = ""
    autosave_delay_ms: int = 3000
    frame_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize all user preferences to a JSON-compatible dictionary for persistence."""
        return {
            "mode": self.mode,
            "show_predictions": self.show_predictions,
            "show_pred_conf": self.show_pred_conf,
            "sleap_env_path": self.sleap_env_path,
            "autosave_delay_ms": self.autosave_delay_ms,
            "frame_mode": self.frame_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseKitConfig":
        """Reconstruct a PoseKitConfig from a dictionary produced by ``to_dict``."""
        return cls(
            mode=data.get("mode", "frame"),
            show_predictions=data.get("show_predictions", True),
            show_pred_conf=data.get("show_pred_conf", False),
            sleap_env_path=data.get("sleap_env_path", ""),
            autosave_delay_ms=data.get("autosave_delay_ms", 3000),
            frame_mode=data.get("frame_mode", False),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/posekit/config/schemas.py tests/test_posekit_config.py
git commit -m "feat: add frame_mode field to PoseKitConfig"
```

---

### Task 3: GUI toggle checkbox

**Files:**
- Modify: `src/hydra_suite/posekit/gui/panels/source_browser_panel.py`
- Modify: `src/hydra_suite/posekit/gui/main_window.py:194-220` (panel wiring in `__init__`)
- Test: `tests/test_posekit_source_browser_panel.py` (new)
- Test: `tests/test_posekit_main_window.py` (add a case)

**Interfaces:**
- Consumes: `PoseKitConfig.frame_mode` (Task 2).
- Produces: `PoseSourceBrowserPanel.chk_frame_mode: QCheckBox`; `MainWindow.chk_frame_mode`; `MainWindow._on_frame_mode_toggled(self, checked: bool) -> None`, which sets `self.config.frame_mode`. Later tasks read `self.config.frame_mode` directly — they do not need to touch the checkbox.

- [ ] **Step 1: Write the failing test for the panel**

```python
"""Tests for the PoseKit source browser panel's Frame Mode checkbox."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.posekit.gui.panels.source_browser_panel import (  # noqa: E402
    PoseSourceBrowserPanel,
)


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_source_browser_panel_has_frame_mode_checkbox_above_labeling_list(qapp):
    panel = PoseSourceBrowserPanel()

    assert panel.chk_frame_mode.text() == "Frame Mode"
    assert not panel.chk_frame_mode.isChecked()
    assert (
        panel.chk_frame_mode.toolTip()
        == "Frame Mode: sampling and labeling operations act on entire frames "
        "(all detected individuals together), not single crops. Required if "
        "you're building a dataset for bottom-up multi-animal pose models."
    )
    # The checkbox is the first widget in the panel's layout, above the
    # "Labeling Set" label.
    layout = panel.layout()
    assert layout.itemAt(0).widget() is panel.chk_frame_mode
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_source_browser_panel.py -v`
Expected: FAIL with `AttributeError: 'PoseSourceBrowserPanel' object has no attribute 'chk_frame_mode'`

- [ ] **Step 3: Add the checkbox to the panel**

In `src/hydra_suite/posekit/gui/panels/source_browser_panel.py`, add `QCheckBox` to the import block:

```python
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
```

At the very top of `__init__` (before `layout.addWidget(QLabel("Labeling Set"))`):

```python
        self.chk_frame_mode = QCheckBox("Frame Mode")
        self.chk_frame_mode.setToolTip(
            "Frame Mode: sampling and labeling operations act on entire "
            "frames (all detected individuals together), not single crops. "
            "Required if you're building a dataset for bottom-up "
            "multi-animal pose models."
        )
        layout.addWidget(self.chk_frame_mode)

        layout.addWidget(QLabel("Labeling Set"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_source_browser_panel.py -v`
Expected: 1 passed

- [ ] **Step 5: Write the failing test for main_window wiring**

Add to `tests/test_posekit_main_window.py`:

```python
def test_on_frame_mode_toggled_updates_config():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    window = SimpleNamespace(config=PoseKitConfig())

    MainWindow._on_frame_mode_toggled(window, True)
    assert window.config.frame_mode is True

    MainWindow._on_frame_mode_toggled(window, False)
    assert window.config.frame_mode is False
```

- [ ] **Step 6: Run test to verify it fails**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py::test_on_frame_mode_toggled_updates_config -v`
Expected: FAIL with `AttributeError: type object 'MainWindow' has no attribute '_on_frame_mode_toggled'`

- [ ] **Step 7: Wire the checkbox in main_window.py**

In `src/hydra_suite/posekit/gui/main_window.py`, after `self.btn_delete_frames = left.btn_delete_frames` (around line 220), add:

```python
        self.chk_frame_mode = left.chk_frame_mode
        self.chk_frame_mode.setChecked(self.config.frame_mode)
        self.chk_frame_mode.toggled.connect(self._on_frame_mode_toggled)
```

Add the new method near `_move_unlabeled_to_labeling` (or any convenient location in the class body):

```python
    def _on_frame_mode_toggled(self, checked: bool) -> None:
        """Persist the Frame Mode checkbox state to the runtime config."""
        self.config.frame_mode = bool(checked)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py::test_on_frame_mode_toggled_updates_config -v`
Expected: 1 passed

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/posekit/gui/panels/source_browser_panel.py src/hydra_suite/posekit/gui/main_window.py tests/test_posekit_source_browser_panel.py tests/test_posekit_main_window.py
git commit -m "feat: add Frame Mode toggle checkbox to posekit left panel"
```

---

### Task 4: Shared commit path (`_add_indices_to_labeling` + `_frame_expansion`)

**Files:**
- Modify: `src/hydra_suite/posekit/gui/main_window.py` (extend the existing `_add_indices_to_labeling` at line 3951, add new `_frame_expansion`)
- Test: `tests/test_posekit_main_window.py` (add cases)

**Interfaces:**
- Consumes: `group_indices_by_frame` (Task 1), `self.config.frame_mode` (Task 2/3), `self._source_id_for_index` (existing), `self.image_paths` (existing), `self.labeling_frames: set[int]` (existing).
- Produces: `MainWindow._frame_expansion(self, indices: set[int]) -> tuple[set[int], int]` (returns `(expanded_indices, distinct_frame_count)`); `MainWindow._add_indices_to_labeling(self, indices: list[int], title: str, disclosed: bool = False) -> bool` (returns `True` if committed/nothing to do, `False` if the user canceled a confirmation). Tasks 5-8 and 10-11 all call `_add_indices_to_labeling`; none of them re-implement expansion or confirmation.

The existing method today (verbatim, for reference — this task replaces it):

```python
    def _add_indices_to_labeling(self, indices: List[int], title: str):
        if not indices:
            return
        for idx in indices:
            self.labeling_frames.add(int(idx))
        self._populate_frames()
        self._select_frame_in_list(self.current_index)
        QMessageBox.information(
            self, title, f"Added {len(indices)} frames to labeling set."
        )
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_posekit_main_window.py`:

```python
def test_frame_expansion_groups_by_source_and_frame():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[
            Path("did10000.jpg"),
            Path("did10001.jpg"),
            Path("did20000.jpg"),
        ],
        _source_id_for_index=lambda idx: "src_a",
    )

    expanded, frame_count = MainWindow._frame_expansion(window, {0})

    assert expanded == {0, 1}
    assert frame_count == 1


def test_add_indices_to_labeling_frame_mode_expands_and_confirms(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    calls = []

    def fake_question(*args, **kwargs):
        calls.append("asked")
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[
            Path("did10000.jpg"),
            Path("did10001.jpg"),
            Path("did20000.jpg"),
        ],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=0,
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
    )

    result = MainWindow._add_indices_to_labeling(window, [0], "Test")

    assert result is True
    assert window.labeling_frames == {0, 1}
    assert calls == ["asked"]


def test_add_indices_to_labeling_frame_mode_cancel_adds_nothing(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=0,
        _populate_frames=lambda: (_ for _ in ()).throw(
            AssertionError("must not refresh UI on cancel")
        ),
        _select_frame_in_list=lambda *a, **k: None,
    )

    result = MainWindow._add_indices_to_labeling(window, [0], "Test")

    assert result is False
    assert window.labeling_frames == set()


def test_add_indices_to_labeling_frame_mode_disclosed_skips_confirmation(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not confirm when disclosed=True")
            )
        ),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=0,
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
    )

    result = MainWindow._add_indices_to_labeling(window, [0], "Test", disclosed=True)

    assert result is True
    assert window.labeling_frames == {0, 1}


def test_add_indices_to_labeling_individual_mode_unchanged(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not confirm outside frame mode")
            )
        ),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=False),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=0,
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
    )

    result = MainWindow._add_indices_to_labeling(window, [0], "Test")

    assert result is True
    assert window.labeling_frames == {0}
```

Add the required imports at the top of `tests/test_posekit_main_window.py` if not already present: `from PySide6.QtWidgets import QMessageBox`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -k "frame_expansion or add_indices_to_labeling" -v`
Expected: FAIL with `AttributeError: type object 'MainWindow' has no attribute '_frame_expansion'`

- [ ] **Step 3: Write the implementation**

In `src/hydra_suite/posekit/gui/main_window.py`, add the import at the top:

```python
from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame
```

Replace the existing `_add_indices_to_labeling` method (line 3951) with:

```python
    def _frame_expansion(self, indices: set) -> tuple:
        """Return (expanded_indices, distinct_frame_count) for the frames touched by `indices`."""
        groups = group_indices_by_frame(
            [p.name for p in self.image_paths],
            [self._source_id_for_index(i) for i in range(len(self.image_paths))],
        )
        idx_to_key = {i: key for key, idxs in groups.items() for i in idxs}
        keys = {idx_to_key[i] for i in indices if i in idx_to_key}
        expanded: set = set()
        for key in keys:
            expanded.update(groups[key])
        return expanded, len(keys)

    def _add_indices_to_labeling(
        self, indices: List[int], title: str, disclosed: bool = False
    ) -> bool:
        """Add `indices` to the labeling set.

        In Frame Mode, expands `indices` to every companion instance
        sharing a frame with any of them, and — unless `disclosed` is
        True (the caller already showed the user the full expansion) —
        confirms via a dialog before committing. Returns True if
        anything was added or there was nothing to add, False if the
        user canceled a Frame Mode confirmation.
        """
        to_add = {int(idx) for idx in indices if int(idx) not in self.labeling_frames}
        if not to_add:
            return True

        companion_count = 0
        frame_count = len(to_add)
        if self.config.frame_mode:
            expanded, frame_count = self._frame_expansion(to_add)
            companions = expanded - to_add
            companion_count = len(companions)
            to_add = expanded - self.labeling_frames
            if companions and not disclosed:
                reply = QMessageBox.question(
                    self,
                    title,
                    f"This will add {frame_count} frame(s) comprising "
                    f"{len(to_add)} total instance(s), including "
                    f"{companion_count} companion instance(s), to the "
                    "labeling set. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply != QMessageBox.Yes:
                    return False

        self.labeling_frames.update(to_add)
        self._populate_frames()
        self._select_frame_in_list(self.current_index)
        QMessageBox.information(
            self, title, f"Added {len(to_add)} frame(s) to labeling set."
        )
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -k "frame_expansion or add_indices_to_labeling" -v`
Expected: 5 passed

- [ ] **Step 5: Run the full existing main_window test file to confirm no regressions**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -v`
Expected: all pass (existing `open_smart_select` still calls `self._add_indices_to_labeling(picked, "Smart Select")` with the old two-arg signature — this still works since `disclosed` defaults to `False`; Task 10 will change this call site to pass `disclosed=True`)

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/posekit/gui/main_window.py tests/test_posekit_main_window.py
git commit -m "feat: add frame-mode expansion and confirmation to posekit's shared labeling-set commit path"
```

---

### Task 5: Manual labeling (`save_current`)

**Files:**
- Modify: `src/hydra_suite/posekit/gui/main_window.py:3428-3468` (`save_current`)
- Test: `tests/test_posekit_main_window.py` (add cases)

**Interfaces:**
- Consumes: `MainWindow._frame_expansion` (Task 4).
- Produces: no new public interface — behavior change only.

Context: `self.labeling_frames` is not mutated directly by `save_current()` today. Instead, `_populate_frames()` (called when `refresh_ui=True`) auto-promotes any frame with a saved label file into `self.labeling_frames` (`"Labeled frames always go to labeling list"`, `if it["is_saved"]: self.labeling_frames.add(idx)`). This task adds the Frame Mode confirmation *before* the label file is written, and explicitly adds companions afterward (since `_populate_frames()`'s auto-promotion only covers the just-saved frame itself, not its companions, which may not yet have labels).

The current method body (verbatim, for reference):

```python
    def save_current(self: object, refresh_ui: object = True) -> object:
        """save_current method documentation."""
        if self._ann is None:
            return
        # Keep cache in sync
        self._cache_current_frame()
        logger.debug(
            "Save current frame=%d refresh_ui=%s", self.current_index, refresh_ui
        )
        img_path = self.image_paths[self.current_index]
        label_path = self._label_path_for(img_path)

        cls = int(self.class_combo.currentIndex())
        self._ann.cls = cls

        # Convert keypoints to image pixel space before saving.
        kpts_save, w, h = self._kpts_to_save_space(self._ann.kpts, img_path)

        bbox = compute_bbox_from_kpts(kpts_save, self.project.bbox_pad_frac, w, h)

        save_yolo_pose_label(
            label_path=label_path,
            cls=cls,
            img_w=w,
            img_h=h,
            kpts_px=kpts_save,
            bbox_xyxy_px=bbox,
            pad_frac=self.project.bbox_pad_frac,
        )
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
        self._dirty = False

        # Only refresh UI if we're staying on the current frame
        if refresh_ui:
            self._populate_frames()
            self._select_frame_in_list(self.current_index)

        self.statusBar().showMessage(f"Saved: {label_path.name}", 2000)
        self._set_saved_status()
        self.save_project()
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_posekit_main_window.py`:

```python
def _make_save_current_window(monkeypatch, frame_mode, current_index=0):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    save_calls = []
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.save_yolo_pose_label",
        lambda **kwargs: save_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.compute_bbox_from_kpts",
        lambda *a, **k: (0, 0, 1, 1),
    )

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=frame_mode),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=current_index,
        _ann=FrameAnn(cls=0, bbox_xyxy=None, kpts=[]),
        _cache_current_frame=lambda: None,
        _label_path_for=lambda p: Path(f"/labels/{p.stem}.txt"),
        class_combo=_DummyCombo(),
        _kpts_to_save_space=lambda kpts, path: (kpts, 10, 10),
        project=SimpleNamespace(bbox_pad_frac=0.1),
        _autosave_timer=SimpleNamespace(isActive=lambda: False, stop=lambda: None),
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
        statusBar=lambda: SimpleNamespace(showMessage=lambda *a, **k: None),
        _set_saved_status=lambda: None,
        save_project=lambda: None,
        _load_ann_from_disk=lambda idx: FrameAnn(cls=0, bbox_xyxy=None, kpts=[]),
        _rebuild_canvas=lambda: None,
    )
    return window, save_calls


def test_save_current_frame_mode_confirms_and_adds_companions(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    window, save_calls = _make_save_current_window(monkeypatch, frame_mode=True)

    MainWindow.save_current(window)

    assert len(save_calls) == 1
    assert window.labeling_frames == {1}  # companion added explicitly


def test_save_current_frame_mode_cancel_discards_edits(monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )
    window, save_calls = _make_save_current_window(monkeypatch, frame_mode=True)
    reload_calls = []
    window._load_ann_from_disk = lambda idx: (
        reload_calls.append(idx) or FrameAnn(cls=0, bbox_xyxy=None, kpts=[])
    )

    MainWindow.save_current(window)

    assert save_calls == []
    assert window.labeling_frames == set()
    assert reload_calls == [0]


def test_save_current_individual_mode_unchanged(monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not confirm outside frame mode")
            )
        ),
    )
    window, save_calls = _make_save_current_window(monkeypatch, frame_mode=False)

    MainWindow.save_current(window)

    assert len(save_calls) == 1
    # Individual mode never explicitly adds anything; only _populate_frames'
    # own auto-promotion (not exercised by this fake) would do so.
    assert window.labeling_frames == set()


def test_save_current_already_in_labeling_set_skips_confirmation(monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not confirm when frame already in labeling set")
            )
        ),
    )
    window, save_calls = _make_save_current_window(monkeypatch, frame_mode=True)
    window.labeling_frames = {0}

    MainWindow.save_current(window)

    assert len(save_calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -k save_current -v`
Expected: FAIL — companions not added / confirmation not shown / edits not discarded (assertion errors comparing `labeling_frames` or `save_calls`)

- [ ] **Step 3: Write the implementation**

Replace `save_current` in `src/hydra_suite/posekit/gui/main_window.py` with:

```python
    def save_current(self: object, refresh_ui: object = True) -> object:
        """save_current method documentation."""
        if self._ann is None:
            return
        # Keep cache in sync
        self._cache_current_frame()
        logger.debug(
            "Save current frame=%d refresh_ui=%s", self.current_index, refresh_ui
        )
        img_path = self.image_paths[self.current_index]
        label_path = self._label_path_for(img_path)

        companions: set = set()
        if self.config.frame_mode and self.current_index not in self.labeling_frames:
            expanded, frame_count = self._frame_expansion({self.current_index})
            companions = expanded - {self.current_index}
            if companions:
                reply = QMessageBox.question(
                    self,
                    "Add frame to labeling set",
                    f"This will add {frame_count} frame(s) comprising "
                    f"{len(expanded)} total instance(s), including "
                    f"{len(companions)} companion instance(s), to the "
                    "labeling set. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply != QMessageBox.Yes:
                    self._ann = self._load_ann_from_disk(self.current_index)
                    self._rebuild_canvas()
                    self._dirty = False
                    self.statusBar().showMessage(
                        "Save canceled — edits discarded.", 3000
                    )
                    return

        cls = int(self.class_combo.currentIndex())
        self._ann.cls = cls

        # Convert keypoints to image pixel space before saving.
        kpts_save, w, h = self._kpts_to_save_space(self._ann.kpts, img_path)

        bbox = compute_bbox_from_kpts(kpts_save, self.project.bbox_pad_frac, w, h)

        save_yolo_pose_label(
            label_path=label_path,
            cls=cls,
            img_w=w,
            img_h=h,
            kpts_px=kpts_save,
            bbox_xyxy_px=bbox,
            pad_frac=self.project.bbox_pad_frac,
        )
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
        self._dirty = False

        if companions:
            self.labeling_frames.update(companions)

        # Only refresh UI if we're staying on the current frame
        if refresh_ui:
            self._populate_frames()
            self._select_frame_in_list(self.current_index)

        self.statusBar().showMessage(f"Saved: {label_path.name}", 2000)
        self._set_saved_status()
        self.save_project()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -k save_current -v`
Expected: 4 passed

- [ ] **Step 5: Run the full main_window test file**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -v`
Expected: all pass, including the pre-existing `test_load_frame_defers_previous_frame_list_refresh_after_save` (unaffected by this change)

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/posekit/gui/main_window.py tests/test_posekit_main_window.py
git commit -m "feat: frame-mode confirmation and companion expansion for manual labeling"
```

---

### Task 6: Unlabeled → Labeling fix

**Files:**
- Modify: `src/hydra_suite/posekit/gui/main_window.py:2686-2696` (`_move_unlabeled_to_labeling`)
- Test: `tests/test_posekit_main_window.py` (add cases)

**Interfaces:**
- Consumes: `MainWindow._add_indices_to_labeling` (Task 4), `MainWindow._collect_selected_indices` (existing, line 3965), `MainWindow._matches_current_source`/`_is_labeled` (existing).
- Produces: no new public interface — behavior change only.

Current method body (verbatim):

```python
    def _move_unlabeled_to_labeling(self):
        """Move unlabeled frames from the current source into the labeling set."""
        for idx, img_path in enumerate(self.image_paths):
            if (
                self._matches_current_source(idx)
                and not self._is_labeled(img_path)
                and idx not in self.labeling_frames
            ):
                self.labeling_frames.add(idx)
        self._populate_frames()
        self._select_frame_in_list(self.current_index, trigger_load=False)
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_posekit_main_window.py`:

```python
def test_move_unlabeled_to_labeling_only_moves_selected(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=False),
        image_paths=[Path("a.png"), Path("b.png"), Path("c.png")],
        labeling_frames=set(),
        current_index=0,
        _matches_current_source=lambda idx: True,
        _is_labeled=lambda p: False,
        _collect_selected_indices=lambda: [1],
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
    )

    MainWindow._move_unlabeled_to_labeling(window)

    assert window.labeling_frames == {1}


def test_move_unlabeled_to_labeling_frame_mode_expands_selection(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        current_index=0,
        _matches_current_source=lambda idx: True,
        _is_labeled=lambda p: False,
        _collect_selected_indices=lambda: [0],
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
    )

    MainWindow._move_unlabeled_to_labeling(window)

    assert window.labeling_frames == {0, 1}


def test_move_unlabeled_to_labeling_no_selection_shows_info(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    info_calls = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: info_calls.append(a)),
    )

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=False),
        image_paths=[Path("a.png")],
        labeling_frames=set(),
        _matches_current_source=lambda idx: True,
        _is_labeled=lambda p: False,
        _collect_selected_indices=lambda: [],
    )

    MainWindow._move_unlabeled_to_labeling(window)

    assert len(info_calls) == 1
    assert window.labeling_frames == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -k move_unlabeled_to_labeling -v`
Expected: FAIL — today's method moves ALL unlabeled frames regardless of selection, so `test_move_unlabeled_to_labeling_only_moves_selected` fails (`labeling_frames` includes indices 1 and 2, not just 1)

- [ ] **Step 3: Write the implementation**

```python
    def _move_unlabeled_to_labeling(self):
        """Move the selected unlabeled frames into the labeling set."""
        candidates = [
            idx
            for idx in self._collect_selected_indices()
            if self._matches_current_source(idx)
            and not self._is_labeled(self.image_paths[idx])
            and idx not in self.labeling_frames
        ]
        if not candidates:
            QMessageBox.information(
                self, "No frames", "Select one or more unlabeled frames first."
            )
            return
        self._add_indices_to_labeling(candidates, "Unlabeled to Labeling")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -k move_unlabeled_to_labeling -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/posekit/gui/main_window.py tests/test_posekit_main_window.py
git commit -m "fix: unlabeled-to-labeling moves only selected frames, expands to companions in frame mode"
```

---

### Task 7: Unlabeled → All frame-mode guard

**Files:**
- Modify: `src/hydra_suite/posekit/gui/main_window.py:2698-2709` (`_move_unlabeled_to_all`)
- Test: `tests/test_posekit_main_window.py` (add cases)

**Interfaces:**
- Consumes: `group_indices_by_frame` (Task 1), `self.config.frame_mode` (Task 2/3).
- Produces: no new public interface — behavior change only.

Current method body (verbatim):

```python
    def _move_unlabeled_to_all(self):
        """Move unlabeled frames from the current source back to the source browser."""
        unlabeled_to_remove = []
        for idx in list(self.labeling_frames):
            if self._matches_current_source(idx) and not self._is_labeled(
                self.image_paths[idx]
            ):
                unlabeled_to_remove.append(idx)
        for idx in unlabeled_to_remove:
            self.labeling_frames.remove(idx)
        self._populate_frames()
        self._select_frame_in_list(self.current_index, trigger_load=False)
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_posekit_main_window.py`:

```python
def test_move_unlabeled_to_all_individual_mode_unchanged():
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=False),
        image_paths=[Path("a.png"), Path("b.png")],
        labeling_frames={0, 1},
        current_index=0,
        _matches_current_source=lambda idx: True,
        _is_labeled=lambda p: False,
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
    )

    MainWindow._move_unlabeled_to_all(window)

    assert window.labeling_frames == set()


def test_move_unlabeled_to_all_frame_mode_skips_partially_labeled_frame(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    info_calls = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: info_calls.append(a)),
    )

    is_labeled_map = {
        Path("did10000.jpg"): False,
        Path("did10001.jpg"): True,  # one instance on frame 1 is labeled
        Path("did20000.jpg"): False,
    }
    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg"), Path("did20000.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames={0, 1, 2},
        current_index=0,
        _matches_current_source=lambda idx: True,
        _is_labeled=lambda p: is_labeled_map[p],
        _populate_frames=lambda: None,
        _select_frame_in_list=lambda *a, **k: None,
    )

    MainWindow._move_unlabeled_to_all(window)

    # Frame (src_a, 1) has a labeled instance -> kept entirely (idx 0 and 1 stay).
    # Frame (src_a, 2) has no labeled instance -> reverted (idx 2 removed).
    assert window.labeling_frames == {0, 1}
    assert len(info_calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -k move_unlabeled_to_all -v`
Expected: FAIL — today's method reverts index 0 too (since it's individually unlabeled), ignoring that its frame-mate (index 1) is labeled

- [ ] **Step 3: Write the implementation**

```python
    def _move_unlabeled_to_all(self):
        """Move unlabeled frames from the current source back to the source browser."""
        unlabeled_to_remove = []
        skipped_frames = 0

        if self.config.frame_mode:
            groups = group_indices_by_frame(
                [p.name for p in self.image_paths],
                [self._source_id_for_index(i) for i in range(len(self.image_paths))],
            )
            idx_to_key = {i: key for key, idxs in groups.items() for i in idxs}
            candidate_keys = set()
            for idx in self.labeling_frames:
                if self._matches_current_source(idx) and not self._is_labeled(
                    self.image_paths[idx]
                ):
                    key = idx_to_key.get(idx)
                    if key is not None:
                        candidate_keys.add(key)
            for key in candidate_keys:
                frame_indices = groups[key]
                if any(self._is_labeled(self.image_paths[i]) for i in frame_indices):
                    skipped_frames += 1
                    continue
                unlabeled_to_remove.extend(
                    i for i in frame_indices if i in self.labeling_frames
                )
        else:
            for idx in list(self.labeling_frames):
                if self._matches_current_source(idx) and not self._is_labeled(
                    self.image_paths[idx]
                ):
                    unlabeled_to_remove.append(idx)

        for idx in unlabeled_to_remove:
            self.labeling_frames.discard(idx)
        self._populate_frames()
        self._select_frame_in_list(self.current_index, trigger_load=False)
        if skipped_frames:
            QMessageBox.information(
                self,
                "Frames kept",
                f"{skipped_frames} frame(s) were kept in the labeling set "
                "because at least one instance on each is already labeled.",
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -k move_unlabeled_to_all -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/posekit/gui/main_window.py tests/test_posekit_main_window.py
git commit -m "feat: frame-mode guard for unlabeled-to-all revert action"
```

---

### Task 8: Random Selection frame-mode change

**Files:**
- Modify: `src/hydra_suite/posekit/gui/main_window.py:2711-2742` (`_add_random_to_labeling`)
- Test: `tests/test_posekit_main_window.py` (add cases)

**Interfaces:**
- Consumes: `group_indices_by_frame` (Task 1), `MainWindow._add_indices_to_labeling` (Task 4).
- Produces: no new public interface — behavior change only.

Current method body (verbatim):

```python
    def _add_random_to_labeling(self):
        """Add random unlabeled frames from All Frames list to labeling set."""
        import random

        count = self.spin_random_count.value()

        # Get all unlabeled frames from the current source browser.
        candidates = []
        for idx, img_path in enumerate(self.image_paths):
            if (
                self._matches_current_source(idx)
                and not self._is_labeled(img_path)
                and idx not in self.labeling_frames
            ):
                candidates.append(idx)

        if not candidates:
            QMessageBox.information(
                self, "No frames", "No unlabeled frames available in All Frames list."
            )
            return

        # Randomly select up to 'count' frames
        to_add = random.sample(candidates, min(count, len(candidates)))
        for idx in to_add:
            self.labeling_frames.add(idx)

        self._populate_frames()
        self._select_frame_in_list(self.current_index, trigger_load=False)
        QMessageBox.information(
            self, "Added frames", f"Added {len(to_add)} frames to labeling set."
        )
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_posekit_main_window.py`:

```python
def test_add_random_to_labeling_individual_mode_uses_add_indices_helper(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    monkeypatch.setattr("random.sample", lambda population, k: population[:k])
    added = []

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=False),
        image_paths=[Path("a.png"), Path("b.png"), Path("c.png")],
        labeling_frames=set(),
        spin_random_count=SimpleNamespace(value=lambda: 2),
        _matches_current_source=lambda idx: True,
        _is_labeled=lambda p: False,
        _add_indices_to_labeling=lambda indices, title: added.append(
            (list(indices), title)
        )
        or True,
    )

    MainWindow._add_random_to_labeling(window)

    assert added == [([0, 1], "Random Selection")]


def test_add_random_to_labeling_frame_mode_samples_frame_ids(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    monkeypatch.setattr("random.sample", lambda population, k: population[:k])
    added = []

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=True),
        image_paths=[Path("did10000.jpg"), Path("did10001.jpg"), Path("did20000.jpg")],
        _source_id_for_index=lambda idx: "src_a",
        labeling_frames=set(),
        spin_random_count=SimpleNamespace(value=lambda: 1),
        _matches_current_source=lambda idx: True,
        _is_labeled=lambda p: False,
        _add_indices_to_labeling=lambda indices, title: added.append(
            (sorted(indices), title)
        )
        or True,
    )

    MainWindow._add_random_to_labeling(window)

    # Frame keys are sorted by dict insertion order from group_indices_by_frame;
    # with only one requested and frame (src_a, 1) inserted first (indices 0/1
    # appear before index 2 in image_paths), it is the one sampled.
    assert added == [([0, 1], "Random Selection")]


def test_add_random_to_labeling_no_candidates_shows_info(monkeypatch):
    from hydra_suite.posekit.config.schemas import PoseKitConfig

    info_calls = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: info_calls.append(a)),
    )

    window = SimpleNamespace(
        config=PoseKitConfig(frame_mode=False),
        image_paths=[Path("a.png")],
        labeling_frames={0},
        spin_random_count=SimpleNamespace(value=lambda: 5),
        _matches_current_source=lambda idx: True,
        _is_labeled=lambda p: False,
    )

    MainWindow._add_random_to_labeling(window)

    assert len(info_calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -k add_random_to_labeling -v`
Expected: FAIL — today's method never calls `_add_indices_to_labeling` and has no frame-mode branch

- [ ] **Step 3: Write the implementation**

```python
    def _add_random_to_labeling(self):
        """Add random unlabeled frames from All Frames list to labeling set."""
        import random

        count = self.spin_random_count.value()

        if self.config.frame_mode:
            groups = group_indices_by_frame(
                [p.name for p in self.image_paths],
                [self._source_id_for_index(i) for i in range(len(self.image_paths))],
            )
            candidate_keys = [
                key
                for key, idxs in groups.items()
                if any(
                    self._matches_current_source(i)
                    and not self._is_labeled(self.image_paths[i])
                    and i not in self.labeling_frames
                    for i in idxs
                )
            ]
            if not candidate_keys:
                QMessageBox.information(
                    self,
                    "No frames",
                    "No unlabeled frames available in All Frames list.",
                )
                return
            chosen_keys = random.sample(
                candidate_keys, min(count, len(candidate_keys))
            )
            to_add = [idx for key in chosen_keys for idx in groups[key]]
            self._add_indices_to_labeling(to_add, "Random Selection")
            return

        # Get all unlabeled frames from the current source browser.
        candidates = []
        for idx, img_path in enumerate(self.image_paths):
            if (
                self._matches_current_source(idx)
                and not self._is_labeled(img_path)
                and idx not in self.labeling_frames
            ):
                candidates.append(idx)

        if not candidates:
            QMessageBox.information(
                self, "No frames", "No unlabeled frames available in All Frames list."
            )
            return

        # Randomly select up to 'count' frames
        to_add = random.sample(candidates, min(count, len(candidates)))
        self._add_indices_to_labeling(to_add, "Random Selection")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -k add_random_to_labeling -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/posekit/gui/main_window.py tests/test_posekit_main_window.py
git commit -m "feat: frame-mode random selection samples whole frames"
```

---

### Task 9: Smart Select cluster-coverage algorithm

**Files:**
- Modify: `src/hydra_suite/posekit/core/extensions.py` (add new function near `pick_frames_stratified`)
- Test: `tests/test_posekit_extensions.py` (new, or add to an existing extensions test file if one already exists — check with `ls tests/ | grep -i extension` first; if none exists, create it)

**Interfaces:**
- Consumes: nothing new — pure function over arrays/dicts the caller supplies.
- Produces: `select_frames_by_cluster_coverage(cluster_id: np.ndarray, eligible_indices: list[int], frame_key_of_index: dict[int, Any], want_n_frames: int) -> list[Any]` — returns selected frame keys in selection order (most-coverage-first). Task 10 calls this and expands the returned keys via `group_indices_by_frame`'s output.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for PoseKit's Smart Select cluster-coverage frame selection."""

from __future__ import annotations

import numpy as np


def test_select_frames_by_cluster_coverage_maximizes_new_cluster_coverage():
    from hydra_suite.posekit.core.extensions import select_frames_by_cluster_coverage

    # 4 individuals across 2 frames. Frame "f1" spans clusters {0, 1}
    # (maximal single-frame coverage); frame "f2" spans only cluster {0}.
    eligible_indices = [0, 1, 2, 3]
    cluster_id = np.array([0, 1, 0, 0])
    frame_key_of_index = {0: "f1", 1: "f1", 2: "f2", 3: "f2"}

    selected = select_frames_by_cluster_coverage(
        cluster_id=cluster_id,
        eligible_indices=eligible_indices,
        frame_key_of_index=frame_key_of_index,
        want_n_frames=1,
    )

    assert selected == ["f1"]


def test_select_frames_by_cluster_coverage_continues_after_full_coverage():
    from hydra_suite.posekit.core.extensions import select_frames_by_cluster_coverage

    # 2 clusters total. Frame "f1" covers both; frame "f2" covers only
    # cluster 0 (fewer total distinct clusters) but is picked once
    # coverage is exhausted and budget remains.
    eligible_indices = [0, 1, 2]
    cluster_id = np.array([0, 1, 0])
    frame_key_of_index = {0: "f1", 1: "f1", 2: "f2"}

    selected = select_frames_by_cluster_coverage(
        cluster_id=cluster_id,
        eligible_indices=eligible_indices,
        frame_key_of_index=frame_key_of_index,
        want_n_frames=2,
    )

    assert selected == ["f1", "f2"]


def test_select_frames_by_cluster_coverage_deterministic_tie_break():
    from hydra_suite.posekit.core.extensions import select_frames_by_cluster_coverage

    # Two frames with identical coverage profiles -- tie-break must be
    # deterministic (smallest frame key wins).
    eligible_indices = [0, 1]
    cluster_id = np.array([0, 1])
    frame_key_of_index = {0: ("src", 5), 1: ("src", 2)}

    selected = select_frames_by_cluster_coverage(
        cluster_id=cluster_id,
        eligible_indices=eligible_indices,
        frame_key_of_index=frame_key_of_index,
        want_n_frames=1,
    )

    assert selected == [("src", 2)]


def test_select_frames_by_cluster_coverage_respects_budget_over_frame_count():
    from hydra_suite.posekit.core.extensions import select_frames_by_cluster_coverage

    eligible_indices = [0, 1, 2]
    cluster_id = np.array([0, 1, 2])
    frame_key_of_index = {0: "f1", 1: "f2", 2: "f3"}

    selected = select_frames_by_cluster_coverage(
        cluster_id=cluster_id,
        eligible_indices=eligible_indices,
        frame_key_of_index=frame_key_of_index,
        want_n_frames=2,
    )

    assert len(selected) == 2


def test_select_frames_by_cluster_coverage_empty_input_returns_empty():
    from hydra_suite.posekit.core.extensions import select_frames_by_cluster_coverage

    selected = select_frames_by_cluster_coverage(
        cluster_id=np.array([]),
        eligible_indices=[],
        frame_key_of_index={},
        want_n_frames=5,
    )

    assert selected == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_extensions.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_frames_by_cluster_coverage'`

- [ ] **Step 3: Write the implementation**

`src/hydra_suite/posekit/core/extensions.py` currently imports `from typing import Dict, List, Optional, Set, Tuple` (no `Any`) — change that line to `from typing import Any, Dict, List, Optional, Set, Tuple`.

Add to `src/hydra_suite/posekit/core/extensions.py` (near `pick_frames_stratified`):

```python
def select_frames_by_cluster_coverage(
    cluster_id: np.ndarray,
    eligible_indices: List[int],
    frame_key_of_index: Dict[int, Any],
    want_n_frames: int,
) -> List[Any]:
    """Greedily select frame keys to maximize embedding-cluster coverage.

    `cluster_id` is aligned with `eligible_indices` (cluster_id[i] is the
    cluster of eligible_indices[i], matching cluster_embeddings_cosine's
    output convention). `frame_key_of_index` maps a subset (or all) of
    `eligible_indices` to an opaque, comparable per-source frame key (see
    `hydra_suite.posekit.core.frame_grouping.group_indices_by_frame`).

    Diversity is scored on the true per-individual embeddings/clusters —
    never averaged across a frame's companions. Each round picks the
    frame covering the most not-yet-covered clusters; ties break on (a)
    total distinct clusters spanned, then (b) the smallest frame key, for
    determinism. Once all clusters are covered, remaining picks continue
    to use the same ranking (which naturally falls back to "most total
    distinct clusters" once new-coverage is zero for every candidate),
    so the full requested budget is used rather than under-filling.

    Returns the selected frame keys, in selection order.
    """
    want_n_frames = max(0, int(want_n_frames))
    if want_n_frames == 0 or not eligible_indices:
        return []

    frame_clusters: Dict[Any, set] = {}
    for local, global_idx in enumerate(eligible_indices):
        key = frame_key_of_index.get(global_idx)
        if key is None:
            continue
        frame_clusters.setdefault(key, set()).add(int(cluster_id[local]))

    covered: set = set()
    selected: List[Any] = []
    remaining = set(frame_clusters.keys())

    while remaining and len(selected) < want_n_frames:
        best_key = min(
            remaining,
            key=lambda k: (
                -len(frame_clusters[k] - covered),
                -len(frame_clusters[k]),
                k,
            ),
        )
        selected.append(best_key)
        covered.update(frame_clusters[best_key])
        remaining.discard(best_key)

    return selected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_extensions.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/posekit/core/extensions.py tests/test_posekit_extensions.py
git commit -m "feat: add greedy cluster-coverage frame selection for posekit smart select"
```

---

### Task 10: SmartSelectDialog UI changes

**Files:**
- Modify: `src/hydra_suite/posekit/gui/dialogs/exploration.py` (`SmartSelectDialog.__init__`, `_preview`)
- Modify: `src/hydra_suite/posekit/gui/main_window.py:3939-3951` (`open_smart_select`)
- Test: `tests/test_posekit_smart_select_dialog.py` (new)

**Interfaces:**
- Consumes: `group_indices_by_frame` (Task 1), `select_frames_by_cluster_coverage` (Task 9), `MainWindow._add_indices_to_labeling` (Task 4).
- Produces: `SmartSelectDialog.__init__` gains `frame_mode: bool = False` and `source_ids: Optional[List[Any]] = None` parameters; `SmartSelectDialog._frame_groups` and `SmartSelectDialog._idx_to_frame_key` attributes, reused by Task 11.

Current `open_smart_select` (verbatim):

```python
    def open_smart_select(self: object) -> object:
        """open_smart_select method documentation."""
        dlg = SmartSelectDialog(self, self.project, self.image_paths, self._is_labeled)
        if dlg.exec() != QDialog.Accepted or not getattr(dlg, "_did_add", False):
            return

        # If they added nothing, ignore
        picked = getattr(dlg, "selected_indices", None)
        if not picked:
            return

        self._add_indices_to_labeling(picked, "Smart Select")
```

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for PoseKit's SmartSelectDialog Frame Mode support."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.posekit.gui.dialogs.exploration import SmartSelectDialog  # noqa: E402


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def _make_dialog(qapp, frame_mode, image_paths, source_ids=None, out_root=None):
    project = SimpleNamespace(
        enhance_enabled=False, out_root=out_root or Path("/tmp/_unused_out_root")
    )
    return SmartSelectDialog(
        None,
        project,
        image_paths,
        lambda p: False,
        frame_mode=frame_mode,
        source_ids=source_ids or [None] * len(image_paths),
    )


def test_smart_select_dialog_frame_mode_disables_stratified_controls(qapp):
    dialog = _make_dialog(
        qapp, True, [Path("did10000.jpg"), Path("did10001.jpg")], ["src_a", "src_a"]
    )

    assert dialog.min_per_spin.isEnabled() is False
    assert dialog.strategy_combo.isEnabled() is False
    assert dialog.min_per_spin.toolTip() == (
        "Not used in Frame Mode — frame selection uses greedy "
        "cluster-coverage instead of per-cluster quotas."
    )


def test_smart_select_dialog_individual_mode_controls_stay_enabled(qapp):
    dialog = _make_dialog(qapp, False, [Path("a.png"), Path("b.png")])

    assert dialog.min_per_spin.isEnabled() is True
    assert dialog.strategy_combo.isEnabled() is True


def test_smart_select_dialog_preview_renders_one_line_per_frame(qapp, tmp_path):
    dialog = _make_dialog(
        qapp,
        True,
        [Path("did10000.jpg"), Path("did10001.jpg"), Path("did20000.jpg")],
        ["src_a", "src_a", "src_a"],
        out_root=tmp_path,
    )
    dialog._eligible_indices = [0, 1, 2]
    # Non-degenerate, distinguishable embeddings: idx0/idx2 identical
    # (frame (src_a, 1) is internally similar), idx1 orthogonal to both
    # (a distinct, separable cluster) -- avoids NaN cosine distances that
    # all-zero vectors would produce.
    dialog._emb = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    dialog.n_spin.setValue(1)
    dialog.k_spin.setValue(2)

    dialog._preview()

    text = dialog.preview.toPlainText()
    assert text.startswith("[frame 1] covers clusters")
    assert "2 instances" in text
    assert sorted(dialog.selected_indices) == [0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_smart_select_dialog.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'frame_mode'`

- [ ] **Step 3: Write the implementation**

In `src/hydra_suite/posekit/gui/dialogs/exploration.py`, add the import near the top:

```python
from hydra_suite.posekit.core.frame_grouping import group_indices_by_frame
```

Change `SmartSelectDialog.__init__`'s signature and add frame-mode setup. Replace:

```python
    def __init__(self, parent, project, image_paths: List[Path], is_labeled_fn) -> None:
        super().__init__(parent)
        self.setWindowTitle("Smart Select (Embeddings)")
        self.setMinimumSize(QSize(720, 420))
        self.project = project
        self.image_paths = image_paths
        self.is_labeled_fn = is_labeled_fn

        self._emb = None
        self._eligible_indices = None
        self._cluster = None

        layout = QVBoxLayout(self)
```

with:

```python
    def __init__(
        self,
        parent,
        project,
        image_paths: List[Path],
        is_labeled_fn,
        frame_mode: bool = False,
        source_ids: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Smart Select (Embeddings)")
        self.setMinimumSize(QSize(720, 420))
        self.project = project
        self.image_paths = image_paths
        self.is_labeled_fn = is_labeled_fn
        self.frame_mode = frame_mode
        self._frame_groups = group_indices_by_frame(
            [p.name for p in image_paths],
            source_ids if source_ids is not None else [None] * len(image_paths),
        )
        self._idx_to_frame_key = {
            i: key for key, idxs in self._frame_groups.items() for i in idxs
        }

        self._emb = None
        self._eligible_indices = None
        self._cluster = None

        layout = QVBoxLayout(self)

        if self.frame_mode:
            lbl_frame_mode = QLabel("Frame Mode is ON — results grouped by source frame")
            lbl_frame_mode.setStyleSheet("font-weight: bold;")
            layout.addWidget(lbl_frame_mode)
```

`src/hydra_suite/posekit/gui/dialogs/exploration.py` currently imports `from typing import List, Optional` (no `Any`) — change that line to `from typing import Any, List, Optional`.

Immediately after the `sel_row` block that creates `self.strategy_combo` (right after `layout.addLayout(sel_row)` and before `self.k_spin.valueChanged.connect(self._update_min_frames)`), add:

```python
        if self.frame_mode:
            self.min_per_spin.setEnabled(False)
            self.min_per_spin.setToolTip(
                "Not used in Frame Mode — frame selection uses greedy "
                "cluster-coverage instead of per-cluster quotas."
            )
            self.strategy_combo.setEnabled(False)
            self.strategy_combo.setToolTip(
                "Not used in Frame Mode — frame selection uses greedy "
                "cluster-coverage instead of per-cluster quotas."
            )
```

Now modify `_preview()` to dispatch to the new algorithm in Frame Mode. Insert a frame-mode branch immediately after `self._cluster = cluster` / `self._autosave_clusters()` / `self.btn_explorer.setEnabled(True)` / `self.lbl_status.setText(...)` and *replace* the `pick_frames_stratified` call plus the preview-text-building block. The method becomes:

```python
    def _preview(self):
        if self._emb is None or self._eligible_indices is None:
            return
        n = int(self.n_spin.value())
        k = int(self.k_spin.value())
        min_per = int(self.min_per_spin.value())
        strategy = self.strategy_combo.currentText().strip()
        cluster_method = self.cluster_method_combo.currentText().strip()

        n_frames = len(self._eligible_indices)
        if cluster_method == "hierarchical" and n_frames > 2500:
            reply = QMessageBox.warning(
                self,
                "Large Dataset Warning",
                f"You have {n_frames} frames selected.\n\n"
                "Hierarchical clustering may be slow or memory-intensive for large datasets.\n\n"
                "Would you like to switch to 'linkage' method instead?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.cluster_method_combo.setCurrentText("linkage")
                cluster_method = "linkage"
            elif reply == QMessageBox.Cancel:
                return

        cluster = cluster_embeddings_cosine(
            self._emb, k=k, method=cluster_method, seed=0
        )
        self._cluster = cluster
        self._autosave_clusters()
        self.btn_explorer.setEnabled(True)
        self.lbl_status.setText("Clusters saved. Use Export → Split method to apply.")

        if self.frame_mode:
            frame_key_of_index = {
                idx: self._idx_to_frame_key[idx]
                for idx in self._eligible_indices
                if idx in self._idx_to_frame_key
            }
            selected_keys = select_frames_by_cluster_coverage(
                cluster_id=cluster,
                eligible_indices=self._eligible_indices,
                frame_key_of_index=frame_key_of_index,
                want_n_frames=n,
            )
            self.selected_indices = sorted(
                idx for key in selected_keys for idx in self._frame_groups.get(key, [])
            )

            lines = []
            for key in selected_keys[:300]:
                frame_indices = self._frame_groups.get(key, [])
                cluster_ids_in_frame = sorted(
                    {
                        int(cluster[self._eligible_indices.index(i)])
                        for i in frame_indices
                        if i in self._eligible_indices
                    }
                )
                labeled_count = sum(
                    1
                    for i in frame_indices
                    if self.is_labeled_fn and self.is_labeled_fn(self.image_paths[i])
                )
                cluster_str = ",".join(str(c) for c in cluster_ids_in_frame)
                lines.append(
                    f"[frame {key[1]}] covers clusters {{{cluster_str}}} — "
                    f"{len(frame_indices)} instances ({labeled_count} labeled)"
                )
            if len(selected_keys) > 300:
                lines.append(f"... ({len(selected_keys) - 300} more)")
            self.preview.setPlainText("\n".join(lines))
            return

        picked = pick_frames_stratified(
            emb=self._emb,
            cluster_id=cluster,
            want_n=n,
            eligible_indices=self._eligible_indices,
            min_per_cluster=min_per,
            seed=0,
            strategy=strategy,
        )

        if self.cb_filter_duplicates.isChecked():
            try:
                # Try simple cosine similarity filtering if imported
                from ...core.extensions import filter_near_duplicates

                threshold = self.dup_threshold_spin.value()
                embeddings_picked = self._emb[
                    [self._eligible_indices.index(i) for i in picked]
                ]
                picked = filter_near_duplicates(
                    embeddings_picked,
                    list(range(len(picked))),
                    threshold=threshold,
                )
                picked = [picked[i] for i in picked]
            except Exception as e:
                logger.warning(f"Duplicate filtering failed: {e}")

        if self.cb_prefer_unlabeled.isChecked() and self.is_labeled_fn:
            labeled = [i for i in picked if self.is_labeled_fn(self.image_paths[i])]
            unlabeled = [
                i for i in picked if not self.is_labeled_fn(self.image_paths[i])
            ]
            picked = unlabeled + labeled[: max(0, n - len(unlabeled))]

        self.selected_indices = picked[:n]

        lines = []
        for idx in self.selected_indices[:300]:
            local = self._eligible_indices.index(idx)
            cid = int(cluster[local])
            labeled_str = (
                " [L]"
                if self.is_labeled_fn and self.is_labeled_fn(self.image_paths[idx])
                else ""
            )
            lines.append(f"[c{cid:03d}]{labeled_str} {self.image_paths[idx].name}")
        if len(self.selected_indices) > 300:
            lines.append(f"... ({len(self.selected_indices) - 300} more)")
        self.preview.setPlainText("\n".join(lines))
```

Add `select_frames_by_cluster_coverage` to the existing import of clustering helpers at the top of `exploration.py` (find the line importing `cluster_embeddings_cosine`/`pick_frames_stratified` and add it to that same import statement).

Finally, in `src/hydra_suite/posekit/gui/main_window.py`, update `open_smart_select`:

```python
    def open_smart_select(self: object) -> object:
        """open_smart_select method documentation."""
        dlg = SmartSelectDialog(
            self,
            self.project,
            self.image_paths,
            self._is_labeled,
            frame_mode=self.config.frame_mode,
            source_ids=[
                self._source_id_for_index(i) for i in range(len(self.image_paths))
            ],
        )
        if dlg.exec() != QDialog.Accepted or not getattr(dlg, "_did_add", False):
            return

        # If they added nothing, ignore
        picked = getattr(dlg, "selected_indices", None)
        if not picked:
            return

        # Smart Select's own preview already discloses the full frame
        # expansion, so no second confirmation is shown here.
        self._add_indices_to_labeling(picked, "Smart Select", disclosed=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_smart_select_dialog.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full main_window test file to confirm `open_smart_select` didn't regress**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_main_window.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/posekit/gui/dialogs/exploration.py src/hydra_suite/posekit/gui/main_window.py tests/test_posekit_smart_select_dialog.py
git commit -m "feat: frame-mode UI and cluster-coverage dispatch for smart select dialog"
```

---

### Task 11: Embedding Explorer frame-mode expansion

**Files:**
- Modify: `src/hydra_suite/posekit/gui/dialogs/exploration.py` (`SmartSelectDialog._open_explorer`)
- Test: `tests/test_posekit_smart_select_dialog.py` (add cases)

**Interfaces:**
- Consumes: `SmartSelectDialog._frame_groups`/`_idx_to_frame_key` (Task 10).
- Produces: no new public interface — behavior change only.

Context discovered during planning: `EmbeddingExplorerDialog` is instantiated *only* from `SmartSelectDialog._open_explorer()` — it has no other caller and no reference to `MainWindow`, so it cannot call the shared `_add_indices_to_labeling` directly. The functional equivalent of "routing through the shared commit path" is implemented here, in `_open_explorer()`, which already has access to `self._frame_groups`/`self._idx_to_frame_key` (Task 10) and merges the explorer's raw point selection into `self.selected_indices` — exactly the place a Frame Mode expansion + confirmation must happen, since (unlike Smart Select's own preview) the Explorer's selection summary never discloses companions.

Current `_open_explorer` (verbatim):

```python
    def _open_explorer(self):
        if self._emb is None:
            QMessageBox.warning(self, "No Embeddings", "Compute embeddings first.")
            return

        dialog = EmbeddingExplorerDialog(
            embeddings=self._emb,
            image_paths=[self.image_paths[i] for i in self._eligible_indices],
            cluster_ids=self._cluster,
            is_labeled_fn=self.is_labeled_fn if self.is_labeled_fn else lambda p: False,
            parent=self,
        )

        if dialog.exec_() == QDialog.Accepted:
            explorer_selected = dialog.selected_indices
            if explorer_selected:
                global_selected = [self._eligible_indices[i] for i in explorer_selected]
                existing = set(self.selected_indices)
                for idx in global_selected:
                    if idx not in existing:
                        self.selected_indices.append(idx)
                        existing.add(idx)
                QMessageBox.information(
                    self,
                    "Added from Explorer",
                    f"Added {len(explorer_selected)} frames.",
                )
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_posekit_smart_select_dialog.py`:

```python
def test_open_explorer_frame_mode_expands_and_confirms(qapp, monkeypatch):
    dialog = _make_dialog(
        qapp,
        True,
        [Path("did10000.jpg"), Path("did10001.jpg"), Path("did20000.jpg")],
        ["src_a", "src_a", "src_a"],
    )
    dialog._eligible_indices = [0, 1, 2]
    dialog._emb = np.zeros((3, 4), dtype=np.float32)
    dialog._cluster = np.array([0, 1, 0])
    dialog.selected_indices = []

    class _FakeExplorer:
        def __init__(self, **kwargs):
            pass

        def exec_(self):
            return QDialog.Accepted

        selected_indices = [0]  # local index 0 -> global index 0 (frame with companion 1)

    monkeypatch.setattr(
        "hydra_suite.posekit.gui.dialogs.exploration.EmbeddingExplorerDialog",
        _FakeExplorer,
    )
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    dialog._open_explorer()

    assert sorted(dialog.selected_indices) == [0, 1]


def test_open_explorer_frame_mode_cancel_adds_nothing(qapp, monkeypatch):
    dialog = _make_dialog(
        qapp,
        True,
        [Path("did10000.jpg"), Path("did10001.jpg")],
        ["src_a", "src_a"],
    )
    dialog._eligible_indices = [0, 1]
    dialog._emb = np.zeros((2, 4), dtype=np.float32)
    dialog._cluster = np.array([0, 1])
    dialog.selected_indices = []

    class _FakeExplorer:
        def __init__(self, **kwargs):
            pass

        def exec_(self):
            return QDialog.Accepted

        selected_indices = [0]

    monkeypatch.setattr(
        "hydra_suite.posekit.gui.dialogs.exploration.EmbeddingExplorerDialog",
        _FakeExplorer,
    )
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )

    dialog._open_explorer()

    assert dialog.selected_indices == []


def test_open_explorer_individual_mode_unchanged(qapp, monkeypatch):
    dialog = _make_dialog(qapp, False, [Path("a.png"), Path("b.png")])
    dialog._eligible_indices = [0, 1]
    dialog._emb = np.zeros((2, 4), dtype=np.float32)
    dialog._cluster = np.array([0, 1])
    dialog.selected_indices = []

    class _FakeExplorer:
        def __init__(self, **kwargs):
            pass

        def exec_(self):
            return QDialog.Accepted

        selected_indices = [0]

    monkeypatch.setattr(
        "hydra_suite.posekit.gui.dialogs.exploration.EmbeddingExplorerDialog",
        _FakeExplorer,
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("must not confirm outside frame mode")
            )
        ),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    dialog._open_explorer()

    assert dialog.selected_indices == [0]
```

Add the required imports to `tests/test_posekit_smart_select_dialog.py`: `from PySide6.QtWidgets import QDialog, QMessageBox`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_smart_select_dialog.py -k open_explorer -v`
Expected: FAIL — today's `_open_explorer` never expands to companions and never confirms

- [ ] **Step 3: Write the implementation**

Replace `_open_explorer` in `src/hydra_suite/posekit/gui/dialogs/exploration.py` with:

```python
    def _open_explorer(self):
        if self._emb is None:
            QMessageBox.warning(self, "No Embeddings", "Compute embeddings first.")
            return

        dialog = EmbeddingExplorerDialog(
            embeddings=self._emb,
            image_paths=[self.image_paths[i] for i in self._eligible_indices],
            cluster_ids=self._cluster,
            is_labeled_fn=self.is_labeled_fn if self.is_labeled_fn else lambda p: False,
            parent=self,
        )

        if dialog.exec_() == QDialog.Accepted:
            explorer_selected = dialog.selected_indices
            if explorer_selected:
                global_selected = {
                    self._eligible_indices[i] for i in explorer_selected
                }

                companion_count = 0
                frame_count = len(global_selected)
                if self.frame_mode:
                    keys = {
                        self._idx_to_frame_key[i]
                        for i in global_selected
                        if i in self._idx_to_frame_key
                    }
                    expanded = {
                        idx for key in keys for idx in self._frame_groups.get(key, [])
                    }
                    companions = expanded - global_selected
                    companion_count = len(companions)
                    frame_count = len(keys)
                    if companions:
                        reply = QMessageBox.question(
                            self,
                            "Added from Explorer",
                            f"This will add {frame_count} frame(s) comprising "
                            f"{len(expanded)} total instance(s), including "
                            f"{companion_count} companion instance(s), to the "
                            "labeling set. Continue?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes,
                        )
                        if reply != QMessageBox.Yes:
                            return
                    global_selected = expanded

                existing = set(self.selected_indices)
                for idx in global_selected:
                    if idx not in existing:
                        self.selected_indices.append(idx)
                        existing.add(idx)
                QMessageBox.information(
                    self,
                    "Added from Explorer",
                    f"Added {len(global_selected)} instance(s) from "
                    f"{frame_count} frame(s).",
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_smart_select_dialog.py -k open_explorer -v`
Expected: 3 passed

- [ ] **Step 5: Run the full smart-select dialog test file**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/test_posekit_smart_select_dialog.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/posekit/gui/dialogs/exploration.py tests/test_posekit_smart_select_dialog.py
git commit -m "feat: frame-mode expansion and confirmation for smart select embedding explorer"
```

---

### Task 12: Full-suite regression pass

**Files:** none (verification only)

**Interfaces:** none.

- [ ] **Step 1: Run every PoseKit test file together**

Run: `source /Users/neurorishika/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps && PYTHONPATH=src python -m pytest tests/ -k posekit -v --ignore=tests/test_identity_postprocess.py`
Expected: all pass (baseline before this feature was 17 passed; this run should show 17 + all new tests added across Tasks 1-11 passing, 0 failed)

- [ ] **Step 2: Run formatting and lint gates**

Run: `make format-check && make lint`
Expected: no errors on the files touched by this plan. If `make format` was run and touched unrelated pre-existing-drift files, revert those specific files with `git checkout -- <file>` before committing (per Global Constraints).

- [ ] **Step 3: Commit any formatting fixes scoped to this feature's files**

```bash
git add -u
git commit -m "chore: formatting pass for posekit frame mode feature" --allow-empty
```
