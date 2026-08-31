"""The staged-escalation preview: seeing SAM3's masks before accepting them.

Until this existed, a staged escalation could only be accepted or rejected
sight-unseen -- the review dialog is a text list, and nothing in the GUI
ever parsed the staging directory's labels for display.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


def _write_staged(root: Path, rel: str, lines: list[str]) -> Path:
    path = root / "labels" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_staged_label_is_found_by_mirroring_the_source_images_tree(tmp_path):
    """A staging dir has labels/ but NO images/, so the plain
    find_label_for_image mirror strategy cannot fire there. The rel path has
    to come from the SOURCE's images/ tree."""
    from hydra_suite.detectkit.gui.utils import find_staged_label_for_image

    source = tmp_path / "src"
    (source / "images" / "sub").mkdir(parents=True)
    image = source / "images" / "sub" / "f0.png"
    image.write_bytes(b"")
    staged = tmp_path / "staged"
    expected = _write_staged(staged, "sub/f0.txt", ["0 0.5 0.5 0.2 0.2"])

    assert find_staged_label_for_image(image, str(source), str(staged)) == expected


def test_staged_label_falls_back_to_a_stem_match(tmp_path):
    from hydra_suite.detectkit.gui.utils import find_staged_label_for_image

    source = tmp_path / "src"
    (source / "images").mkdir(parents=True)
    image = source / "images" / "f0.png"
    image.write_bytes(b"")
    staged = tmp_path / "staged"
    expected = _write_staged(staged, "f0.txt", ["0 0.5 0.5 0.2 0.2"])

    assert find_staged_label_for_image(image, str(source), str(staged)) == expected


def test_no_staged_label_returns_none(tmp_path):
    from hydra_suite.detectkit.gui.utils import find_staged_label_for_image

    source = tmp_path / "src"
    (source / "images").mkdir(parents=True)
    image = source / "images" / "f0.png"
    image.write_bytes(b"")
    (tmp_path / "staged" / "labels").mkdir(parents=True)

    assert (
        find_staged_label_for_image(image, str(source), str(tmp_path / "staged"))
        is None
    )


def test_staged_class_names_come_from_the_staging_dirs_classes_txt(tmp_path):
    """The staged classes.txt holds the PROMPT, which is what the overlay
    label should read -- not the project's class list."""
    from hydra_suite.detectkit.gui.utils import staged_class_names

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "classes.txt").write_text("worker ant\n")
    assert staged_class_names(str(staged)) == ["worker ant"]
    assert staged_class_names(str(tmp_path / "missing")) == ["object"]


def test_show_image_draws_the_pending_escalation_and_clears_it_otherwise():
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow.show_image)
    assert "_refresh_escalation_overlay" in source

    refresh = inspect.getsource(MainWindow._refresh_escalation_overlay)
    # Cleared first, so a frame with no staged label cannot keep showing the
    # previous frame's masks.
    assert "clear_escalation_detections" in refresh
    assert "set_escalation_detections" in refresh
    assert "find_staged_label_for_image" in refresh


def test_the_escalation_layer_is_cleared_even_when_the_frame_fails_to_load():
    """`if not ok: return` fired BEFORE the refresh, so navigating to a
    corrupt frame left the previous frame's magenta masks floating over the
    previous frame's pixmap with GT and predictions already gone."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow.show_image)
    clear_at = source.index("clear_escalation_detections")
    bail_at = source.index("if not ok:")
    assert clear_at < bail_at


def test_the_overlay_does_not_decode_the_frame_a_third_time():
    """canvas.load_image already decoded it; a second decode for (h, w) cost
    ~100 ms per keypress on 4512^2 frames, and the escalation overlay added
    a third."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    assert "cv2.imread" not in inspect.getsource(MainWindow._refresh_escalation_overlay)
    assert "cv2.imread" not in inspect.getsource(MainWindow.show_image)


def test_reviewing_escalations_refreshes_the_overlay_directly():
    """Accept/Reject cleared the overlay only INCIDENTALLY, via the dataset
    panel resetting its selection to row 0. A selection-preserving refresh
    would have left accepted or rejected masks on screen with nothing
    anywhere calling for a redraw."""
    import inspect

    from hydra_suite.detectkit.gui import escalation_actions

    source = inspect.getsource(escalation_actions.on_review_escalations)
    assert "_refresh_escalation_overlay" in source


def test_canvas_reports_the_loaded_image_size():
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    QApplication.instance() or QApplication([])
    canvas = OBBCanvas()
    assert canvas.image_size() is None
    canvas.set_image_array(np.zeros((37, 61, 3), dtype=np.uint8))
    assert canvas.image_size() == (37, 61)


def test_the_overlay_renders_at_the_escalations_own_target_level():
    """PendingEscalation.target_level is load-bearing, not decorative.

    SAM2 can stage OBB (it converts existing boxes in place), and drawing an
    OBB quad as polygon-native gave it the polygon style (dotted + filled)
    AND a derived OBB of the very same quad -- a duplicated outline in the
    wrong style. The level has to come from the record.
    """
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    refresh = inspect.getsource(MainWindow._refresh_escalation_overlay)
    assert "_resolve_pending_level(pending)" in refresh
    assert "GeometryLevel.POLYGON" not in refresh  # no longer hardcoded


def test_pending_level_parses_the_record_and_degrades_on_junk():
    """target_level is an unvalidated string from project JSON, exactly like
    OBBSource.level -- a hand-edited or future-version file must not crash
    the overlay on every image selection."""
    from hydra_suite.detectkit.gui.main_window import _resolve_pending_level
    from hydra_suite.detectkit.gui.models import PendingEscalation
    from hydra_suite.training.geometry_levels import GeometryLevel

    assert (
        _resolve_pending_level(PendingEscalation(target_level="obb"))
        == GeometryLevel.OBB
    )
    assert (
        _resolve_pending_level(PendingEscalation(target_level="polygon"))
        == GeometryLevel.POLYGON
    )
    # Junk degrades to POLYGON: a staged mask is a polygon unless the record
    # says otherwise, and both escalation producers stage polygons by default.
    assert (
        _resolve_pending_level(PendingEscalation(target_level="not_a_level"))
        == GeometryLevel.POLYGON
    )
