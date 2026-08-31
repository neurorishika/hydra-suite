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


def test_escalation_overlay_is_a_polygon_native_layer():
    """SAM3/SAM2 stage POLYGON labels, so the native level is POLYGON and the
    OBB/AABB drawn under it are the derivations a promotion would produce."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    refresh = inspect.getsource(MainWindow._refresh_escalation_overlay)
    assert "GeometryLevel.POLYGON" in refresh
