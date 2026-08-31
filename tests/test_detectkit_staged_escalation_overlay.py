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


def test_the_staged_layer_refreshes_through_the_same_path_as_every_other():
    """The escalation layer's refresh used to fire only incidentally, and
    its clear used to sit below an early return. Both are structural now:
    one _refresh_overlays call, one idempotent set_layer per key."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    assert "_refresh_overlays" in inspect.getsource(
        MainWindow._refresh_escalation_overlay
    )


def test_the_escalation_layer_is_cleared_even_when_the_frame_fails_to_load():
    """`if not ok: return` fired BEFORE the refresh, so navigating to a
    corrupt frame left the previous frame's magenta masks floating over the
    previous frame's pixmap with GT and predictions already gone.

    Structurally now: the remove_layer loop over PROVIDERS runs before the
    load_image bail, so no layer can outlive a failed frame change.
    """
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow.show_image)
    clear_at = source.index("self._canvas.remove_layer(provider.key)")
    bail_at = source.index("if not self._canvas.load_image(image_path):")
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
    """Accept/Reject must refresh the overlay DIRECTLY, not incidentally via
    a selection reset -- a selection-preserving refresh would leave accepted
    or rejected masks on screen with nothing calling for a redraw. This is
    now MainWindow._after_review_change, shared by every review handler
    (accept/reject/bulk/revert/rethreshold)."""
    import inspect

    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow._after_review_change)
    assert "_refresh_overlays" in source
    assert '"gt"' in source and '"escalation"' in source


def test_canvas_reports_the_loaded_image_size():
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from hydra_suite.detectkit.gui.canvas import OBBCanvas

    QApplication.instance() or QApplication([])
    canvas = OBBCanvas()
    assert canvas.image_size() is None
    canvas.set_image_array(np.zeros((37, 61, 3), dtype=np.uint8))
    assert canvas.image_size() == (37, 61)
