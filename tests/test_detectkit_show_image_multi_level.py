"""Regression: show_image must render every geometry level, not just native."""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


def test_show_image_calls_multi_level_api_with_source_level_and_reviewed():
    """show_image must resolve the current source's level/reviewed and pass
    them to set_gt_detections_multi_level, not the old single-layer
    set_gt_detections."""
    from hydra_suite.detectkit.gui.main_window import MainWindow

    source = inspect.getsource(MainWindow.show_image)
    assert "set_gt_detections_multi_level" in source
    assert "set_gt_detections(" not in source  # the old single-layer call is gone
    assert "native_level" in source
    assert "reviewed" in source
    assert "GeometryLevel.from_str" in source
    assert "except ValueError" in source  # from_str must be guarded, see Step 3


def test_resolve_native_level_and_reviewed_reads_the_matching_source():
    """Behavioral check (not just source-text): the level/reviewed resolved
    for a given source_path must actually come from the OBBSource whose
    .path matches, not some other source or a wrong default."""
    from hydra_suite.detectkit.gui.main_window import _resolve_source_render_state
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.training.geometry_levels import GeometryLevel

    proj = DetectKitProject(class_names=["ant"])
    proj.sources = [
        OBBSource(path="/a", name="a", level="obb", reviewed=True),
        OBBSource(path="/b", name="b", level="polygon", reviewed=False),
    ]

    native_level, reviewed = _resolve_source_render_state(proj, "/b")
    assert native_level == GeometryLevel.POLYGON
    assert reviewed is False


def test_resolve_native_level_and_reviewed_defaults_when_source_missing():
    from hydra_suite.detectkit.gui.main_window import _resolve_source_render_state
    from hydra_suite.detectkit.gui.models import DetectKitProject
    from hydra_suite.training.geometry_levels import GeometryLevel

    proj = DetectKitProject(class_names=["ant"])
    native_level, reviewed = _resolve_source_render_state(proj, "/nonexistent")
    assert native_level == GeometryLevel.OBB
    assert reviewed is True


def test_resolve_native_level_and_reviewed_falls_back_on_unknown_level_string():
    """A hand-edited/future-version project JSON could carry a level string
    GeometryLevel.from_str doesn't recognize -- this must degrade to OBB
    with a warning, not crash show_image on every image selection."""
    from hydra_suite.detectkit.gui.main_window import _resolve_source_render_state
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
    from hydra_suite.training.geometry_levels import GeometryLevel

    proj = DetectKitProject(class_names=["ant"])
    proj.sources = [
        OBBSource(path="/c", name="c", level="not_a_real_level", reviewed=True)
    ]

    native_level, reviewed = _resolve_source_render_state(proj, "/c")
    assert native_level == GeometryLevel.OBB  # fallback, not a raised ValueError
    assert reviewed is True
