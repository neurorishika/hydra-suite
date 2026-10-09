#!/usr/bin/env python
"""Render the shared SAHI settings widget in every role and every DetectKit host.

Offscreen (``QT_QPA_PLATFORM=offscreen``) and hermetic: ``HYDRA_DATA_DIR`` /
``HYDRA_CONFIG_DIR`` point at a temp dir for the run, so no user setting or
published model leaks into the pictures. Writes PNGs into ``--out``:

* ``role_<role>.png`` / ``role_<role>_advanced.png`` -- the bare widget with a
  representative state (derived badges visible), Advanced collapsed/expanded;
* ``host_*.png`` -- the real DetectKit hosts at their natural size.

The TrackerKit host is added by slice S4b.

Usage::

    python tools/sahi_widget_gallery.py --out docs/superpowers/assets/sahi-unification
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DEFAULT_OUT = Path("docs/superpowers/assets/sahi-unification")


@contextlib.contextmanager
def _hermetic_dirs():
    """Point HYDRA_DATA_DIR / HYDRA_CONFIG_DIR at a temp dir, then restore."""
    saved = {key: os.environ.get(key) for key in ("HYDRA_DATA_DIR", "HYDRA_CONFIG_DIR")}
    with tempfile.TemporaryDirectory(prefix="sahi_gallery_") as tmp:
        root = Path(tmp)
        for key, sub in (("HYDRA_DATA_DIR", "data"), ("HYDRA_CONFIG_DIR", "config")):
            (root / sub).mkdir()
            os.environ[key] = str(root / sub)
        try:
            yield root
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _grab(widget, path: Path, app) -> Path:
    widget.show()
    for _ in range(3):
        app.processEvents()
    widget.grab().save(str(path))
    widget.hide()
    return path


def _grab_scroll_page(window, inner, path: Path, app) -> Path:
    """Grab the whole scroll page holding ``inner`` (not just the viewport)."""
    from PySide6.QtWidgets import QScrollArea, QTabWidget

    for tabs in window.findChildren(QTabWidget):
        for index in range(tabs.count()):
            if tabs.widget(index).isAncestorOf(inner):
                tabs.setCurrentIndex(index)
    scroll = inner.parentWidget()
    while scroll is not None and not isinstance(scroll, QScrollArea):
        scroll = scroll.parentWidget()
    window.show()
    for _ in range(3):
        app.processEvents()
    page = scroll.widget() if scroll is not None else window
    page.grab().save(str(path))
    window.hide()
    return path


# ---------------------------------------------------------------- roles


def _role_widget(role: str):
    from hydra_suite.utils.tiling_spec import TilingSpec
    from hydra_suite.widgets.slice_settings import SliceSettingsWidget

    titles = {
        "infer_yolo": "Sliced inference (SAHI)",
        "train_yolo": "Sliced dataset / inference (SAHI)",
        "train_sam3": "Tiling (SAHI geometry)",
        "escalate_sam3": "Tiling (SAHI)",
        "escalate_sam2": "Tiling (SAHI)",
    }
    w = SliceSettingsWidget(role=role, title=titles[role])
    if role == "infer_yolo":
        w.set_model_input_size(1024)
        w.set_spec(
            TilingSpec(
                enabled=True,
                geometry_mode="auto_object",
                object_tile_fractions=(0.1,),
                reference_body_px=48.0,
                overlap=0.2,
            ),
            extras={"merge_threshold": 0.5, "merge_policy_raw": "nmm"},
        )
        w.set_reference_body(48.0, "stamped")
        w.set_preview_frame_options([(2048, 1536, 12), (1920, 1080, 3)])
    elif role == "train_yolo":
        w.set_spec(
            TilingSpec(
                enabled=True,
                geometry_mode="auto_object",
                object_tile_fractions=(0.05, 0.1, 0.15, 0.2),
                reference_body_px=48.0,
                overlap=0.2,
            ),
            extras={
                "negative_tile_fraction": 0.15,
                "full_frame_mix": True,
                "balance_multiscale_loss": True,
                "balance_multiscale_loss_power": 0.5,
                "min_area_ratio": 0.25,
            },
        )
        w.set_preview_frame_options([(2048, 1536, 12)])
    elif role == "train_sam3":
        w.set_model_input_size(1008)
        w.set_spec(
            TilingSpec(
                enabled=True,
                geometry_mode="auto_object",
                object_tile_fractions=(),
                reference_body_px=48.0,
                overlap=0.25,
                fragment_policy="crowd",
                merge_policy="nms",
                merge_metric="polygon_iou",
            ),
            extras={
                "object_tile_fraction": 0.055,
                "keep_empty_tiles": True,
                "full_frame_mix": False,
                "min_area_ratio": 0.25,
                "tile_overlap": 0.25,
            },
        )
        w.set_preview_frame_options([(2048, 1536, 12)])
    elif role == "escalate_sam3":
        w.set_spec(
            TilingSpec(
                enabled=True,
                geometry_mode="auto_object",
                object_tile_fractions=(0.05,),
                overlap=0.5,
            ),
            extras={"merge_iou": 0.5, "seam_margin_px": 4},
        )
        w.set_reference_body(82.2, "dataset")
    else:
        w.set_spec(
            TilingSpec(
                enabled=True, geometry_mode="auto_object", object_tile_fractions=(0.1,)
            )
        )
        w.set_reference_body(40.0, "profile")
    return w


def render_roles(out: Path, app) -> list[Path]:
    from hydra_suite.widgets.slice_settings import ROLES

    written = []
    for role in ROLES:
        widget = _role_widget(role)
        for expanded, suffix in ((False, ""), (True, "_advanced")):
            widget.set_advanced_expanded(expanded)
            widget.adjustSize()
            written.append(_grab(widget, out / f"role_{role}{suffix}.png", app))
        widget.deleteLater()
    return written


# ---------------------------------------------------------------- hosts


def _project(root: Path):
    from hydra_suite.detectkit.gui.models import (
        DetectKitProject,
        OBBSource,
        SliceTrainingSettings,
    )

    project = DetectKitProject(project_dir=root / "project", class_names=["ant"])
    project.sources = [
        OBBSource(path=str(root / "colony_a"), name="colony_a", level="obb"),
        OBBSource(path=str(root / "colony_b"), name="colony_b", level="aabb"),
    ]
    project.slice_settings = SliceTrainingSettings(
        enabled=True,
        geometry_mode="auto_object",
        target_size_fractions=[0.05, 0.1, 0.15, 0.2],
        reference_body_px=48.0,
    )
    return project


def render_hosts(out: Path, app, root: Path) -> list[Path]:
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as sam3
    from hydra_suite.detectkit.gui.dialogs.escalate_sam2_dialog import (
        EscalateSam2Dialog,
    )
    from hydra_suite.detectkit.gui.dialogs.inference_settings import (
        InferenceSettingsDialog,
    )
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog
    from hydra_suite.detectkit.gui.models import InferenceRunSettings

    project = _project(root)
    written = []

    training = TrainingDialog(project)
    training.sam3_panel._probed_once = True  # no `conda run` probe thread
    training.slice_group.set_preview_frame_options([(2048, 1536, 12)])
    written.append(
        _grab_scroll_page(
            training, training.slice_group, out / "host_detectkit_training.png", app
        )
    )

    # The SAM3 panel inside the real (dark-themed) TrainingDialog, so the
    # disabled-state contrast is exercised under the app's own stylesheet.
    # Rendered enabled: on a machine without CUDA the availability probe
    # disables the whole panel, which is not what is under review.
    panel = training.sam3_panel
    panel._apply_availability(SimpleNamespace(usable=True, reason=""), "hydra-sam3")
    panel.slice_group.set_preview_frame_options([(2048, 1536, 12)])
    written.append(
        _grab_scroll_page(
            training, panel.slice_group, out / "host_sam3_training.png", app
        )
    )

    settings = InferenceRunSettings.from_project(project, confidence_threshold=0.25)
    inference = InferenceSettingsDialog(settings, settings, model_input_size=1024)
    inference.slice_widget.set_preview_frame_options([(2048, 1536, 12)])
    written.append(_grab(inference, out / "host_inference_settings.png", app))

    class _Available:  # skip the 3.45 GB download note: not what is shown
        usable, checkpoint_missing, reason = True, False, ""

    original_probe = sam3.probe_checkpoint
    sam3.probe_checkpoint = lambda *_a, **_k: _Available()
    try:
        escalation = sam3.SemanticEscalationDialog(
            project.sources,
            82.2,
            project=project,
            body_px_origin="the median longest side of your existing labels",
            body_px_source="dataset",
        )
        written.append(_grab(escalation, out / "host_sam3_escalation.png", app))
    finally:
        sam3.probe_checkpoint = original_probe

    sam2 = EscalateSam2Dialog(
        project.sources,
        project=project,
        reference_body_px=40.0,
        body_px_source="dataset",
    )
    sam2._tile_fraction.setValue(0.1)
    written.append(_grab(sam2, out / "host_sam2_escalation.png", app))
    return written


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with _hermetic_dirs() as root:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        written = render_roles(out, app)
        written += render_hosts(out, app, root)
    for path in written:
        print(path)
    return written


if __name__ == "__main__":
    main(sys.argv[1:])
