#!/usr/bin/env python
"""Render the shared SAHI settings widget in every role and every DetectKit host.

Offscreen (``QT_QPA_PLATFORM=offscreen``) and hermetic: ``HYDRA_DATA_DIR`` /
``HYDRA_CONFIG_DIR`` point at a temp dir for the run, so no user setting or
published model leaks into the pictures. Writes PNGs into ``--out``:

* ``role_<role>.png`` / ``role_<role>_advanced.png`` -- the bare widget with a
  representative state (derived badges visible), Advanced collapsed/expanded;
* ``host_*.png`` -- the real DetectKit hosts at their natural size, and the
  TrackerKit main window's "Find Animals" page (YOLO direct, SAHI on, a model
  sidecar with two calibration profiles): ``host_trackerkit.png`` on the
  primary profile with a loaded 2448 x 2048 video's frame size,
  ``host_trackerkit_custom.png`` on the custom-geometry profile with Advanced
  expanded and no video (the labelled example frame), and
  ``host_trackerkit_off.png`` with SAHI unticked (collapsed to the checkbox).
  TrackerKit uses the compact layout (``layout="compact"``: paired rows, one
  summary line) with the preview below the controls (``preview_position``).

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

    scroll = inner.parentWidget()
    while scroll is not None and not isinstance(scroll, QScrollArea):
        scroll = scroll.parentWidget()
    window.show()
    for _ in range(3):
        app.processEvents()
    # Select the tab AFTER showing: TrackerKit restores its first tab on
    # show, and a hidden page is never laid out (a stale, squeezed grab).
    for tabs in window.findChildren(QTabWidget):
        for index in range(tabs.count()):
            if tabs.widget(index).isAncestorOf(inner):
                tabs.setCurrentIndex(index)
    for _ in range(3):
        app.processEvents()
    if not inner.isVisible():
        raise RuntimeError("gallery: the page to grab is not shown")
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


TRACKERKIT_SIDECAR = {
    "schema_version": 2,
    "training_geometry": {
        "geometry_mode": "auto_object",
        "imgsz": 1024,
        "reference_body_px": 48.0,
        "object_tile_fraction": 0.1,
        "overlap": 0.2,
    },
    "primary_profile_id": "balanced",
    "profiles": [
        {
            "id": "balanced",
            "name": "Balanced",
            "settings": {
                "enabled": True,
                "geometry_mode": "auto_object",
                "object_tile_fraction": 0.1,
                "overlap": 0.2,
                "trained_body_px": 48.0,
            },
        },
        {
            "id": "fast",
            "name": "Fast scan",
            "settings": {
                "enabled": True,
                "geometry_mode": "custom",
                "slice_width": 1024,
                "slice_height": 800,
                "overlap": 0.1,
            },
        },
    ],
}


def _seed_trackerkit_model(root: Path) -> Path:
    """A stub direct-OBB model + two-profile sidecar in the temp model repo."""
    import json

    from hydra_suite.core.inference.model_paths import get_models_root_directory

    models_root = Path(get_models_root_directory())
    model = models_root / "obb" / "ant_direct.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("stub model", encoding="utf-8")
    registry = {
        "schema_version": 2,
        "entries": {
            "obb/ant_direct.pt": {
                "task_family": "obb",
                "usage_role": "obb_direct",
                "size": "26s",
                "species": "ant",
                "model_info": "ant_direct",
            }
        },
    }
    (models_root / "model_registry.json").write_text(
        json.dumps(registry), encoding="utf-8"
    )
    (model.parent / (model.name + ".slice_meta.json")).write_text(
        json.dumps(TRACKERKIT_SIDECAR), encoding="utf-8"
    )
    return model


def _assert_yolo_page(panel) -> None:
    """Fail loudly rather than publish a shot of the wrong detection page."""
    if panel.combo_detection_method.currentText() != "YOLO OBB" or (
        panel.stack_detection.currentIndex() != 1
    ):
        raise RuntimeError("TrackerKit gallery: the YOLO page is not shown")


def render_trackerkit(out: Path, app, root: Path) -> list[Path]:
    """TrackerKit's Find Animals page with the SAHI widget in its YOLO group."""
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    model = _seed_trackerkit_model(root)
    # The machine-global advanced config must neither leak in nor be written.
    saved = (MainWindow._save_advanced_config, MainWindow._load_advanced_config)
    MainWindow._save_advanced_config = lambda self: None
    MainWindow._load_advanced_config = lambda self: {}
    try:
        window = MainWindow()
    finally:
        MainWindow._save_advanced_config, MainWindow._load_advanced_config = saved
    window.resize(1500, 1000)
    window._show_workspace()
    panel = window._detection_panel
    window.tabs.setCurrentWidget(panel)
    panel.combo_detection_method.setCurrentIndex(1)  # YOLO
    panel._refresh_yolo_model_combo(preferred_model_path=str(model))
    window._set_yolo_model_selection(str(model))
    panel.combo_yolo_obb_mode.setCurrentIndex(0)  # Direct
    panel.apply_slice_meta_for_model(str(model))
    panel.chk_slice_enabled.setChecked(True)
    # What loading a 2448 x 2048 video does (session.py); the custom shot
    # below shows the labelled example frame instead (no video).
    panel.set_slice_preview_frame_size(2448, 2048)
    _assert_yolo_page(panel)
    written = [
        _grab_scroll_page(
            window, panel.slice_settings, out / "host_trackerkit.png", app
        )
    ]
    panel.set_slice_preview_frame_size(None, None)
    panel.combo_slice_profile.setCurrentIndex(
        panel.combo_slice_profile.findData("fast")
    )
    panel.slice_settings.btn_slice_advanced.setChecked(True)  # the user path
    _assert_yolo_page(panel)
    written.append(
        _grab_scroll_page(
            window, panel.slice_settings, out / "host_trackerkit_custom.png", app
        )
    )
    panel.chk_slice_enabled.setChecked(False)  # the user path: collapses
    _assert_yolo_page(panel)
    written.append(
        _grab_scroll_page(
            window, panel.slice_settings, out / "host_trackerkit_off.png", app
        )
    )
    window.close()
    window.deleteLater()
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
        written += render_trackerkit(out, app, root)
    for path in written:
        print(path)
    return written


if __name__ == "__main__":
    main(sys.argv[1:])
