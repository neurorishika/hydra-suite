import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.inference_stager import stage_predictions


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "sources" / "src"
    (root / "images" / "sub").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "classes.txt").write_text("ant\n")
    for rel in ("a.png", "sub/b.png"):
        Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(
            root / "images" / rel
        )
    return OBBSource(path=str(root), name="src", level="obb")


def _dets():
    return [
        {
            "class_id": 0,
            "polygon_px": [(10, 10), (50, 10), (50, 40), (10, 40)],
            "confidence": 0.9,
        }
    ]


def test_a_label_file_is_written_per_predicted_frame(tmp_path, source):
    per_image = {
        str(Path(source.path) / "images" / "a.png"): _dets(),
        str(Path(source.path) / "images" / "sub" / "b.png"): _dets(),
    }

    review = stage_predictions(
        source,
        tmp_path,
        per_image,
        model_path="/models/best.pt",
        inference_kind="obb_direct",
        confidence=0.4,
        device="mps",
    )

    labels = Path(review.staged_path) / "labels"
    assert (labels / "a.txt").is_file()
    assert (labels / "sub" / "b.txt").is_file()


def test_staged_paths_mirror_the_images_tree(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "sub" / "b.png"): _dets()}

    review = stage_predictions(
        source,
        tmp_path,
        per_image,
        model_path="/m.pt",
        inference_kind="obb_direct",
        confidence=0.4,
        device="cpu",
    )

    staged = sorted(
        p.relative_to(Path(review.staged_path) / "labels").as_posix()
        for p in (Path(review.staged_path) / "labels").rglob("*.txt")
    )
    assert staged == ["sub/b.txt"]


def test_the_producer_is_inference_and_the_level_follows_the_kind(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "a.png"): _dets()}

    obb = stage_predictions(
        source,
        tmp_path,
        per_image,
        model_path="/m.pt",
        inference_kind="obb_direct",
        confidence=0.4,
        device="cpu",
    )
    assert obb.producer == "inference"
    assert obb.target_level == "obb"

    source.staged_review = None
    seg = stage_predictions(
        source,
        tmp_path,
        per_image,
        model_path="/m.pt",
        inference_kind="sequential_segment",
        confidence=0.4,
        device="cpu",
    )
    assert seg.target_level == "polygon"


def test_run_json_records_the_model_confidence_and_device(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "a.png"): _dets()}

    review = stage_predictions(
        source,
        tmp_path,
        per_image,
        model_path="/models/best.pt",
        inference_kind="obb_direct",
        confidence=0.42,
        device="mps",
    )

    run = json.loads((Path(review.staged_path) / "run.json").read_text())
    assert run["producer"] == "inference"
    assert run["params"]["model_path"] == "/models/best.pt"
    assert run["params"]["confidence"] == 0.42
    assert run["params"]["device"] == "mps"


def test_classes_txt_is_written_from_the_project_not_the_source(tmp_path, source):
    """`class_id` in predictions indexes the PROJECT's class list.

    A source whose own `classes.txt` differs in ORDER from the project's is
    exactly the silent-corruption case: if the staged file were copied from
    the source (as it used to be), `resolve_staged_class_ids` would map
    staged->source by name against the wrong list and degenerate to
    identity, applying raw model ids as source ids.
    """
    # The source's own classes.txt (order differs from the project's).
    (Path(source.path) / "classes.txt").write_text("larva\nant\n")
    per_image = {str(Path(source.path) / "images" / "a.png"): _dets()}

    review = stage_predictions(
        source,
        tmp_path,
        per_image,
        model_path="/m.pt",
        inference_kind="obb_direct",
        confidence=0.4,
        device="cpu",
        class_names=["ant", "larva"],
    )

    assert (Path(review.staged_path) / "classes.txt").read_text() == "ant\nlarva\n"


def test_staging_lands_inside_the_projects_pending_escalations_dir(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "a.png"): _dets()}

    review = stage_predictions(
        source,
        tmp_path,
        per_image,
        model_path="/m.pt",
        inference_kind="obb_direct",
        confidence=0.4,
        device="cpu",
    )

    assert "pending_escalations" in Path(review.staged_path).parts


def test_frames_with_no_detections_are_not_staged(tmp_path, source):
    per_image = {
        str(Path(source.path) / "images" / "a.png"): [],
        str(Path(source.path) / "images" / "sub" / "b.png"): _dets(),
    }

    review = stage_predictions(
        source,
        tmp_path,
        per_image,
        model_path="/m.pt",
        inference_kind="obb_direct",
        confidence=0.4,
        device="cpu",
    )

    staged = list((Path(review.staged_path) / "labels").rglob("*.txt"))
    assert [p.name for p in staged] == ["b.txt"]


def test_images_outside_images_dir_are_skipped_not_staged_at_the_flat_fallback(
    tmp_path, source
):
    """An image outside the source's `images/` tree cannot be reviewed.

    `review_key_for_image` requires the image to be under `images/`, so a
    label staged for an out-of-tree image would be permanently unreachable
    by per-frame Accept/Reject, and `accept_all` would fail looking up its
    frame size. Skip it instead, exactly like an unreadable image.
    """
    stray = Path(source.path) / "stray.png"
    Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(stray)
    per_image = {
        str(stray): _dets(),
        str(Path(source.path) / "images" / "a.png"): _dets(),
    }

    review = stage_predictions(
        source,
        tmp_path,
        per_image,
        model_path="/m.pt",
        inference_kind="obb_direct",
        confidence=0.4,
        device="cpu",
    )

    staged = list((Path(review.staged_path) / "labels").rglob("*.txt"))
    assert [p.name for p in staged] == ["a.txt"]


def test_staging_only_out_of_tree_images_is_refused_rather_than_a_stuck_review(
    tmp_path, source
):
    stray = Path(source.path) / "stray.png"
    Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(stray)
    per_image = {str(stray): _dets()}

    with pytest.raises(RuntimeError, match="no detections"):
        stage_predictions(
            source,
            tmp_path,
            per_image,
            model_path="/m.pt",
            inference_kind="obb_direct",
            confidence=0.4,
            device="cpu",
        )

    assert source.staged_review is None


def test_staging_nothing_at_all_is_refused_rather_than_creating_a_dead_review(
    tmp_path, source
):
    """A zero-frame review would be unfinishable.

    `is_complete` needs total > 0, reject-all rejects nothing, and revert has
    no snapshot -- the user would be stuck with a review only a hand-edit of
    the project JSON could clear.
    """
    per_image = {str(Path(source.path) / "images" / "a.png"): []}

    with pytest.raises(RuntimeError, match="no detections"):
        stage_predictions(
            source,
            tmp_path,
            per_image,
            model_path="/m.pt",
            inference_kind="obb_direct",
            confidence=0.4,
            device="cpu",
        )

    assert source.staged_review is None


def test_staging_over_an_open_review_is_refused(tmp_path, source):
    per_image = {str(Path(source.path) / "images" / "a.png"): _dets()}
    stage_predictions(
        source,
        tmp_path,
        per_image,
        model_path="/m.pt",
        inference_kind="obb_direct",
        confidence=0.4,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="already has a staged review"):
        stage_predictions(
            source,
            tmp_path,
            per_image,
            model_path="/m.pt",
            inference_kind="obb_direct",
            confidence=0.4,
            device="cpu",
        )


def _wired_window(monkeypatch, tmp_path, predictions):
    """A DetectKitMainWindow with just enough stubbed to run the handler.

    Every modal is patched out: an unpatched QMessageBox in a GUI test hangs
    the suite rather than failing it. `_project` MUST be set -- the handler's
    first guard reads it, and a fresh window has it as None.
    """
    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication

    from hydra_suite.detectkit.gui import main_window as mw
    from hydra_suite.detectkit.gui.models import OBBSource

    # Repo Qt pattern, no pytest-qt (not installed). The caller is
    # responsible for window.deleteLater() -- see the tests below.
    QApplication.instance() or QApplication([])
    window = mw.DetectKitMainWindow()
    source = OBBSource(path=str(tmp_path / "src"), name="src")
    window._project = SimpleNamespace(
        project_dir=str(tmp_path),
        active_model_path="m.pt",
        sources=[source],
        class_names=["ant"],
    )
    window._dataset_predictions = dict(predictions)
    window._dataset_prediction_signature = ("sig", "m.pt")

    monkeypatch.setattr(window, "_current_source_obj", lambda: source)
    monkeypatch.setattr(window, "_dataset_signature", lambda settings: ("sig", "m.pt"))
    monkeypatch.setattr(
        window,
        "_effective_inference_settings",
        lambda settings: SimpleNamespace(device="mps"),
    )
    monkeypatch.setattr(
        window._tools_panel,
        "get_overlay_settings",
        lambda: SimpleNamespace(confidence_threshold=0.40),
    )
    monkeypatch.setattr(window, "_save_current_project", lambda: None)
    monkeypatch.setattr(window, "_sync_review_bar", lambda: None)
    monkeypatch.setattr(window, "_refresh_overlays", lambda keys=None: None)
    monkeypatch.setattr(
        mw,
        "detectkit_resolve_inference_models",
        lambda project, model_path: ("sequential_segment", "p.pt", "s.pt"),
    )
    monkeypatch.setattr(
        mw.QMessageBox, "information", staticmethod(lambda *a, **k: None)
    )
    monkeypatch.setattr(mw.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    return mw, window


def _det(conf):
    return {
        "class_id": 0,
        "polygon_px": [(0, 0), (10, 0), (10, 10), (0, 10)],
        "confidence": conf,
    }


def test_staging_action_refuses_when_no_predictions_are_held(monkeypatch, tmp_path):
    mw, window = _wired_window(monkeypatch, tmp_path, {})
    called: list = []
    monkeypatch.setattr(mw, "stage_predictions", lambda *a, **k: called.append(a))

    window._on_stage_predictions()
    window.deleteLater()

    assert called == []


def test_staging_action_stages_only_what_is_visible_at_the_slider(
    monkeypatch, tmp_path
):
    """The floor-retained candidates the user never saw must not be staged.

    _dataset_predictions is held at INFERENCE_CONFIDENCE_FLOOR (0.01) so the
    slider stays useful without re-running the model. Staging the raw dict
    would stage candidates the user never reviewed while run.json claimed
    the slider value.
    """
    mw, window = _wired_window(
        monkeypatch, tmp_path, {"/img/a.png": [_det(0.9), _det(0.02)]}
    )
    seen: dict = {}
    monkeypatch.setattr(
        mw,
        "stage_predictions",
        lambda src, project_dir, per_image, **kw: seen.update(
            per_image=per_image, kw=kw
        ),
    )

    window._on_stage_predictions()
    window.deleteLater()

    assert [d["confidence"] for d in seen["per_image"]["/img/a.png"]] == [0.9]
    assert seen["kw"]["confidence"] == 0.40


def test_staging_action_records_the_real_kind_and_device(monkeypatch, tmp_path):
    """OverlaySettings carries neither field; they come from the run's own
    resolution. A sequential_segment run staged as obb_direct would declare
    polygon labels at OBB level."""
    mw, window = _wired_window(monkeypatch, tmp_path, {"/img/a.png": [_det(0.9)]})
    seen: dict = {}
    monkeypatch.setattr(
        mw,
        "stage_predictions",
        lambda src, project_dir, per_image, **kw: seen.update(kw),
    )

    window._on_stage_predictions()
    window.deleteLater()

    assert seen["inference_kind"] == "sequential_segment"
    assert seen["device"] == "mps"
