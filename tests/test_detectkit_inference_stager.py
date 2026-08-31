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


def test_classes_txt_is_copied_from_the_source(tmp_path, source):
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

    assert (Path(review.staged_path) / "classes.txt").read_text() == "ant\n"


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
