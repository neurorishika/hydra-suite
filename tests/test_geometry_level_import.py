from hydra_suite.detectkit.gui.models import OBBSource


def test_obbsource_level_defaults_to_obb():
    src = OBBSource(path="/x", name="s")
    assert src.level == "obb"


def test_obbsource_level_roundtrips():
    src = OBBSource(path="/x", name="s", level="polygon")
    assert OBBSource.from_dict(src.to_dict()).level == "polygon"


def test_obbsource_from_dict_missing_level_is_obb():
    # Simulates a pre-migration project JSON with no "level" key.
    legacy = {"path": "/x", "name": "s", "validated": True, "source_kind": "detectkit"}
    assert OBBSource.from_dict(legacy).level == "obb"


import json
from pathlib import Path

import numpy as np
from PIL import Image

from hydra_suite.detectkit.gui.source_import import materialize_detectkit_source


def _img(path: Path, w=20, h=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(path)


def test_import_yolo_obb_is_obb_level(tmp_path):
    root = tmp_path / "src"
    _img(root / "images" / "a.png")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n", encoding="utf-8"
    )
    (root / "classes.txt").write_text("object\n", encoding="utf-8")
    mat = materialize_detectkit_source(root, tmp_path / "proj", force_import=True)
    assert mat.level == "obb"


def test_import_yolo_detect_is_aabb_level(tmp_path):
    root = tmp_path / "src"
    _img(root / "images" / "a.png")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "classes.txt").write_text("object\n", encoding="utf-8")
    mat = materialize_detectkit_source(root, tmp_path / "proj", force_import=True)
    # detect input keeps aabb information: stored as an axis-aligned quad, level aabb.
    assert mat.level == "aabb"
    lines = (
        (Path(mat.canonical_path) / "labels" / "a.txt").read_text().strip().splitlines()
    )
    assert len(lines[0].split()) == 9  # class + 8 coords (quad), not cx cy w h


def test_import_coco_segmentation_preserved_as_polygon(tmp_path):
    root = tmp_path / "src"
    _img(root / "images" / "a.png", 20, 20)
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 20, "height": 20}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[2, 2, 18, 2, 18, 18, 10, 19, 2, 18]],
                "bbox": [2, 2, 16, 16],
            }
        ],
        "categories": [{"id": 1, "name": "object"}],
    }
    (root / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")
    mat = materialize_detectkit_source(root, tmp_path / "proj", force_import=True)
    assert mat.level == "polygon"
    line = (
        (Path(mat.canonical_path) / "labels" / "a.txt")
        .read_text()
        .strip()
        .splitlines()[0]
    )
    assert (
        len(line.split()) == 11
    )  # class + 5 points preserved (not collapsed to a quad)
