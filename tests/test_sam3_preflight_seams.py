"""Tests for the real (non-monkeypatched) implementations of preflight's
environment-probe seams -- Task-8 fix round 2, finding 2 and the disk-mount
minor fix. `test_sam3_preflight.py` always replaces these seams; these tests
exercise what ships in production.
"""

import json

from hydra_suite.training.sam3_lora import preflight as pf


def test_instance_count_reads_train_split_excluding_crowd(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "iscrowd": 0},
            {"id": 2, "image_id": 1, "iscrowd": 0},
            {"id": 3, "image_id": 1, "iscrowd": 1},
        ],
    }
    (train_dir / "_annotations.coco.json").write_text(
        json.dumps(coco), encoding="utf-8"
    )

    assert pf._instance_count(str(tmp_path)) == 2


def test_instance_count_zero_when_split_missing(tmp_path):
    assert pf._instance_count(str(tmp_path)) == 0


def test_instance_count_zero_when_json_unreadable(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "_annotations.coco.json").write_text("not json", encoding="utf-8")

    assert pf._instance_count(str(tmp_path)) == 0


def test_free_disk_gb_walks_up_to_existing_ancestor(tmp_path):
    missing = tmp_path / "does" / "not" / "exist" / "yet"

    # Must not raise even though `missing` does not exist, and must measure
    # the SAME filesystem as `tmp_path` (its nearest existing ancestor), not
    # `/` via `.anchor`.
    free_missing = pf._free_disk_gb(str(missing))
    free_existing = pf._free_disk_gb(str(tmp_path))

    assert free_missing == free_existing
    assert free_missing > 0
