from pathlib import Path

import cv2
import numpy as np

from hydra_suite.detectkit.jobs.direct_calibration import (
    EXHAUSTIVE_LABEL_WARNING,
    collect_evidence,
)

LABEL_LINE = "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"


def _dataset(tmp_path: Path, split: str, names: list[str]) -> Path:
    images = tmp_path / "images" / split
    labels = tmp_path / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for name in names:
        cv2.imwrite(str(images / f"{name}.png"), np.zeros((200, 300, 3), np.uint8))
        (labels / f"{name}.txt").write_text(LABEL_LINE)
    yaml = tmp_path / "data.yaml"
    yaml.write_text(
        f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n"
    )
    return yaml


def test_val_split_is_the_default_evidence(tmp_path):
    _dataset(tmp_path, "train", ["a", "b", "c"])
    yaml = _dataset(tmp_path, "val", ["v0", "v1"])
    evidence = collect_evidence(dataset_yaml=yaml, sources=[])
    assert evidence.split == "val"
    assert len(evidence.frames) == 2 and evidence.instances == 2
    assert evidence.size_range == ((200, 300), (200, 300))


def test_missing_val_split_falls_back_to_train_and_reports_it(tmp_path):
    yaml = _dataset(tmp_path, "train", ["a", "b"])
    evidence = collect_evidence(dataset_yaml=yaml, sources=[], split="val")
    assert evidence.split == "train"


def test_sampling_keeps_one_recording_together(tmp_path):
    yaml = _dataset(
        tmp_path,
        "val",
        [f"rec1_{i:03d}" for i in range(6)] + [f"rec2_{i:03d}" for i in range(6)],
    )
    evidence = collect_evidence(dataset_yaml=yaml, sources=[], budget=6)
    stems = {Path(p).stem.split("_")[0] for p, _labels in evidence.frames}
    assert stems == {"rec1"}, "budget must consume whole recordings, not stride"
    assert evidence.sampled_from == 12


def test_evidence_fingerprint_changes_with_labels(tmp_path):
    yaml = _dataset(tmp_path, "val", ["v0", "v1"])
    before = collect_evidence(dataset_yaml=yaml, sources=[]).fingerprint
    (tmp_path / "labels" / "val" / "v0.txt").write_text(
        "0 0.3 0.3 0.4 0.3 0.4 0.4 0.3 0.4\n"
    )
    assert collect_evidence(dataset_yaml=yaml, sources=[]).fingerprint != before


def test_exhaustive_label_warning_is_stated():
    assert "exhaustively labelled" in EXHAUSTIVE_LABEL_WARNING


def test_single_oversized_recording_is_truncated_to_budget(tmp_path):
    yaml = _dataset(tmp_path, "val", [f"recA_{i:03d}" for i in range(10)])
    evidence = collect_evidence(dataset_yaml=yaml, sources=[], budget=3)
    assert len(evidence.frames) == 3
    assert evidence.sampled_from == 10


def test_labels_dir_resolved_when_root_path_contains_images_segment(tmp_path):
    root = tmp_path / "images" / "pilot1"
    images = root / "images" / "val"
    labels = root / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    cv2.imwrite(str(images / "v0.png"), np.zeros((200, 300, 3), np.uint8))
    (labels / "v0.txt").write_text(LABEL_LINE)
    yaml = tmp_path / "data.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: ant\n"
    )
    evidence = collect_evidence(dataset_yaml=yaml, sources=[])
    assert evidence.split == "val"
    assert len(evidence.frames) == 1 and evidence.instances == 1


def test_images_dir_without_images_segment_yields_no_frames(tmp_path):
    images = tmp_path / "frames" / "val"
    images.mkdir(parents=True)
    cv2.imwrite(str(images / "v0.png"), np.zeros((200, 300, 3), np.uint8))
    yaml = tmp_path / "data.yaml"
    yaml.write_text(f"path: {tmp_path}\nval: frames/val\nnames:\n  0: ant\n")
    evidence = collect_evidence(dataset_yaml=yaml, sources=[], split="val")
    assert evidence.frames == []
    assert evidence.split in ("val", "train")


def test_sources_fallback_reports_split_as_sources(tmp_path, monkeypatch):
    import hydra_suite.detectkit.jobs.direct_calibration as direct_calibration

    fake_frame = (tmp_path / "s0.png", [])

    def _fake_stratified(sources, *, budget):
        return [fake_frame]

    monkeypatch.setattr(
        direct_calibration, "stratified_calibration_frames", _fake_stratified
    )
    evidence = direct_calibration.collect_evidence(
        dataset_yaml=None, sources=["fake-source"]
    )
    assert evidence.split == "sources"
    assert evidence.frames == [fake_frame]
