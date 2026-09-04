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


class _FakeSource:
    """Mimics a RegionSource. merge_policy here is the REGION policy vocabulary."""

    merge_policy = "plain"

    def merge_plan(self, _frame_idx):
        return None


def _fake_models(request, candidate):
    """4-tuple matching load_calibration_models: (models, runtime, config, imgsz)."""
    from hydra_suite.core.inference.direct_calibration_sweep import config_for_point

    config = config_for_point(
        str(request.model_path),
        slice_params=candidate.slice_params(),
        merge=request.merge_settings[0],
        confidence=request.confidences[0],
        max_targets=request.max_targets,
        runtime_tier="cpu",
        model_task="obb",
    )
    return object(), object(), config, 640


def _evidence_frames(tmp_path):
    """Two tiny real frames with one polygon label each.

    Real files matter here: the sweep loop decodes each frame with
    ``cv2.imread`` one at a time, and this is exactly the loop under test.
    """
    from hydra_suite.data.al.escalation import LabelRecord
    from hydra_suite.utils.geometry_levels import GeometryLevel

    frames = []
    for name in ("f0", "f1"):
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), np.zeros((64, 64, 3), np.uint8))
        label = LabelRecord(
            class_id=0,
            confidence=1.0,
            points=np.array([[4, 4], [20, 4], [20, 20], [4, 20]], dtype=np.float32),
            level=GeometryLevel.POLYGON,
        )
        frames.append((path, [label]))
    return frames


def _request(tmp_path, confidences=(0.35,), merges=None):
    from hydra_suite.core.inference.direct_calibration_grid import build_candidate_grid
    from hydra_suite.core.inference.direct_calibration_sweep import MergeSettings
    from hydra_suite.detectkit.jobs.direct_calibration import (
        DirectCalibrationRequest,
        EvidenceSet,
    )

    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    evidence = EvidenceSet(
        frames=_evidence_frames(tmp_path),
        split="val",
        instances=2,
        size_range=((64, 64), (64, 64)),
        sampled_from=2,
        fingerprint="deadbeef",
    )
    return DirectCalibrationRequest(
        model_path=model,
        task="obb",
        evidence=evidence,
        candidates=build_candidate_grid(
            {
                "geometry_mode": "auto_object",
                "imgsz": 640,
                "object_tile_fraction": 0.4,
                "overlap": 0.2,
            }
        )[:2],
        confidences=confidences,
        merge_settings=merges or (MergeSettings("greedy_nmm", "ios", 0.5),),
        runtime_tier="cpu",
        max_targets=64,
        evidence_dir=tmp_path / "evidence",
    )


def test_one_inference_pass_per_geometry_regardless_of_sweep_size(
    monkeypatch, tmp_path
):
    from hydra_suite.core.inference.direct_calibration_sweep import MergeSettings
    from hydra_suite.detectkit.jobs import direct_calibration as job

    calls = []
    monkeypatch.setattr(job, "load_calibration_models", _fake_models)
    monkeypatch.setattr(
        job,
        "collect_obb_parts_by_frame",
        lambda frames, *a, **k: (
            calls.append(1) or ([[] for _ in frames], _FakeSource())
        ),
    )
    request = _request(
        tmp_path,
        confidences=(0.1, 0.2, 0.3, 0.4),
        merges=(
            MergeSettings("greedy_nmm", "ios", 0.5),
            MergeSettings("nmm", "iou", 0.6),
        ),
    )
    outcome = job.run_direct_calibration(request)
    n_frames = len(request.evidence.frames)
    assert (
        len(calls) == len(request.candidates) * n_frames
    ), "one model call per frame per geometry, independent of the sweep size"
    assert len(outcome.points) == len(request.candidates) * 4 * 2, (
        "the confidence x merge sweep must add ZERO model calls but must still "
        "emit one point per (candidate, confidence, merge) combination"
    )
    for preview in outcome.previews:
        for _path, gt_polygons, pred_polygons in preview.frames:
            for polygon in (*gt_polygons, *pred_polygons):
                assert isinstance(polygon, np.ndarray) and polygon.ndim == 2
    assert not any(
        isinstance(value, np.ndarray) and value.ndim == 3
        for point in outcome.points
        for value in vars(point).values()
    ), "no decoded image array may be retained on a point"


def test_cancellation_returns_partial_and_never_claims_completeness(
    monkeypatch, tmp_path
):
    from hydra_suite.detectkit.jobs import direct_calibration as job

    monkeypatch.setattr(job, "load_calibration_models", _fake_models)
    monkeypatch.setattr(
        job,
        "collect_obb_parts_by_frame",
        lambda frames, *a, **k: ([[] for _ in frames], _FakeSource()),
    )
    outcome = job.run_direct_calibration(_request(tmp_path), should_stop=lambda: True)
    assert outcome.partial is True and outcome.points == []


def test_failed_candidate_becomes_a_failed_row_not_a_silent_omission(
    monkeypatch, tmp_path
):
    from hydra_suite.detectkit.jobs import direct_calibration as job

    monkeypatch.setattr(job, "load_calibration_models", _fake_models)

    def boom(*_a, **_k):
        raise ValueError("tile budget exceeded")

    monkeypatch.setattr(job, "collect_obb_parts_by_frame", boom)
    outcome = job.run_direct_calibration(_request(tmp_path))
    assert len(outcome.points) == len(_request(tmp_path).candidates)
    assert all(point.failed_reason for point in outcome.points)
    assert "tile budget exceeded" in outcome.points[0].failed_reason


def test_points_record_the_detection_cap_they_were_measured_under(
    monkeypatch, tmp_path
):
    from hydra_suite.detectkit.jobs import direct_calibration as job

    monkeypatch.setattr(job, "load_calibration_models", _fake_models)
    monkeypatch.setattr(
        job,
        "collect_obb_parts_by_frame",
        lambda frames, *a, **k: ([[] for _ in frames], _FakeSource()),
    )
    outcome = job.run_direct_calibration(_request(tmp_path))
    assert all(point.max_detections == 64 for point in outcome.points)
