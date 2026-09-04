"""Integration test for the DetectKit AL worker.

The worker no longer takes a per-frame `detector_fn(frame, conf, iou)` closure:
it builds one `InferenceRunner` from an `ALDetectorSpec` and runs a single
batched, cached detection pass (`get_or_compute_raw`) over the whole candidate
list, then scores every candidate from its cached raw `OBBResult`.

These tests therefore inject a runner double at the same construction seam the
production code uses -- `al_worker._build_detection_context` -- exactly as the
pre-existing re-read test already patches `al_worker._build_frame_source`. The
double implements `detect_batch_raw` only (the contract `get_or_compute_raw`
documents for callers without real weights), so the whole file stays free of
model fixtures while covering the same behaviours the old `fake_detector`
closures covered.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.core.inference.config import OBBConfig, OBBDirectConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.detectkit.gui.models import DetectKitProject
from hydra_suite.detectkit.jobs.al_worker import ALDetectorSpec
from hydra_suite.utils.geometry import obb_corners_from_dims

_SPEC = ALDetectorSpec(kind="obb_direct", model_path="/unused/model.pt")


def test_gui_al_worker_is_a_thin_protected_sidecar_coordinator(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hydra_suite.detectkit.gui.models import OBBSource
    from hydra_suite.detectkit.jobs import al_worker as mod

    source = OBBSource(path=str(tmp_path / "round"), name="round")

    class FakeOperation:
        def __init__(self, request, **kwargs):
            self.request = request
            self.kwargs = kwargs

        def run(self, progress):
            progress(50, "scoring")
            return SimpleNamespace(
                success=True,
                canceled=False,
                payload={
                    "source_path": source.path,
                    "n_picked": 1,
                    "selected_frames": [7],
                    "source": source.to_dict(),
                },
            )

        def cancel(self):
            pass

    monkeypatch.setattr(mod, "ProtectedOperation", FakeOperation)
    project = DetectKitProject(project_dir=tmp_path, sources=[])
    worker = mod.ALWorker(
        mod.ALRequest(
            input_kind="project",
            input_path=str(tmp_path),
            project=project,
            budget=1,
            detector=_SPEC,
        )
    )
    seen = []
    worker.result_ready.connect(lambda *args: seen.append(args))
    worker.execute()

    assert seen == [(source.path, 1, [7])]
    assert project.sources[0].path == source.path
    assert "run_active_learning(" not in inspect.getsource(mod.ALWorker.execute)


def _seed_image_folder(tmp_path: Path, n: int = 6) -> Path:
    folder = tmp_path / "frames"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(folder / f"f_{i:03d}.png"), img)
    return folder


def _write_video(path: Path, n: int = 8, size: tuple[int, int] = (64, 64)) -> Path:
    """Write a short video whose frames are all visually distinct."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size, True
    )
    rng = np.random.default_rng(1)
    try:
        for _ in range(n):
            writer.write(
                rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
            )
    finally:
        writer.release()
    return path


def _raw_from_tuples(frame_idx: int, tuples) -> OBBResult:
    """Build a raw `OBBResult` that `detections_from_obb_result` round-trips
    back into exactly `tuples` ((cx, cy, major, minor, theta, conf))."""
    n = len(tuples)
    if n == 0:
        return OBBResult(
            frame_idx=frame_idx,
            centroids=np.zeros((0, 2), dtype=np.float32),
            angles=np.zeros(0, dtype=np.float32),
            sizes=np.zeros(0, dtype=np.float32),
            shapes=np.zeros((0, 2), dtype=np.float32),
            confidences=np.zeros(0, dtype=np.float32),
            corners=np.zeros((0, 4, 2), dtype=np.float32),
            detection_ids=OBBResult.make_detection_ids(frame_idx, 0),
        )
    centroids = np.array([[t[0], t[1]] for t in tuples], dtype=np.float32)
    angles = np.array([t[4] for t in tuples], dtype=np.float32)
    # `detections_from_obb_result` inverts (size, aspect) into
    # (major, minor) = (sqrt(size * aspect), sqrt(size / aspect)).
    sizes = np.array([t[2] * t[3] for t in tuples], dtype=np.float32)
    shapes = np.array(
        [[t[2] * t[3] * math.pi / 4.0, t[2] / t[3]] for t in tuples], dtype=np.float32
    )
    confidences = np.array([t[5] for t in tuples], dtype=np.float32)
    corners = np.stack(
        [obb_corners_from_dims(t[0], t[1], t[2], t[3], t[4]) for t in tuples]
    ).astype(np.float32)
    return OBBResult(
        frame_idx=frame_idx,
        centroids=centroids,
        angles=angles,
        sizes=sizes,
        shapes=shapes,
        confidences=confidences,
        corners=corners,
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


class _FakeRunner:
    """Minimal `detect_batch_raw` double -- the contract `get_or_compute_raw`
    documents for callers that cannot load real weights."""

    def __init__(self, dets_for):
        self._dets_for = dets_for
        self.calls: list[list[int]] = []

    def detect_batch_raw(self, frames, frame_indices=None, roi_mask=None):
        frames = list(frames)
        if frame_indices is None:
            frame_indices = list(range(len(frames)))
        self.calls.append(list(frame_indices))
        return [
            _raw_from_tuples(idx, self._dets_for(pos, idx))
            for pos, idx in enumerate(frame_indices)
        ]


def _patch_detection(monkeypatch, dets_for) -> _FakeRunner:
    """Route `run_active_learning`'s detector construction to a fake runner."""
    from hydra_suite.detectkit.jobs import al_worker as al_worker_mod

    runner = _FakeRunner(dets_for)
    obb_config = OBBConfig(
        mode="direct",
        direct=OBBDirectConfig(model_path="/unused/model.pt"),
        confidence_threshold=0.25,
        iou_threshold=0.7,
    )
    monkeypatch.setattr(
        al_worker_mod,
        "_build_detection_context",
        lambda req: (runner, obb_config),
    )
    return runner


def test_active_learning_cancellation_stops_before_detection(tmp_path, monkeypatch):
    from hydra_suite.detectkit.jobs import al_worker as al_worker_mod
    from hydra_suite.detectkit.jobs.al_worker import (
        ActiveLearningCancelled,
        ALRequest,
        run_active_learning,
    )

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])
    folder = _seed_image_folder(tmp_path, n=6)
    detection_started = False

    def _unexpected_detection(_req):
        nonlocal detection_started
        detection_started = True
        raise AssertionError("detection should not start after cancellation")

    monkeypatch.setattr(
        al_worker_mod, "_build_detection_context", _unexpected_detection
    )
    checks = 0

    def should_stop() -> bool:
        nonlocal checks
        checks += 1
        return checks > 2

    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=1,
        detector=_SPEC,
        candidate_pool=al_worker_mod.CandidatePoolConfig(
            dedup_method="none", max_candidates=None
        ),
    )

    with pytest.raises(ActiveLearningCancelled):
        run_active_learning(request, should_stop=should_stop)

    assert detection_started is False
    assert project.sources == []


def test_active_learning_cancellation_removes_partial_export(tmp_path, monkeypatch):
    from hydra_suite.detectkit.jobs import al_worker as al_worker_mod
    from hydra_suite.detectkit.jobs.al_worker import (
        ActiveLearningCancelled,
        ALRequest,
        run_active_learning,
    )

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])
    folder = _seed_image_folder(tmp_path, n=3)
    _patch_detection(monkeypatch, lambda _pos, _idx: [(10, 10, 8, 4, 0.0, 0.95)])
    cancelled = False

    def cancel_during_export(*, round_dir, frames, images, **_kwargs):
        nonlocal cancelled
        Path(round_dir).mkdir(parents=True)
        (Path(round_dir) / "partial.txt").write_text("partial")
        cancelled = True
        images[frames[0].frame_id]
        raise AssertionError("the lazy image read must observe cancellation")

    monkeypatch.setattr(al_worker_mod, "export_al_dataset", cancel_during_export)
    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=1,
        detector=_SPEC,
        diversity_window=0,
        probabilistic=False,
        candidate_pool=al_worker_mod.CandidatePoolConfig(
            dedup_method="none", max_candidates=None
        ),
    )

    with pytest.raises(ActiveLearningCancelled):
        run_active_learning(request, should_stop=lambda: cancelled)

    assert project.sources == []
    assert not list((project_dir / "sources").glob("al_round_*"))


def test_al_worker_writes_seeded_labels_and_registers_source(tmp_path, monkeypatch):
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])

    folder = _seed_image_folder(tmp_path, n=6)

    _patch_detection(
        monkeypatch,
        lambda pos, idx: [
            (10, 10, 8, 4, 0.0, 0.95),
            (30, 30, 8, 4, 0.0, 0.55),
            (50, 50, 8, 4, 0.0, 0.30),
        ],
    )

    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=3,
        preset="balanced",
        expected_count=2,
        detector=_SPEC,
        diversity_window=0,
        probabilistic=False,
    )

    result = run_active_learning(request)

    assert result.n_picked == 3
    new_source_dir = Path(result.source_path)
    assert (new_source_dir / "images").is_dir()
    assert (new_source_dir / "labels").is_dir()
    image_files = list((new_source_dir / "images").iterdir())
    label_files = list((new_source_dir / "labels").iterdir())
    assert len(image_files) == 3
    assert len(label_files) == 3
    for lf in label_files:
        lines = lf.read_text().strip().splitlines()
        assert len(lines) == 3  # all three model predictions seeded as YOLO OBB lines

    assert any(s.path == str(new_source_dir) for s in project.sources)


def test_al_worker_detects_every_candidate_in_one_batched_call(tmp_path, monkeypatch):
    """The whole candidate list must go through a SINGLE `detect_batch_raw`
    call -- that batching (not a per-frame closure) is the point of the
    restructure."""
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])
    folder = _seed_image_folder(tmp_path, n=6)

    runner = _patch_detection(monkeypatch, lambda pos, idx: [(10, 10, 8, 4, 0.0, 0.95)])

    run_active_learning(
        ALRequest(
            input_kind="folder",
            input_path=str(folder),
            project=project,
            budget=2,
            preset="balanced",
            expected_count=1,
            detector=_SPEC,
            diversity_window=0,
            probabilistic=False,
        )
    )

    assert len(runner.calls) == 1
    assert runner.calls[0] == list(range(6))


def test_run_active_learning_populates_detection_cache(tmp_path, monkeypatch):
    """A video-backed round writes its raw detections to the same
    `.inference_cache_<stem>/detection.npz` tracking uses, and a second round
    over the same video reads it back instead of re-detecting."""
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning
    from hydra_suite.utils.video_artifacts import build_inference_cache_dir

    video_path = _write_video(tmp_path / "clip.mp4", n=8)

    def _request(round_name: str):
        # A separate project dir per round: AL round folders are named from a
        # second-resolution timestamp, so two rounds run back-to-back inside
        # one project would collide on the same folder name.
        project_dir = tmp_path / round_name
        project_dir.mkdir()
        return ALRequest(
            input_kind="video",
            input_path=str(video_path),
            project=DetectKitProject(project_dir=project_dir, sources=[]),
            budget=2,
            preset="balanced",
            expected_count=1,
            detector=_SPEC,
            diversity_window=0,
            probabilistic=False,
        )

    runner = _patch_detection(monkeypatch, lambda pos, idx: [(10, 10, 8, 4, 0.0, 0.95)])
    result = run_active_learning(_request("proj_a"))

    assert result.n_picked >= 1
    cache_dir = build_inference_cache_dir(str(video_path))
    assert (cache_dir / "al" / "detection.npz").exists()

    # Second round over the same video: fully covered by the cache written
    # above, so no further model call happens.
    first_calls = len(runner.calls)
    run_active_learning(_request("proj_b"))
    assert len(runner.calls) == first_calls


def test_al_round_never_touches_trackings_detection_cache(tmp_path, monkeypatch):
    """An AL round must write its own cache file. `DetectionCacheHandle.close()`
    rewrites `<dir>/detection.npz` from its own buffer alone, so if AL pointed at
    `.inference_cache_<stem>/` directly it would silently destroy a complete
    tracking detection cache and replace it with a sparse candidate-only one."""
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning
    from hydra_suite.utils.video_artifacts import build_inference_cache_dir

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    video_path = _write_video(tmp_path / "clip.mp4", n=8)

    # Stand in for a tracking run's existing detection cache.
    cache_dir = build_inference_cache_dir(str(video_path), create=True)
    tracking_cache = cache_dir / "detection.npz"
    tracking_cache.write_bytes(b"tracking-cache-sentinel")

    _patch_detection(monkeypatch, lambda pos, idx: [(10, 10, 8, 4, 0.0, 0.95)])
    run_active_learning(
        ALRequest(
            input_kind="video",
            input_path=str(video_path),
            project=DetectKitProject(project_dir=project_dir, sources=[]),
            budget=2,
            preset="balanced",
            expected_count=1,
            detector=_SPEC,
            diversity_window=0,
            probabilistic=False,
        )
    )

    assert tracking_cache.read_bytes() == b"tracking-cache-sentinel"
    assert (cache_dir / "al" / "detection.npz").exists()


def test_candidate_pool_is_capped_by_default(tmp_path, monkeypatch):
    """`CandidatePoolConfig.max_candidates` must default to a finite cap: the AL
    round holds every candidate frame in memory and sends them through the model
    as ONE unwindowed batch, so an unbounded pool is an OOM risk on long videos."""
    from hydra_suite.data.al.candidate_pool import CandidatePoolConfig
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    cap = CandidatePoolConfig().max_candidates
    assert cap is not None and cap > 0

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    folder = _seed_image_folder(tmp_path, n=cap + 5)  # every frame is distinct

    runner = _patch_detection(monkeypatch, lambda pos, idx: [(10, 10, 8, 4, 0.0, 0.9)])
    run_active_learning(
        ALRequest(
            input_kind="folder",
            input_path=str(folder),
            project=DetectKitProject(project_dir=project_dir, sources=[]),
            budget=2,
            preset="balanced",
            expected_count=1,
            detector=_SPEC,
            diversity_window=0,
            probabilistic=False,
        )
    )

    assert len(runner.calls) == 1
    assert len(runner.calls[0]) == cap


def test_al_worker_registers_only_authoritative_source_for_multi_level_export(
    tmp_path, monkeypatch
):
    """A round exported at multiple levels (obb authoritative + aabb derived)
    must register exactly ONE project source -- the authoritative root -- not
    one sibling per level. The derived level's folder still gets written to
    disk by export_al_dataset (unchanged), it's just not registered."""
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])

    folder = _seed_image_folder(tmp_path, n=3)

    _patch_detection(monkeypatch, lambda pos, idx: [(10, 10, 8, 4, 0.0, 0.95)])

    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=3,
        preset="balanced",
        expected_count=1,
        detector=_SPEC,
        diversity_window=0,
        probabilistic=False,
        export_levels=["obb", "aabb"],
        native_level="obb",
    )

    result = run_active_learning(request)

    assert len(project.sources) == 1
    registered = project.sources[0]
    assert registered.level == "obb"
    assert registered.derived_from is None
    assert registered.name.startswith("al_round_")
    assert "_obb" not in registered.name
    assert "_aabb" not in registered.name
    assert registered.path == result.source_path

    # The derived aabb sibling still exists on disk (export.py is unchanged)
    # even though it was not registered as a project source.
    aabb_root = Path(result.source_path).parent / "aabb"
    assert aabb_root.is_dir()


def test_al_worker_refuses_polygon_export_when_no_frame_has_detections(
    tmp_path, monkeypatch
):
    """Regression: `native_level` must gate independently of what LabelRecords
    actually exist.

    `derive_down`'s per-record check only fires if a record reaches it. If
    every picked frame has zero detections (plausible for uncertainty-driven
    top-K picks), no record is ever built, so `native_level` -- not record
    inspection -- is the only thing that can refuse a request for a
    geometry level an obb-only model cannot produce.
    """
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])

    folder = _seed_image_folder(tmp_path, n=6)

    _patch_detection(monkeypatch, lambda pos, idx: [])

    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=3,
        preset="balanced",
        expected_count=0,
        detector=_SPEC,
        diversity_window=0,
        probabilistic=False,
        export_level="polygon",
        export_levels=["polygon"],
        native_level="obb",
    )

    with pytest.raises(ValueError, match="polygon"):
        run_active_learning(request)

    # Nothing partial was registered on the project when the round refused.
    assert project.sources == []


def test_al_worker_drops_frames_that_fail_to_re_read(tmp_path, monkeypatch):
    """If FrameSource.read returns None during the post-select write loop,
    that frame is logged-and-skipped, and ALResult reflects only successful writes.
    """
    from hydra_suite.data.al.frame_source import FrameRef
    from hydra_suite.detectkit.gui.models import DetectKitProject
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])

    folder = _seed_image_folder(tmp_path, n=4)

    # Build a folder source, then wrap its `read` so the THIRD invocation per
    # frame_id (i.e., the post-select readability probe) returns None for one
    # specific frame_id. Reads 1 and 2 (the candidate-pool scan and the
    # batched-detection pass) must succeed for all candidates so the same
    # frames make it into `picked_ids`.
    from hydra_suite.data.al.frame_source import ImageFolderFrameSource

    real_source = ImageFolderFrameSource(str(folder))

    class _FailOnThirdRead:
        def __init__(self, base, fail_frame_id):
            self._base = base
            self._fail_frame_id = fail_frame_id
            self._read_counts: dict[int, int] = {}

        def __iter__(self):
            return iter(self._base)

        def read(self, ref: FrameRef):
            self._read_counts[ref.frame_id] = self._read_counts.get(ref.frame_id, 0) + 1
            if (
                ref.frame_id == self._fail_frame_id
                and self._read_counts[ref.frame_id] >= 3
            ):
                return None
            return self._base.read(ref)

        def length(self):
            return self._base.length()

    wrapped = _FailOnThirdRead(real_source, fail_frame_id=1)

    _patch_detection(
        monkeypatch,
        lambda pos, idx: [
            (10, 10, 8, 4, 0.0, 0.55),
            (30, 30, 8, 4, 0.0, 0.40),
        ],
    )

    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=4,
        preset="balanced",
        expected_count=2,
        detector=_SPEC,
        diversity_window=0,
        probabilistic=False,
    )

    # Patch the FrameSource builder so `run_active_learning` uses our wrapper.
    from hydra_suite.detectkit.jobs import al_worker as al_worker_mod

    monkeypatch.setattr(al_worker_mod, "_build_frame_source", lambda req: wrapped)
    result = run_active_learning(request)

    # Frame 1 should have been picked but failed re-read; result reflects writes.
    assert result.n_picked == 3
    assert 1 not in result.selected_frames
    written_dir = Path(result.source_path)
    image_files = sorted(p.name for p in (written_dir / "images").iterdir())
    label_files = sorted(p.name for p in (written_dir / "labels").iterdir())
    assert len(image_files) == 3
    assert len(label_files) == 3
    assert "f_000001.jpg" not in image_files


def test_al_worker_drops_candidates_that_fail_the_detection_read(tmp_path, monkeypatch):
    """A candidate whose frame cannot be decoded for the batched detection pass
    is dropped before detection -- it is never scored and never picked."""
    from hydra_suite.data.al.frame_source import FrameRef, ImageFolderFrameSource
    from hydra_suite.detectkit.jobs import al_worker as al_worker_mod
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])
    folder = _seed_image_folder(tmp_path, n=4)

    real_source = ImageFolderFrameSource(str(folder))

    class _FailOnSecondRead:
        def __init__(self, base, fail_frame_id):
            self._base = base
            self._fail_frame_id = fail_frame_id
            self._read_counts: dict[int, int] = {}

        def __iter__(self):
            return iter(self._base)

        def read(self, ref: FrameRef):
            self._read_counts[ref.frame_id] = self._read_counts.get(ref.frame_id, 0) + 1
            if (
                ref.frame_id == self._fail_frame_id
                and self._read_counts[ref.frame_id] >= 2
            ):
                return None
            return self._base.read(ref)

        def length(self):
            return self._base.length()

    wrapped = _FailOnSecondRead(real_source, fail_frame_id=2)
    runner = _patch_detection(monkeypatch, lambda pos, idx: [(10, 10, 8, 4, 0.0, 0.9)])
    monkeypatch.setattr(al_worker_mod, "_build_frame_source", lambda req: wrapped)

    result = run_active_learning(
        ALRequest(
            input_kind="folder",
            input_path=str(folder),
            project=project,
            budget=4,
            preset="balanced",
            expected_count=1,
            detector=_SPEC,
            diversity_window=0,
            probabilistic=False,
        )
    )

    assert runner.calls == [[0, 1, 3]]  # frame 2 never reached detection
    assert 2 not in result.selected_frames


def test_n_picked_counts_images_on_disk_not_probe_survivors(tmp_path, monkeypatch):
    """`written_ids` counts frames that passed the readability probe. The
    exporter then drops any frame whose records did not survive, so reporting
    the probe count claimed more images than exist on disk."""
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])
    folder = _seed_image_folder(tmp_path, n=6)

    # The first frame of the batch yields nothing; the rest are normal.
    _patch_detection(
        monkeypatch,
        lambda pos, idx: [] if pos == 0 else [(10, 10, 8, 4, 0.0, 0.95)],
    )

    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=3,
        preset="balanced",
        expected_count=1,
        detector=_SPEC,
        diversity_window=0,
        probabilistic=False,
    )

    result = run_active_learning(request)

    images = list((Path(result.source_path) / "images").iterdir())
    labels = list((Path(result.source_path) / "labels").iterdir())
    assert result.n_picked == len(images) == len(labels)
    assert len(result.selected_frames) == result.n_picked
    # No empty label file was written for the dropped frame.
    for lf in labels:
        assert lf.read_text().strip() != ""


def test_run_active_learning_requires_a_detector_spec(tmp_path):
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    with pytest.raises(ValueError, match="detector"):
        run_active_learning(
            ALRequest(
                input_kind="folder",
                input_path=str(_seed_image_folder(tmp_path, n=2)),
                project=DetectKitProject(project_dir=project_dir, sources=[]),
                budget=1,
            )
        )


def test_build_detection_context_threads_base_conf_as_sequential_stage1_override(
    tmp_path, monkeypatch
):
    """Task 10 fix round (Critical finding C1): `_build_detection_context` must
    pass `detect_confidence_threshold=req.base_conf` through to
    `build_obb_config_for_al` -- otherwise sequential mode's stage-1 detect
    gate silently locks at `build_obb_only_config`'s dataclass default (0.25)
    regardless of what the AL round actually requests, which is exactly what
    the retired per-frame detector closure never did (it applied one caller
    `conf` to both stages). This test asserts the real production callsite
    (not just `build_obb_config_for_al` in isolation, already covered by
    `tests/test_al_inference_adapter.py`) does the threading."""
    from hydra_suite.detectkit.jobs import al_worker as al_worker_mod
    from hydra_suite.detectkit.jobs.al_worker import ALDetectorSpec, ALRequest

    captured: dict = {}

    def _fake_build_obb_config_for_al(kind, primary, secondary, **kwargs):
        from hydra_suite.core.inference.config import InferenceConfig

        captured["kind"] = kind
        captured.update(kwargs)
        obb_config = OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path=primary),
            confidence_threshold=kwargs["confidence_threshold"],
            iou_threshold=kwargs["iou_threshold"],
        )
        return InferenceConfig(obb=obb_config)

    class _FakeInferenceRunner:
        def __init__(self, config, video_path=None):
            self.config = config
            self.video_path = video_path

    monkeypatch.setattr(
        al_worker_mod, "build_obb_config_for_al", _fake_build_obb_config_for_al
    )
    monkeypatch.setattr(
        "hydra_suite.core.inference.runner.InferenceRunner", _FakeInferenceRunner
    )

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    req = ALRequest(
        input_kind="video",
        input_path="unused.mp4",
        project=DetectKitProject(project_dir=project_dir, sources=[]),
        budget=1,
        detector=ALDetectorSpec(
            kind="sequential",
            model_path="/detect.pt",
            secondary_model_path="/obb.pt",
        ),
        base_conf=0.05,
        base_iou=0.5,
    )

    al_worker_mod._build_detection_context(req)

    assert captured["kind"] == "sequential"
    assert captured["confidence_threshold"] == 0.05
    assert captured["detect_confidence_threshold"] == 0.05


def test_build_detection_context_threads_a_generous_max_targets(tmp_path, monkeypatch):
    """Fix 2: the production callsite must lift the tracking-oriented
    `max_targets=8` default, and must scale it above a project's own declared
    animal count so the ceiling can never bite before `count_deviation` can
    measure an overcount."""
    from hydra_suite.data.al.inference_adapter import AL_DEFAULT_MAX_TARGETS
    from hydra_suite.detectkit.jobs import al_worker as al_worker_mod
    from hydra_suite.detectkit.jobs.al_worker import ALRequest

    captured: dict = {}

    def _fake_build_obb_config_for_al(kind, primary, secondary, **kwargs):
        from hydra_suite.core.inference.config import InferenceConfig

        captured.update(kwargs)
        return InferenceConfig(
            obb=OBBConfig(
                mode="direct",
                direct=OBBDirectConfig(model_path=primary),
                confidence_threshold=kwargs["confidence_threshold"],
                iou_threshold=kwargs["iou_threshold"],
            )
        )

    class _FakeInferenceRunner:
        def __init__(self, config, video_path=None):
            self.config = config

    monkeypatch.setattr(
        al_worker_mod, "build_obb_config_for_al", _fake_build_obb_config_for_al
    )
    monkeypatch.setattr(
        "hydra_suite.core.inference.runner.InferenceRunner", _FakeInferenceRunner
    )

    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    def _req(expected_count):
        return ALRequest(
            input_kind="video",
            input_path="unused.mp4",
            project=DetectKitProject(project_dir=project_dir, sources=[]),
            budget=1,
            expected_count=expected_count,
            detector=_SPEC,
        )

    al_worker_mod._build_detection_context(_req(20))
    assert captured["max_targets"] == AL_DEFAULT_MAX_TARGETS

    al_worker_mod._build_detection_context(_req(500))
    assert captured["max_targets"] == 1000


def test_candidate_stride_spreads_a_capped_scan_over_a_long_source():
    """Fix 4 (unit): the stride must be 1 for short sources and >1 for sources
    long enough that a stride-1 capped scan would only see their beginning."""
    from hydra_suite.detectkit.jobs.al_worker import _candidate_stride

    assert _candidate_stride(100, None) == 1  # uncapped pool: never stride
    assert _candidate_stride(100, 128) == 1  # shorter than the cap
    assert _candidate_stride(128, 128) == 1  # exactly the cap
    assert _candidate_stride(0, 128) == 1  # unknown length: no guessing
    assert _candidate_stride(12800, 128) == 100  # 100x the cap => stride 100


def test_capped_candidate_pool_spans_the_whole_video_not_just_its_start(
    tmp_path, monkeypatch
):
    """Fix 4: `max_candidates` is a memory guard, and `build_candidate_pool`
    stops as soon as it fills -- so on a long video a stride-1 scan would score
    only the opening frames. `_build_frame_source` must stride the scan so a
    still-capped pool covers the entire source."""
    from hydra_suite.data.al.candidate_pool import CandidatePoolConfig
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    cap = 8
    n_frames = 400
    video_path = _write_video(tmp_path / "long.mp4", n=n_frames)

    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    runner = _patch_detection(monkeypatch, lambda pos, idx: [(10, 10, 8, 4, 0.0, 0.9)])
    run_active_learning(
        ALRequest(
            input_kind="video",
            input_path=str(video_path),
            project=DetectKitProject(project_dir=project_dir, sources=[]),
            budget=2,
            preset="balanced",
            expected_count=1,
            detector=_SPEC,
            diversity_window=0,
            probabilistic=False,
            candidate_pool=CandidatePoolConfig(max_candidates=cap),
        )
    )

    assert len(runner.calls) == 1
    scanned = runner.calls[0]
    # Still bounded by the memory cap...
    assert len(scanned) <= cap
    # ...but reaching well past the first `cap` frames of the video.
    assert max(scanned) > n_frames // 2
