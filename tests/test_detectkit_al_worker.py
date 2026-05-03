"""Integration test for the DetectKit AL worker."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hydra_suite.detectkit.gui.models import DetectKitProject


def _seed_image_folder(tmp_path: Path, n: int = 6) -> Path:
    folder = tmp_path / "frames"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(folder / f"f_{i:03d}.png"), img)
    return folder


def test_al_worker_writes_seeded_labels_and_registers_source(tmp_path):
    from hydra_suite.detectkit.jobs.al_worker import ALRequest, run_active_learning

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    project = DetectKitProject(project_dir=project_dir, sources=[])

    folder = _seed_image_folder(tmp_path, n=6)

    def fake_detector(frame, conf, iou):
        return [
            (10, 10, 8, 4, 0.0, 0.95),
            (30, 30, 8, 4, 0.0, 0.55),
            (50, 50, 8, 4, 0.0, 0.30),
        ]

    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=3,
        preset="balanced",
        expected_count=2,
        detector_fn=fake_detector,
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


def test_al_worker_drops_frames_that_fail_to_re_read(tmp_path):
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

    # Build a folder source, then wrap its `read` so the second invocation per
    # frame_id (i.e., the post-select re-read) returns None for one specific
    # frame_id. Initial scoring-loop reads must succeed for all candidates so
    # the same frames make it into `picked_ids`.
    from hydra_suite.data.al.frame_source import ImageFolderFrameSource

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

    wrapped = _FailOnSecondRead(real_source, fail_frame_id=1)

    def fake_detector(frame, conf, iou):
        return [
            (10, 10, 8, 4, 0.0, 0.55),
            (30, 30, 8, 4, 0.0, 0.40),
        ]

    request = ALRequest(
        input_kind="folder",
        input_path=str(folder),
        project=project,
        budget=4,
        preset="balanced",
        expected_count=2,
        detector_fn=fake_detector,
        diversity_window=0,
        probabilistic=False,
    )

    # Patch the FrameSource builder so `run_active_learning` uses our wrapper.
    from hydra_suite.detectkit.jobs import al_worker as al_worker_mod

    original_builder = al_worker_mod._build_frame_source
    al_worker_mod._build_frame_source = lambda req: wrapped
    try:
        result = run_active_learning(request)
    finally:
        al_worker_mod._build_frame_source = original_builder

    # Frame 1 should have been picked but failed re-read; result reflects writes.
    assert result.n_picked == 3
    assert 1 not in result.selected_frames
    written_dir = Path(result.source_path)
    image_files = sorted(p.name for p in (written_dir / "images").iterdir())
    label_files = sorted(p.name for p in (written_dir / "labels").iterdir())
    assert len(image_files) == 3
    assert len(label_files) == 3
    assert "f_000001.jpg" not in image_files
