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
