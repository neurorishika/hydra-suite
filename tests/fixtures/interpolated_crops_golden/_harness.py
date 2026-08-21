"""Shared harness for the interpolated-crops characterization golden.

This module is deliberately standalone (no pytest, no test-suite imports
beyond ``hydra_suite`` itself) so the EXACT SAME code can build the fixture
and drive ``run_interpolated_crops`` against two different checkouts of
``hydra_suite``: the pre-refactor commit (golden capture, via a throwaway
worktree + ``PYTHONPATH`` override) and the current post-refactor tree (the
characterization test in
``tests/test_interpolated_crops_characterization_golden.py``).

Coverage note (see task-13-report.md for the full writeup): this harness
exercises the CNN-identity and head-tail signal types with real (untrained
but architecturally real) classifier weights, plus the occlusion / gap /
geometry-sourcing-priority machinery. Pose and AprilTag are deliberately
OMITTED -- no small CPU-fast pose-model fixture exists anywhere in the test
suite (building one needs a network download of a YOLO-pose base checkpoint
plus non-trivial backend-loading work), and the lab AprilTag fork, while
importable in this environment, has no existing small-fixture generator for
decodable synthetic tag imagery. ``ENABLE_POSE_EXTRACTOR`` and
``USE_APRILTAGS`` are both left off.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic classifier fixtures (mirrors tests/test_classifier_fixtures.py's
# torchvision_flat_identity / tiny_flat_headtail session fixtures, but as
# plain functions so they can run outside pytest in the throwaway golden-
# capture worktree).
# ---------------------------------------------------------------------------


def build_cnn_identity_model(path: Path) -> Path:
    """TinyClassifier-backed flat checkpoint, 3 identity classes -- for
    CNN_CLASSIFIERS. Uses the ``tinyclassifier`` backbone (same as
    ``tests/test_classifier_fixtures.py``'s ``tiny_multi_identity``) rather
    than a real torchvision backbone like resnet18, purely to keep the
    committed checkpoint small (~100s of KB instead of ~45MB) -- the
    architecture doesn't matter for this golden, only that CNN inference is
    real and input-sensitive.
    """
    if path.exists():
        return path
    from hydra_suite.training.torchvision_model import (
        build_torchvision_classifier,
        save_torchvision_checkpoint,
    )

    model = build_torchvision_classifier(
        "tinyclassifier", num_classes=3, trainable_layers=-1
    )
    save_torchvision_checkpoint(
        model=model,
        backbone="tinyclassifier",
        class_names=["antA", "antB", "antC"],
        factor_names=["flat"],
        input_size=(64, 64),
        best_val_acc=None,
        history={},
        trainable_layers=-1,
        backbone_lr_scale=1.0,
        monochrome=False,
        path=str(path),
    )
    return path


def build_headtail_model(path: Path) -> Path:
    """TinyClassifier v2 checkpoint with canonical head-tail labels."""
    if path.exists():
        return path
    import torch

    from hydra_suite.training.tiny_model import _build_tiny_classifier_class

    TinyClassifier = _build_tiny_classifier_class()
    model = TinyClassifier(n_classes=5, hidden_layers=1, hidden_dim=32, dropout=0.1)
    ckpt: dict[str, Any] = {
        "schema_version": 2,
        "arch": "tinyclassifier",
        "input_size": [64, 64],
        "factor_names": ["flat"],
        "class_names_per_factor": [["up", "down", "left", "right", "unknown"]],
        "class_names": ["up", "down", "left", "right", "unknown"],
        "num_classes": 5,
        "monochrome": False,
        "model_state_dict": model.state_dict(),
        "hidden_layers": 1,
        "hidden_dim": 32,
        "dropout": 0.1,
    }
    torch.save(ckpt, str(path))
    return path


# ---------------------------------------------------------------------------
# Synthetic video + tracking CSV
# ---------------------------------------------------------------------------

VIDEO_W, VIDEO_H = 400, 400
N_FRAMES = 25  # FrameID 0..24
REFERENCE_BODY_SIZE = 20.0


def build_synthetic_video(path: Path) -> Path:
    if path.exists():
        return path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (VIDEO_W, VIDEO_H))
    yy, xx = np.mgrid[0:VIDEO_H, 0:VIDEO_W]
    for f in range(N_FRAMES):
        frame = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        frame[:, :, 0] = (xx * 3 + yy * 5 + f * 7) % 256
        frame[:, :, 1] = (xx * 2 + yy * 7 + f * 11) % 256
        frame[:, :, 2] = (xx * 5 + yy * 2 + f * 13) % 256
        writer.write(frame)
    writer.release()
    return path


def build_tracking_csv(path: Path) -> Path:
    """Two trajectories with occluded runs exercising both the mechanism-1
    (CSV-value) geometry-sourcing priority and the NaN fallback, plus one
    multi-task frame range (both trajectories occluded simultaneously,
    frames 5-9) and one single-task frame range (only trajectory 2 occluded,
    frames 15-17, while trajectory 1 stays tracked)."""
    if path.exists():
        return path
    rows: list[dict[str, Any]] = []

    # --- Trajectory 1 --------------------------------------------------
    for f in range(0, 5):
        rows.append(
            dict(
                FrameID=f,
                TrajectoryID=1,
                X=50 + 2 * f,
                Y=50 + 1 * f,
                Theta=0.05 * f,
                State="tracked",
                DetectionID=f,
            )
        )
    # Occluded run 5-9: frames 5,6 pre-filled by mechanism (1); 7,8,9 NaN
    # (fallback to independent linear interpolation).
    rows.append(
        dict(
            FrameID=5,
            TrajectoryID=1,
            X=60.0,
            Y=55.0,
            Theta=0.25,
            State="occluded",
            DetectionID=None,
        )
    )
    rows.append(
        dict(
            FrameID=6,
            TrajectoryID=1,
            X=62.0,
            Y=56.0,
            Theta=0.30,
            State="occluded",
            DetectionID=None,
        )
    )
    for f in (7, 8, 9):
        rows.append(
            dict(
                FrameID=f,
                TrajectoryID=1,
                X=float("nan"),
                Y=float("nan"),
                Theta=float("nan"),
                State="occluded",
                DetectionID=None,
            )
        )
    for f in range(10, 25):
        rows.append(
            dict(
                FrameID=f,
                TrajectoryID=1,
                X=70 + 2 * (f - 10),
                Y=60 + 1 * (f - 10),
                Theta=0.30 + 0.05 * (f - 10),
                State="tracked",
                DetectionID=f,
            )
        )

    # --- Trajectory 2 ---------------------------------------------------
    def _traj2_track(f: int) -> dict[str, Any]:
        return dict(
            FrameID=f,
            TrajectoryID=2,
            X=140 + 2 * f,
            Y=140 + 1 * f,
            Theta=1.0 + 0.02 * f,
            State="tracked",
            DetectionID=1000 + f,
        )

    for f in range(0, 5):
        rows.append(_traj2_track(f))
    # Occluded 5-9 (all NaN -- overlaps trajectory 1's occluded run: a
    # multi-task frame range).
    for f in range(5, 10):
        rows.append(
            dict(
                FrameID=f,
                TrajectoryID=2,
                X=float("nan"),
                Y=float("nan"),
                Theta=float("nan"),
                State="occluded",
                DetectionID=None,
            )
        )
    for f in range(10, 15):
        rows.append(_traj2_track(f))
    # Occluded 15-17 (all NaN) -- trajectory 1 is tracked throughout, so
    # these are single-task frames.
    for f in (15, 16, 17):
        rows.append(
            dict(
                FrameID=f,
                TrajectoryID=2,
                X=float("nan"),
                Y=float("nan"),
                Theta=float("nan"),
                State="occluded",
                DetectionID=None,
            )
        )
    for f in range(18, 25):
        rows.append(_traj2_track(f))

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Params + driver
# ---------------------------------------------------------------------------


def build_params(
    cnn_model_path: Path, headtail_model_path: Path, output_dir: Path
) -> dict:
    return {
        "INDIVIDUAL_DATASET_OUTPUT_DIR": str(output_dir),
        "INDIVIDUAL_DATASET_NAME": "golden_run",
        "ENABLE_INDIVIDUAL_IMAGE_SAVE": True,
        "REFERENCE_BODY_SIZE": REFERENCE_BODY_SIZE,
        "RESIZE_FACTOR": 1.0,
        "RUNTIME_TIER": "cpu",
        "ENABLE_POSE_EXTRACTOR": False,
        "USE_APRILTAGS": False,
        "CNN_CLASSIFIERS": [
            {
                "label": "identity",
                "model_path": str(cnn_model_path),
                "confidence": 0.5,
                "batch_size": 8,
                "scoring_mode": "atomic",
            }
        ],
        "YOLO_HEADTAIL_MODEL_PATH": str(headtail_model_path),
        "YOLO_HEADTAIL_CONF_THRESHOLD": 0.5,
        "HEADTAIL_BATCH_SIZE": 8,
        "INTERP_POSE_INFERENCE_BATCH_SIZE": 8,
    }


def build_fixture(
    root: Path,
    *,
    cnn_model_src: Path | None = None,
    headtail_model_src: Path | None = None,
) -> tuple[Path, Path, dict]:
    """Build (or reuse cached) CSV/video/model fixtures under ``root``.

    ``cnn_model_src``/``headtail_model_src``, when given, are copied in
    verbatim instead of being freshly (randomly) initialized -- this is how
    the golden-capture run and the "current code" test run share BIT-
    IDENTICAL classifier weights despite running as two separate Python
    processes against two different checkouts. Without this, comparing CNN/
    head-tail output between golden and current would be comparing two
    independently-random-initialized models, which is not a meaningful
    equivalence check.

    Returns (csv_path, video_path, params).
    """
    root.mkdir(parents=True, exist_ok=True)
    if cnn_model_src is not None:
        import shutil

        cnn_model_path = root / "cnn_identity.pth"
        shutil.copy(cnn_model_src, cnn_model_path)
    else:
        cnn_model_path = build_cnn_identity_model(root / "cnn_identity.pth")
    if headtail_model_src is not None:
        import shutil

        headtail_model_path = root / "headtail.pth"
        shutil.copy(headtail_model_src, headtail_model_path)
    else:
        headtail_model_path = build_headtail_model(root / "headtail.pth")
    csv_path = build_tracking_csv(root / "tracking.csv")
    video_path = build_synthetic_video(root / "video.mp4")
    out_dir = root / "output"
    params = build_params(cnn_model_path, headtail_model_path, out_dir)
    return csv_path, video_path, params


def run_harness(root: Path, **fixture_kwargs):
    """Build the fixture and run ``run_interpolated_crops`` against it.

    Returns the finished-payload dict.
    """
    from hydra_suite.core.post.interpolated_crops import run_interpolated_crops

    csv_path, video_path, params = build_fixture(root, **fixture_kwargs)
    detection_cache_path = str(root / "no_such_cache.npz")  # forces fallback sizing
    return run_interpolated_crops(
        str(csv_path),
        str(video_path),
        detection_cache_path,
        params,
    )


def collect_golden(root: Path, golden_dir: Path) -> None:
    """Run the harness and copy its 4 artifact CSVs into ``golden_dir`` under
    fixed names, matching the brief's Step 1 (interpolated_pose.csv,
    interpolated_cnn_<label>.csv, interpolated_tags.csv,
    interpolated_headtail.csv). Pose/tag CSVs are intentionally not produced
    (see module docstring) -- their absence is itself part of the captured
    golden state and is asserted by the characterization test.
    """
    import shutil

    golden_dir.mkdir(parents=True, exist_ok=True)
    payload = run_harness(root)

    # Persist the EXACT model weight files used for this capture, so a later
    # "current code" run can reuse them verbatim rather than freshly (and
    # differently) randomly initializing its own copies.
    shutil.copy(root / "cnn_identity.pth", golden_dir / "cnn_identity.pth")
    shutil.copy(root / "headtail.pth", golden_dir / "headtail.pth")

    mapping = {
        "pose_csv_path": "interpolated_pose.csv",
        "tag_csv_path": "interpolated_tags.csv",
        "headtail_csv_path": "interpolated_headtail.csv",
        "roi_csv_path": "interpolated_rois.csv",
        "mapping_path": "interpolated_mapping.csv",
    }
    for key, dest_name in mapping.items():
        src = payload.get(key)
        if src and os.path.exists(src):
            shutil.copy(src, golden_dir / dest_name)

    for label, path in (payload.get("cnn_csv_paths") or {}).items():
        if path and os.path.exists(path):
            shutil.copy(path, golden_dir / f"interpolated_cnn_{label}.csv")

    # Record the payload's scalar summary too, for sanity-checking the
    # capture (row counts etc.) without needing to re-derive it.
    import json

    summary = {
        k: v for k, v in payload.items() if isinstance(v, (int, float, str, bool))
    }
    (golden_dir / "_payload_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/interp_golden_root")
    golden_dir = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/interp_golden_out")
    )
    collect_golden(root, golden_dir)
    print(f"Golden artifacts written to {golden_dir}")
