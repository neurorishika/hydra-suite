#!/usr/bin/env python
"""Benchmark TrackerKit OBB detection on sampled real video frames.

Examples
--------
python tools/benchmark_trackerkit_obb.py \
  --video /path/to/video.mp4 \
  --model /path/to/obb.pt \
  --runtime tensorrt \
  --batch-size 1 \
  --resize-factor 0.25 \
  --tracking-realtime-mode

python tools/benchmark_trackerkit_obb.py \
  --video /path/to/video.mp4 \
  --model /path/to/obb.pt \
  --headtail-model /path/to/headtail.pth \
  --runtime tensorrt \
  --batch-size 1 \
  --resize-factor 0.25 \
  --tracking-realtime-mode
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

from hydra_suite.trackerkit.obb_runtime_probe import main

if __name__ == "__main__":
    raise SystemExit(main())
