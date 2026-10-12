"""Shared constants for PoseKit GUI: file extensions, defaults, and thresholds."""

from typing import Set

IMG_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PROJECT_NAME = "pose_project.json"
DEFAULT_DATASET_IMAGES_DIR = "images"
DEFAULT_POSEKIT_PROJECT_DIR = "posekit_project"
DEFAULT_SKELETON_DIRNAME = "configs"
DEFAULT_KPT_RADIUS = 5.0
DEFAULT_LABEL_FONT_SIZE = 10
DEFAULT_AUTOSAVE_DELAY_MS = 3000
LARGE_DATASET_SIEVE_THRESHOLD = 10000
DEFAULT_EDGE_WIDTH = 2.0

# Frame-list sort mode that shows (and navigates) Smart Select's batch order.
SORT_SMART_BATCH = "Smart Select batch"
