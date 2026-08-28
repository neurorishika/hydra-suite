"""Integration helpers for invoking X-AnyLabeling conversion workflows.

This module provides a thin wrapper around the external `xanylabeling` CLI to
convert project labels into YOLO-OBB format from within MAT workflows.
"""

import logging
import subprocess
from pathlib import Path

from hydra_suite.utils.conda_utils import conda_subprocess_kwargs

logger = logging.getLogger(__name__)


def _build_convert_cmd(mode: str) -> list[str]:
    """Build the xlabel2yolo conversion CLI args for the given shape *mode*.

    *mode* must be one of the x-anylabeling ``convert`` modes for
    xlabel2yolo: "detect" (AABB), "obb" (rotated box), or "segment"
    (polygon/mask) -- see ``xal_mode_for_level`` in
    ``detectkit/gui/panels/dataset_panel.py``, which is the single place
    that maps a DetectKit geometry level to this mode string.
    """
    return [
        "xanylabeling",
        "convert",
        "--task",
        "xlabel2yolo",
        "--mode",
        mode,
        "--images",
        "./images",
        "--labels",
        "./images",
        "--output",
        "./labels",
        "--classes",
        "classes.txt",
    ]


# Preserved for backwards compatibility with existing importers; reflects
# the "obb" default. Prefer calling convert_project(..., mode=...) directly.
HARD_CODED_CMD = _build_convert_cmd("obb")


def convert_project(
    project_dir: str,
    output_dir: str,
    conda_env: str | None = None,
    mode: str = "obb",
) -> tuple[bool, str]:
    """Convert an X-AnyLabeling project to YOLO using the x-anylabeling CLI.

    Args:
        project_dir: path to X-AnyLabeling project folder
        output_dir: destination folder for converted dataset
        conda_env: conda env name for running xanylabeling
        mode: xlabel2yolo shape mode -- "detect", "obb", or "segment". Must
            match the shape type actually present in the project's xlabel
            JSON (i.e. the source's geometry level), or the converter will
            not find the shapes it's looking for.

    Returns:
        (success: bool, log: str)
    """
    project_path = Path(project_dir)
    if not project_path.exists():
        return False, f"Project path not found: {project_dir}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = []
    if conda_env:
        cmd.extend(["conda", "run", "-n", conda_env])
    cmd.extend(_build_convert_cmd(mode))

    logger.info("Running X-AnyLabeling conversion: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_path),
            capture_output=True,
            text=True,
            timeout=3600,
            **(conda_subprocess_kwargs() if conda_env else {}),
        )
    except Exception as e:
        return False, f"Failed to run conversion: {e}"

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if result.returncode != 0:
        return False, f"Conversion failed\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    # Ensure labels were generated in project ./labels directory
    labels_dir = project_path / "labels"
    if not labels_dir.exists():
        return False, f"Conversion output not found at {labels_dir}"

    return True, f"Conversion succeeded\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
