"""Integration helpers for invoking X-AnyLabeling conversion workflows.

This module provides a thin wrapper around the external `xanylabeling` CLI to
convert project labels into YOLO-OBB format from within MAT workflows.
"""

import logging
import subprocess
from pathlib import Path

from hydra_suite.utils.conda_utils import conda_subprocess_kwargs

logger = logging.getLogger(__name__)

HARD_CODED_CMD = [
    "xanylabeling",
    "convert",
    "--task",
    "xlabel2yolo",
    "--mode",
    "obb",
    "--images",
    "./images",
    "--labels",
    "./images",
    "--output",
    "./labels",
    "--classes",
    "classes.txt",
]


def convert_project(
    project_dir: str, output_dir: str, conda_env: str | None = None
) -> tuple[bool, str]:
    """Convert an X-AnyLabeling project to YOLO-OBB using the hardcoded CLI.

    Args:
        project_dir: path to X-AnyLabeling project folder
        output_dir: destination folder for converted dataset
        conda_env: conda env name for running xanylabeling

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
    cmd.extend(HARD_CODED_CMD)

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
