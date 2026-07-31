"""SLEAP-specific utility functions for model export and config extraction."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hydra_suite.core.identity.pose.utils import (
    align_hw_to_stride,
    load_structured_config,
    nested_get,
    safe_pos_int,
)
from hydra_suite.utils.conda_utils import conda_subprocess_kwargs

logger = logging.getLogger(__name__)


def extract_hw_from_sleap_config(
    cfg: Dict[str, Any],
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """Extract height/width and stride from SLEAP config."""
    prep = nested_get(cfg, ["data_config", "preprocessing"])
    hw: Optional[Tuple[int, int]] = None
    if isinstance(prep, dict):
        crop_size = prep.get("crop_size")
        if isinstance(crop_size, (list, tuple)) and len(crop_size) >= 2:
            ch = safe_pos_int(crop_size[0])
            cw = safe_pos_int(crop_size[1])
            if ch and cw:
                hw = (int(ch), int(cw))
        elif isinstance(crop_size, dict):
            ch = safe_pos_int(crop_size.get("height", crop_size.get("h")))
            cw = safe_pos_int(crop_size.get("width", crop_size.get("w")))
            if ch and cw:
                hw = (int(ch), int(cw))
        else:
            c = safe_pos_int(crop_size)
            if c:
                hw = (int(c), int(c))

        if hw is None:
            mh = safe_pos_int(prep.get("max_height"))
            mw = safe_pos_int(prep.get("max_width"))
            if mh and mw:
                hw = (int(mh), int(mw))
            elif mh:
                hw = (int(mh), int(mh))
            elif mw:
                hw = (int(mw), int(mw))

    stride = safe_pos_int(
        nested_get(cfg, ["model_config", "backbone_config", "unet", "max_stride"])
    )
    if stride is None:
        stride = safe_pos_int(
            nested_get(
                cfg, ["model_config", "backbone_config", "unet", "output_stride"]
            )
        )
    return hw, stride


def derive_sleap_export_input_hw(model_path_str: str) -> Optional[Tuple[int, int]]:
    """Derive input height/width for SLEAP model export from training config."""
    model_path = Path(str(model_path_str or "")).expanduser().resolve()
    model_dir = model_path if model_path.is_dir() else model_path.parent
    if not model_dir.exists() or not model_dir.is_dir():
        return None

    candidates = [
        model_dir / "training_config.yaml",
        model_dir / "training_config.yml",
        model_dir / "training_config.json",
        model_dir / "initial_config.yaml",
        model_dir / "initial_config.yml",
        model_dir / "initial_config.json",
    ]
    chosen_hw: Optional[Tuple[int, int]] = None
    stride = 32
    for cfg_path in candidates:
        if not cfg_path.exists() or not cfg_path.is_file():
            continue
        cfg = load_structured_config(cfg_path)
        if not isinstance(cfg, dict):
            continue
        hw, stride_candidate = extract_hw_from_sleap_config(cfg)
        if stride_candidate is not None and stride_candidate > 0:
            stride = int(stride_candidate)
        if hw is not None and chosen_hw is None:
            chosen_hw = hw
        if chosen_hw is not None and stride is not None:
            break

    if chosen_hw is None:
        return None
    return align_hw_to_stride(chosen_hw, stride=max(1, int(stride)))


def looks_like_sleap_export_path(path_str: str, runtime_flavor: str) -> bool:
    """Check if path looks like a SLEAP export directory/file."""
    path = Path(path_str).expanduser().resolve()
    runtime = str(runtime_flavor or "").strip().lower()
    if not path.exists():
        return False

    engine_suffixes = {".engine", ".trt"}
    if path.is_file():
        if runtime == "onnx":
            return path.suffix.lower() == ".onnx"
        if runtime == "tensorrt":
            return path.suffix.lower() in engine_suffixes
        return path.suffix.lower() in {".onnx", *engine_suffixes}

    has_meta = (path / "metadata.json").exists() or (
        path / "export_metadata.json"
    ).exists()
    if runtime == "onnx":
        has_artifact = any(path.rglob("*.onnx"))
    elif runtime == "tensorrt":
        has_artifact = any(path.rglob("*.engine")) or any(path.rglob("*.trt"))
    else:
        has_artifact = (
            any(path.rglob("*.onnx"))
            or any(path.rglob("*.engine"))
            or any(path.rglob("*.trt"))
        )
    return bool(has_meta or has_artifact)


def normalize_export_result_path(
    export_result: Any, expected_suffix: str
) -> Optional[Path]:
    """Normalize export result to Path with expected suffix."""
    candidates: List[Path] = []
    if isinstance(export_result, (str, Path)):
        candidates.append(Path(export_result))
    elif isinstance(export_result, (list, tuple)):
        for item in export_result:
            if isinstance(item, (str, Path)):
                candidates.append(Path(item))
    for p in candidates:
        p = p.expanduser().resolve()
        if p.exists() and p.is_file():
            if not expected_suffix or p.suffix.lower() == expected_suffix.lower():
                return p
    for p in candidates:
        parent = p.expanduser().resolve().parent
        if not parent.exists():
            continue
        matches = sorted(parent.glob(f"*{expected_suffix}"))
        if matches:
            return matches[-1].resolve()
    return None


def run_cli_command(cmd: List[str], timeout_sec: int = 1800) -> Tuple[bool, str]:
    """Run CLI command and return (success, output)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            **(conda_subprocess_kwargs() if cmd and cmd[0] == "conda" else {}),
        )
    except Exception as exc:
        return False, str(exc)
    if proc.returncode == 0:
        return True, proc.stdout or ""
    err = (proc.stderr or proc.stdout or "").strip()
    return False, err
