#!/usr/bin/env python
"""One-shot migration of legacy runtime-vocabulary config/preset JSON files.

FT5 made config loading raise a loud ``ValueError`` when ``runtime_tier`` is
missing (clean break from the old per-stage ``compute_runtime`` strings).
This script lets users migrate old config/preset JSON files to the Gen-2
``runtime_tier`` field once, offline.

This script is deliberately SELF-CONTAINED: it does not import
``hydra_suite.core.inference.config.migrate_runtime_to_tier`` (or anything
else from ``hydra_suite``) so that it keeps working even after that helper is
deleted in a later cleanup pass. The tier-mapping logic is embedded below.

Usage:
    python scripts/migrate_runtime_config.py <file.json> [more.json ...]

For each file: read JSON, migrate it (see ``migrate_config_dict``), and write
the result back in place. A ``.bak`` sidecar with the original contents is
created next to the file (an existing ``.bak`` is never overwritten).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Deprecated per-stage / legacy keys stripped wherever they appear (top-level
# and nested).
_DEPRECATED_KEYS = {
    "compute_runtime",
    "detect_compute_runtime",
    "obb_compute_runtime",
    "pose_runtime_flavor",
    "pose_sleap_device",
    "COMPUTE_RUNTIME",
    "OBB_DIRECT_COMPUTE_RUNTIME",
    "OBB_SEQUENTIAL_DETECT_COMPUTE_RUNTIME",
    "OBB_SEQUENTIAL_OBB_COMPUTE_RUNTIME",
    "HEADTAIL_COMPUTE_RUNTIME",
    "POSE_YOLO_COMPUTE_RUNTIME",
    "POSE_SLEAP_COMPUTE_RUNTIME",
    "CNN_PHASES_COMPUTE_RUNTIME",
}

_VALID_TIERS = {"cpu", "gpu", "gpu_fast"}

# Fast (ONNX / TensorRT) runtimes map straight to the "gpu_fast" tier.
_FAST_RUNTIMES = {"onnx_cpu", "onnx_cuda", "onnx_coreml", "tensorrt"}

# Native GPU-class runtimes map to the "gpu" tier.
_GPU_RUNTIMES = {"cuda", "mps"}


def _normalize_runtime(raw: Any) -> str | None:
    """Normalize a legacy runtime string to its canonical form.

    Mirrors the retired ``migrate_runtime_to_tier`` normalization: lowercase,
    strip whitespace, collapse aliases ("onnx_mps" -> "onnx_coreml",
    "trt"/"tensor_rt" -> "tensorrt"), and strip a device index suffix like
    "cuda:0" -> "cuda".
    """
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value.startswith("cuda:"):
        value = "cuda"
    if value in ("onnx_mps",):
        value = "onnx_coreml"
    if value in ("trt", "tensor_rt"):
        value = "tensorrt"
    return value


def _tier_from_runtimes(runtimes: set[str]) -> str:
    """Compute a Gen-2 runtime_tier from a set of normalized legacy runtimes."""
    if runtimes & _FAST_RUNTIMES:
        return "gpu_fast"
    if runtimes & _GPU_RUNTIMES:
        return "gpu"
    return "cpu"


def _collect_runtimes(d: dict) -> set[str]:
    """Collect every legacy runtime string found anywhere relevant in ``d``.

    Handles both the flat-preset format (top-level ``compute_runtime`` /
    ``pose_runtime_flavor``) and the nested InferenceConfig format
    (``obb.direct.compute_runtime``, etc.).
    """
    collected: set[str] = set()

    def _add(raw: Any) -> None:
        normalized = _normalize_runtime(raw)
        if normalized:
            collected.add(normalized)

    # --- Flat preset format ---
    _add(d.get("compute_runtime"))
    _add(d.get("pose_runtime_flavor"))

    # --- Nested InferenceConfig format ---
    obb = d.get("obb")
    if isinstance(obb, dict):
        direct = obb.get("direct")
        if isinstance(direct, dict):
            _add(direct.get("compute_runtime"))
        sequential = obb.get("sequential")
        if isinstance(sequential, dict):
            _add(sequential.get("detect_compute_runtime"))
            _add(sequential.get("obb_compute_runtime"))

    headtail = d.get("headtail")
    if isinstance(headtail, dict):
        _add(headtail.get("compute_runtime"))

    cnn_phases = d.get("cnn_phases")
    if isinstance(cnn_phases, list):
        for phase in cnn_phases:
            if isinstance(phase, dict):
                _add(phase.get("compute_runtime"))

    pose = d.get("pose")
    if isinstance(pose, dict):
        yolo = pose.get("yolo")
        if isinstance(yolo, dict):
            _add(yolo.get("compute_runtime"))
        sleap = pose.get("sleap")
        if isinstance(sleap, dict):
            _add(sleap.get("compute_runtime"))

    return collected


def _strip_deprecated(value: Any) -> Any:
    """Recursively return a copy of ``value`` with deprecated keys removed."""
    if isinstance(value, dict):
        return {
            key: _strip_deprecated(val)
            for key, val in value.items()
            if key not in _DEPRECATED_KEYS
        }
    if isinstance(value, list):
        return [_strip_deprecated(item) for item in value]
    return value


def migrate_config_dict(d: dict) -> dict:
    """Migrate a legacy config/preset dict to the Gen-2 ``runtime_tier`` scheme.

    Returns a NEW dict (the input is never mutated):
      - If ``d`` already has a valid ``runtime_tier`` (cpu/gpu/gpu_fast), it is
        kept as-is (idempotent), and deprecated keys are still stripped.
      - Otherwise the tier is computed from legacy per-stage runtime strings
        collected from both the flat-preset and nested-InferenceConfig
        formats, and ``runtime_tier`` is set on the returned dict.
    """
    existing_tier = d.get("runtime_tier")
    if isinstance(existing_tier, str) and existing_tier in _VALID_TIERS:
        tier = existing_tier
    else:
        tier = _tier_from_runtimes(_collect_runtimes(d))

    migrated = _strip_deprecated(d)
    migrated["runtime_tier"] = tier
    return migrated


def _migrate_file(path: Path) -> None:
    original_text = path.read_text()
    original = json.loads(original_text)
    old_tier = original.get("runtime_tier")

    migrated = migrate_config_dict(original)

    bak_path = path.with_suffix(path.suffix + ".bak")
    if not bak_path.exists():
        bak_path.write_text(original_text)

    path.write_text(json.dumps(migrated, indent=2) + "\n")

    print(f"{path}: {old_tier!r} -> runtime_tier={migrated['runtime_tier']!r}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(
            "usage: python scripts/migrate_runtime_config.py <file.json> [more.json ...]",
            file=sys.stderr,
        )
        return 1

    for arg in argv:
        _migrate_file(Path(arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
