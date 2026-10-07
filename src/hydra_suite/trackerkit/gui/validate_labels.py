#!/usr/bin/env python3
"""
Utility script to validate YOLO pose labels for NaN and numerical issues.
Run this before training to catch problematic labels.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from hydra_suite.utils.hidden_files import is_hidden_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _validate_class_id(parts: List[str], issues: List[str]) -> bool:
    """Validate the class ID field. Returns False if parsing fails fatally."""
    try:
        cls = int(parts[0])
        if cls < 0:
            issues.append(f"Negative class ID: {cls}")
        return True
    except Exception:
        return False


def _validate_bbox(
    parts: List[str], issues: List[str], min_bbox_dim: float, epsilon: float
) -> bool:
    """Validate bbox coordinates. Returns False if parsing fails fatally."""
    try:
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])
    except Exception:
        return False

    for name, val in [("cx", cx), ("cy", cy), ("bw", bw), ("bh", bh)]:
        if not np.isfinite(val):
            issues.append(f"Non-finite bbox {name}: {val}")

    if not (-epsilon <= cx <= 1.0 + epsilon):
        issues.append(f"Bbox cx out of range [0, 1]: {cx:.6f}")
    if not (-epsilon <= cy <= 1.0 + epsilon):
        issues.append(f"Bbox cy out of range [0, 1]: {cy:.6f}")
    if not (0.0 <= bw <= 1.0 + epsilon):
        issues.append(f"Bbox w out of range [0, 1]: {bw:.6f}")
    if not (0.0 <= bh <= 1.0 + epsilon):
        issues.append(f"Bbox h out of range [0, 1]: {bh:.6f}")

    if bw < min_bbox_dim:
        issues.append(f"Bbox width too small: {bw:.6f} < {min_bbox_dim}")
    if bh < min_bbox_dim:
        issues.append(f"Bbox height too small: {bh:.6f} < {min_bbox_dim}")
    return True


def _validate_keypoints(
    parts: List[str], kpt_count: int, issues: List[str], epsilon: float
) -> None:
    """Validate keypoint coordinates, visibility, and ranges."""
    visible_count = 0
    for i in range(5, 5 + 3 * kpt_count, 3):
        kpt_idx = (i - 5) // 3
        try:
            x = float(parts[i])
            y = float(parts[i + 1])
            v = int(float(parts[i + 2]))
        except Exception:
            issues.append(f"Cannot parse keypoint {kpt_idx}")
            continue

        if not np.isfinite(x):
            issues.append(f"Non-finite x for keypoint {kpt_idx}: {x}")
        if not np.isfinite(y):
            issues.append(f"Non-finite y for keypoint {kpt_idx}: {y}")

        if v not in (0, 1, 2):
            issues.append(
                f"Invalid visibility {v} for keypoint {kpt_idx} (must be 0, 1, or 2)"
            )

        if v > 0:
            visible_count += 1
            if not (-epsilon <= x <= 1.0 + epsilon):
                issues.append(f"Keypoint {kpt_idx} x out of range: {x:.6f}")
            if not (-epsilon <= y <= 1.0 + epsilon):
                issues.append(f"Keypoint {kpt_idx} y out of range: {y:.6f}")

    if visible_count == 0:
        issues.append("No visible keypoints (all visibility = 0)")


def validate_label_file(
    label_path: Path, kpt_count: int, min_bbox_dim: float = 0.001, epsilon: float = 1e-8
) -> Tuple[bool, List[str]]:
    """
    Validate a single YOLO pose label file.

    Returns:
        (is_valid, list_of_issues)
    """
    issues: List[str] = []

    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return False, [f"Cannot read file: {e}"]

    if not text:
        return False, ["File is empty"]

    parts = text.split()
    expected_parts = 5 + 3 * kpt_count

    if len(parts) < expected_parts:
        return False, [f"Expected {expected_parts} values, got {len(parts)}"]

    if not _validate_class_id(parts, issues):
        return False, ["Invalid class ID"]

    if not _validate_bbox(parts, issues, min_bbox_dim, epsilon):
        return False, ["Cannot parse bbox coordinates"]

    _validate_keypoints(parts, kpt_count, issues, epsilon)

    is_valid = len(issues) == 0
    return is_valid, issues


def validate_dataset(
    labels_dir: Path, kpt_count: int, fix_issues: bool = False, verbose: bool = False
) -> Tuple[int, int, int]:
    """
    Validate all label files in a directory.

    Returns:
        (valid_count, invalid_count, fixed_count)
    """
    label_files = sorted(
        p for p in labels_dir.glob("**/*.txt") if not is_hidden_file(p)
    )

    if not label_files:
        logger.warning(f"No label files found in {labels_dir}")
        return 0, 0, 0

    logger.info(f"Validating {len(label_files)} label files...")

    valid_count = 0
    invalid_count = 0
    fixed_count = 0

    for label_path in label_files:
        is_valid, issues = validate_label_file(label_path, kpt_count)

        if is_valid:
            valid_count += 1
            if verbose:
                logger.info(f"✓ {label_path.name}")
        else:
            invalid_count += 1
            logger.warning(f"✗ {label_path.name}")
            for issue in issues:
                logger.warning(f"  - {issue}")

            if fix_issues:
                # TODO: Implement automatic fixes for common issues
                logger.info("  Auto-fix not yet implemented")

    return valid_count, invalid_count, fixed_count


def main() -> object:
    """main function documentation."""
    parser = argparse.ArgumentParser(
        description="Validate YOLO pose labels for NaN and numerical issues"
    )
    parser.add_argument(
        "labels_dir", type=Path, help="Directory containing label files (.txt)"
    )
    parser.add_argument(
        "--kpt-count", type=int, required=True, help="Number of keypoints per sample"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix issues automatically (not yet implemented)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print validation result for each file"
    )

    args = parser.parse_args()

    if not args.labels_dir.exists():
        logger.error(f"Labels directory not found: {args.labels_dir}")
        return 1

    valid, invalid, fixed = validate_dataset(
        args.labels_dir, args.kpt_count, fix_issues=args.fix, verbose=args.verbose
    )

    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Valid labels:   {valid}")
    logger.info(f"Invalid labels: {invalid}")
    if fixed > 0:
        logger.info(f"Fixed labels:   {fixed}")
    logger.info(f"Total labels:   {valid + invalid}")

    if invalid > 0:
        logger.warning(
            f"\n⚠️  Found {invalid} invalid label(s). Fix these before training!"
        )
        return 1
    else:
        logger.info("\n✅ All labels are valid!")
        return 0


if __name__ == "__main__":
    exit(main())
