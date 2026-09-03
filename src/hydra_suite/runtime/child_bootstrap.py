"""Minimal executable that installs child limits before loading the workload."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from .resource_limits import apply_child_limits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--address-space-bytes", type=int)
    parser.add_argument("--mps-high-watermark-ratio", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("a child command is required after --")
    apply_child_limits(
        address_space_bytes=args.address_space_bytes,
        mps_high_watermark_ratio=args.mps_high_watermark_ratio,
    )
    os.execvpe(command[0], command, os.environ)
    return 127  # pragma: no cover - exec replaces the process


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())
