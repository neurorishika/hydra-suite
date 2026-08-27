#!/usr/bin/env python
"""Stamp an existing classifier artifact with the Layer-2 fit policy it was trained under.

Usage: python scripts/stamp_fit_policy.py <model.pth | bundle.multihead.json> --policy letterbox|squash

Models trained before 2026-08-05 (commit 3a2163ac) used torchvision
Resize((sz,sz)) -> squash. Models trained after that with the shared
CanonicalFitTransform (training/canonical_transform.py) -> letterbox.

Without an explicit fit_policy stamp, Layer-2 preprocessing at inference
(``ClassifierMetadata.fit_policy``, see
src/hydra_suite/core/individual/classification/backend.py) assumes the
legacy "squash" default and logs a loud warning -- this script lets a human
who knows what an artifact was actually trained with stamp it explicitly,
silencing that warning and (if it was actually a letterbox artifact)
fixing a real accuracy regression.

Before writing, a ``<original>.bak`` sibling is created with the original
bytes (skipped if one already exists, to avoid clobbering a prior backup --
see --help for behavior). Pass --dry-run to preview what would be
stamped/backed-up without writing anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

POLICIES = ("letterbox", "squash")


def _backup(p: Path, *, dry_run: bool) -> None:
    """Copy ``p`` to ``p`` + ``.bak`` before it is overwritten.

    If a ``.bak`` already exists, skip (warn) rather than overwrite it --
    the first backup is presumed to be the one worth keeping (the original,
    pre-any-stamping artifact); a later run must not clobber it with an
    already-stamped (or otherwise modified) copy.
    """
    bak = p.with_name(p.name + ".bak")
    if bak.exists():
        print(f"  (backup already exists, not overwriting: {bak})", file=sys.stderr)
        return
    if dry_run:
        print(f"  [dry-run] would back up {p} -> {bak}")
        return
    shutil.copy2(p, bak)
    print(f"  backed up {p} -> {bak}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "artifact", help="Path to a .pth checkpoint or .multihead.json manifest"
    )
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be stamped/backed-up without writing anything",
    )
    args = ap.parse_args(argv)
    p = Path(args.artifact)

    if not p.exists():
        print(f"{p}: no such file", file=sys.stderr)
        return 2

    if p.name.lower().endswith(".json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            print(f"{p}: not a dict manifest", file=sys.stderr)
            return 2
        _backup(p, dry_run=args.dry_run)
        if args.dry_run:
            print(f"[dry-run] would stamp {p} fit_policy={args.policy}")
        else:
            data["fit_policy"] = args.policy
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"stamped {p} fit_policy={args.policy}")
        for factor_model in data.get("factor_models") or []:
            if not isinstance(factor_model, dict):
                continue
            rel = factor_model.get("path")
            if not rel:
                continue
            sub = (p.parent / str(rel)).expanduser()
            if sub.exists():
                sub_argv = [str(sub), "--policy", args.policy]
                if args.dry_run:
                    sub_argv.append("--dry-run")
                rc = main(sub_argv)
                if rc != 0:
                    return rc
        return 0
    else:
        import torch

        # Checkpoints hold dict/list metadata alongside tensors, so this must
        # unpickle the full object -- trusted local artifacts only.
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            print(f"{p}: not a dict checkpoint", file=sys.stderr)
            return 2
        _backup(p, dry_run=args.dry_run)
        if args.dry_run:
            print(f"[dry-run] would stamp {p} fit_policy={args.policy}")
            return 0
        ckpt["fit_policy"] = args.policy
        torch.save(ckpt, str(p))

    print(f"stamped {p} fit_policy={args.policy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
