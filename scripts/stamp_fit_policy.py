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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

POLICIES = ("letterbox", "squash")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "artifact", help="Path to a .pth checkpoint or .multihead.json manifest"
    )
    ap.add_argument("--policy", required=True, choices=POLICIES)
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
        data["fit_policy"] = args.policy
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        for factor_model in data.get("factor_models") or []:
            if not isinstance(factor_model, dict):
                continue
            rel = factor_model.get("path")
            if not rel:
                continue
            sub = (p.parent / str(rel)).expanduser()
            if sub.exists():
                rc = main([str(sub), "--policy", args.policy])
                if rc != 0:
                    return rc
    else:
        import torch

        # Checkpoints hold dict/list metadata alongside tensors, so this must
        # unpickle the full object -- trusted local artifacts only.
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            print(f"{p}: not a dict checkpoint", file=sys.stderr)
            return 2
        ckpt["fit_policy"] = args.policy
        torch.save(ckpt, str(p))

    print(f"stamped {p} fit_policy={args.policy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
