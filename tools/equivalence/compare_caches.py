"""Compare two legacy-format detection caches (*_detection_cache_*.npz).

Decisive diagnostic for the heading (theta) / lifecycle divergence: both the
legacy and new pipelines write this same cache and feed it to the SAME tracking
code. So:

  - If the caches MATCH (position, theta, order, heading) -> the divergence is in
    tracking input handling, not detection.
  - If they DIFFER (theta off by pi, different order, different heading hints) ->
    the divergence is at the detection/cache-write stage.

Per frame it reports: detection count, whether detections are in the SAME ORDER,
position residual (order-independent nearest match), and theta residual both raw
and modulo-pi (to separate a 180-degree direction flip from a real angle change).

Usage:
  python tools/equivalence/compare_caches.py LEGACY.npz NEW.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def ang_diff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def frame_ids(npz) -> list[str]:
    return sorted(k[:-5] for k in npz.files if k.endswith("_meas"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy")
    ap.add_argument("new")
    ap.add_argument("--pos-tol", type=float, default=0.5)
    args = ap.parse_args()

    # Caches are produced by our own pipeline; we only read plain float arrays
    # (_meas, _heading_hints), so pickle is neither needed nor enabled.
    a = np.load(args.legacy)
    b = np.load(args.new)
    fa, fb = frame_ids(a), frame_ids(b)
    print(f"legacy frames={len(fa)}  new frames={len(fb)}")
    common = [f for f in fa if f in set(fb)]
    if len(fa) != len(fb):
        print(f"!! frame-set differs; comparing {len(common)} common frames")

    tot = dict(
        frames=0,
        count_mismatch=0,
        order_mismatch=0,
        pos_unmatched=0,
        theta_flip=0,
        theta_real=0,
        heading_mismatch=0,
    )
    worst_pos = 0.0
    worst_theta_modpi = 0.0
    examples = []

    for f in common:
        ma = a[f"{f}_meas"].reshape(-1, 3)
        mb = b[f"{f}_meas"].reshape(-1, 3)
        tot["frames"] += 1
        if ma.shape[0] != mb.shape[0]:
            tot["count_mismatch"] += 1
            if len(examples) < 8:
                examples.append(f"{f}: count {ma.shape[0]} vs {mb.shape[0]}")
            continue
        if ma.shape[0] == 0:
            continue

        # Same-order check (positions identical index-for-index)
        if ma.shape == mb.shape and np.allclose(
            ma[:, :2], mb[:, :2], atol=args.pos_tol
        ):
            same_order = True
        else:
            same_order = False
            tot["order_mismatch"] += 1

        # Order-independent nearest-position matching
        cost = np.linalg.norm(ma[:, None, :2] - mb[None, :, :2], axis=2)
        bi = cost.argmin(axis=1)
        for i, j in enumerate(bi):
            if cost[i, j] > args.pos_tol:
                tot["pos_unmatched"] += 1
                continue
            worst_pos = max(worst_pos, float(cost[i, j]))
            raw = float(ang_diff(ma[i, 2], mb[j, 2]))
            modpi = min(raw, abs(np.pi - raw))
            worst_theta_modpi = max(worst_theta_modpi, modpi)
            if raw > 0.05 and abs(raw - np.pi) < 0.05:
                tot["theta_flip"] += 1
            elif raw > 0.05:
                tot["theta_real"] += 1

        # heading hints (directed orientation) if present
        ha = a[f"{f}_heading_hints"] if f"{f}_heading_hints" in a.files else None
        hb = b[f"{f}_heading_hints"] if f"{f}_heading_hints" in b.files else None
        if ha is not None and hb is not None:
            ha = np.asarray(ha).ravel()
            hb = np.asarray(hb).ravel()
            if (
                ha.shape == hb.shape
                and ha.size
                and not np.allclose(ang_diff(ha, hb), 0, atol=0.05)
            ):
                tot["heading_mismatch"] += 1

        if not same_order and len(examples) < 8:
            examples.append(f"{f}: positions match but ORDER differs")

    print("\n=== per-frame summary ===")
    for k, v in tot.items():
        print(f"  {k:18s} {v}")
    print(f"  worst pos |Δ|        {worst_pos:.3e} px")
    print(
        f"  worst theta |Δ| modπ {worst_theta_modpi:.3e} rad (≈0 ⇒ diffs are pure 180° flips)"
    )
    if examples:
        print("\n  examples:")
        for e in examples:
            print(f"    {e}")

    detections_identical = (
        tot["count_mismatch"] == 0
        and tot["pos_unmatched"] == 0
        and tot["theta_real"] == 0
        and tot["heading_mismatch"] == 0
    )
    print("\nINTERPRETATION:")
    if detections_identical and tot["theta_flip"] == 0 and tot["order_mismatch"] == 0:
        print("  caches IDENTICAL ⇒ divergence is in TRACKING, not detection.")
    elif detections_identical and tot["order_mismatch"] > 0:
        print("  positions/angles match but ORDER differs ⇒ detection ordering drives")
        print("  track seeding/first-frame heading. Fix ordering to match legacy.")
    elif tot["theta_flip"] > 0 and tot["theta_real"] == 0:
        print("  detections match in position; theta differs by pure 180° flips ⇒")
        print(
            "  heading-direction is written differently at the DETECTION/cache stage."
        )
    else:
        print("  detections differ (count/position/angle) ⇒ root cause is in the OBB/")
        print("  filtering/cache-write stage, upstream of tracking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
