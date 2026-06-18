"""Compare two tracking-output CSVs for equivalence.

Two complementary views:

1. Positional (primary, ID-agnostic): for each frame, match tracked detections
   between the two runs by nearest (X, Y) within a gate. Reports residuals plus
   unmatched counts. Robust to track-ID / trajectory-ID renumbering, which can
   differ even when the underlying detections are identical.

2. Keyed (secondary): when both files share the same key grid
   (FrameID + track/trajectory id), compare column-by-column row-aligned.

Run it twice to interpret results:
  - new_a vs new_b  -> determinism baseline (the noise floor of one pipeline)
  - legacy vs new_a -> equivalence (must be within, or close to, that floor)

Exit code 0 = within tolerance, 1 = differences beyond tolerance.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def ang_diff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def _id_col(df: pd.DataFrame) -> str | None:
    for c in ("TrackID", "TrajectoryID", "DetectionID"):
        if c in df.columns:
            return c
    return None


def positional(a: pd.DataFrame, b: pd.DataFrame, gate: float):
    """Per-frame nearest-position matching of tracked rows (non-NaN X,Y)."""
    av = a.dropna(subset=["X", "Y"])
    bv = b.dropna(subset=["X", "Y"])
    frames = sorted(set(av["FrameID"]) | set(bv["FrameID"]))
    pos_res, ang_res = [], []
    matched = unmatched_a = unmatched_b = 0
    for f in frames:
        fa = av[av["FrameID"] == f]
        fb = bv[bv["FrameID"] == f]
        pa = fa[["X", "Y"]].to_numpy(float)
        pb = fb[["X", "Y"]].to_numpy(float)
        if len(pa) == 0 or len(pb) == 0:
            unmatched_a += len(pa)
            unmatched_b += len(pb)
            continue
        cost = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        if _HAVE_SCIPY:
            ri, ci = linear_sum_assignment(cost)
        else:  # greedy fallback
            ri, ci, used = [], [], set()
            order = np.dstack(
                np.unravel_index(np.argsort(cost, axis=None), cost.shape)
            )[0]
            seen_r = set()
            for r, c in order:
                if r in seen_r or c in used:
                    continue
                seen_r.add(r)
                used.add(c)
                ri.append(r)
                ci.append(c)
            ri, ci = np.array(ri, int), np.array(ci, int)
        ta = fa["Theta"].to_numpy(float)
        tb = fb["Theta"].to_numpy(float)
        mr = 0
        for r, c in zip(ri, ci):
            if cost[r, c] <= gate:
                pos_res.append(cost[r, c])
                if not (np.isnan(ta[r]) or np.isnan(tb[c])):
                    ang_res.append(float(ang_diff(ta[r], tb[c])))
                mr += 1
        matched += mr
        unmatched_a += len(pa) - mr
        unmatched_b += len(pb) - mr
    pos_res = np.array(pos_res) if pos_res else np.array([0.0])
    ang_res = np.array(ang_res) if ang_res else np.array([0.0])
    return {
        "matched": matched,
        "unmatched_legacy": unmatched_a,
        "unmatched_new": unmatched_b,
        "pos_max": float(pos_res.max()),
        "pos_mean": float(pos_res.mean()),
        "pos_p99": float(np.percentile(pos_res, 99)),
        "theta_max": float(ang_res.max()),
        "theta_mean": float(ang_res.mean()),
    }


def keyed(a: pd.DataFrame, b: pd.DataFrame, atol: float):
    idc = _id_col(a)
    if idc is None or "FrameID" not in a.columns or list(a.columns) != list(b.columns):
        return None
    key = ["FrameID", idc]
    a2 = a.sort_values(key).reset_index(drop=True)
    b2 = b.sort_values(key).reset_index(drop=True)
    if a2.shape[0] != b2.shape[0] or not (a2[key].values == b2[key].values).all():
        return {"aligned": False, "rows_legacy": a2.shape[0], "rows_new": b2.shape[0]}
    out = {"aligned": True, "cols": {}}
    for c in a2.columns:
        if c in key:
            continue
        cl, cr = a2[c], b2[c]
        if pd.api.types.is_numeric_dtype(cl):
            l = cl.to_numpy(float)
            r = cr.to_numpy(float)
            both = ~np.isnan(l) & ~np.isnan(r)
            d = np.abs(
                (ang_diff(l[both], r[both]) if c == "Theta" else l[both] - r[both])
            )
            out["cols"][c] = {
                "max": float(d.max()) if d.size else 0.0,
                "nan_mismatch": int((np.isnan(l) != np.isnan(r)).sum()),
            }
        else:
            mism = int((cl.fillna("∅").astype(str) != cr.fillna("∅").astype(str)).sum())
            out["cols"][c] = {"mismatch": mism}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy")
    ap.add_argument("new")
    ap.add_argument(
        "--gate", type=float, default=2.0, help="Max px to call a positional match."
    )
    ap.add_argument(
        "--pos-atol", type=float, default=0.5, help="Pass if pos_p99 <= this (px)."
    )
    ap.add_argument(
        "--theta-atol",
        type=float,
        default=0.05,
        help="Pass if theta_mean <= this (rad).",
    )
    args = ap.parse_args()

    a = pd.read_csv(args.legacy)
    b = pd.read_csv(args.new)
    print(f"legacy {a.shape}  new {b.shape}")
    if list(a.columns) != list(b.columns):
        print("  note: column sets differ")
        print("   legacy:", list(a.columns))
        print("   new   :", list(b.columns))

    p = positional(a, b, args.gate)
    print("\n[positional, ID-agnostic]")
    print(
        f"  matched={p['matched']}  unmatched_legacy={p['unmatched_legacy']}  unmatched_new={p['unmatched_new']}"
    )
    print(
        f"  pos |Δ| (px): max={p['pos_max']:.3e} mean={p['pos_mean']:.3e} p99={p['pos_p99']:.3e}"
    )
    print(f"  theta |Δ| (rad): max={p['theta_max']:.3e} mean={p['theta_mean']:.3e}")

    k = keyed(a, b, args.pos_atol)
    if k is None:
        print("\n[keyed] skipped (different schema / no shared id column)")
    elif not k["aligned"]:
        print(
            f"\n[keyed] not aligned: legacy {k['rows_legacy']} rows vs new {k['rows_new']} rows"
        )
    else:
        print("\n[keyed] aligned; per-column differences:")
        for c, v in k["cols"].items():
            if "max" in v and (v["max"] > args.pos_atol or v["nan_mismatch"]):
                print(
                    f"  {c:24s} max|Δ|={v['max']:.3e} nan_mismatch={v['nan_mismatch']}"
                )
            elif "mismatch" in v and v["mismatch"]:
                print(f"  {c:24s} mismatched rows={v['mismatch']}")

    unmatched = p["unmatched_legacy"] + p["unmatched_new"]
    ok = (
        p["pos_p99"] <= args.pos_atol
        and p["theta_mean"] <= args.theta_atol
        and unmatched == 0
    )
    print(
        f"\nVERDICT: {'EQUIVALENT ✅' if ok else 'DIFFERENCES ❌'}"
        f"  (pos_p99<={args.pos_atol}px, theta_mean<={args.theta_atol}rad, unmatched==0)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
