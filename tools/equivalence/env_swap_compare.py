#!/usr/bin/env python3
"""Compare two tracking CSVs produced by different Python ENVIRONMENTS.

Verdicts:
  BYTE-IDENTICAL       files are identical.
  BLAS-FLOOR           identical except float rounding in the columns listed in
                       BLAS_COLUMNS, each within REL_TOL. These come from the
                       Kalman covariance (numba ``@`` -> scipy's BLAS): conda
                       scipy links OpenBLAS, pip scipy on macOS links Accelerate,
                       and float32 GEMM rounds differently. Positions, theta,
                       IDs, states and row counts are untouched. Accepted as the
                       noise floor of an environment swap (user decision,
                       2026-10-03).
  DIFFERS              anything else.

Exit 0 for BYTE-IDENTICAL / BLAS-FLOOR, 1 for DIFFERS. Stdlib only.
"""

from __future__ import annotations

import csv
import sys

BLAS_COLUMNS = {"PositionUncertainty", "AssignmentConfidence"}
REL_TOL = 1e-5


def compare(a_path: str, b_path: str) -> tuple[str, str]:
    with open(a_path, newline="") as fa, open(b_path, newline="") as fb:
        a, b = list(csv.reader(fa)), list(csv.reader(fb))
    if a == b:
        return "BYTE-IDENTICAL", f"{len(a) - 1} rows"
    if len(a) != len(b) or a[0] != b[0]:
        return "DIFFERS", f"rows {len(a) - 1} vs {len(b) - 1} or header changed"
    header = a[0]
    worst: dict[str, float] = {}
    for row, (x, y) in enumerate(zip(a[1:], b[1:]), start=1):
        for col, (u, v) in enumerate(zip(x, y)):
            if u == v:
                continue
            name = header[col]
            if name not in BLAS_COLUMNS:
                return "DIFFERS", f"row {row} column {name}: {u!r} vs {v!r}"
            try:
                fu, fv = float(u), float(v)
            except ValueError:
                return "DIFFERS", f"row {row} column {name}: {u!r} vs {v!r}"
            rel = abs(fu - fv) / max(abs(fu), abs(fv), 1e-30)
            if rel > REL_TOL:
                return (
                    "DIFFERS",
                    f"row {row} column {name}: rel diff {rel:.2e} > {REL_TOL}",
                )
            worst[name] = max(worst.get(name, 0.0), rel)
    detail = ", ".join(f"{k} max rel {v:.1e}" for k, v in sorted(worst.items()))
    return "BLAS-FLOOR", f"{len(a) - 1} rows; only {detail}"


def main(argv: list[str]) -> int:
    verdict, detail = compare(argv[1], argv[2])
    print(f"{verdict} ({detail})")
    return 0 if verdict != "DIFFERS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
