"""Measure each clip's determinism floor with the AUTOTUNER's OWN gate.

``compare.py`` is the repository's tracking-equivalence gate; its angular
criterion is ``theta_mean`` and its ``theta_max`` includes the documented
bistable head/tail pi-flips, so it cannot answer the question the autotuner
cares about: is this clip's own run-to-run angular noise below the tuner's
``EquivalencePolicy.angle_max_tolerance`` (0.05 rad, a **per-row max** after
head/tail-flip exclusion)?

If a clip's own A-vs-A repeat already exceeds that tolerance, the tuner can
never accept any candidate on it -- the clip is simply non-tunable, and that is
a property of the clip, not a bug to be papered over by loosening the gate.

Point this at a ``run_matrix.sh`` output root; it reads each clip's
``new_a``/``new_b`` directories (the two same-tree repeats).

**Read ``angle_max_rad`` and ``position_p99``; treat ``passed``/``unmatched_rows``
as diagnostics, not a verdict.** ``compare_outputs`` matches rows by nearest
(X, Y) and cannot match a row whose X/Y is NaN -- a lost or coasting track --
so it reports those as unmatched and fails, even for two BYTE-IDENTICAL files.
That is not this tool misreporting: it is the shipped behaviour that makes the
tuner abort real calibration with ``baseline_nondeterministic_beyond_contract``
(see the plan's "MPS gate results" section). ``passed=False`` with
``angle_max=0.0`` and ``position_p99=0.0`` is that defect, not clip noise.

Usage::

    PYTHONPATH=src python tools/equivalence/autotune_determinism_floor.py \
        /tmp/equiv_autotune/mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="matrix output root, e.g. /tmp/equiv_autotune/mps")
    ap.add_argument("--a", default="new_a")
    ap.add_argument("--b", default="new_b")
    args = ap.parse_args()

    import pandas as pd

    from hydra_suite.core.inference.autotune.equivalence import (
        CalibrationOutputs,
        EquivalencePolicy,
        compare_outputs,
    )

    root = Path(args.root)
    policy = EquivalencePolicy()
    rows = []
    for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        a_dir, b_dir = clip_dir / args.a, clip_dir / args.b
        if not a_dir.is_dir() or not b_dir.is_dir():
            continue

        def load(run_dir: Path) -> CalibrationOutputs | str:
            final = sorted(run_dir.glob("*_tracking_final.csv"))
            if not final:
                return "no *_tracking_final.csv"
            forward = sorted(run_dir.glob("*_tracking_forward.csv"))
            # Clips with streaming individual analysis emit no forward CSV;
            # comparing final against itself twice would double-count, so the
            # forward slot gets an empty frame with the same columns instead.
            final_df = pd.read_csv(final[0])
            forward_df = (
                pd.read_csv(forward[0]) if forward else final_df.iloc[0:0].copy()
            )
            return CalibrationOutputs(forward_df, final_df)

        left, right = load(a_dir), load(b_dir)
        if isinstance(left, str) or isinstance(right, str):
            rows.append(
                {
                    "clip": clip_dir.name,
                    "error": left if isinstance(left, str) else right,
                }
            )
            continue
        verdict = compare_outputs(
            left, right, policy=policy, for_determinism_floor=True
        )
        rows.append(
            {
                "clip": clip_dir.name,
                "rows_a_forward": int(len(left.forward)),
                "rows_b_forward": int(len(right.forward)),
                "rows_a_final": int(len(left.final)),
                "rows_b_final": int(len(right.final)),
                "position_p99": verdict.position_p99,
                "angle_max_rad": verdict.angle_max,
                "angle_tolerance_rad": policy.angle_max_tolerance,
                "tunable": verdict.angle_max <= policy.angle_max_tolerance,
                "unmatched_rows": verdict.unmatched_rows,
                "row_counts_match": verdict.row_counts_match,
                "nan_pattern_mismatches": verdict.nan_pattern_mismatches,
                "categorical_mismatches": verdict.categorical_mismatches,
                "passed": verdict.passed,
                "details": list(verdict.details),
            }
        )

    print(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
