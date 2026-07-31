"""Benchmark the three runtime tiers (cpu / gpu / gpu_fast) across the
equivalence-harness fixture clips, on whatever box this runs on.

For each fixture x tier it invokes ``runner.py`` as a SUBPROCESS (so Qt +
coremltools never share a process — that combo can crash) into a temp outdir and
reads ``fps`` / ``wall_clock_s`` from the run's ``meta.json``.

``gpu_fast`` builds a TensorRT engine / CoreML .mlpackage on first use; that
one-time export inflates the first run, so gpu_fast is run twice and the second
(steady-state) timing is reported (first-run wall-clock kept as ``export_s``).

Usage:
    PYTHONPATH=src python tools/equivalence/tier_benchmark.py \
        --box mps --tiers cpu,gpu,gpu_fast --out /tmp/tier_bench_mps.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FX = HERE / "fixtures"
SKEL = FX / "ooceraea_biroi.json"

# name | needs skeleton
FIXTURES = [
    ("emi_obb_identity", False),
    ("ant_pose_headtail", True),
    ("ant_obb_sleap", True),
    ("worm_bgsub", False),
    ("ant_cnn_identity", True),
    ("fly_obb", False),
]


def _run_once(name: str, tier: str, needs_skel: bool, outdir: Path) -> dict:
    video = FX / "clips" / f"{name}.mp4"
    config = FX / "configs" / f"{name}.json"
    cmd = [
        sys.executable,
        str(HERE / "runner.py"),
        "--orig-config",
        str(config),
        "--video",
        str(video),
        "--outdir",
        str(outdir),
        "--runtime",
        tier,
        "--label",
        tier,
    ]
    if needs_skel:
        cmd += ["--skeleton", str(SKEL)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    meta_path = outdir / "meta.json"
    fps = None
    if meta_path.exists():
        try:
            fps = json.load(open(meta_path)).get("fps")
        except Exception:
            fps = None
    return {
        "ok": proc.returncode == 0 and fps is not None,
        "fps": fps,
        "wall_s": round(wall, 2),
        "returncode": proc.returncode,
        "tail": (proc.stderr or proc.stdout or "")[-400:] if fps is None else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--box", required=True, help="Label for this machine (e.g. mps / cuda)."
    )
    ap.add_argument("--tiers", default="cpu,gpu,gpu_fast")
    ap.add_argument(
        "--only", default="", help="Comma-list of fixture names to restrict to."
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    fixtures = [f for f in FIXTURES if not only or f[0] in only]

    results: dict = {"box": args.box, "tiers": tiers, "rows": []}
    with tempfile.TemporaryDirectory(prefix="tierbench-") as tmp:
        tmproot = Path(tmp)
        for name, needs_skel in fixtures:
            row = {"fixture": name}
            for tier in tiers:
                if tier == "gpu_fast":
                    # First run builds the artifact (export_s); second is steady-state.
                    r1 = _run_once(
                        name, tier, needs_skel, tmproot / f"{name}_{tier}_warm"
                    )
                    r2 = _run_once(name, tier, needs_skel, tmproot / f"{name}_{tier}")
                    r2["export_s"] = r1["wall_s"]
                    if not r2["ok"] and r1["ok"]:
                        r2 = {**r1, "note": "second run failed; reporting first"}
                    row[tier] = r2
                else:
                    row[tier] = _run_once(
                        name, tier, needs_skel, tmproot / f"{name}_{tier}"
                    )
                cell = row[tier]
                print(
                    f"{args.box} {name} {tier}: "
                    f"{'fps=' + str(cell.get('fps')) if cell.get('ok') else 'FAIL rc=' + str(cell.get('returncode'))}",
                    flush=True,
                )
            results["rows"].append(row)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}", flush=True)

    # Markdown summary.
    print("\n| fixture | " + " | ".join(tiers) + " |")
    print("|---|" + "|".join(["---"] * len(tiers)) + "|")
    for row in results["rows"]:
        cells = []
        for tier in tiers:
            c = row.get(tier, {})
            cells.append(f"{c.get('fps')}" if c.get("ok") else "FAIL")
        print(f"| {row['fixture']} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
