"""Replay the post-tracking merge on real tracking CSVs.

Two modes, because the fixture clips are far too small to exercise the
merge-overlap pair enumeration at the scale where it matters:

``--window START END``
    Restrict both passes to a frame window, then run BOTH the current
    ``_merge_overlapping_agreeing_trajectories`` and a verbatim copy of the
    original all-pairs body over the identical inputs and assert the results
    are frame-for-frame identical.  Quadratic cost drops ~100x on a 10% window,
    so the oracle is affordable on real data.

``--full``
    Run the whole merge once with the current code and report wall-clock per
    stage.  No oracle -- this measures, it does not verify.

Usage:
    python tools/merge_replay_check.py --video /path/X1_d1.mp4 \
        --config /path/X1_d1_config.json --window 0 4000
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time

import numpy as np
import pandas as pd

from hydra_suite.core.post import processing as P
from hydra_suite.trackerkit.cli_config import load_tracker_cli_session


def reference_merge_overlapping(
    trajectories,
    agreement_distance,
    min_overlap,
    min_length,
    identity_disagree_min_run=5,
    identity_drives_splits: bool = True,
):
    """The original all-pairs body, verbatim, as the equivalence oracle."""
    if not trajectories:
        return trajectories
    has_detection_id = (
        "DetectionID" in trajectories[0].columns if trajectories else False
    )
    max_spatial_jump = agreement_distance * 5
    max_iterations = 50
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        merged_any = False
        used = set()
        new_trajectories = []
        traj_lookups, traj_frame_sets, traj_bounds = [], [], []
        for traj in trajectories:
            lookup = P._build_frame_lookup(traj, require_valid_x=True)
            frame_set = set(lookup.keys())
            traj_lookups.append(lookup)
            traj_frame_sets.append(frame_set)
            traj_bounds.append(
                (min(frame_set), max(frame_set)) if frame_set else (np.inf, -np.inf)
            )
        for i in range(len(trajectories)):
            if i in used:
                continue
            traj_a = trajectories[i]
            for j in range(i + 1, len(trajectories)):
                if j in used:
                    continue
                segs = P._try_merge_trajectory_pair(
                    i,
                    j,
                    trajectories,
                    traj_lookups,
                    traj_frame_sets,
                    traj_bounds,
                    has_detection_id,
                    agreement_distance,
                    min_overlap,
                    min_length,
                    max_spatial_jump,
                    identity_disagree_min_run=identity_disagree_min_run,
                    identity_drives_splits=identity_drives_splits,
                )
                if segs is not None:
                    used.add(i)
                    used.add(j)
                    new_trajectories.extend(segs)
                    merged_any = True
                    break
            if i not in used:
                new_trajectories.append(traj_a)
                used.add(i)
        trajectories = new_trajectories
        if not merged_any:
            break
    return trajectories


def build_session(video, config_path):
    from hydra_suite.core.tracking.session import SessionCallbacks, TrackingSessionCore

    cli = load_tracker_cli_session(video, config_path=config_path)
    callbacks = SessionCallbacks(
        progress=lambda *_: None,
        status=lambda *_: None,
        warning=lambda *_: None,
        stage_changed=lambda *_: None,
        should_stop=lambda: False,
    )
    core = TrackingSessionCore(
        video_path=video,
        config=cli.config,
        params=cli.params,
        paths={"raw_csv_path": cli.raw_csv_path, "detection_cache_path": ""},
        callbacks=callbacks,
    )
    return cli, core


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--window", nargs=2, type=int, metavar=("START", "END"))
    ap.add_argument("--full", action="store_true")
    ap.add_argument(
        "--compare-final",
        metavar="CSV",
        help="Run the complete merge (resolve + interpolate + rescale), write it "
        "with the session's own writer, and diff against this reference "
        "_tracking_final.csv produced by the pre-change code.",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cli, core = build_session(args.video, args.config)
    base = cli.raw_csv_path.rsplit(".", 1)[0]

    t0 = time.perf_counter()
    forward = core._postprocess_csv(f"{base}_forward.csv")
    backward = core._postprocess_csv(f"{base}_backward.csv")
    print(f"postprocess: {time.perf_counter() - t0:.1f}s")

    def as_list(trajs):
        if isinstance(trajs, pd.DataFrame):
            return [g for _, g in trajs.groupby("TrajectoryID")]
        return list(trajs)

    fwd, bwd = as_list(forward), as_list(backward)
    print(f"trajectories: forward={len(fwd)} backward={len(bwd)}")

    params = cli.params
    if args.window:
        lo, hi = args.window

        def clip(trajs):
            out = []
            for df in trajs:
                sub = df[(df["FrameID"] >= lo) & (df["FrameID"] < hi)]
                if len(sub) >= int(params.get("MIN_TRAJECTORY_LENGTH", 5)):
                    out.append(sub.reset_index(drop=True))
            return out

        fwd, bwd = clip(fwd), clip(bwd)
        print(f"window [{lo},{hi}): forward={len(fwd)} backward={len(bwd)}")

    # Reproduce resolve_trajectories up to the overlap-merge step, then run the
    # two implementations on the identical input.
    agreement = float(params.get("AGREEMENT_DISTANCE", 15.0))
    min_overlap = int(params.get("MIN_OVERLAP_FRAMES", 5))
    min_length = int(params.get("MIN_TRAJECTORY_LENGTH", 5))
    disagree_run = int(params.get("IDENTITY_DISAGREE_MIN_RUN", 5))
    gates = bool(params.get("IDENTITY_GATES_TRAJECTORY_STRUCTURE", True))

    if args.compare_final:
        from hydra_suite.core.tracking.session import _save_trajectories_to_csv

        t0 = time.perf_counter()
        final_df = core._merge(fwd, bwd)
        print(f"full merge: {time.perf_counter() - t0:.1f}s -> {len(final_df)} rows")

        # Write beside nothing of the user's: this is a verification artifact.
        out = os.path.join(
            os.environ.get("MERGE_REPLAY_OUT", os.path.expanduser("~")),
            "replay_tracking_final.csv",
        )
        assert _save_trajectories_to_csv(final_df, out), "writer refused to write"

        def digest(path):
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        a, b = digest(out), digest(args.compare_final)
        print(f"replay    sha256 {a}  {os.path.getsize(out)} bytes")
        print(f"reference sha256 {b}  {os.path.getsize(args.compare_final)} bytes")
        print("BYTE-IDENTICAL \u2705" if a == b else "DIFFERS \u274c")
        return 0 if a == b else 1

    t0 = time.perf_counter()
    resolved = P.resolve_trajectories(fwd, bwd, params)
    print(
        f"resolve_trajectories (current code): {time.perf_counter() - t0:.1f}s "
        f"-> {len(resolved)} trajectories"
    )

    if args.full:
        return 0

    # Oracle leg: rebuild the same pre-overlap state and compare.
    candidates = P._find_merge_candidates(fwd, bwd, agreement, min_overlap)
    applied = P._apply_merge_candidates(
        candidates,
        fwd,
        bwd,
        agreement,
        min_length,
        identity_disagree_min_run=disagree_run,
        identity_drives_splits=gates,
    )
    applied = P._remove_spatially_redundant_trajectories(
        applied, agreement, min_overlap
    )

    t0 = time.perf_counter()
    new = P._merge_overlapping_agreeing_trajectories(
        [d.copy() for d in applied],
        agreement,
        min_overlap,
        min_length,
        identity_disagree_min_run=disagree_run,
        identity_drives_splits=gates,
    )
    new_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = reference_merge_overlapping(
        [d.copy() for d in applied],
        agreement,
        min_overlap,
        min_length,
        identity_disagree_min_run=disagree_run,
        identity_drives_splits=gates,
    )
    ref_s = time.perf_counter() - t0

    print(f"overlap-merge input: {len(applied)} trajectories")
    print(f"  new       : {new_s:8.1f}s -> {len(new)} trajectories")
    print(f"  all-pairs : {ref_s:8.1f}s -> {len(ref)} trajectories")
    print(f"  speedup   : {ref_s / max(new_s, 1e-9):8.1f}x")

    assert len(new) == len(ref), f"count differs: {len(new)} vs {len(ref)}"
    for k, (a, b) in enumerate(zip(new, ref)):
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=f"trajectory[{k}]")
    print("IDENTICAL ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
