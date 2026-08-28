#!/usr/bin/env python3
"""Wall-clock benchmark: OLD (pre-Task-6) DetectKit AL round vs the CURRENT
(Tasks 1-9) ``InferenceRunner``-based AL round, on the same real fixture.

This is THE measurement that demonstrates the actual point of the AL-pipeline
optimization effort (Tasks 1-9): the OLD ``run_active_learning`` scored every
candidate frame with a raw per-frame ``detector_fn`` closure -- one call for
the base detection plus two more inside ``score_nms_instability``'s
perturbation sweep, i.e. THREE-plus unbatched, uncached model calls per
candidate frame, run strictly sequentially. Tasks 2/3/5/9 replaced that with
one batched, cached ``InferenceRunner.detect_batch_raw`` pass over the whole
candidate list, with every downstream signal (including NMS instability)
derived from the cached raw ``OBBResult`` via cheap NumPy re-filtering instead
of more model calls.

Method (same "which hydra_suite is importable" trick ``tools/equivalence/``
already uses, per this repo's ``CLAUDE.md``): a throwaway, detached git
worktree checks out the OLD commit's ``src/`` tree; the CURRENT tree's
``src/`` is used unchanged. Each variant's timed run happens in its own
subprocess (via ``python -m tools.equivalence.al_benchmark --_worker ...``,
run with ``sys.path`` pointing at just that one ``src/`` tree) so the two
`hydra_suite` package trees never collide inside one interpreter -- the same
constraint the tracking-equivalence harness works around with two worktrees.

Runs CPU-only (device_preference="cpu" for the OLD detector_fn's
``load_torch_model``; the NEW ``ALDetectorSpec`` leaves ``runtime_tier=None``,
which ``build_obb_only_config`` resolves to its own ``compute_runtime="cpu"``
default) so both variants use the exact same accelerator -- a GPU's
dispatch-latency noise would otherwise swamp the (purely architectural)
batching/caching win this script exists to demonstrate, and CPU is the most
call-count-sensitive setting anyway (no per-call GPU-dispatch amortization to
hide the difference).

Usage::

    python tools/equivalence/al_benchmark.py \\
        --video tools/equivalence/fixtures/clips/fly_obb.mp4 \\
        --model ~/Library/Application\\ Support/hydra-suite/models/obb/20260503-171130_26x_fly_train7.pt

This does not need to be a pytest test -- it's a manual verification tool, run
once to confirm the actual goal (this is a performance effort) and its output
recorded in the Task 10 report / PR description, not committed as an
automated gate (a wall-clock number is inherently machine- and load-dependent).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# See tools/equivalence/runner.py's identical guard: conda/torch builds often
# link libomp twice; without this, OpenMP aborts the process ("OMP Error #15").
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO = Path(__file__).resolve().parents[2]

# The commit right before Task 6 (i.e. Task 5's completion commit) -- the last
# point in the AL-pipeline-optimization history before ANY of the batching /
# caching / restructure work (Tasks 6-9) landed. See
# .superpowers/sdd/2026-08-27-al-pipeline-optimization/progress.md, which
# names this exact commit range for Task 6
# ("review-38fff295..cb15fdbb.diff").
DEFAULT_OLD_REF = "38fff295"


def _build_common_kwargs(args: argparse.Namespace) -> dict:
    from hydra_suite.data.al.candidate_pool import CandidatePoolConfig

    return dict(
        input_kind="video",
        input_path=args.video,
        budget=args.budget,
        preset="balanced",
        expected_count=args.expected_count,
        diversity_window=0,
        probabilistic=False,
        candidate_pool=CandidatePoolConfig(max_candidates=args.max_candidates),
        base_conf=args.base_conf,
        base_iou=args.base_iou,
    )


def _run_worker(args: argparse.Namespace) -> dict:
    """Runs INSIDE a subprocess whose sys.path[0] is one hydra_suite src tree.

    Builds an ALRequest in whichever shape that tree's ``al_worker.ALRequest``
    actually has (introspected via dataclasses.fields -- the OLD tree has
    ``detector_fn: Callable``, the current tree has ``detector: ALDetectorSpec``)
    and times ``run_active_learning`` end to end. Prints one JSON line to
    stdout: ``{"elapsed_s": ..., "n_picked": ..., "n_candidates_capped": ...}``.
    """
    import hydra_suite.detectkit.jobs.al_worker as al_worker_mod
    from hydra_suite.detectkit.gui.models import DetectKitProject

    ALRequest = al_worker_mod.ALRequest
    run_active_learning = al_worker_mod.run_active_learning

    project_dir = Path(tempfile.mkdtemp(prefix="al_benchmark_proj_"))
    project = DetectKitProject(project_dir=project_dir, sources=[])

    common = _build_common_kwargs(args)
    common["project"] = project

    field_names = {f.name for f in dataclasses.fields(ALRequest)}
    if "detector" in field_names:
        # CURRENT (Task 5+9) shape: a declarative spec, one InferenceRunner.
        ALDetectorSpec = al_worker_mod.ALDetectorSpec
        spec = ALDetectorSpec(
            kind=args.kind,
            model_path=args.model,
            secondary_model_path=args.secondary_model,
            crop_pad_ratio=args.crop_pad_ratio,
            runtime_tier=None,  # -> build_obb_only_config's compute_runtime="cpu" default
        )
        req = ALRequest(detector=spec, **common)
    else:
        # OLD (pre-Task-6) shape: a per-frame detector_fn(frame, conf, iou)
        # closure over a directly-loaded torch executor -- exactly what
        # `main_window._load_active_detector_fn` used to build.
        from hydra_suite.detectkit.gui.prediction_preview import (
            load_torch_model,
            predict_obb_for_frame_export,
        )

        if args.kind == "sequential":
            from hydra_suite.detectkit.gui.prediction_preview import (
                predict_obb_for_frame_sequential,
            )

            detect_model, detect_device = load_torch_model(args.model, "cpu")
            obb_model, obb_device = load_torch_model(args.secondary_model, "cpu")

            def detector_fn(frame, conf, iou):
                return predict_obb_for_frame_sequential(
                    detect_model,
                    obb_model,
                    frame,
                    detect_device=detect_device,
                    obb_device=obb_device,
                    conf=conf,
                    iou=iou,
                    crop_pad_ratio=args.crop_pad_ratio,
                )

        else:
            model, device = load_torch_model(args.model, "cpu")

            def detector_fn(frame, conf, iou):
                return predict_obb_for_frame_export(
                    model, frame, device=device, conf=conf, iou=iou
                )

        req = ALRequest(detector_fn=detector_fn, **common)

    t0 = time.perf_counter()
    result = run_active_learning(req)
    elapsed = time.perf_counter() - t0

    return {"elapsed_s": elapsed, "n_picked": result.n_picked}


def _spawn_worker(src: Path, args: argparse.Namespace) -> dict:
    """Runs `_run_worker` in a fresh subprocess with `src` as sys.path[0].

    A fresh subprocess (rather than manipulating sys.path in-process) is
    required: Python caches imported modules by name, so a second `import
    hydra_suite...` in the same interpreter after switching sys.path would
    silently reuse the FIRST tree's already-imported modules.
    """
    worker_args = [
        "--_worker",
        "--video",
        str(args.video),
        "--model",
        str(args.model),
        "--kind",
        args.kind,
        "--budget",
        str(args.budget),
        "--max-candidates",
        str(args.max_candidates),
        "--expected-count",
        str(args.expected_count),
        "--base-conf",
        str(args.base_conf),
        "--base-iou",
        str(args.base_iou),
        "--crop-pad-ratio",
        str(args.crop_pad_ratio),
    ]
    if args.secondary_model:
        worker_args += ["--secondary-model", str(args.secondary_model)]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())] + worker_args,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker subprocess (src={src}) failed with code {proc.returncode}:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    # The worker's own stdout may carry warnings from imports (coremltools,
    # sklearn, etc. -- see the equivalence-test scratch runs); the timing
    # result is the LAST line, printed as a single JSON object.
    last_line = proc.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def _ensure_old_worktree(old_ref: str, keep: bool) -> Path:
    short = subprocess.run(
        ["git", "rev-parse", "--short", old_ref],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    wt_path = REPO / ".worktrees" / f"al-benchmark-old-{short}"
    if wt_path.exists():
        print(f"### reusing existing baseline worktree {wt_path}")
        return wt_path
    print(f"### creating baseline worktree {wt_path} @ {old_ref} ({short})")
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(wt_path), old_ref],
        cwd=str(REPO),
        check=True,
    )
    return wt_path


def _cleanup_worktree(wt_path: Path) -> None:
    print(f"### removing baseline worktree {wt_path}")
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(wt_path)],
        cwd=str(REPO),
        check=False,
    )
    subprocess.run(["git", "worktree", "prune"], cwd=str(REPO), check=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Fixture video path.")
    p.add_argument(
        "--model",
        required=True,
        help="OBB-direct model path, or the stage-1 detect model for --kind sequential.",
    )
    p.add_argument(
        "--secondary-model",
        default=None,
        help="Stage-2 crop-OBB model path (required for --kind sequential).",
    )
    p.add_argument("--kind", choices=["obb_direct", "sequential"], default="obb_direct")
    p.add_argument("--old-ref", default=DEFAULT_OLD_REF, help="Baseline commit-ish.")
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--max-candidates", type=int, default=150)
    p.add_argument("--expected-count", type=int, default=3)
    p.add_argument("--base-conf", type=float, default=0.05)
    p.add_argument("--base-iou", type=float, default=0.5)
    p.add_argument("--crop-pad-ratio", type=float, default=0.15)
    p.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Don't remove the baseline worktree afterward (faster re-runs).",
    )
    p.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: re-invocation of this script as the timed subprocess
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args._worker:
        # Re-invoked as the timed subprocess: sys.path[0] was set by the
        # parent's PYTHONPATH, so `import hydra_suite...` resolves to exactly
        # the one src tree the parent selected for this run.
        print(json.dumps(_run_worker(args)))
        return 0

    if args.kind == "sequential" and not args.secondary_model:
        print("--secondary-model is required for --kind sequential", file=sys.stderr)
        return 2

    video = Path(args.video).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    if not video.exists():
        print(f"video not found: {video}", file=sys.stderr)
        return 2
    if not model.exists():
        print(f"model not found: {model}", file=sys.stderr)
        return 2
    args.video = str(video)
    args.model = str(model)
    if args.secondary_model:
        secondary = Path(args.secondary_model).expanduser().resolve()
        if not secondary.exists():
            print(f"secondary model not found: {secondary}", file=sys.stderr)
            return 2
        args.secondary_model = str(secondary)

    old_wt = _ensure_old_worktree(args.old_ref, args.keep_worktree)
    try:
        old_src = old_wt / "src"
        new_src = REPO / "src"

        print(
            f"### OLD  (commit {args.old_ref}, src={old_src}) -- "
            f"per-frame detector_fn closure, unbatched/uncached"
        )
        old_result = _spawn_worker(old_src, args)
        print(
            f"    elapsed = {old_result['elapsed_s']:.3f}s, n_picked = {old_result['n_picked']}"
        )

        print(
            f"### NEW  (HEAD, src={new_src}) -- "
            f"InferenceRunner.detect_batch_raw, one batched+cached pass"
        )
        new_result = _spawn_worker(new_src, args)
        print(
            f"    elapsed = {new_result['elapsed_s']:.3f}s, n_picked = {new_result['n_picked']}"
        )

        speedup = (
            old_result["elapsed_s"] / new_result["elapsed_s"]
            if new_result["elapsed_s"] > 0
            else float("inf")
        )
        print()
        print(
            f"### RESULT: OLD {old_result['elapsed_s']:.3f}s -> NEW {new_result['elapsed_s']:.3f}s"
            f"  (speedup {speedup:.2f}x)"
        )
        return 0
    finally:
        if not args.keep_worktree:
            _cleanup_worktree(old_wt)
        else:
            print(f"### kept baseline worktree at {old_wt} (--keep-worktree)")


if __name__ == "__main__":
    raise SystemExit(main())
