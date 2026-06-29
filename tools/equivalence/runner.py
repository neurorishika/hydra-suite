"""Run ONE tracking session into an isolated output dir, capturing env metadata.

Used by the equivalence harness to compare the new inference pipeline (this
worktree) against the legacy pipeline (main), across devices (cpu/mps/cuda).

Isolation: the source video is symlinked into --outdir, so all derived artifacts
(detection caches as ``<stem>_caches/``, trajectory CSVs) land in --outdir and
the user's real data directory is never touched.

Which pipeline runs is decided entirely by which ``hydra_suite`` is importable
(set PYTHONPATH to a source tree); this script records that path in meta.json.

Pure side-outputs (rendered video, training datasets, saved crop images,
confidence-density map) are disabled to save time/disk; the full
detect -> headtail -> pose -> identity -> track path is preserved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

# Conda/torch builds often link libomp twice; without this, OpenMP aborts the
# process ("OMP Error #15"). Must be set before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("equiv.runner")

DISABLE = {
    "video_output_enabled": False,
    "enable_confidence_density_map": False,
    "enable_dataset_generation": False,
    "enable_individual_dataset": False,
    "enable_individual_image_save": False,
    "final_media_export_videos_enabled": False,
}

# Map a single --runtime choice onto every per-stage runtime field.
# pose has its own "flavor" vocabulary (cpu/mps/cuda), handled separately.
_POSE_FLAVOR = {
    "cpu": "cpu",
    "mps": "mps",
    "cuda": "cuda",
    "onnx_cpu": "cpu",
    "onnx_cuda": "cuda",
    "tensorrt": "cuda",
}


def runtime_overrides(runtime: str) -> dict:
    if runtime == "config":
        return {}  # leave the config's own runtime untouched
    return {
        "compute_runtime": runtime,
        "cnn_runtime": runtime,
        "headtail_runtime": runtime,
        "pose_runtime_flavor": _POSE_FLAVOR.get(runtime, "cpu"),
        "pose_sleap_device": _POSE_FLAVOR.get(runtime, "cpu"),
        "enable_tensorrt": runtime == "tensorrt",
    }


def build_config(
    orig_config_path: str,
    video_link: Path,
    outdir: Path,
    runtime: str,
    skeleton: str | None = None,
) -> Path:
    with open(orig_config_path) as fh:
        cfg = json.load(fh)
    stem = video_link.stem
    cfg["file_path"] = str(video_link)
    cfg["csv_path"] = str(outdir / f"{stem}_tracking.csv")
    cfg["video_output_path"] = str(outdir / f"{stem}_tracking.mp4")
    cfg["use_cached_detections"] = False  # recompute fresh every run
    if skeleton:
        cfg["pose_skeleton_file"] = str(Path(skeleton).expanduser().resolve())
    cfg.update(DISABLE)
    cfg.update(runtime_overrides(runtime))
    out_cfg = outdir / "equiv_config.json"
    with open(out_cfg, "w") as fh:
        json.dump(cfg, fh, indent=2)
    return out_cfg


def _git_describe(src_path: Path) -> dict:
    try:
        root = subprocess.check_output(
            ["git", "-C", str(src_path), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return {"git_root": root, "git_commit": commit, "git_branch": branch}
    except Exception:
        return {"git_root": None, "git_commit": None, "git_branch": None}


def capture_meta(label: str, runtime: str, hydra_file: str) -> dict:
    meta = {
        "label": label,
        "requested_runtime": runtime,
        "hydra_suite_file": hydra_file,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    meta.update(_git_describe(Path(hydra_file).parent))
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["torch_cuda_available"] = bool(torch.cuda.is_available())
        meta["torch_mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception as exc:  # pragma: no cover
        meta["torch_error"] = str(exc)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-config", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--runtime",
        default="config",
        choices=["config", "cpu", "mps", "cuda", "onnx_cpu", "onnx_cuda", "tensorrt"],
        help="Override all stage runtimes; 'config' keeps the config's own runtime.",
    )
    ap.add_argument("--label", default="run", help="Label recorded in meta.json.")
    ap.add_argument(
        "--skeleton",
        default=None,
        help="Override pose_skeleton_file (portable clip configs leave it blank).",
    )
    args = ap.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src = Path(args.video)
    video_link = outdir / src.name
    if video_link.exists() or video_link.is_symlink():
        video_link.unlink()
    video_link.symlink_to(src.resolve())

    # Purge any derived caches from a previous run in this outdir. This is
    # CRITICAL for equivalence: the worker reuses an InferenceRunner cache when
    # caches_all_valid() (a config-hash check that ignores video length/content)
    # passes, silently overriding our use_cached_detections=False. If the clip
    # was regenerated with a different frame count under the same name, a stale
    # cache would be served and detections truncated to the old length. Deleting
    # the per-video cache dir + legacy cache dir guarantees a fresh detection pass.
    import shutil

    for stale in (*outdir.glob(".inference_cache_*"), *outdir.glob("*_caches")):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)

    cfg_path = build_config(
        args.orig_config, video_link, outdir, args.runtime, skeleton=args.skeleton
    )

    import hydra_suite

    meta = capture_meta(args.label, args.runtime, hydra_suite.__file__)
    with open(outdir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("hydra_suite src: %s", hydra_suite.__file__)
    log.info(
        "branch=%s commit=%s runtime=%s",
        meta.get("git_branch"),
        (meta.get("git_commit") or "")[:10],
        args.runtime,
    )

    # Time the full tracking run for the harness's performance check. The new
    # pipeline must not be slower than legacy: a regression like a cold per-frame
    # pose service (seconds/frame) shows up here even when the CSVs are equivalent.
    import time as _time

    from hydra_suite.trackerkit.cli import run_tracking_cli

    try:
        with open(cfg_path) as _fh:
            _cfg = json.load(_fh)
        n_frames = int(_cfg.get("end_frame", 0)) - int(_cfg.get("start_frame", 0)) + 1
    except Exception:
        n_frames = 0

    # Deterministic seeding so the run is reproducible and legacy-vs-new is
    # comparable. The bgsub background priming (core/background/model.py) samples
    # frames via the global `random` module with no seed of its own; without this
    # both legacy and new pick different frames every run, making worm_bgsub
    # non-deterministic (new_a != new_b) and never equal to legacy. The bgsub
    # path is identical code in both checkouts, so seeding identically here yields
    # an identical background prime. No-op for the YOLO/OBB clips.
    import random as _random

    import numpy as _np

    _random.seed(0)
    _np.random.seed(0)

    _t0 = _time.perf_counter()
    rc = run_tracking_cli([str(video_link)], config_path=str(cfg_path))
    elapsed = _time.perf_counter() - _t0

    meta["tracking_seconds"] = round(elapsed, 3)
    meta["n_frames"] = n_frames
    meta["fps"] = round(n_frames / elapsed, 2) if elapsed > 0 and n_frames else None
    meta["exit_code"] = int(rc)
    with open(outdir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    produced = sorted(p.name for p in outdir.glob("*tracking*.csv"))
    log.info(
        "exit code: %s  %.1fs  %.1f fps  produced CSVs: %s",
        rc,
        elapsed,
        (meta["fps"] or 0.0),
        produced,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
