"""Spike: does cross-frame crop accumulation (Option B) actually pay?

Measurement only -- touches no pipeline code. Builds the REAL stages from a
real fixture config + clip, runs the REAL batch stage functions two ways over
the SAME detections, and reports wall time + numeric agreement:

  arm "per_frame"   : run_*_batch([f], [obb])  once per frame   (today)
  arm "accumulated" : run_*_batch(frames, obbs) once per group  (Option B)

Crop warp (extract_canonical_crops_batch) is timed separately from backend
forward, because B does NOT shrink warp -- only the forward.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_it(fn, repeats: int, warmup: int):
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), min(samples), max(samples)


def build(config_path: Path, clip: Path, tier: str, n_frames: int, skeleton: str = ""):
    import cv2

    from hydra_suite.core.inference.config import build_inference_config_from_params
    from hydra_suite.core.inference.runner import InferenceRunner
    from hydra_suite.trackerkit.cli_config import (
        TrackerCliVideoProbe,
        build_tracking_parameters,
    )

    cap = cv2.VideoCapture(str(clip))
    probe = TrackerCliVideoProbe(
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cfg_raw = json.loads(config_path.read_text())
    params = build_tracking_parameters(cfg_raw, video_probe=probe)
    params["RUNTIME_TIER"] = tier
    if skeleton:
        params["POSE_SKELETON_FILE"] = skeleton
    config = build_inference_config_from_params(params)

    if os.environ.get("SPIKE_NO_POSE"):
        config.pose = None
    if os.environ.get("SPIKE_CLS_BATCH"):
        _b = int(os.environ["SPIKE_CLS_BATCH"])
        if config.headtail is not None:
            config.headtail.batch_size = _b
        for _c in config.cnn_phases:
            _c.batch_size = _b
    if os.environ.get("SPIKE_POSE_BATCH"):
        config.pose.batch_size = int(os.environ["SPIKE_POSE_BATCH"])
    runner = InferenceRunner(config, cache_dir=None, video_path=str(clip))
    models = runner._models
    runtime = runner.runtime

    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return config, models, runtime, frames


def detections_for(frames, models, config, runtime):
    """Run detection ONCE; return [(frame, filtered_obb)] for non-empty frames."""
    from hydra_suite.core.inference.result import OBBResult
    from hydra_suite.core.inference.stages.filtering import filter_for_source
    from hydra_suite.core.inference.stages.obb import (
        _RawOBBTensors,
        effective_raw_detection_cap,
        materialize_tensors,
        run_obb,
    )

    pairs = []
    for i, frame in enumerate(frames):
        raw_list = run_obb([frame], models.obb, config.obb, runtime, roi_mask=None)
        raw = raw_list[0]
        obb = (
            materialize_tensors(raw, effective_raw_detection_cap(config.obb))
            if isinstance(raw, _RawOBBTensors)
            else raw
        )
        obb = OBBResult(
            frame_idx=i,
            centroids=obb.centroids,
            angles=obb.angles,
            sizes=obb.sizes,
            shapes=obb.shapes,
            confidences=obb.confidences,
            corners=obb.corners,
            detection_ids=OBBResult.make_detection_ids(i, obb.num_detections),
            class_ids=obb.class_ids,
        )
        filtered, _idx = filter_for_source(config, obb, None)
        if filtered.num_detections:
            pairs.append((frame, filtered))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tier", default="gpu")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--groups", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--skeleton", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch

    config, models, runtime, frames = build(
        Path(args.config), Path(args.clip), args.tier, args.frames, args.skeleton
    )
    pairs = detections_for(frames, models, config, runtime)
    dets = [p[1].num_detections for p in pairs]
    report: dict = {
        "clip": Path(args.clip).name,
        "config": Path(args.config).name,
        "tier": args.tier,
        "device": runtime.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "frames_with_detections": len(pairs),
        "detections_per_frame": dets,
        "median_dets_per_frame": statistics.median(dets) if dets else 0,
        "stages_enabled": {
            "headtail": models.headtail is not None,
            "cnn": len(models.cnn),
            "pose": models.pose is not None,
        },
        "batch_knobs": {
            "headtail": getattr(config.headtail, "batch_size", None),
            "cnn": [c.batch_size for c in config.cnn_phases],
            "pose": getattr(config.pose, "batch_size", None),
            "detection_batch_size": config.detection_batch_size,
        },
        "results": [],
    }

    from hydra_suite.core.inference.stages.cnn import run_cnn_batch
    from hydra_suite.core.inference.stages.crops import extract_canonical_crops_batch
    from hydra_suite.core.inference.stages.headtail import run_headtail_batch
    from hydra_suite.core.inference.stages.pose import run_pose_batch

    geometry = config.canonical

    for g in args.groups:
        groups = [pairs[i : i + g] for i in range(0, len(pairs), g)]
        groups = [grp for grp in groups if len(grp) == g]  # full groups only
        if not groups:
            continue
        crops_total = sum(o.num_detections for grp in groups for _f, o in grp)
        entry = {"group_size": g, "n_groups": len(groups), "total_crops": crops_total}

        # ---- crop warp ----
        def warp_pf(groups=groups):
            for grp in groups:
                for f, o in grp:
                    extract_canonical_crops_batch([f], [o], geometry, runtime)

        def warp_acc(groups=groups):
            for grp in groups:
                fs = [f for f, _ in grp]
                os_ = [o for _, o in grp]
                extract_canonical_crops_batch(fs, os_, geometry, runtime)

        entry["warp"] = {
            "per_frame_s": _time_it(warp_pf, args.repeats, args.warmup),
            "accumulated_s": _time_it(warp_acc, args.repeats, args.warmup),
        }

        # ---- head-tail ----
        if models.headtail is not None:

            def ht_pf(groups=groups):
                out = {}
                for grp in groups:
                    for f, o in grp:
                        out.update(
                            run_headtail_batch(
                                [f],
                                [o],
                                models.headtail,
                                config.headtail,
                                runtime,
                                geometry,
                            )
                        )
                return out

            def ht_acc(groups=groups):
                out = {}
                for grp in groups:
                    fs = [f for f, _ in grp]
                    os_ = [o for _, o in grp]
                    out.update(
                        run_headtail_batch(
                            fs,
                            os_,
                            models.headtail,
                            config.headtail,
                            runtime,
                            geometry,
                        )
                    )
                return out

            entry["headtail"] = {
                "per_frame_s": _time_it(ht_pf, args.repeats, args.warmup),
                "accumulated_s": _time_it(ht_acc, args.repeats, args.warmup),
                "agreement": compare_headtail(ht_pf(), ht_acc(), ht_acc()),
            }

        # ---- CNN ----
        if models.cnn:
            mdl, cfg_cnn = models.cnn[0], config.cnn_phases[0]

            def cnn_pf(groups=groups, mdl=mdl, cfg_cnn=cfg_cnn):
                out = {}
                for grp in groups:
                    for f, o in grp:
                        out.update(
                            run_cnn_batch([f], [o], mdl, cfg_cnn, runtime, geometry)
                        )
                return out

            def cnn_acc(groups=groups, mdl=mdl, cfg_cnn=cfg_cnn):
                out = {}
                for grp in groups:
                    fs = [f for f, _ in grp]
                    os_ = [o for _, o in grp]
                    out.update(run_cnn_batch(fs, os_, mdl, cfg_cnn, runtime, geometry))
                return out

            entry["cnn"] = {
                "label": cfg_cnn.label,
                "per_frame_s": _time_it(cnn_pf, args.repeats, args.warmup),
                "accumulated_s": _time_it(cnn_acc, args.repeats, args.warmup),
                "agreement": compare_cnn(cnn_pf(), cnn_acc(), cnn_acc()),
            }

        # ---- pose ----
        if models.pose is not None:

            def pose_pf(groups=groups):
                out = {}
                for grp in groups:
                    for f, o in grp:
                        b = extract_canonical_crops_batch([f], [o], geometry, runtime)
                        out.update(
                            run_pose_batch(
                                b, models.pose, config.pose, runtime, geometry
                            )
                        )
                return out

            def pose_acc(groups=groups):
                out = {}
                for grp in groups:
                    fs = [f for f, _ in grp]
                    os_ = [o for _, o in grp]
                    b = extract_canonical_crops_batch(fs, os_, geometry, runtime)
                    out.update(
                        run_pose_batch(b, models.pose, config.pose, runtime, geometry)
                    )
                return out

            entry["pose"] = {
                "per_frame_s": _time_it(pose_pf, args.repeats, args.warmup),
                "accumulated_s": _time_it(pose_acc, args.repeats, args.warmup),
                "agreement": compare_pose(pose_pf(), pose_acc(), pose_acc()),
            }

        report["results"].append(entry)
        print(json.dumps(entry, indent=2, default=str), flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {args.out}")
    return 0


def _probs(res):
    for attr in ("probabilities", "raw_probabilities", "scores", "confidences"):
        v = getattr(res, attr, None)
        if v is not None:
            return np.asarray(v, dtype=np.float64)
    return None


def compare_headtail(a, b, b2):
    """(per_frame vs accumulated) against (accumulated vs accumulated) floor."""

    def delta(x, y):
        flips = 0
        maxd = 0.0
        n = 0
        for k in sorted(set(x) & set(y)):
            hx = np.asarray(getattr(x[k], "heading_hints", []), dtype=object)
            hy = np.asarray(getattr(y[k], "heading_hints", []), dtype=object)
            m = min(len(hx), len(hy))
            n += m
            for i in range(m):
                if bool(hx[i]) != bool(hy[i]):
                    flips += 1
            px, py = _probs(x[k]), _probs(y[k])
            if px is not None and py is not None and px.shape == py.shape:
                maxd = max(maxd, float(np.max(np.abs(px - py))) if px.size else 0.0)
        return {"n": n, "flips": flips, "max_abs_prob_delta": maxd}

    return {"determinism_floor": delta(b, b2), "per_frame_vs_accumulated": delta(a, b)}


def compare_cnn(a, b, b2):
    """CNNResult.predictions[d].factors[f].raw_probabilities -- per-factor argmax."""

    def delta(x, y):
        flips = 0
        maxd = 0.0
        n = 0
        for k in sorted(set(x) & set(y)):
            px_list = getattr(x[k], "predictions", [])
            py_list = getattr(y[k], "predictions", [])
            for dx, dy in zip(px_list, py_list):
                for fx, fy in zip(dx.factors, dy.factors):
                    rx = np.asarray(fx.raw_probabilities, dtype=np.float64)
                    ry = np.asarray(fy.raw_probabilities, dtype=np.float64)
                    if rx.shape != ry.shape or rx.size == 0:
                        continue
                    n += 1
                    maxd = max(maxd, float(np.max(np.abs(rx - ry))))
                    if int(np.argmax(rx)) != int(np.argmax(ry)):
                        flips += 1
        return {"n_factor_preds": n, "argmax_flips": flips, "max_abs_prob_delta": maxd}

    return {"determinism_floor": delta(b, b2), "per_frame_vs_accumulated": delta(a, b)}


def compare_pose(a, b, b2):
    def delta(x, y):
        maxd = 0.0
        n = 0
        for k in sorted(set(x) & set(y)):
            kx = getattr(x[k], "keypoints", None)
            ky = getattr(y[k], "keypoints", None)
            if kx is None or ky is None:
                continue
            kx = np.asarray(kx, dtype=np.float64)
            ky = np.asarray(ky, dtype=np.float64)
            if kx.shape != ky.shape or kx.size == 0:
                continue
            n += kx.shape[0]
            d = np.abs(kx - ky)
            d = d[np.isfinite(d)]
            if d.size:
                maxd = max(maxd, float(np.max(d)))
        return {"n": n, "max_abs_px_delta": maxd}

    return {"determinism_floor": delta(b, b2), "per_frame_vs_accumulated": delta(a, b)}


if __name__ == "__main__":
    sys.exit(main())
