"""Benchmark + correctness probe for warm, streaming SLEAP inference.

RUN THIS INSIDE THE `sleap` CONDA ENV (where `sleap_nn` is importable), e.g.:

    conda run --no-capture-output -n sleap python tools/equivalence/sleap_inference_bench.py \
        --model-dir "/Users/.../models/pose/SLEAP/20260214-224154_unet_ant_single_instance" \
        --video tools/equivalence/fixtures/clips/ant_pose_headtail.mp4 \
        --frames 32 --device mps --batch 16 --crop-size 128

Goal: decide HOW to do per-frame SLEAP inference in the new pipeline without the
6 s/frame CLI fallback. It loads the predictor ONCE (warm) and compares several
strategies on the SAME N input frames:

  cold_reload   : rebuild the predictor every call (the current ~6 s/frame cost)
  warm_batch    : predictor warm, all N frames in ONE predict() call (legacy-style)
  warm_perframe : predictor warm, N separate 1-frame predict() calls (inline realtime)
  direct_batch  : predictor warm, manual preprocess -> _run_inference_on_batch (lowest overhead)

For each working strategy it reports per-call latency and extracts keypoints
(`pred_instance_peaks`); it then checks that warm_* strategies AGREE with
warm_batch (so we know the fast path is also correct). Strategies that error are
reported, not fatal — this is a probe, not the final implementation.

Nothing here is wired into the pipeline; it only informs the design.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


def load_frames(video: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise SystemExit(f"No frames read from {video}")
    log(f"loaded {len(frames)} frames {frames[0].shape} from {video}")
    return frames


def load_preprocess_config(model_dir: str):
    """Mirror the service: read data_config.preprocessing from training_config.yaml."""
    cfg_path = Path(model_dir) / "training_config.yaml"
    if not cfg_path.exists():
        return None
    try:
        import yaml

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return (raw.get("data_config") or {}).get("preprocessing") or None
    except Exception as exc:
        log(f"  (could not load preprocess_config: {exc})")
        return None


def build_predictor(model_dir: str, device: str, batch: int, preprocess_config):
    from sleap_nn.inference.predictors import Predictor

    kwargs = dict(model_paths=[model_dir], device=device, batch_size=batch)
    if preprocess_config is not None:
        try:
            from omegaconf import OmegaConf

            kwargs["preprocess_config"] = OmegaConf.create(preprocess_config)
        except Exception:
            kwargs["preprocess_config"] = preprocess_config
    return Predictor.from_model_paths(**kwargs)


def write_pngs(frames: list[np.ndarray], out_dir: Path) -> list[Path]:
    """Write each crop as a lossless PNG (this is exactly what the service's
    temp-file path already does), returning the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, fr in enumerate(frames):
        p = out_dir / f"crop_{i:05d}.png"
        cv2.imwrite(str(p), fr)
        paths.append(p)
    return paths


def image_video(png_paths: list[Path]):
    """Build an in-memory ImageVideo from PNG paths (sleap_io 0.6.5 has no
    numpy->Video, so a lossless image-backed Video is the readable equivalent)."""
    import sleap_io as sio

    return sio.Video.from_filename([str(p) for p in png_paths])


def run_via_pipeline(pred, inference_object) -> list:
    """make_pipeline(inference_object) + predict(make_labels=False) -> raw dicts.

    inference_object is positional and accepts a path str / Path / sio.Video /
    sio.Labels (per SingleInstancePredictor.make_pipeline). We pass an in-memory
    ImageVideo so there is no per-call temp-video encoding.
    """
    pred.make_pipeline(inference_object)
    out = pred.predict(make_labels=False)
    return list(out) if out is not None else []


def build_prepared_batch(crops_bgr: list[np.ndarray], pred):
    """Replicate the provider + _process_batch preprocessing from IN-MEMORY crops.

    Returns (imgs, fidxs, vidxs, org_szs, instances, eff_scales) ready for
    Predictor._run_inference_on_batch — no Video, no reader thread, no disk. This
    is exactly the code path the service streaming fix would use. Format matches
    what the probe revealed: imgs[i]=(1,1,C,max_h,max_w) uint8, org_szs[i]=(1,1,2).
    """
    import torch
    from sleap_nn.data.resizing import apply_sizematcher

    pc = pred.preprocess_config
    mh, mw = pc["max_height"], pc["max_width"]
    ensure_rgb = bool(pc.get("ensure_rgb", True))
    ensure_gray = bool(pc.get("ensure_grayscale", False))
    imgs, fidxs, vidxs, org_szs, eff_scales = [], [], [], [], []
    for i, crop in enumerate(crops_bgr):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # sio reads RGB; cv2 is BGR
        chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
        img = torch.from_numpy(np.expand_dims(chw, 0).copy())  # (1,C,H,W) uint8
        H, W = crop.shape[:2]
        orig = torch.Tensor([H, W]).unsqueeze(0)  # (1,2)
        img, eff = apply_sizematcher(img, mh, mw)
        if ensure_rgb and img.shape[-3] != 3:
            img = img.repeat(1, 3, 1, 1)
        elif ensure_gray and img.shape[-3] != 1:
            import torchvision.transforms.functional as TF

            img = TF.rgb_to_grayscale(img, num_output_channels=1)
        imgs.append(img.unsqueeze(0))  # (1,1,C,mh,mw)
        org_szs.append(orig.unsqueeze(0))  # (1,1,2)
        eff_scales.append(torch.tensor(eff))
        fidxs.append(i)
        vidxs.append(0)
    return imgs, fidxs, vidxs, org_szs, [], eff_scales


def peaks_of(raw: list) -> Optional[np.ndarray]:
    """Concatenate pred_instance_peaks across the returned batch dicts."""
    chunks = []
    for d in raw or []:
        if isinstance(d, dict) and "pred_instance_peaks" in d:
            arr = np.asarray(d["pred_instance_peaks"], dtype=np.float32)
            chunks.append(arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr)
    if not chunks:
        return None
    try:
        return np.concatenate(chunks, axis=0)
    except Exception:
        return None


def bench(fn, repeat: int) -> dict:
    times = []
    last = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
    return {
        "ms_mean": 1000 * statistics.mean(times),
        "ms_min": 1000 * min(times),
        "result": last,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument(
        "--frames",
        type=int,
        default=32,
        help="number of frames to use as test crops",
    )
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=3, help="timing repeats per strategy")
    ap.add_argument(
        "--crop-size",
        type=int,
        default=0,
        help="resize each frame to this square size to mimic canonical crops "
        "(0 = keep full frame). Production runs SLEAP on small crops, so e.g. "
        "--crop-size 128 gives more representative per-call timing.",
    )
    args = ap.parse_args()

    frames = load_frames(args.video, args.frames)
    if args.crop_size > 0:
        frames = [cv2.resize(fr, (args.crop_size, args.crop_size)) for fr in frames]
        log(
            f"resized frames to {args.crop_size}x{args.crop_size} (crop-representative timing)"
        )
    pre = load_preprocess_config(args.model_dir)
    log(f"preprocess_config: {pre}")

    tmp = Path(tempfile.mkdtemp(prefix="sleap_bench_"))
    png_paths = write_pngs(frames, tmp / "crops")
    big_video = image_video(png_paths)
    per_frame_videos = [image_video([p]) for p in png_paths]

    results: dict[str, dict] = {}

    # --- warm predictor (built ONCE) ---
    log("\n[build warm predictor] ...")
    t0 = time.perf_counter()
    pred = build_predictor(args.model_dir, args.device, args.batch, pre)
    log(f"  predictor built in {time.perf_counter() - t0:.1f}s ({type(pred).__name__})")

    # warm_batch: all N frames in one call
    try:
        log("\n[warm_batch] all N frames, one predict() ...")
        r = bench(lambda: run_via_pipeline(pred, big_video), args.repeat)
        peaks = peaks_of(r["result"])
        results["warm_batch"] = {
            "total_ms": r["ms_mean"],
            "per_frame_ms": r["ms_mean"] / len(frames),
            "peaks": peaks,
            "n_peaks": None if peaks is None else peaks.shape[0],
        }
        log(
            f"  total={r['ms_mean']:.0f}ms  per-frame={r['ms_mean'] / len(frames):.1f}ms"
            f"  peaks={None if peaks is None else peaks.shape}"
        )
    except Exception as exc:
        results["warm_batch"] = {"error": repr(exc)}
        log(f"  FAILED: {exc!r}")

    # warm_perframe: N separate 1-frame calls (inline realtime simulation)
    try:
        log("\n[warm_perframe] N separate 1-frame predict() calls ...")
        per_call = []
        all_peaks = []
        for p in per_frame_videos:
            t0 = time.perf_counter()
            raw = run_via_pipeline(pred, p)
            per_call.append(time.perf_counter() - t0)
            pk = peaks_of(raw)
            if pk is not None:
                all_peaks.append(pk)
        peaks = np.concatenate(all_peaks, axis=0) if all_peaks else None
        results["warm_perframe"] = {
            "per_frame_ms": 1000 * statistics.mean(per_call),
            "per_frame_ms_min": 1000 * min(per_call),
            "peaks": peaks,
        }
        log(
            f"  per-frame mean={1000 * statistics.mean(per_call):.1f}ms"
            f"  min={1000 * min(per_call):.1f}ms  peaks={None if peaks is None else peaks.shape}"
        )
    except Exception as exc:
        results["warm_perframe"] = {"error": repr(exc)}
        log(f"  FAILED: {exc!r}")

    # infer_only: isolate _run_inference_on_batch cost (NO make_pipeline / no reader
    # thread). This is the floor a "direct streaming" path would hit, and it reveals
    # the exact prepared-batch tensor format (so we can build that path correctly).
    try:
        log("\n[infer_only] _run_inference_on_batch with NO pipeline per call ...")
        pred.make_pipeline(big_video)
        if getattr(pred, "inference_model", None) is None:
            pred._initialize_inference_model()
        pred.pipeline.start()
        imgs, fidxs, vidxs, org_szs, instances, eff_scales, _done = (
            pred._process_batch()
        )
        pred.pipeline.join()
        log(
            f"  prepared batch: n={len(imgs)}  imgs[0]={tuple(imgs[0].shape)} "
            f"{imgs[0].dtype}  org_szs[0]={tuple(org_szs[0].shape)}  "
            f"instances_key={getattr(pred, 'instances_key', None)}"
        )

        def _infer():
            return list(
                pred._run_inference_on_batch(
                    imgs, fidxs, vidxs, org_szs, instances, eff_scales
                )
            )

        r = bench(_infer, args.repeat)
        peaks = peaks_of(r["result"])
        results["infer_only"] = {
            "per_frame_ms": r["ms_mean"] / max(1, len(imgs)),
            "peaks": peaks,
        }
        log(
            f"  per-frame={r['ms_mean'] / max(1, len(imgs)):.2f}ms (n={len(imgs)})"
            f"  peaks={None if peaks is None else peaks.shape}"
        )
    except Exception as exc:
        results["infer_only"] = {"error": repr(exc)}
        log(f"  FAILED: {exc!r}")

    # direct_inmem: the ACTUAL service path — build the batch from in-memory crop
    # arrays and call the warm _run_inference_on_batch. Tested in BOTH conditions:
    #   *_batch    = all N crops in one call (batch-across-frames mode)
    #   *_perframe = 1 crop per call (worst-case inline realtime mode)
    for label, groups in (
        ("direct_inmem_batch", [frames]),
        ("direct_inmem_perframe", [[f] for f in frames]),
    ):
        try:
            log(f"\n[{label}] in-memory crops -> _run_inference_on_batch ...")
            per_crop_times = []
            all_peaks = []
            for crops in groups:
                t0 = time.perf_counter()
                batch = build_prepared_batch(crops, pred)
                raw = list(pred._run_inference_on_batch(*batch))
                per_crop_times.append((time.perf_counter() - t0) / max(1, len(crops)))
                pk = peaks_of(raw)
                if pk is not None:
                    all_peaks.append(pk)
            peaks = np.concatenate(all_peaks, axis=0) if all_peaks else None
            results[label] = {
                "per_frame_ms": 1000 * statistics.mean(per_crop_times),
                "peaks": peaks,
            }
            log(
                f"  per-frame mean={1000 * statistics.mean(per_crop_times):.2f}ms"
                f"  peaks={None if peaks is None else peaks.shape}"
            )
        except Exception as exc:
            results[label] = {"error": repr(exc)}
            log(f"  FAILED: {exc!r}")

    # cold_reload: rebuild predictor each call (current ~6s/frame behavior) — few only
    try:
        log("\n[cold_reload] rebuild predictor each call (baseline, 3 frames) ...")
        per_call = []
        for p in per_frame_videos[:3]:
            t0 = time.perf_counter()
            cold = build_predictor(args.model_dir, args.device, args.batch, pre)
            run_via_pipeline(cold, p)
            per_call.append(time.perf_counter() - t0)
            del cold
        results["cold_reload"] = {"per_frame_ms": 1000 * statistics.mean(per_call)}
        log(f"  per-frame mean={1000 * statistics.mean(per_call):.0f}ms")
    except Exception as exc:
        results["cold_reload"] = {"error": repr(exc)}
        log(f"  FAILED: {exc!r}")

    # --- correctness: warm_perframe must agree with warm_batch ---
    log("\n=== CORRECTNESS (keypoint agreement vs warm_batch) ===")
    ref = results.get("warm_batch", {}).get("peaks")
    if ref is None:
        log("  warm_batch produced no peaks — cannot compare.")
    else:
        for name in (
            "warm_perframe",
            "infer_only",
            "direct_inmem_batch",
            "direct_inmem_perframe",
        ):
            pk = results.get(name, {}).get("peaks")
            if pk is None:
                log(f"  {name}: no peaks")
            elif pk.shape != ref.shape:
                log(f"  {name}: SHAPE differs {pk.shape} vs {ref.shape}")
            else:
                d = float(np.nanmax(np.abs(pk - ref)))
                log(
                    f"  {name}: max|Δ| = {d:.4f} px  -> {'MATCH' if d < 1.0 else 'DIFFERS'}"
                )

    # --- summary table ---
    log("\n=== PERFORMANCE SUMMARY (per-frame) ===")
    for name in (
        "cold_reload",
        "warm_batch",
        "warm_perframe",
        "infer_only",
        "direct_inmem_batch",
        "direct_inmem_perframe",
    ):
        r = results.get(name, {})
        if "error" in r:
            log(f"  {name:22s} ERROR: {r['error']}")
        else:
            pf = r.get("per_frame_ms")
            log(f"  {name:22s} {pf:.2f} ms/frame" if pf else f"  {name:22s} (n/a)")
    log("\nInterpretation:")
    log("  - warm_perframe ≈ infer_only       -> make_pipeline overhead is negligible;")
    log("    per-frame make_pipeline+predict is fine for inline realtime.")
    log(
        "  - warm_perframe >> infer_only      -> make_pipeline (reader-thread) overhead"
    )
    log(
        "    dominates; a direct _run_inference_on_batch path removes it. infer_only is"
    )
    log("    the floor that direct path would reach.")
    log(f"\n(temp files in {tmp} — delete when done)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
