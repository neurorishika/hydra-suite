"""Regenerate the short equivalence clips + portable configs from source videos.

Provenance/dev tool: run on a machine that has the full source videos and their
tracking configs. Produces, under fixtures/clips/ and fixtures/configs/:

  - a short clip per entry (frame-accurate trim via ffmpeg, remapped to 0..N-1)
  - a portable tracking config: frame range remapped, side-outputs left to the
    runner, detections recomputed, and every machine-specific MODEL path
    rewritten relative to the hydra-suite models dir so it resolves on any box
    that fetched the models archive. Pose skeleton is supplied at run time via
    runner --skeleton (kept out of the config so configs stay portable).

The clips + their required model files are uploaded as GitHub Release assets
(see manifest.json / fetch_fixtures.sh); only the small portable bits are
committed (configs/, skeleton, manifest, scripts).

Edit CLIPS below to change windows or add feature combinations. Each clip is a
representative of one DEMO so the matrix spans distinct settings:

  emi_obb_identity   OBB-direct + identity online decoder (no headtail/pose)
  ant_pose_headtail  OBB + headtail + SLEAP pose + identity (realtime)
  ant_obb_sleap      OBB-direct + SLEAP pose + identity-analysis (DEMO 1)
  worm_bgsub         background_subtraction detection path (DEMO 2)
  ant_cnn_identity   OBB + CNN multihead identity + headtail + pose-dir (DEMO 3)
  fly_obb            OBB-direct, fly, fresh detection (DEMO 4)

Usage:
  python generate_clips.py                # clips + configs
  python generate_clips.py --configs-only # only rewrite portable configs (fast,
                                          # no ffmpeg; clips already extracted)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from hydra_suite.paths import get_models_dir

HERE = Path(__file__).resolve().parent
CLIPS_DIR = HERE / "clips"
CONFIGS_DIR = HERE / "configs"
MODELS_DIR = Path(get_models_dir()).resolve()

DEMO = "/Users/neurorishika/Projects/Rockefeller/Ruta/Presentation/DEMO"
MTD = "/Users/neurorishika/Projects/Rockefeller/RutaKronauer/MultiTrackerData"

# name | source video | source config | start_frame | n_frames | note
CLIPS = [
    dict(
        name="emi_obb_identity",
        video=f"{MTD}/ant/emi_short.mp4",
        config=f"{MTD}/ant/emi_short_config.json",
        start=0,
        n=500,
        note="OBB-direct + identity online decoder (no headtail/pose)",
    ),
    dict(
        name="ant_pose_headtail",
        video=f"{MTD}/ant2/000001_cropped_roi.mp4",
        config=f"{MTD}/ant2/000001_cropped_roi_config.json",
        start=3629,
        n=500,
        note="OBB + headtail + SLEAP pose + identity (realtime)",
    ),
    dict(
        name="ant_obb_sleap",
        video=f"{DEMO}/DEMO 1/ant.mp4",
        config=f"{DEMO}/DEMO 1/ant_config.json",
        start=0,
        n=500,
        note="OBB-direct + SLEAP pose + identity-analysis, no headtail (DEMO 1)",
    ),
    dict(
        name="worm_bgsub",
        video=f"{DEMO}/DEMO 2/DEMO_BG/worm.avi",
        config=f"{DEMO}/DEMO 2/DEMO_BG/worm_config.json",
        start=0,
        n=500,
        note="background_subtraction detection path (DEMO 2)",
    ),
    dict(
        name="ant_cnn_identity",
        video=f"{DEMO}/DEMO 3/ant.mp4",
        config=f"{DEMO}/DEMO 3/ant_config.json",
        start=0,
        n=500,
        note="OBB + CNN multihead identity + headtail + SLEAP pose + pose-dir (DEMO 3)",
    ),
    dict(
        name="fly_obb",
        video=f"{DEMO}/DEMO 4/melanogaster.mp4",
        config=f"{DEMO}/DEMO 4/melanogaster_config.json",
        start=0,
        n=500,
        note="OBB-direct, fly, fresh detection (DEMO 4)",
    ),
]


def extract_clip(video: str, start: int, n: int, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    end = start + n - 1
    # Frame-accurate select; visually-lossless x264 (crf 18). The clip only needs
    # to be a valid, identical input to both pipelines, so we trade exact pixels
    # for a far smaller asset.
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video,
        "-vf",
        f"select=between(n\\,{start}\\,{end})",
        "-vsync",
        "0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        str(out),
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _relativize(value):
    """Rewrite absolute model paths under the models dir to relative form.

    Recurses through dicts/lists so nested model paths (e.g. cnn_classifiers[].
    model_path) are handled too. Non-path strings and paths outside the models
    dir are left untouched.
    """
    if isinstance(value, str):
        if not value:
            return value
        try:
            p = Path(value).resolve()
        except (OSError, ValueError):
            return value
        if MODELS_DIR in p.parents or p == MODELS_DIR:
            return str(p.relative_to(MODELS_DIR))
        return value
    if isinstance(value, list):
        return [_relativize(v) for v in value]
    if isinstance(value, dict):
        return {k: _relativize(v) for k, v in value.items()}
    return value


# Secondary-detection model paths that are only consumed by the sequential OBB
# mode; in direct mode they go unused, so we blank them to keep the models
# archive minimal (and to avoid packing models the run never loads).
_SEQ_KEYS = (
    "yolo_detect_model_path",
    "yolo_crop_obb_model_path",
    "yolo_seq_detect_model_path",
    "yolo_seq_crop_obb_model_path",
)


def portable_config(src_config: str, n: int, out: Path) -> None:
    with open(src_config) as fh:
        cfg = json.load(fh)
    cfg["start_frame"] = 0
    cfg["end_frame"] = n - 1
    cfg["use_cached_detections"] = False
    # Blank machine-specific I/O paths; runner.py sets these per run, and the pose
    # skeleton is supplied per run via runner --skeleton (keeps configs portable).
    for k in ("file_path", "csv_path", "video_output_path", "pose_skeleton_file"):
        cfg[k] = ""
    # Make every model path resolve against the fetched models dir on any machine.
    cfg = _relativize(cfg)
    # Drop sequential-mode detection models when running OBB-direct.
    if str(cfg.get("yolo_obb_mode", "direct")).lower() == "direct":
        for k in _SEQ_KEYS:
            if k in cfg:
                cfg[k] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(cfg, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs-only",
        action="store_true",
        help="Only rewrite portable configs (skip ffmpeg clip extraction).",
    )
    args = ap.parse_args()
    for c in CLIPS:
        clip = CLIPS_DIR / f"{c['name']}.mp4"
        cfg = CONFIGS_DIR / f"{c['name']}.json"
        print(f"[{c['name']}] {c['note']}")
        if not args.configs_only:
            print(f"  extracting {c['n']} frames @ {c['start']} -> {clip}")
            extract_clip(c["video"], c["start"], c["n"], clip)
            print(f"  clip = {clip.stat().st_size / 1e6:.1f} MB")
        portable_config(c["config"], c["n"], cfg)
        print(f"  wrote portable config -> {cfg}")
    print(
        "\ndone. Next: tools/equivalence/fixtures/make_manifest.py, then upload "
        "clips + models.tar.gz as a release."
    )


if __name__ == "__main__":
    main()
