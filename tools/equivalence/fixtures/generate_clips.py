"""Regenerate the short equivalence clips + portable configs from source videos.

Provenance/dev tool: run on a machine that has the full source videos and their
tracking configs. Produces, under fixtures/clips/ and fixtures/configs/:

  - a short lossless clip per entry (frame-accurate trim via ffmpeg)
  - a portable tracking config (frame range remapped to 0..N-1, side-outputs off,
    detections recomputed; model paths kept relative so they resolve via the
    machine's models dir; pose skeleton supplied at run time via runner --skeleton)

The clips + their required model files are then uploaded as GitHub Release assets
(see manifest.json / fetch_fixtures.sh); only the small portable bits are committed.

Edit CLIPS below to change windows or add feature combinations.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIPS_DIR = HERE / "clips"
CONFIGS_DIR = HERE / "configs"

# name | source video | source config | start_frame | n_frames | note
CLIPS = [
    dict(
        name="emi_obb_identity",
        video="/Users/neurorishika/Projects/Rockefeller/RutaKronauer/MultiTrackerData/ant/emi_short.mp4",
        config="/Users/neurorishika/Projects/Rockefeller/RutaKronauer/MultiTrackerData/ant/emi_short_config.json",
        start=0,
        n=60,
        note="OBB-direct + identity online decoder (no headtail/pose)",
    ),
    dict(
        name="ant_pose_headtail",
        video="/Users/neurorishika/Projects/Rockefeller/RutaKronauer/MultiTrackerData/ant2/000001_cropped_roi.mp4",
        config="/Users/neurorishika/Projects/Rockefeller/RutaKronauer/MultiTrackerData/ant2/000001_cropped_roi_config.json",
        start=3629,
        n=60,
        note="OBB + headtail + SLEAP pose + identity (realtime)",
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


def portable_config(src_config: str, n: int, out: Path) -> None:
    with open(src_config) as fh:
        cfg = json.load(fh)
    cfg["start_frame"] = 0
    cfg["end_frame"] = n - 1
    cfg["use_cached_detections"] = False
    # Blank machine-specific paths; runner.py sets these per run, and the pose
    # skeleton is supplied per run via runner --skeleton (keeps configs portable).
    for k in ("file_path", "csv_path", "video_output_path", "pose_skeleton_file"):
        cfg[k] = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(cfg, fh, indent=2)


def main() -> None:
    for c in CLIPS:
        clip = CLIPS_DIR / f"{c['name']}.mp4"
        cfg = CONFIGS_DIR / f"{c['name']}.json"
        print(f"[{c['name']}] {c['note']}")
        print(f"  extracting {c['n']} frames @ {c['start']} -> {clip}")
        extract_clip(c["video"], c["start"], c["n"], clip)
        portable_config(c["config"], c["n"], cfg)
        print(f"  wrote portable config -> {cfg}")
    print(
        "done. Next: tools/equivalence/fixtures/make_manifest.py, then upload clips+models as a release."
    )


if __name__ == "__main__":
    main()
