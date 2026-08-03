"""Run an external ViTPose checkpoint over sampled crops and write sheets.

    python -m tools.vitpose.external_ckpt.cli \
        --species ant --ckpt /path/ViTPose_base_ant9kp_256x256.pth \
        --out /tmp/vitpose_probe
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .crops import crop_matrix, select_samples, warp_crop
from .model import load_external_checkpoint, predict
from .render import confidence_table, contact_sheet, draw_pose, label_tile
from .skeleton import builtin_skeleton

DEMO = Path("/Users/neurorishika/Projects/Rockefeller/Ruta/Presentation/DEMO")


@dataclass(frozen=True)
class SpeciesPreset:
    video: Path
    csv: Path
    body_size_px: float


SPECIES: dict[str, SpeciesPreset] = {
    "ant": SpeciesPreset(
        video=DEMO / "DEMO 3" / "ant.mp4",
        csv=DEMO / "DEMO 3" / "ant_tracking_final.csv",
        body_size_px=76.81,
    ),
    "fly": SpeciesPreset(
        video=DEMO / "DEMO 4" / "melanogaster.mp4",
        csv=DEMO / "DEMO 4" / "melanogaster_tracking_final.csv",
        body_size_px=104.14,
    ),
}


def read_frames(video_path: Path, frame_ids: list[int]) -> dict[int, np.ndarray]:
    """Seek-and-grab. Frames are requested in ascending order so the decoder
    only ever moves forward."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    out: dict[int, np.ndarray] = {}
    try:
        for fid in sorted(set(frame_ids)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cannot read frame {fid} of {video_path}")
            out[fid] = frame
    finally:
        cap.release()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--species", required=True, choices=sorted(SPECIES))
    p.add_argument("--ckpt", type=Path, required=True, help="external .pth checkpoint")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--body-size", type=float, default=None, dest="body_size")
    p.add_argument("--n", type=int, default=12, help="samples per crop mode")
    p.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="crop side as a multiple of reference body size",
    )
    p.add_argument("--out-px", type=int, default=256, dest="out_px")
    p.add_argument("--device", default="mps")
    p.add_argument("--conf-thr", type=float, default=0.2, dest="conf_thr")
    p.add_argument("--out", type=Path, default=Path("/tmp/vitpose_probe"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preset = SPECIES[args.species]
    video = args.video or preset.video
    csv = args.csv or preset.csv
    body_size = args.body_size or preset.body_size_px
    side_px = args.scale * body_size

    spec = builtin_skeleton(args.species)
    model, num_keypoints = load_external_checkpoint(args.ckpt)
    if num_keypoints != spec.num_keypoints:
        raise SystemExit(
            f"checkpoint has {num_keypoints} keypoints but the {args.species} "
            f"skeleton declares {spec.num_keypoints}"
        )
    print(f"loaded {args.ckpt.name}: {num_keypoints} keypoints, strict OK")

    samples = select_samples(csv, args.n)
    frames = read_frames(video, [s.frame_id for s in samples])
    args.out.mkdir(parents=True, exist_ok=True)

    for mode, rotate in (("axis", False), ("rot", True)):
        crops = [
            warp_crop(
                frames[s.frame_id],
                crop_matrix(s.cx, s.cy, s.theta, side_px, args.out_px, rotate),
                args.out_px,
            )
            for s in samples
        ]
        coords, conf = predict(model, crops, args.device)
        tiles = [
            label_tile(
                draw_pose(crops[i], coords[i], conf[i], spec, conf_thr=args.conf_thr),
                f"f={samples[i].frame_id} t={samples[i].track_id}",
            )
            for i in range(len(crops))
        ]
        sheet_path = args.out / f"{args.species}_{mode}.png"
        cv2.imwrite(str(sheet_path), contact_sheet(tiles, cols=4))
        table_path = args.out / f"{args.species}_{mode}_confidence.txt"
        table_path.write_text(confidence_table(conf, spec) + "\n")
        print(f"wrote {sheet_path}")
        print(f"wrote {table_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
