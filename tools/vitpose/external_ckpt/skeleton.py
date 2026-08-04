"""Skeleton descriptors for the collaborator's external ViTPose checkpoints.

Converted once from their mmpose `dataset_info` configs into plain JSON so this
tool needs no mmpose/mmcv install. Config colours are RGB; we store BGR because
everything downstream draws with OpenCV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SKELETON_DIR = Path(__file__).parent / "skeletons"

_BUILTIN = {
    "ant": "ant_9kp.json",
    "fly": "fly_29kp.json",
}


@dataclass(frozen=True)
class SkeletonSpec:
    name: str
    keypoint_names: list[str]
    keypoint_colors_bgr: list[tuple[int, int, int]]
    skeleton_edges: list[tuple[int, int]]
    edge_colors_bgr: list[tuple[int, int, int]]

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoint_names)


def _to_bgr(rgb: list[int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (int(b), int(g), int(r))


def load_skeleton(path: Path) -> SkeletonSpec:
    payload = json.loads(Path(path).read_text())
    names = list(payload["keypoint_names"])
    if payload["num_keypoints"] != len(names):
        raise ValueError(
            f"{path}: num_keypoints={payload['num_keypoints']} but "
            f"{len(names)} names"
        )
    return SkeletonSpec(
        name=payload["name"],
        keypoint_names=names,
        keypoint_colors_bgr=[_to_bgr(c) for c in payload["keypoint_colors_rgb"]],
        skeleton_edges=[(int(a), int(b)) for a, b in payload["skeleton_edges"]],
        edge_colors_bgr=[_to_bgr(c) for c in payload["edge_colors_rgb"]],
    )


def builtin_skeleton(species: str) -> SkeletonSpec:
    if species not in _BUILTIN:
        raise ValueError(
            f"unknown species {species!r} (expected one of {sorted(_BUILTIN)})"
        )
    return load_skeleton(SKELETON_DIR / _BUILTIN[species])
