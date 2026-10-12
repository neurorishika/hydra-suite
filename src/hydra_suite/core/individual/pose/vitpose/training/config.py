from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import VARIANTS
from ..geometry import PoseGeometry

_FIELDS = {
    "init_checkpoint",
    "variant",
    "num_keypoints",
    "dataset_dir",
    "output_dir",
    "device",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "drop_path",
    "sigma",
    "grad_clip",
    "val_fraction",
    "seed",
    "resume_from",
    "input_size",
    "scale_jitter",
    "aspect_jitter",
    "flip_pairs",
    "fliplr",
    "flipud",
}


@dataclass
class RunConfig:
    init_checkpoint: str
    variant: str
    num_keypoints: int
    dataset_dir: str
    output_dir: str
    device: str = "cpu"
    epochs: int = 40
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 0.1
    drop_path: float = 0.1
    sigma: float = 2.0
    grad_clip: float = 1.0
    val_fraction: float = 0.2
    seed: int = 0
    resume_from: str | None = None
    input_size: list[int] | None = None
    # Centred window augmentation (see training/window_jitter.py). ``None``
    # keeps the legacy hard-coded random zoom; a number (incl. 0) enables the
    # window jitter: window size ~ U(1-j, 1+j), aspect ratio log-uniform in
    # [1/(1+a), 1+a] at constant area. The crop centre never moves.
    scale_jitter: float | None = None
    aspect_jitter: float = 0.0
    # Mirror augmentation: probability of a left-right / up-down flip, and the
    # left/right keypoint pairs swapped by a flip (0-based index pairs).
    flip_pairs: list[list[int]] | None = None
    fliplr: float = 0.0
    flipud: float = 0.0

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "RunConfig":
        return validate_run_config(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_run_config(d: dict) -> RunConfig:
    unknown = set(d) - _FIELDS
    if unknown:
        raise ValueError(f"unknown run.json keys: {sorted(unknown)}")
    if d.get("variant") not in VARIANTS:
        raise ValueError(
            f"variant must be one of {sorted(VARIANTS)} (uppercase); got {d.get('variant')!r}"
        )
    if int(d.get("num_keypoints", 0)) <= 0:
        raise ValueError("num_keypoints must be positive")
    if int(d.get("epochs", 40)) <= 0:
        raise ValueError("epochs must be positive")
    vf = float(d.get("val_fraction", 0.2))
    if not (0.0 < vf < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    sj = d.get("scale_jitter")
    if sj is not None and not (0.0 <= float(sj) <= 0.9):
        raise ValueError("scale_jitter must be in [0, 0.9] or null")
    if not (0.0 <= float(d.get("aspect_jitter", 0.0)) <= 1.0):
        raise ValueError("aspect_jitter must be in [0, 1]")
    for key in ("fliplr", "flipud"):
        value = d.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number in [0, 1]")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{key} must be in [0, 1]")
    pairs = d.get("flip_pairs")
    if pairs is not None:
        k = int(d.get("num_keypoints", 0))
        ok = isinstance(pairs, (list, tuple)) and all(
            isinstance(p, (list, tuple))
            and len(p) == 2
            and all(
                isinstance(i, int) and not isinstance(i, bool) and 0 <= i < k for i in p
            )
            and p[0] != p[1]
            for p in pairs
        )
        flat = [i for p in pairs for i in p] if ok else []
        if not ok or len(flat) != len(set(flat)):
            raise ValueError(
                "flip_pairs must be disjoint [a, b] pairs of keypoint indices"
            )
        d = dict(d)
        d["flip_pairs"] = [[int(a), int(b)] for a, b in pairs]
    size = d.get("input_size")
    if size is not None:
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("input_size must be a two-element list [H, W]")
        try:
            PoseGeometry.from_hw(size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid input_size {size!r}: {exc}") from exc
        d = dict(d)
        d["input_size"] = [int(size[0]), int(size[1])]
    return RunConfig(**d)
