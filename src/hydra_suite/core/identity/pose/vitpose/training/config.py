from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import VARIANTS

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
    if int(d.get("epochs", 0)) <= 0:
        raise ValueError("epochs must be positive")
    vf = float(d.get("val_fraction", 0.2))
    if not (0.0 < vf < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    return RunConfig(**d)
