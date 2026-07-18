from __future__ import annotations

import re
import sys
from pathlib import Path

from hydra_suite.core.identity.pose.vitpose.training.config import validate_run_config
from hydra_suite.posekit.core.vitpose_checkpoints import resolve_checkpoint

_MODULE = "hydra_suite.core.identity.pose.vitpose.training"
_LINE = re.compile(
    r"^EPOCH (?P<epoch>\d+) train_loss=(?P<tl>[\d.eE+-]+) val_loss=(?P<vl>[\d.eE+-]+) "
    r"pck@0\.05=(?P<p5>[\d.eE+-]+) pck@0\.1=(?P<p1>[\d.eE+-]+)\s*$"
)


def prepare_run(params: dict, run_dir: Path, cache_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_checkpoint(params["init_checkpoint"], cache_dir)
    merged = dict(params)
    merged["init_checkpoint"] = str(resolved)
    merged["output_dir"] = str(run_dir)
    cfg = validate_run_config(merged)  # raises ValueError on bad params
    run_json = run_dir / "run.json"
    cfg.to_json(run_json)
    return run_json


def build_training_command(run_json: Path) -> list[str]:
    return [sys.executable, "-m", _MODULE, "--config", str(run_json)]


def parse_progress_line(line: str) -> dict | None:
    m = _LINE.match(line.strip())
    if not m:
        return None
    return {
        "epoch": int(m["epoch"]),
        "train_loss": float(m["tl"]),
        "val_loss": float(m["vl"]),
        "pck@0.05": float(m["p5"]),
        "pck@0.1": float(m["p1"]),
    }
