from __future__ import annotations

import argparse
from pathlib import Path

from .config import RunConfig
from .train import train


def main() -> None:
    ap = argparse.ArgumentParser(prog="vitpose.training")
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    result = train(RunConfig.from_json(args.config))
    print(
        f"DONE best_pck={result['best_pck']:.4f} best_epoch={result['best_epoch']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
