"""Contained entry point for SAM3 checkpoint merge and serialization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hydra_suite.training.contracts import Sam3LoraParams

MAX_REQUEST_BYTES = 64 * 1024


def _read_request(path: Path) -> dict:
    size = path.stat().st_size
    if size <= 0 or size > MAX_REQUEST_BYTES:
        raise RuntimeError("SAM3 publish request exceeds its safe size bound")
    with path.open("rb") as stream:
        raw = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise RuntimeError("SAM3 publish request grew beyond its safe size bound")
    request = json.loads(raw)
    if not isinstance(request, dict):
        raise RuntimeError("SAM3 publish request is not an object")
    return request


def _write_result(path: Path, payload: dict[str, str]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = _read_request(args.request)

    # Heavy imports occur only after child_bootstrap has applied containment.
    from .publish_worker import publish_sam3_artifact

    artifact, sidecar = publish_sam3_artifact(
        run_id=request["run_id"],
        adapters_path=request["adapters_path"],
        base_checkpoint=request["base_checkpoint"],
        build_manifest=request["build_manifest"],
        params=Sam3LoraParams(**request["params"]),
        source_fingerprint=request["source_fingerprint"],
        models_root=request["models_root"],
        publish_attempt_id=request["publish_attempt_id"],
    )
    _write_result(
        args.result,
        {"artifact_path": str(artifact), "sidecar_path": str(sidecar)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
