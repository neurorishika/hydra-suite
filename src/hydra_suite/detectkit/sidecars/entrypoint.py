"""Minimal CLI entry point loaded after the resource-limit bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hydra_suite.runtime.safe_text import bounded_terminal_text

from .operations import OPERATIONS
from .protocol import SidecarResult, SidecarStatus, read_request, write_result


def _progress(percent: int, message: str) -> None:
    record = {
        "detectkit_sidecar": 1,
        "type": "progress",
        "percent": max(0, min(100, int(percent))),
        "message": bounded_terminal_text(message),
    }
    print(json.dumps(record, separators=(",", ":")), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    request_path = Path(args.request)
    result_path = Path(args.result)
    request = read_request(request_path)
    try:
        executor = OPERATIONS.get(request.operation.value)
        if executor is None:
            raise ValueError(f"operation is not implemented: {request.operation.value}")
        payload = executor(dict(request.payload), _progress)
        result = SidecarResult(
            request.request_id,
            request.operation,
            SidecarStatus.SUCCESS,
            payload=payload,
        )
    except Exception as exc:  # the parent owns classification and cleanup
        result = SidecarResult(
            request.request_id,
            request.operation,
            SidecarStatus.FAILED,
            message=bounded_terminal_text(exc),
        )
        write_result(result_path, result)
        print(bounded_terminal_text(exc), file=sys.stderr, flush=True)
        return 1
    write_result(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
