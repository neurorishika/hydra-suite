"""Child entry point for contained DetectKit dataset preparation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from hydra_suite.training import TrainingOrchestrator
from hydra_suite.training.dataset_io import fsync_directory, read_bounded_text

from .dataset_preparation_sidecar import MAX_REQUEST_BYTES, decode_request
from .training import preflight_sources, prepare_role_datasets


def _emit(kind: str, message: str) -> None:
    print(json.dumps({"type": kind, "message": str(message)[:32_768]}), flush=True)


def _write_result(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--staging-root", required=True)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--disk-required-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    request_path = Path(args.request).resolve()
    result_path = Path(args.result).resolve()
    staging_root = Path(args.staging_root).resolve()
    final_root = Path(args.final_root).resolve()
    try:
        payload = json.loads(
            read_bounded_text(request_path, max_bytes=MAX_REQUEST_BYTES)
        )
        request = decode_request(payload)
        if staging_root.exists() or final_root.exists():
            raise RuntimeError("private dataset preparation target already exists")
        final_root.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(final_root.parent).free < int(args.disk_required_bytes):
            raise RuntimeError(
                "available disk space fell below the immutable preparation budget"
            )
        staging_root.mkdir(parents=True)
        report = preflight_sources(request.sources)
        if not report.valid:
            messages = "; ".join(issue.message for issue in report.issues[:32])
            raise RuntimeError(f"Dataset source preflight failed: {messages}")
        prepared = prepare_role_datasets(
            TrainingOrchestrator(staging_root),
            request,
            log=lambda message: _emit("log", message),
            status=lambda message: _emit("status", message),
            should_cancel=lambda: False,
        )
        os.replace(staging_root, final_root)
        fsync_directory(final_root.parent)
        remapped = {
            role: str(final_root / Path(path).relative_to(staging_root))
            for role, path in prepared.role_dataset_dirs.items()
        }
        _write_result(
            result_path,
            {
                "success": True,
                "role_dataset_dirs": remapped,
                "roles": [role.value for role in prepared.roles],
                "measured_reference_body_px": prepared.measured_reference_body_px,
                "preflight": report.to_dict(),
            },
        )
        return 0
    except Exception as exc:  # child must publish one bounded diagnostic
        shutil.rmtree(staging_root, ignore_errors=True)
        try:
            _write_result(result_path, {"success": False, "error": str(exc)[:32_768]})
        except Exception:
            pass
        _emit("log", f"Dataset preparation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
