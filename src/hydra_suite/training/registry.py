"""Local training run registry for MAT."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows does not provide fcntl.
    fcntl = None

from .contracts import TrainingRunSpec

_PROCESS_REGISTRY_LOCK = threading.RLock()
_MAX_REGISTRY_BYTES = 8 * 1024 * 1024
_MAX_REGISTRY_RECORDS = 10_000
_MAX_REGISTRY_JSON_DEPTH = 64
_MAX_REGISTRY_JSON_VALUES = 100_000
_MAX_REGISTRY_STRING_CODEPOINTS = (_MAX_REGISTRY_BYTES - 16) // 12


def _validate_registry_json_shape(encoded: bytes) -> None:
    """Reject compact structures that amplify beyond the bounded raw file."""

    depth = 0
    separators = 0
    in_string = False
    escaped = False
    for byte in encoded:
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue
        if byte == ord('"'):
            in_string = True
        elif byte in (ord("["), ord("{")):
            depth += 1
            if depth > _MAX_REGISTRY_JSON_DEPTH:
                raise RuntimeError("Training registry exceeds its safe nesting cap")
        elif byte in (ord("]"), ord("}")):
            depth = max(0, depth - 1)
        elif byte == ord(","):
            separators += 1
            if separators >= _MAX_REGISTRY_JSON_VALUES:
                raise RuntimeError("Training registry exceeds its safe value cap")


def _validate_registry_value_shape(
    value: object, *, depth: int = 0, separators: list[int] | None = None
) -> None:
    """Apply the decoder's shape caps before JSON encoding can allocate."""

    if separators is None:
        separators = [0]
    if isinstance(value, str):
        # JSON may expand one non-BMP code point to a 12-byte surrogate pair.
        # This bound ensures one encoder chunk cannot exceed the file cap.
        if len(value) > _MAX_REGISTRY_STRING_CODEPOINTS:
            raise RuntimeError("Training registry string exceeds its safe size cap")
        return
    if isinstance(value, dict):
        next_depth = depth + 1
        if next_depth > _MAX_REGISTRY_JSON_DEPTH:
            raise RuntimeError("Training registry exceeds its safe nesting cap")
        separators[0] += max(0, len(value) - 1)
        if separators[0] >= _MAX_REGISTRY_JSON_VALUES:
            raise RuntimeError("Training registry exceeds its safe value cap")
        for key, item in value.items():
            _validate_registry_value_shape(key, depth=next_depth, separators=separators)
            _validate_registry_value_shape(
                item, depth=next_depth, separators=separators
            )
        return
    if isinstance(value, (list, tuple)):
        next_depth = depth + 1
        if next_depth > _MAX_REGISTRY_JSON_DEPTH:
            raise RuntimeError("Training registry exceeds its safe nesting cap")
        separators[0] += max(0, len(value) - 1)
        if separators[0] >= _MAX_REGISTRY_JSON_VALUES:
            raise RuntimeError("Training registry exceeds its safe value cap")
        for item in value:
            _validate_registry_value_shape(
                item, depth=next_depth, separators=separators
            )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _use_project_root_override() -> bool:
    return getattr(_project_root, "__module__", __name__) != __name__


def get_runs_root() -> Path:
    """Return the directory where training run metadata and artifacts are stored."""
    if _use_project_root_override():
        root = _project_root() / "training" / "runs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    from hydra_suite.paths import get_training_runs_dir

    return get_training_runs_dir()


def get_registry_path() -> Path:
    """Return the path to the JSON file that indexes all training runs."""
    return get_runs_root() / "registry.json"


@contextmanager
def _registry_lock():
    """Serialize registry transactions across threads and POSIX processes."""

    with _PROCESS_REGISTRY_LOCK:
        lock_path = get_runs_root() / ".registry.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_registry_unlocked() -> dict[str, Any]:
    path = get_registry_path()
    if not path.exists():
        return {"runs": []}
    try:
        with path.open("rb") as handle:
            encoded = handle.read(_MAX_REGISTRY_BYTES + 1)
        if len(encoded) > _MAX_REGISTRY_BYTES:
            raise RuntimeError(
                "Training registry exceeds its safe size cap; archive old runs "
                "before starting another training job"
            )
        _validate_registry_json_shape(encoded)
        data = json.loads(encoded)
    except RuntimeError:
        raise
    except Exception:
        return {"runs": []}
    if not isinstance(data, dict):
        return {"runs": []}
    _validate_registry_value_shape(data)
    runs = data.get("runs")
    if not isinstance(runs, list):
        data["runs"] = []
    elif len(runs) > _MAX_REGISTRY_RECORDS:
        raise RuntimeError(
            "Training registry exceeds its safe run-record cap; archive old runs "
            "before starting another training job"
        )
    return data


def _save_registry_unlocked(registry: dict[str, Any]) -> None:
    path = get_registry_path()
    runs = registry.get("runs")
    if isinstance(runs, list) and len(runs) > _MAX_REGISTRY_RECORDS:
        raise RuntimeError("Training registry exceeds its safe run-record cap")
    _validate_registry_value_shape(registry)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".registry.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            encoded_bytes = 0
            for chunk in json.JSONEncoder(indent=2).iterencode(registry):
                encoded_bytes += len(chunk.encode("utf-8"))
                if encoded_bytes > _MAX_REGISTRY_BYTES:
                    raise RuntimeError("Training registry exceeds its safe size cap")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_registry() -> dict[str, Any]:
    """Load the training run registry from disk, returning an empty structure on failure."""

    with _registry_lock():
        return _load_registry_unlocked()


def save_registry(registry: dict[str, Any]) -> None:
    """Atomically write the training run registry dict to disk as JSON."""

    with _registry_lock():
        _save_registry_unlocked(registry)


def _update_hash_from_file(digest: Any, path: Path) -> None:
    """Hash a metadata file with fixed-size reads."""

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)


def _iter_dataset_files(directory: Path, *, depth: int = 0):
    """Yield files with streaming scandir and bounded recursion depth."""

    if depth > 128:
        raise ValueError(f"Dataset directory nesting exceeds safe depth: {directory}")
    with os.scandir(directory) as entries:
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    yield from _iter_dataset_files(Path(entry.path), depth=depth + 1)
                elif entry.is_file(follow_symlinks=True):
                    yield entry
            except FileNotFoundError:
                # A concurrently removed entry is absent from this observed
                # snapshot; never turn churn into an unbounded retry.
                continue


def dataset_fingerprint(dataset_dir: str | Path) -> str:
    """Compute a stable fingerprint without materializing the dataset tree."""

    root = Path(dataset_dir).expanduser().resolve()
    h = hashlib.sha256()
    h.update(b"hydra-dataset-fingerprint-v2\0")
    h.update(str(root).encode("utf-8"))

    manifest = root / "manifest.json"
    if manifest.exists():
        _update_hash_from_file(h, manifest)

    yaml_path = root / "dataset.yaml"
    if yaml_path.exists():
        _update_hash_from_file(h, yaml_path)

    # Directory order is filesystem-dependent. Combine independent entry
    # digests using commutative fixed-width accumulators, preserving stable
    # output without retaining and sorting every path.
    modulus = 1 << 256
    digest_sum = 0
    digest_xor = 0
    file_count = 0
    for entry in _iter_dataset_files(root):
        try:
            st = entry.stat(follow_symlinks=True)
        except FileNotFoundError:
            continue
        rel = Path(entry.path).relative_to(root).as_posix()
        entry_hash = hashlib.sha256()
        entry_hash.update(rel.encode("utf-8"))
        entry_hash.update(b"\0")
        # File size is unsigned; POSIX mtimes may validly predate the epoch.
        entry_hash.update(struct.pack(">Qq", st.st_size, st.st_mtime_ns))
        value = int.from_bytes(entry_hash.digest(), "big")
        digest_sum = (digest_sum + value) % modulus
        digest_xor ^= value
        file_count += 1
    h.update(struct.pack(">Q", file_count))
    h.update(digest_sum.to_bytes(32, "big"))
    h.update(digest_xor.to_bytes(32, "big"))

    return h.hexdigest()


def new_run_id(role: str) -> str:
    """Generate a unique run ID combining timestamp, role name, and random suffix."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tail = hashlib.sha1(os.urandom(16)).hexdigest()[:8]
    return f"{stamp}_{role}_{tail}"


def create_run_record(
    spec: TrainingRunSpec,
    run_id: str,
    run_dir: str | Path,
    dataset_fp: str,
    parent_run_id: str = "",
) -> dict[str, Any]:
    """Create and persist initial run record."""

    now = datetime.now().isoformat(timespec="seconds")
    rec = {
        "run_id": run_id,
        "started_at": now,
        "finished_at": "",
        "status": "running",
        "role": spec.role.value,
        "dataset_fingerprint": dataset_fp,
        "command": [],
        "metrics_paths": [],
        "artifact_paths": [],
        "published_model_path": "",
        "published_registry_entry": "",
        "parent_run_id": parent_run_id,
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "spec": spec.to_dict(),
    }
    with _registry_lock():
        reg = _load_registry_unlocked()
        reg.setdefault("runs", []).append(rec)
        _save_registry_unlocked(reg)
    return rec


def update_run_record(run_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    """Merge *patch* into the registry record for *run_id* and persist; return the updated record or None."""
    with _registry_lock():
        reg = _load_registry_unlocked()
        for rec in reg.get("runs", []):
            if rec.get("run_id") == run_id:
                rec.update(patch)
                _save_registry_unlocked(reg)
                return rec
    return None


def finalize_run_record(
    run_id: str,
    *,
    status: str,
    command: list[str] | None = None,
    metrics_paths: list[str] | None = None,
    artifact_paths: list[str] | None = None,
    published_model_path: str = "",
    published_registry_entry: str = "",
    error_message: str = "",
    failure_details: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Finalize a run record with terminal status."""

    patch = {
        "status": status,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "command": list(command or []),
        "metrics_paths": list(metrics_paths or []),
        "artifact_paths": list(artifact_paths or []),
        "published_model_path": published_model_path,
        "published_registry_entry": published_registry_entry,
    }
    if error_message:
        patch["error_message"] = error_message
    if failure_details:
        for key in (
            "failure_kind",
            "resource_preflight",
            "containment",
            "resource_telemetry",
            "retry_history",
        ):
            if key in failure_details:
                patch[key] = failure_details[key]
    return update_run_record(run_id, patch)
