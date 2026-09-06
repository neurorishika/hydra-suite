"""Process-safe persistent storage for validated inference tuning profiles."""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from typing import IO, Any, Mapping

from hydra_suite.paths import get_data_dir

from .fingerprint import TUNING_SCHEMA_VERSION, TuningProfileKey
from .models import (
    CandidateEvidence,
    EquivalenceVerdict,
    InferenceTuningProfile,
    InferenceTuningSettings,
    ProfileState,
)

try:  # pragma: no branch - one platform implementation is always available
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

MAX_PROFILE_BYTES = 4 * 1024 * 1024
MAX_PROFILE_RECORDS = 512


def profile_store_root() -> Path:
    return get_data_dir() / "inference_tuning_profiles"


class SingleFlightClaim(AbstractContextManager["SingleFlightClaim"]):
    """A per-profile OS lock; lock-file contents are diagnostic only."""

    def __init__(self, path: Path, *, timeout_seconds: float = 0.0) -> None:
        self.path = path
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self.acquired = False
        self._handle: IO[str] | None = None

    def __enter__(self) -> "SingleFlightClaim":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                _try_lock(handle)
                self.acquired = True
                self._handle = handle
                handle.seek(0)
                handle.truncate()
                json.dump(
                    {"pid": os.getpid(), "acquired_at_unix_ns": time.time_ns()},
                    handle,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    handle.close()
                    return self
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def __exit__(self, *_args: object) -> None:
        if self._handle is None:
            return
        try:
            _unlock(self._handle)
        finally:
            self._handle.close()
            self._handle = None
            self.acquired = False


class InferenceTuningProfileStore:
    """One bounded atomic record per exact key plus per-key single-flight locks."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else profile_store_root()

    def _record_path(self, key: TuningProfileKey) -> Path:
        return self.root / f"{key.digest}.json"

    def _lock_path(self, key: TuningProfileKey) -> Path:
        return self.root / "locks" / f"{key.digest}.lock"

    def claim(
        self, key: TuningProfileKey, *, timeout_seconds: float = 0.0
    ) -> SingleFlightClaim:
        return SingleFlightClaim(self._lock_path(key), timeout_seconds=timeout_seconds)

    def load(self, key: TuningProfileKey) -> InferenceTuningProfile | None:
        path = self._record_path(key)
        if not path.is_file():
            return None
        try:
            with path.open("rb") as stream:
                encoded = stream.read(MAX_PROFILE_BYTES + 1)
            if len(encoded) > MAX_PROFILE_BYTES:
                return None
            raw = json.loads(encoded)
            if not isinstance(raw, dict):
                return None
            if raw.get("schema_version") != TUNING_SCHEMA_VERSION:
                return None
            profile = _profile_from_dict(raw.get("profile"))
            if profile.key != key or profile.profile_id != key.digest[:24]:
                return None
            return profile
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def save(self, profile: InferenceTuningProfile) -> None:
        if not isinstance(profile.key, TuningProfileKey):
            raise TypeError("profile.key must be a TuningProfileKey")
        expected_id = profile.key.digest[:24]
        if profile.profile_id != expected_id:
            raise ValueError("profile_id must be derived from the exact key digest")
        encoded = json.dumps(
            {
                "schema_version": TUNING_SCHEMA_VERSION,
                "profile": _profile_to_dict(profile),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > MAX_PROFILE_BYTES:
            raise ValueError("inference tuning profile exceeds its size cap")
        self.root.mkdir(parents=True, exist_ok=True)
        self._prune_before_write(excluding=self._record_path(profile.key))
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{expected_id}.", suffix=".tmp", dir=self.root
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            destination = self._record_path(profile.key)
            os.replace(temporary, destination)
            _fsync_directory(self.root)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def mark_provisional(self, key: TuningProfileKey) -> bool:
        """Invalidate reuse without changing the stored winning settings."""

        profile = self.load(key)
        if profile is None:
            return False
        raw = _profile_to_dict(profile)
        raw["state"] = ProfileState.PROVISIONAL.value
        replacement = _profile_from_dict(raw)
        with self.claim(key, timeout_seconds=0.0) as claim:
            if not claim.acquired:
                return False
            self.save(replacement)
        return True

    def _prune_before_write(self, *, excluding: Path) -> None:
        try:
            records = [
                item
                for item in self.root.glob("*.json")
                if item != excluding and item.is_file()
            ]
            excess = len(records) - MAX_PROFILE_RECORDS + 1
            if excess > 0:
                for path in sorted(records, key=lambda item: item.stat().st_mtime_ns)[
                    :excess
                ]:
                    path.unlink(missing_ok=True)
        except OSError:
            # Retention is best-effort; the record itself still gets an atomic write.
            return


def _profile_to_dict(profile: InferenceTuningProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "key": profile.key.to_dict(),
        "baseline": profile.baseline.to_dict(),
        "requested": profile.requested.to_dict(),
        "admitted": profile.admitted.to_dict(),
        "selected": profile.selected.to_dict(),
        "candidates": [_candidate_to_dict(item) for item in profile.candidates],
        "state": profile.state.value,
        "selection_reason": profile.selection_reason,
        "rejected": [list(item) for item in profile.rejected],
        "calibration_summary": [list(item) for item in profile.calibration_summary],
        "created_at_unix_ns": profile.created_at_unix_ns,
        "last_validation_unix_ns": profile.last_validation_unix_ns,
        "observed_production_throughput": list(profile.observed_production_throughput),
    }


def _candidate_to_dict(candidate: CandidateEvidence) -> dict[str, Any]:
    result = asdict(candidate)
    result["settings"] = candidate.settings.to_dict()
    if candidate.equivalence is not None:
        result["equivalence"] = asdict(candidate.equivalence)
    return result


def _candidate_from_dict(value: Mapping[str, Any]) -> CandidateEvidence:
    raw = dict(value)
    raw["settings"] = InferenceTuningSettings.from_dict(raw["settings"])
    if raw.get("equivalence") is not None:
        verdict = dict(raw["equivalence"])
        verdict["details"] = tuple(verdict.get("details", ()))
        raw["equivalence"] = EquivalenceVerdict(**verdict)
    for name in (
        "throughput_samples",
        "stage_seconds_samples",
        "artifact_ids",
        "thermal_c_range",
    ):
        if raw.get(name) is not None:
            raw[name] = tuple(raw[name])
    return CandidateEvidence(**raw)


def _profile_from_dict(value: object) -> InferenceTuningProfile:
    if not isinstance(value, Mapping):
        raise TypeError("profile must be an object")
    raw = dict(value)
    raw["key"] = TuningProfileKey.from_dict(raw["key"])
    for name in ("baseline", "requested", "admitted", "selected"):
        raw[name] = InferenceTuningSettings.from_dict(raw[name])
    raw["candidates"] = tuple(_candidate_from_dict(item) for item in raw["candidates"])
    raw["state"] = ProfileState(raw["state"])
    raw["rejected"] = tuple(tuple(item) for item in raw.get("rejected", ()))
    raw["calibration_summary"] = tuple(
        tuple(item) for item in raw.get("calibration_summary", ())
    )
    raw["observed_production_throughput"] = tuple(
        raw.get("observed_production_throughput", ())
    )
    return InferenceTuningProfile(**raw)


def _try_lock(handle: IO[str]) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    if msvcrt is None:  # pragma: no cover
        raise RuntimeError("no interprocess file locking implementation")
    handle.seek(0)
    if handle.read(1) == "":
        handle.write("0")
        handle.flush()
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)


def _unlock(handle: IO[str]) -> None:
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    if msvcrt is not None:  # pragma: no cover
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":  # pragma: no cover - Windows has no directory fsync
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
