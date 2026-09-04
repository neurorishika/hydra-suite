"""Small, versioned control protocol for protected DetectKit sidecars.

Only metadata belongs in these JSON messages. Images, masks, polygons, complete
prediction sets, and calibration evidence are exchanged through bounded files.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_PAYLOAD_DEPTH = 24
MAX_PAYLOAD_VALUES = 20_000
MAX_TEXT_CHARS = 16_384
MAX_IDENTIFIER_CHARS = 128


class Operation(str, Enum):
    DATASET_INFERENCE = "dataset-inference"
    EVALUATION = "evaluation"
    SEMANTIC_ESCALATION = "semantic-escalation"
    SEMANTIC_CALIBRATION = "semantic-calibration"
    SEMANTIC_PREVIEW = "semantic-preview"


class SidecarStatus(str, Enum):
    SUCCESS = "success"
    CANCELED = "canceled"
    FAILED = "failed"


def _validate_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_IDENTIFIER_CHARS:
        raise ValueError(f"{name} is invalid")
    if any(ord(char) < 0x20 for char in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _validate_json_value(value: Any, *, depth: int = 0, counter: list[int]) -> None:
    if depth > MAX_PAYLOAD_DEPTH:
        raise ValueError("payload exceeds maximum nesting depth")
    counter[0] += 1
    if counter[0] > MAX_PAYLOAD_VALUES:
        raise ValueError("payload exceeds maximum value count")
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        if len(value) > MAX_TEXT_CHARS:
            raise ValueError("payload text exceeds safe length")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, depth=depth + 1, counter=counter)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > MAX_IDENTIFIER_CHARS:
                raise ValueError("payload object key is invalid")
            _validate_json_value(item, depth=depth + 1, counter=counter)
        return
    raise TypeError("payload contains a non-JSON value")


def _validated_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("payload must be an object")
    _validate_json_value(value, counter=[0])
    return dict(value)


@dataclass(frozen=True, slots=True)
class SidecarRequest:
    request_id: str
    operation: Operation
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _validate_identifier(self.request_id, "request_id")
        )
        object.__setattr__(self, "operation", Operation(self.operation))
        object.__setattr__(self, "payload", _validated_payload(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class SidecarResult:
    request_id: str
    operation: Operation
    status: SidecarStatus
    message: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _validate_identifier(self.request_id, "request_id")
        )
        object.__setattr__(self, "operation", Operation(self.operation))
        object.__setattr__(self, "status", SidecarStatus(self.status))
        if not isinstance(self.message, str) or len(self.message) > MAX_TEXT_CHARS:
            raise ValueError("result message exceeds safe length")
        object.__setattr__(self, "payload", _validated_payload(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "status": self.status.value,
            "message": self.message,
            "payload": dict(self.payload),
        }


def _read_mapping(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        encoded = stream.read(MAX_MESSAGE_BYTES + 1)
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ValueError(f"sidecar message exceeds {MAX_MESSAGE_BYTES}-byte cap")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("sidecar message contains duplicate fields")
            result[key] = value
        return result

    raw = json.loads(encoded, object_pairs_hook=no_duplicates)
    if not isinstance(raw, dict):
        raise TypeError("sidecar message root must be an object")
    return raw


def _require_fields(raw: dict[str, Any], expected: set[str]) -> None:
    if set(raw) != expected:
        raise ValueError("sidecar message fields do not match the protocol")
    version = raw.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != PROTOCOL_VERSION
    ):
        raise ValueError(f"unsupported sidecar protocol version: {version!r}")


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ValueError(f"sidecar message exceeds {MAX_MESSAGE_BYTES}-byte cap")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_request(path: Path, request: SidecarRequest) -> None:
    _write_atomic(path, request.to_dict())


def read_request(path: Path) -> SidecarRequest:
    raw = _read_mapping(path)
    _require_fields(raw, {"version", "request_id", "operation", "payload"})
    try:
        operation = Operation(raw["operation"])
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported sidecar operation") from exc
    return SidecarRequest(raw["request_id"], operation, raw["payload"])


def write_result(path: Path, result: SidecarResult) -> None:
    _write_atomic(path, result.to_dict())


def read_result(path: Path, *, expected: SidecarRequest | None = None) -> SidecarResult:
    raw = _read_mapping(path)
    _require_fields(
        raw,
        {"version", "request_id", "operation", "status", "message", "payload"},
    )
    try:
        result = SidecarResult(
            raw["request_id"],
            Operation(raw["operation"]),
            SidecarStatus(raw["status"]),
            raw["message"],
            raw["payload"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid sidecar result: {exc}") from exc
    if expected is not None and (
        result.request_id != expected.request_id
        or result.operation is not expected.operation
    ):
        raise ValueError("sidecar result identity does not match its request")
    return result
