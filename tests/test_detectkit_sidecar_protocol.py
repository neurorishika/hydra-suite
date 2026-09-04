from __future__ import annotations

import json

import pytest


def test_sidecar_request_round_trip_and_atomic_result(tmp_path):
    from hydra_suite.detectkit.sidecars.protocol import (
        Operation,
        SidecarRequest,
        SidecarResult,
        SidecarStatus,
        read_request,
        read_result,
        write_request,
        write_result,
    )

    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request = SidecarRequest(
        request_id="req-1",
        operation=Operation.DATASET_INFERENCE,
        payload={"source_path": "/data/source", "batch": 1},
    )
    write_request(request_path, request)
    assert read_request(request_path) == request

    result = SidecarResult(
        request_id="req-1",
        operation=Operation.DATASET_INFERENCE,
        status=SidecarStatus.SUCCESS,
        payload={"image_count": 3},
    )
    write_result(result_path, result)
    assert read_result(result_path, expected=request) == result
    assert not result_path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda raw: raw.update(version=999), "version"),
        (lambda raw: raw.update(operation="not-real"), "operation"),
        (lambda raw: raw.update(payload=[]), "payload"),
        (lambda raw: raw.update(extra="field"), "fields"),
    ],
)
def test_sidecar_request_rejects_malformed_messages(tmp_path, mutation, message):
    from hydra_suite.detectkit.sidecars.protocol import read_request

    path = tmp_path / "request.json"
    raw = {
        "version": 1,
        "request_id": "req-1",
        "operation": "dataset-inference",
        "payload": {},
    }
    mutation(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        read_request(path)


def test_sidecar_protocol_bounds_file_and_nested_payload(tmp_path):
    from hydra_suite.detectkit.sidecars.protocol import MAX_MESSAGE_BYTES, read_request

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_MESSAGE_BYTES + 1))
    with pytest.raises(ValueError, match="cap"):
        read_request(oversized)

    nested = tmp_path / "nested.json"
    value: object = "leaf"
    for _ in range(40):
        value = {"x": value}
    nested.write_text(
        json.dumps(
            {
                "version": 1,
                "request_id": "req-1",
                "operation": "dataset-inference",
                "payload": {"nested": value},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="depth"):
        read_request(nested)


def test_sidecar_result_must_match_request_identity(tmp_path):
    from hydra_suite.detectkit.sidecars.protocol import (
        Operation,
        SidecarRequest,
        read_result,
    )

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "request_id": "different",
                "operation": "dataset-inference",
                "status": "success",
                "message": "",
                "payload": {},
            }
        ),
        encoding="utf-8",
    )
    request = SidecarRequest("expected", Operation.DATASET_INFERENCE, {})
    with pytest.raises(ValueError, match="identity"):
        read_result(path, expected=request)
