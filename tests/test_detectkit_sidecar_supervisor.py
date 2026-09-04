import json
from types import SimpleNamespace

import pytest

from hydra_suite.runtime.process_supervisor import ExitKind


@pytest.mark.parametrize(
    "kind",
    [
        ExitKind.HOST_SOFT_LIMIT,
        ExitKind.HOST_HARD_LIMIT,
        ExitKind.ACCELERATOR_OOM,
        ExitKind.CANCELED,
        ExitKind.ORDINARY_FAILURE,
    ],
)
def test_protected_operation_preserves_distinct_exit_classification(kind):
    from hydra_suite.detectkit.sidecars.supervisor import _failed_outcome

    result = _failed_outcome(
        SimpleNamespace(
            classified_exit=SimpleNamespace(kind=kind, message="classified"),
            peak_tree_rss_bytes=2 * 1024**3,
            peak_accelerator_bytes=3,
            dropped_output_lines=7,
        ),
        4 * 1024**3,
    )

    assert result.failure_kind == kind.value
    assert result.canceled is (kind is ExitKind.CANCELED)
    assert "4.0 GiB" in result.message
    assert "2.0 GiB" in result.message
    assert result.dropped_output_lines == 7


def test_progress_protocol_is_typed_and_bounded():
    from hydra_suite.detectkit.sidecars.supervisor import _parse_progress

    valid = json.dumps(
        {
            "detectkit_sidecar": 1,
            "type": "progress",
            "percent": 150,
            "message": "working",
        }
    )
    assert _parse_progress(valid) == (100, "working")
    assert _parse_progress(json.dumps({"detectkit_sidecar": 1})) is None
    assert _parse_progress("{" + "x" * 20_000) is None
    oversized = json.dumps(
        {
            "detectkit_sidecar": 1,
            "type": "progress",
            "percent": 2,
            "message": "x" * 4097,
        }
    )
    assert _parse_progress(oversized) is None
