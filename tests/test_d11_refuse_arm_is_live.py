"""The D11 refuse arm must have a production caller, not just an API.

D11 ruled: warn on every surface, refuse ONLY when a run explicitly names a
comparison baseline. `enforce_drift_verdicts` shipped in core with that
behaviour, but both dataset builders still called `log_drift_verdicts`, so
the refuse arm was reachable only from tests. A guard nobody calls is a
guard that does not exist.

These tests pin the wiring itself rather than re-testing the guard's own
logic (covered by tests/test_geometry_drift_severity.py).
"""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "hydra_suite"
_BUILDERS = (
    _SRC / "training" / "sliced_dataset.py",
    _SRC / "training" / "sam3_lora" / "dataset_build.py",
)


def _enforce_calls(path: pathlib.Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "enforce_drift_verdicts"
    ]


def test_both_builders_call_the_enforcing_guard() -> None:
    for path in _BUILDERS:
        assert _enforce_calls(path), (
            f"{path.name} does not call enforce_drift_verdicts. The D11 refuse "
            "arm has no production caller, so a run naming a comparison "
            "baseline would only warn on a geometry divergence."
        )


def test_every_enforce_call_passes_a_comparison_baseline() -> None:
    """Without the kwarg, `enforce_` degrades silently to `log_` behaviour."""
    for path in _BUILDERS:
        for call in _enforce_calls(path):
            names = {kw.arg for kw in call.keywords}
            assert "comparison_baseline" in names, (
                f"{path.name}:{call.lineno} calls enforce_drift_verdicts "
                "without comparison_baseline. The guard then never refuses, "
                "which is indistinguishable from the old log-only call."
            )


def test_no_builder_still_uses_the_log_only_guard() -> None:
    """A leftover log_drift_verdicts call would be a silently un-enforced path."""
    for path in _BUILDERS:
        tree = ast.parse(path.read_text())
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "log_drift_verdicts"
        ]
        assert not offenders, (
            f"{path.name} still calls log_drift_verdicts at {offenders}; that "
            "path can never refuse even against a named baseline."
        )
