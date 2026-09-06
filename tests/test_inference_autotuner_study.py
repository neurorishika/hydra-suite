from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "experiments"
    / "inference_autotuner_study.py"
)
_SPEC = importlib.util.spec_from_file_location("inference_autotuner_study", _PATH)
study = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = study
assert _SPEC.loader is not None
_SPEC.loader.exec_module(study)


def test_materialize_case_config_isolated_and_bounded():
    base = {
        "start_frame": 10,
        "end_frame": 999,
        "enable_backward_tracking": True,
        "cnn_classifiers": [{"label": "one", "batch_size": 64}],
        "nested": {"unchanged": True},
    }
    result = study.materialize_case_config(
        base, {"detection_batch_size": 8, "cnn_batch_size": 4}, frames=64
    )

    assert result["start_frame"] == 10
    assert result["end_frame"] == 73
    assert result["detection_batch_size"] == 8
    assert result["cnn_classifiers"][0]["batch_size"] == 4
    assert "cnn_batch_size" not in result
    assert result["enable_backward_tracking"] is False
    assert result["enable_postprocessing"] is False
    assert result["enable_profiling"] is True
    assert base["end_frame"] == 999
    assert base["enable_backward_tracking"] is True


def test_flatten_spans_uses_unambiguous_full_paths():
    tree = {
        "name": "root",
        "total_s": 3.0,
        "children": [
            {
                "name": "inference",
                "total_s": 2.0,
                "children": [{"name": "window", "total_s": 1.5, "children": []}],
            }
        ],
    }

    flattened = study.flatten_spans(tree)

    assert flattened["root/inference/window"]["total_s"] == 1.5
    assert set(flattened) == {"root", "root/inference", "root/inference/window"}


def test_selected_cases_rejects_unknown_label():
    matrix = {"cases": [{"label": "one"}, {"label": "two"}]}

    try:
        study._selected_cases(matrix, {"missing"})
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("unknown labels must fail rather than run nothing")


def test_expected_failure_is_preserved_in_selected_case():
    matrix = {"cases": [{"label": "boundary", "expect_failure": True}]}

    selected = study._selected_cases(matrix, {"boundary"})

    assert selected[0]["expect_failure"] is True
