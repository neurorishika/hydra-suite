"""Predictor geometry must enter the staging hash, or stale candidate caches
are silently reused after an imgsz change."""

from types import SimpleNamespace

from hydra_suite.detectkit.jobs.semantic_escalation import staged_dirname_for


def _src(tmp_path):
    return SimpleNamespace(path=str(tmp_path), name="src")


def test_imgsz_changes_the_staging_dirname(tmp_path):
    a = staged_dirname_for(_src(tmp_path), "sam3", "ant", imgsz=644)
    b = staged_dirname_for(_src(tmp_path), "sam3", "ant", imgsz=1008)
    assert a != b


def test_prompt_and_variant_still_change_it(tmp_path):
    base = staged_dirname_for(_src(tmp_path), "sam3", "ant")
    assert staged_dirname_for(_src(tmp_path), "sam3", "beetle") != base
    assert staged_dirname_for(_src(tmp_path), "sam3-x", "ant") != base


def test_same_inputs_are_stable(tmp_path):
    a = staged_dirname_for(_src(tmp_path), "sam3", "ant", imgsz=1008)
    b = staged_dirname_for(_src(tmp_path), "sam3", "ant", imgsz=1008)
    assert a == b
