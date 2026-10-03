"""Tests for ``hydra doctor`` and the headless ``hydra`` CLI dispatcher."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from hydra_suite.runtime import doctor


def test_hydra_cli_doctor_path_never_imports_qt():
    code = (
        "import sys\n"
        "import hydra_suite.launcher.cli\n"
        "import hydra_suite.runtime.doctor\n"
        "assert 'PySide6' not in sys.modules, sorted(m for m in sys.modules if 'PySide' in m)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hydra_cli_version(capsys):
    from hydra_suite.launcher import cli

    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.startswith("hydra-suite ")


def test_hydra_cli_dispatches_doctor(monkeypatch):
    from hydra_suite.launcher import cli

    seen = {}
    monkeypatch.setattr(
        doctor, "main", lambda argv: seen.setdefault("argv", argv) and 0 or 0
    )
    assert cli.main(["doctor", "--json"]) == 0
    assert seen["argv"] == ["--json"]


def _fake_checks(*statuses):
    return [
        doctor.Check(f"c{i}", s, "detail", "fix it") for i, s in enumerate(statuses)
    ]


@pytest.mark.parametrize(
    "statuses, code",
    [
        ((doctor.OK, doctor.SKIP), 0),
        ((doctor.OK, doctor.WARN), 0),  # warnings never fail the run
        ((doctor.OK, doctor.FAIL, doctor.WARN), 1),
    ],
)
def test_exit_code_reflects_failures_only(monkeypatch, capsys, statuses, code):
    monkeypatch.setattr(doctor, "run_checks", lambda *a, **k: _fake_checks(*statuses))
    assert doctor.main(["--tier", "cpu"]) == code


def test_json_output(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "run_checks", lambda *a, **k: _fake_checks(doctor.OK))
    doctor.main(["--tier", "cpu", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["tier"] == "cpu"
    assert payload["checks"][0]["status"] == "ok"


def test_render_shows_fix_only_for_problems():
    text = doctor.render(_fake_checks(doctor.OK, doctor.FAIL), "cpu")
    assert text.count("fix: fix it") == 1
    assert "1 failed" in text


def test_opencv_check_flags_two_owners(monkeypatch):
    versions = {"opencv-python": "4.13.0", "opencv-python-headless": "4.13.0"}
    monkeypatch.setattr(doctor, "_installed", versions.get)
    check = doctor.check_opencv()
    assert check.status == doctor.FAIL
    assert "opencv-python 4.13.0" in check.detail


def test_opencv_check_accepts_single_headless(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "_installed", {"opencv-python-headless": "4.13.0"}.get)
    monkeypatch.setattr(doctor.sys, "prefix", str(tmp_path))  # no conda-meta
    assert doctor.check_opencv().status == doctor.OK


def test_opencv_check_flags_conda_owned_cv2(monkeypatch, tmp_path):
    (tmp_path / "conda-meta").mkdir()
    (tmp_path / "conda-meta" / "py-opencv-4.13.0-h123_0.json").write_text("{}")
    monkeypatch.setattr(doctor, "_installed", {"opencv-python-headless": "4.13.0"}.get)
    monkeypatch.setattr(doctor.sys, "prefix", str(tmp_path))
    check = doctor.check_opencv()
    assert check.status == doctor.FAIL and "conda py-opencv" in check.detail


def test_sam3_training_is_optional_unless_required(monkeypatch):
    from hydra_suite.training.sam3_lora import availability

    monkeypatch.setattr(
        availability,
        "probe_sam3_training_availability",
        lambda: availability.Sam3TrainingAvailability(False, "env missing\nlong tail"),
    )
    assert doctor.check_sam3_training(required=False).status == doctor.SKIP
    required = doctor.check_sam3_training(required=True)
    assert required.status == doctor.FAIL and required.detail.endswith("env missing")


def test_python_check():
    assert doctor.check_python().status == doctor.OK
