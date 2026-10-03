"""``detectkit escalate``: headless SAM2/SAM3 escalation entry point."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hydra_suite.detectkit import app, escalate_cli


def test_detectkit_dispatches_escalate(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        escalate_cli, "main", lambda argv: seen.setdefault("argv", argv) or 0
    )
    app.main(["escalate", "sam2", "--project", "/p"])
    assert seen["argv"] == ["sam2", "--project", "/p"]


def test_parser_defaults():
    args = escalate_cli.build_parser().parse_args(["sam2", "--project", "/p"])
    assert (args.device, args.source, args.overwrite) == ("auto", [], False)
    args = escalate_cli.build_parser().parse_args(
        ["sam3", "--project", "/p", "--prompt", "ant", "--device", "cpu"]
    )
    assert (args.prompt, args.device, args.variant) == ("ant", "cpu", "sam3")


def test_sam3_requires_a_prompt():
    with pytest.raises(SystemExit):
        escalate_cli.build_parser().parse_args(["sam3", "--project", "/p"])


def test_unknown_source_is_a_clear_error():
    project = SimpleNamespace(sources=[SimpleNamespace(name="a")])
    with pytest.raises(SystemExit, match="unknown source"):
        escalate_cli._selected(project, ["b"])


def test_sam2_runs_on_the_requested_device(monkeypatch, tmp_path):
    from hydra_suite.core.inference.sam2 import executor as executor_mod
    from hydra_suite.detectkit.gui import project as project_mod
    from hydra_suite.detectkit.jobs import sam2_escalation

    src = SimpleNamespace(name="a", level="obb", path=str(tmp_path / "a"))
    project = SimpleNamespace(project_dir=str(tmp_path), sources=[src])
    monkeypatch.setattr(escalate_cli, "_open", lambda _p: project)
    monkeypatch.setattr(project_mod, "save_project", lambda _p: None)
    seen = {}
    monkeypatch.setattr(
        executor_mod.Sam2SegmentExecutor,
        "from_variant",
        staticmethod(lambda variant, device=None: seen.setdefault("device", device)),
    )
    monkeypatch.setattr(
        sam2_escalation,
        "run_escalation",
        lambda req, ex, **kw: sam2_escalation.EscalationResult(staged=["a"]),
    )
    args = escalate_cli.build_parser().parse_args(
        ["sam2", "--project", str(tmp_path), "--device", "cpu"]
    )
    assert escalate_cli.run_sam2(args) == 0
    assert seen["device"] == "cpu"


def test_sam3_reports_missing_dependencies(monkeypatch, capsys):
    from hydra_suite.core.inference.semantic import checkpoints

    monkeypatch.setattr(
        checkpoints,
        "probe_dependencies",
        lambda: checkpoints.Sam3Availability(False, "clip missing"),
    )
    args = escalate_cli.build_parser().parse_args(
        ["sam3", "--project", "/nope", "--prompt", "ant"]
    )
    assert escalate_cli.run_sam3(args) == 2
    assert "clip missing" in capsys.readouterr().err
