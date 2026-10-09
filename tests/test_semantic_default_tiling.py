"""F3: headless SAM3 escalation tiles exactly as the dialog would open."""

from types import SimpleNamespace

from hydra_suite.detectkit.jobs import semantic_escalation as se


def _project(**kw):
    base = dict(
        semantic_escalation_settings={},
        semantic_calibration={},
        project_dir="/tmp/sahi_s3_project",
        slice_settings=SimpleNamespace(reference_body_px=0.0),
        sources=[],
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_saved_settings_for_same_variant_win(monkeypatch):
    monkeypatch.setattr(se, "sidecar_for", lambda key: {"object_tile_fraction": 0.2})
    p = _project(
        semantic_escalation_settings={
            "variant": "sam3",
            "tile_fraction": 0.07,
            "reference_body_px": 40.0,
            "overlap": 0.3,
        }
    )
    got = se.default_semantic_tiling(p, "sam3")
    assert (
        got["tile_fraction"],
        got["reference_body_px"],
        got["overlap"],
        got["origin"],
    ) == (0.07, 40.0, 0.3, "saved")


def test_saved_full_frame_is_respected(monkeypatch):
    monkeypatch.setattr(se, "sidecar_for", lambda key: None)
    p = _project(
        semantic_escalation_settings={
            "variant": "sam3",
            "tile_fraction": 0.0,
            "reference_body_px": 40.0,
        }
    )
    assert se.default_semantic_tiling(p, "sam3")["tile_fraction"] is None


def test_stamped_fraction_and_body_when_nothing_saved(monkeypatch):
    """Review Focus 3: a finetuned model tiles at its trained scale headlessly."""
    monkeypatch.setattr(
        se,
        "sidecar_for",
        lambda key: {"object_tile_fraction": 0.055, "reference_body_px": 53.4},
    )
    got = se.default_semantic_tiling(_project(), "ft-model")
    assert (got["tile_fraction"], got["reference_body_px"], got["origin"]) == (
        0.055,
        53.4,
        "stamped",
    )


def test_stock_variant_with_known_body_tiles_at_the_seed(monkeypatch):
    """Plan review B1: the dialog's opening state, not full frame."""
    monkeypatch.setattr(se, "sidecar_for", lambda key: None)
    p = _project(slice_settings=SimpleNamespace(reference_body_px=82.2))
    got = se.default_semantic_tiling(p, "sam3")
    assert (got["tile_fraction"], got["reference_body_px"], got["origin"]) == (
        0.05,
        82.2,
        "default",
    )


def test_stock_variant_with_no_body_is_full_frame(monkeypatch):
    monkeypatch.setattr(se, "sidecar_for", lambda key: None)
    got = se.default_semantic_tiling(_project(), "sam3")
    assert got["tile_fraction"] is None and got["origin"] == "full_frame"


def test_calibration_record_recommended_point(monkeypatch):
    monkeypatch.setattr(se, "sidecar_for", lambda key: None)
    record = {
        "variant": "sam3",
        "recommended_index": 1,
        "parameters": {"reference_body_px": 61.0},
        "points": [{"tile_fraction": 0.03}, {"tile_fraction": 0.1}],
    }
    got = se.default_semantic_tiling(_project(semantic_calibration=record), "sam3")
    assert (got["tile_fraction"], got["reference_body_px"], got["origin"]) == (
        0.1,
        61.0,
        "calibration",
    )
    other = dict(record, variant="ft-model")
    assert (
        se.default_semantic_tiling(_project(semantic_calibration=other), "sam3")[
            "origin"
        ]
        != "calibration"
    )


def test_malformed_calibration_record_never_raises(monkeypatch):
    monkeypatch.setattr(se, "sidecar_for", lambda key: None)
    record = {"variant": "sam3", "recommended_index": 7, "points": "nope"}
    got = se.default_semantic_tiling(_project(semantic_calibration=record), "sam3")
    assert got["origin"] == "full_frame"


def test_cli_sam3_accepts_tiling_flags():
    from hydra_suite.detectkit.escalate_cli import build_parser

    ns = build_parser().parse_args(
        [
            "sam3",
            "--project",
            "p",
            "--prompt",
            "ant",
            "--tile-fraction",
            "0.06",
            "--reference-body-px",
            "50",
        ]
    )
    assert ns.tile_fraction == 0.06 and ns.reference_body_px == 50.0


def _run_cli(monkeypatch, argv, sidecar):
    from hydra_suite.detectkit import escalate_cli as cli

    captured = {}
    monkeypatch.setattr(cli, "_open", lambda path: _project())
    monkeypatch.setattr(cli, "_selected", lambda project, names: [])
    monkeypatch.setattr(cli, "_resolve_class_name", lambda *a: "ant")
    monkeypatch.setattr(se, "sidecar_for", lambda key: sidecar)
    import hydra_suite.core.inference.semantic.checkpoints as ck
    import hydra_suite.detectkit.sidecars.operations as ops

    monkeypatch.setattr(
        ck, "probe_dependencies", lambda: SimpleNamespace(usable=True, reason="")
    )
    monkeypatch.setattr(
        ops,
        "run_semantic_escalation_sidecar",
        lambda payload, progress: captured.update(payload) or {"semantic_result": {}},
    )
    ns = cli.build_parser().parse_args(argv)
    assert cli.run_sam3(ns) == 0
    return captured["params"]


def test_cli_sam3_payload_carries_resolved_tiling(monkeypatch):
    params = _run_cli(
        monkeypatch,
        ["sam3", "--project", "p", "--prompt", "ant", "--variant", "ft-model"],
        {"object_tile_fraction": 0.055, "reference_body_px": 53.4},
    )
    assert params["tile_fraction"] == 0.055 and params["reference_body_px"] == 53.4


def test_cli_sam3_flags_win_and_zero_is_full_frame(monkeypatch):
    base = ["sam3", "--project", "p", "--prompt", "ant", "--variant", "ft-model"]
    stamp = {"object_tile_fraction": 0.055, "reference_body_px": 53.4}
    params = _run_cli(
        monkeypatch,
        base + ["--tile-fraction", "0.08", "--reference-body-px", "70"],
        stamp,
    )
    assert params["tile_fraction"] == 0.08 and params["reference_body_px"] == 70.0
    params = _run_cli(monkeypatch, base + ["--tile-fraction", "0"], stamp)
    assert params["tile_fraction"] is None


def test_dialog_opens_a_finetuned_model_at_its_stamped_scale(monkeypatch, tmp_path):
    """The dialog and the CLI share one resolver: same numbers for the same model."""
    import pytest

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as mod
    from hydra_suite.detectkit.gui.models import DetectKitProject

    class _Available:
        usable = True
        checkpoint_missing = False
        reason = ""

    stamp = {"object_tile_fraction": 0.055, "reference_body_px": 53.4}
    monkeypatch.setattr(mod, "probe_checkpoint", lambda *_a, **_k: _Available())
    monkeypatch.setattr(mod, "available_models", lambda: ["ft-model", "sam3"])
    monkeypatch.setattr(mod, "sidecar_for", lambda key: None)
    monkeypatch.setattr(se, "sidecar_for", lambda key: stamp)
    project = DetectKitProject(project_dir=tmp_path)
    dialog = mod.SemanticEscalationDialog([], 0.0, project=project)
    assert dialog.selected_variant() == "ft-model"
    assert dialog.parameters()["tile_fraction"] == pytest.approx(0.055)
    assert dialog.parameters()["reference_body_px"] == pytest.approx(53.4)
    headless = se.default_semantic_tiling(project, "ft-model", body_chain_px=0.0)
    assert headless["tile_fraction"] == pytest.approx(0.055)
    assert headless["reference_body_px"] == pytest.approx(53.4)


# -- M1: a saved dict that does not apply to the opening variant ------------


def _dialog_and_cli(monkeypatch, tmp_path, saved, body, models=("sam3",)):
    import argparse

    import pytest

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from hydra_suite.detectkit import escalate_cli as cli
    from hydra_suite.detectkit.gui import escalation_actions as acts
    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as mod
    from hydra_suite.detectkit.gui.models import DetectKitProject

    class _Available:
        usable = True
        checkpoint_missing = False
        reason = ""

    monkeypatch.setattr(mod, "probe_checkpoint", lambda *_a, **_k: _Available())
    monkeypatch.setattr(mod, "available_models", lambda: list(models))
    monkeypatch.setattr(mod, "sidecar_for", lambda key: None)
    monkeypatch.setattr(se, "sidecar_for", lambda key: None)
    project = DetectKitProject(project_dir=tmp_path)
    project.slice_settings.reference_body_px = body
    project.semantic_escalation_settings = dict(saved)
    chain, origin = acts.resolve_reference_body_px(project)
    dialog = mod.SemanticEscalationDialog(
        [], chain, body_px_origin=origin, project=project
    )
    params = dialog.parameters()
    ns = argparse.Namespace(
        variant=dialog.selected_variant(), tile_fraction=None, reference_body_px=None
    )
    tiling = cli._sam3_tiling(project, ns)
    return params, tiling


def _effective(fraction, body):
    from hydra_suite.core.inference.semantic.tiling import resolve_tile_px

    return resolve_tile_px(float(body or 0.0), (fraction or None))


def test_stale_saved_variant_dialog_and_cli_both_open_at_the_seed(
    monkeypatch, tmp_path
):
    saved = {"variant": "gone-model", "tile_fraction": 0.07, "reference_body_px": 40}
    params, tiling = _dialog_and_cli(monkeypatch, tmp_path, saved, 82.2)
    assert (params["tile_fraction"], params["reference_body_px"]) == (0.05, 82.2)
    assert (tiling["tile_fraction"], tiling["reference_body_px"]) == (0.05, 82.2)


def test_stale_saved_variant_with_no_body_is_full_frame_in_both(monkeypatch, tmp_path):
    saved = {"variant": "gone-model", "tile_fraction": 0.07, "reference_body_px": 40}
    params, tiling = _dialog_and_cli(monkeypatch, tmp_path, saved, 0.0)
    assert _effective(params["tile_fraction"], params["reference_body_px"]) is None
    assert tiling["tile_fraction"] is None


def test_saved_body_without_a_fraction_agrees_between_dialog_and_cli(
    monkeypatch, tmp_path
):
    params, tiling = _dialog_and_cli(
        monkeypatch, tmp_path, {"reference_body_px": 33}, 82.2
    )
    assert _effective(params["tile_fraction"], params["reference_body_px"]) == (
        _effective(tiling["tile_fraction"], tiling["reference_body_px"])
    )
    assert (params["tile_fraction"], params["reference_body_px"]) == (
        tiling["tile_fraction"],
        tiling["reference_body_px"],
    )


# -- M2: --reference-body-px alone ------------------------------------------


def test_cli_body_flag_alone_tiles_a_stock_variant_at_the_seed(monkeypatch):
    params = _run_cli(
        monkeypatch,
        ["sam3", "--project", "p", "--prompt", "ant", "--reference-body-px", "50"],
        None,
    )
    assert params["tile_fraction"] == 0.05 and params["reference_body_px"] == 50.0


def test_cli_body_flag_alone_keeps_a_stamped_fraction(monkeypatch):
    params = _run_cli(
        monkeypatch,
        [
            "sam3",
            "--project",
            "p",
            "--prompt",
            "ant",
            "--variant",
            "ft-model",
            "--reference-body-px",
            "50",
        ],
        {"object_tile_fraction": 0.055, "reference_body_px": 53.4},
    )
    assert params["tile_fraction"] == 0.055 and params["reference_body_px"] == 50.0
