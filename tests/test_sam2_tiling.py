"""Shared tile sizing and SAM2 owner-tile segmentation."""

import types

from hydra_suite.core.inference.semantic.tiling import TilingSettings


def test_tiling_settings_default_is_full_frame():
    t = TilingSettings()
    assert t.resolved_tile_px() is None
    assert t.plan_for((400, 600)).tiles == [(0, 0, 600, 400)]


def test_tiling_settings_resolves_and_plans():
    t = TilingSettings(reference_body_px=50.0, tile_fraction=0.25)
    assert t.resolved_tile_px() == 200
    assert len(t.plan_for((400, 600)).tiles) > 1


def test_tile_larger_than_frame_falls_back_to_full_frame():
    t = TilingSettings(tile_px=1000)
    assert t.plan_for((400, 600)).tiles == [(0, 0, 600, 400)]


def test_escalation_request_tiling_defaults_preserve_positional_args():
    from hydra_suite.detectkit.jobs.sam2_escalation import EscalationRequest

    req = EscalationRequest(types.SimpleNamespace(), ["a"], "v", True)
    assert req.overwrite is True and req.tiling == TilingSettings()


def test_semantic_request_exposes_the_same_tiling():
    from hydra_suite.detectkit.jobs.semantic_escalation import SemanticEscalationRequest

    req = SemanticEscalationRequest(
        types.SimpleNamespace(),
        ["a"],
        "sam3",
        "ant",
        reference_body_px=50.0,
        tile_fraction=0.25,
        overlap=0.3,
    )
    assert req.tiling == TilingSettings(50.0, 0.25, None, 0.3)
