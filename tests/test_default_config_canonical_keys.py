"""The bundled default.json must carry the canonicalization knobs used by
CanonicalFitTransform-based crop consumers (F7d): canonical_margin and
reference_body_size, alongside the existing reference_aspect_ratio.

Loaded via the same accessor the rest of the codebase uses to read bundled
config presets: hydra_suite.paths.get_default_config.
"""

from __future__ import annotations

from hydra_suite.paths import get_default_config


def test_default_config_has_canonical_margin_and_reference_body_size():
    cfg = get_default_config("default.json")
    assert cfg is not None
    assert "reference_aspect_ratio" in cfg  # sanity: sibling key exists
    assert "canonical_margin" in cfg
    assert "reference_body_size" in cfg
    assert cfg["canonical_margin"] == 1.3
    assert cfg["reference_body_size"] == 20.0


def test_ooceraea_biroi_config_has_canonical_margin_and_reference_body_size():
    cfg = get_default_config("ooceraea_biroi.json")
    assert cfg is not None
    assert "canonical_margin" in cfg
    assert "reference_body_size" in cfg
    assert cfg["canonical_margin"] == 1.3
    assert cfg["reference_body_size"] == 20.0
