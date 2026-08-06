"""The canonical margin must be settable, and identical on both entry points.

`core/inference/config.py` used to read `yolo_headtail_canonical_margin` -- a key
nothing in `src/` ever wrote -- so the inference margin was pinned at 1.3 and
could not be configured, while the crop-dataset exporter used a different knob
entirely. Under global canonicalization the margin is the operator's ONLY dial
against clipped animals, so it has to actually reach `InferenceConfig`.

After merging main's shared engine-param builder there is one builder
(`build_engine_params`) and one advanced-config table
(`cli_config._default_advanced_config`, which main's own
`_default_advanced_config_fallback` declares the single source of truth), so
these assertions target those rather than a branch-local module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hydra_suite.core.inference.config import build_inference_config_from_params
from hydra_suite.trackerkit.cli_config import _default_advanced_config
from hydra_suite.trackerkit.engine_params import RuntimeContext, build_engine_params

SRC = Path(__file__).resolve().parents[1] / "src"


def _runtime() -> RuntimeContext:
    return RuntimeContext(fps=30.0, total_frames=100, frame_width=640, frame_height=480)


def test_both_canonical_keys_are_in_the_single_defaults_table():
    table = _default_advanced_config()
    assert "canonical_margin" in table
    assert "reference_aspect_ratio" in table


def test_the_dead_margin_key_is_gone():
    """`yolo_headtail_canonical_margin` was read but never written."""
    hits = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if "yolo_headtail_canonical_margin" in p.read_text(encoding="utf-8")
    ]
    assert hits == [], f"dead margin key still referenced in {hits}"


@pytest.mark.parametrize("margin", [1.3, 1.75, 2.0])
def test_a_configured_margin_reaches_the_engine_params(margin):
    params = build_engine_params(
        {"file_path": "/tmp/x.mp4", "fps": 30.0, "canonical_margin": margin},
        runtime=_runtime(),
    )
    assert params["ADVANCED_CONFIG"]["canonical_margin"] == margin


def test_margin_and_aspect_reach_the_inference_geometry():
    """The end of the chain: an operator's dials become the canvas."""
    params = build_engine_params(
        {
            "file_path": "/tmp/x.mp4",
            "fps": 30.0,
            "canonical_margin": 1.75,
            "reference_aspect_ratio": 2.44,
            "reference_body_size": 20.0,
        },
        runtime=_runtime(),
    )
    geometry = build_inference_config_from_params(params).canonical

    assert geometry.margin == pytest.approx(1.75)
    assert geometry.aspect_ratio == pytest.approx(2.44)
    # major = body * sqrt(aspect) = 20 * 1.562 = 31.2; canvas_w = 31.2 * 1.75
    assert geometry.canvas_w == 56
    assert geometry.canvas_h == 24


def test_aspect_ratio_default_agrees_everywhere():
    """A fourth, disagreeing 4.0 default survived main's param unification."""
    table_default = _default_advanced_config()["reference_aspect_ratio"]
    built = build_engine_params(
        {"file_path": "/tmp/x.mp4", "fps": 30.0}, runtime=_runtime()
    )["ADVANCED_CONFIG"]["reference_aspect_ratio"]

    assert table_default == 2.0
    assert built == 2.0
