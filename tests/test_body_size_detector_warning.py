"""reference_body_size is detector-dependent; the UI must say so.

Measured on one recording (Libby_Tagged, identical frames, both detection
caches): an OBB detect model reported a geometric mean of 76.5 px, a
segmentation model 47.9 px -- a 1.6x gap. The box LENGTHS differed by only
21% (115.4 vs 91.0); the WIDTHS differed 2x (51.0 vs 25.7), and the metric is
sqrt(major * minor), so width counts as much as length.

Carrying the wrong value across models is silently destructive: the
object-size window scales as the SQUARE. At body=49.83 the window becomes
390-3900 px^2 while the OBB model's median detection is 5852 px^2, so 1.8% of
real detections survive (93.5% at the correct 76.81).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "hydra_suite" / "trackerkit"
PANEL = SRC / "gui" / "panels" / "detection_panel.py"
MAIN = SRC / "gui" / "main_window.py"


def test_panel_shows_a_persistent_detector_dependence_warning():
    text = PANEL.read_text()
    assert "label_body_size_warning" in text, "no warning widget under the control"
    warning = text[text.index("label_body_size_warning") :][:1200]
    assert "DETECTOR" in warning, "warning must say the value comes from the detector"
    assert "reject" in warning, (
        "warning must name the consequence (detections being rejected), not "
        "just that the number changes"
    )


def test_panel_names_the_body_scaled_parameters():
    """A warning that says 'some parameters' is not actionable."""
    text = PANEL.read_text()
    assert "BODY_SCALED_PARAMS" in text
    for name in ("object size", "distance", "velocity"):
        assert name in text[text.index("BODY_SCALED_PARAMS") :][:600], name


def test_tooltip_does_not_call_it_a_diameter():
    """It is sqrt(major*minor) over the detection box, not a body diameter."""
    text = PANEL.read_text()
    tip = text[text.index("spin_reference_body_size.setToolTip") :][:1400]
    assert "diameter" not in tip.lower(), "tooltip still calls it a diameter"
    assert "GEOMETRIC MEAN" in tip or "geometric mean" in tip


def test_autoset_dialog_warns_on_a_material_change():
    text = MAIN.read_text()
    fn = text[text.index("def _auto_set_body_size_from_detection") :]
    fn = fn[: fn.index("def _auto_set_aspect_ratio_from_detection")]
    assert "previous" in fn, "dialog must compare against the replaced value"
    assert "SQUARE" in fn, "dialog must explain the squared object-size effect"
    assert re.search(
        r"0\.85\s*<=\s*ratio\s*<=\s*1\.18", fn
    ), "a tolerance band must exist so small nudges do not nag"


@pytest.mark.parametrize(
    "previous,new,should_warn",
    [
        (76.81, 49.83, True),  # the real detector-switch case
        (49.83, 76.81, True),  # and its inverse
        (76.81, 78.0, False),  # a nudge
        (76.81, 76.81, False),  # no change
        (0.0, 50.0, False),  # nothing to compare against yet
    ],
)
def test_warning_band_semantics(previous, new, should_warn):
    """The band the UI code uses, stated independently of Qt."""
    ratio = new / previous if previous > 0 else 1.0
    warns = previous > 0 and not (0.85 <= ratio <= 1.18)
    assert warns is should_warn


def test_squared_effect_is_what_broke_the_fixture():
    """Pin the arithmetic the warning is about."""
    body_ok, body_bad = 76.81, 49.83
    # object-size limits scale with body^2
    assert (body_bad / body_ok) ** 2 == pytest.approx(0.42, abs=0.01)
    # the OBB model's median detection area, which the bad window excludes
    assert 3900 < 5852 and 926 < 5852 < 9267
