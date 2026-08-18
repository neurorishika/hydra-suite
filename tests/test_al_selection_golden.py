"""Characterization golden for AL frame selection.

Frame selection deliberately CHANGES in this work (absolute floors replace
min-max normalization, the fragmentation channel is restored, edge scoring is
fixed). A byte-identity oracle against the old behaviour would therefore be
wrong, and an oracle derived from the new code would be tautological. This
golden pins the new behaviour against a fixed synthetic signal set, so future
refactors that claim to preserve selection must actually preserve it.

To regenerate after an INTENTIONAL scoring change:
    python -m pytest tests/test_al_selection_golden.py --update-golden
and review the diff as part of the change.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hydra_suite.data.al.acquisition import PRESETS, select
from hydra_suite.data.al.signals import ALSignals

GOLDEN = Path(__file__).parent / "goldens" / "al_selection_characterization.json"


def _fixed_signals():
    """120 deterministic frames spanning every channel's dynamic range."""
    rng = np.random.default_rng(20260817)
    signals = []
    for fid in range(120):
        signals.append(
            ALSignals(
                frame_id=fid,
                n_detections=int(rng.integers(0, 8)),
                mean_confidence=float(rng.uniform(0.1, 1.0)),
                uncertainty_score=float(rng.uniform(0.0, 1.0)),
                count_deviation=float(rng.uniform(0.0, 1.0)),
                crowd_score=float(rng.uniform(0.0, 1.0)),
                fragmentation_score=float(rng.uniform(0.0, 1.0)),
                edge_score=float(rng.uniform(0.0, 1.0)),
                extras={
                    "assignment": float(rng.uniform(0.0, 1.0)),
                    "track_loss": float(rng.uniform(0.0, 1.0)),
                    "position_uncertainty": float(rng.uniform(0.0, 1.0)),
                },
            )
        )
    return signals


def test_selection_matches_golden(request):
    picked = select(
        _fixed_signals(),
        weights=PRESETS["tracker_default"],
        k=20,
        diversity_window=5,
        probabilistic=False,  # deterministic: no rng in the golden
        min_score=0.30,
    )
    if request.config.getoption("--update-golden", default=False):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps({"picked": picked}, indent=2))
        pytest.skip("golden updated")

    expected = json.loads(GOLDEN.read_text())["picked"]
    assert picked == expected
