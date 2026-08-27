"""Task 2: ``IdentityFinalSmoothedLabel``/``IdentityFinalSmoothedConfidence``
are an ungated record of the classifier's per-frame smoothed posterior --
not the realtime overlay's display-threshold-gated decision.

``_annotate_smoothed_labels`` must always write the argmax known label and
its posterior for rows with cache evidence; only rows with no matched cache
evidence (no DetectionID join) get ``unknown``/0.0. The realtime decoder's
own display-threshold gate (``substrate.solve_unique_assignment``) is a
separate code path and is untouched by this change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import columns as C
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.offline import _annotate_smoothed_labels
from hydra_suite.core.individual.identity.smoothing import smoothed_label_and_conf


def _lp(p_a):
    p = np.array([1e-6, p_a, 1.0 - p_a - 1e-6])
    p = p / p.sum()
    return np.log(p)


def test_smoothed_label_and_conf_without_threshold_reports_argmax():
    cat = IdentityCatalog.from_labels(["ant_a", "ant_b"])
    out = smoothed_label_and_conf([_lp(0.55), _lp(0.05)], cat, display_threshold=None)
    assert out[0][0] == "ant_a" and abs(out[0][1] - 0.55) < 1e-3
    assert out[1][0] == "ant_b" and abs(out[1][1] - 0.95) < 1e-3


def test_annotate_writes_low_confidence_rows_and_unknown_for_no_evidence():
    cat = IdentityCatalog.from_labels(["ant_a", "ant_b"])
    df = pd.DataFrame(
        {"TrajectoryID": [7, 7, 7], "FrameID": [1, 2, 3], "DetectionID": [1, 2, np.nan]}
    )
    smoothed = {7: [(1, _lp(0.55)), (2, _lp(0.99))]}  # frame 3 has no evidence
    out = _annotate_smoothed_labels(
        df, smoothed, cat, {"IDENTITY_DISPLAY_THRESHOLD": 0.95}
    )
    assert out[C.FINAL_SMOOTHED_LABEL].tolist() == ["ant_a", "ant_a", "unknown"]
    assert abs(out[C.FINAL_SMOOTHED_CONFIDENCE].iloc[0] - 0.55) < 1e-3
    assert out[C.FINAL_SMOOTHED_CONFIDENCE].iloc[2] == 0.0
