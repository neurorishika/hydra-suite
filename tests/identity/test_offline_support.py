import math

import pandas as pd

from hydra_suite.core.individual.identity.offline import _combined_support

K = [f"l{i}" for i in range(25)]


def _row(cnn_log=None, tag_log=None, online="unknown", conf=0.0):
    return pd.Series(
        {
            "CNNLogEvidence": cnn_log or {},
            "TagLogEvidence": tag_log or {},
            "OnlineLabel": online,
            "OnlineConfidence": conf,
        }
    )


def test_single_cnn_source_is_not_flattened_by_small_weight():
    cnn = {l: (math.log(0.99999) if l == "l0" else math.log(1e-5 / 24)) for l in K}
    sup = _combined_support(_row(cnn_log=cnn), K, {"FRAGMENT_CNN_WEIGHT": 0.1})
    assert sup["l0"] > 0.999


def test_uninformative_prior_does_not_count_as_a_source():
    cnn = {l: (math.log(0.99999) if l == "l0" else math.log(1e-5 / 24)) for l in K}
    sup = _combined_support(
        _row(cnn_log=cnn, online="not_in_catalog", conf=0.9),
        K,
        {"FRAGMENT_CNN_WEIGHT": 0.4, "ONLINE_PRIOR_WEIGHT": 0.25},
    )
    assert sup["l0"] > 0.999


def test_informative_prior_is_blended_convexly():
    cnn = {l: (math.log(0.6) if l == "l0" else math.log(0.4 / 24)) for l in K}
    sup_no = _combined_support(_row(cnn_log=cnn), K, {"FRAGMENT_CNN_WEIGHT": 1.0})
    sup_pr = _combined_support(
        _row(cnn_log=cnn, online="l1", conf=0.9),
        K,
        {"FRAGMENT_CNN_WEIGHT": 1.0, "ONLINE_PRIOR_WEIGHT": 1.0},
    )
    assert sup_pr["l1"] > sup_no["l1"] and sup_pr["l0"] < sup_no["l0"]
    assert abs(sum(sup_pr.values()) - 1.0) < 1e-9


def test_no_sources_gives_uniform():
    sup = _combined_support(_row(), K, {})
    assert all(abs(v - 1 / 25) < 1e-9 for v in sup.values())
