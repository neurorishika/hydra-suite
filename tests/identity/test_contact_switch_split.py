"""Contact-gated identity-switch splitting (evidence_trust.find_contact_switches).

A trajectory is cut at a contact only when the clean (non-contact) evidence
on the two sides names different animals; contacts with too little clean
evidence on a side keep their motion-based continuity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity import offline
from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence import IdentityEvidence
from hydra_suite.core.individual.identity.evidence_trust import (
    EvidenceTrust,
    TrustCalibration,
    TrustSettings,
    contact_episodes,
    find_contact_switches,
)

N = 100


def _trust(nn_bl_by_det, **settings):
    cal = TrustCalibration(10.0, 10.0, 100.0, 4.0, 1000, 1000, True, True, False)
    s = TrustSettings(enabled=True, split_min_log_odds=4.0, **settings)
    return EvidenceTrust(
        s, cal, trusted={}, det_nn_bl=pd.Series(nn_bl_by_det, dtype=float)
    )


def _setup(contacts, label_of, no_detection=()):
    """Rows, raw evidence and per-detection nn (BL) for one trajectory.

    ``contacts``: frame ranges in contact (nn 0.2 BL; elsewhere 3 BL).
    ``label_of(frame)``: 0/1 -> which of two labels the evidence favours.
    """
    in_contact = np.zeros(N, bool)
    for a, b in contacts:
        in_contact[a:b] = True
    rows, seq, nn = [], [], {}
    for f in range(N):
        det = np.nan if f in no_detection else f
        rows.append({"TrajectoryID": 1, "FrameID": f, "DetectionID": det})
        nn[f] = 0.2 if in_contact[f] else 3.0
        p = np.full(3, 0.05)
        p[1 + label_of(f)] = 0.9
        seq.append((f, np.log(p)))
    return pd.DataFrame(rows), seq, nn


def test_contact_episodes_fill_and_min_length():
    c = np.zeros(30, bool)
    c[5:8] = True
    c[9:12] = True  # one-frame gap -> bridged with fill=2
    c[20:22] = True  # too short
    s, e = contact_episodes(c, min_frames=3, fill=2)
    assert list(zip(s, e)) == [(5, 12)]


def test_switch_across_a_contact_cuts_both_ends_of_it():
    df, seq, nn = _setup([(45, 55)], lambda f: 0 if f < 50 else 1)
    cuts = find_contact_switches(df, seq, _trust(nn))
    assert cuts == [44, 54]


def test_same_label_across_a_contact_is_not_cut():
    df, seq, nn = _setup([(45, 55)], lambda f: 0)
    assert find_contact_switches(df, seq, _trust(nn)) == []


def test_thin_side_keeps_motion_continuity():
    # Two contacts with only 3 clean frames (20-22) between them: with 4
    # clean frames required per side, neither contact can be compared on its
    # inner side, so the trajectory is left whole.
    df, seq, nn = _setup([(10, 20), (23, 33)], lambda f: 0 if f < 23 else 1)
    assert find_contact_switches(df, seq, _trust(nn, split_min_views=4)) == []
    # With fewer frames required the second contact is decidable.
    assert find_contact_switches(df, seq, _trust(nn, split_min_views=3)) == [22, 32]


def test_rows_without_a_detection_count_as_contact():
    df, seq, nn = _setup([], lambda f: 0 if f < 50 else 1, no_detection=range(45, 55))
    assert find_contact_switches(df, seq, _trust(nn)) == [44, 54]


def test_disabled_or_inactive_trust_never_cuts():
    df, seq, nn = _setup([(45, 55)], lambda f: 0 if f < 50 else 1)
    no_geom = _trust(nn)
    no_geom.det_nn_bl = None
    assert find_contact_switches(df, seq, no_geom) == []
    off = _trust(nn)
    off.reason_inactive = "disabled"
    assert find_contact_switches(df, seq, off) == []


def _solve(tmp_path, label_of, **settings):
    catalog = IdentityCatalog.from_labels(["a", "b"])
    path = tmp_path / "ev.npz"
    cache = IdentityEvidenceCache(path, catalog_labels=catalog.labels, mode="w")
    df, seq, nn = _setup([(45, 55)], label_of)
    for f, lp in seq:
        cache.save_frame(f, [IdentityEvidence.from_cnn(f, f, "cnn", lp)])
    cache.flush()
    df["X"] = 3.0 * df["FrameID"]
    df["Y"] = 0.0
    # PELT splitting OFF: the contact split runs with the trust layer alone.
    params = {"ENABLE_PELT_SPLITTING": False, "MIN_FRAGMENT_FRAMES": 5}
    return offline.run_fragment_solver(
        df,
        catalog,
        params,
        cache=IdentityEvidenceCache(path, mode="r"),
        evidence_trust=_trust(nn, **settings),
    )


def test_solver_splits_a_swap_at_the_contact_without_pelt(tmp_path):
    out = _solve(tmp_path, lambda f: 0 if f < 50 else 1)
    before = out[out.FrameID < 45]["IdentityFinalLabel"].unique()
    after = out[out.FrameID >= 55]["IdentityFinalLabel"].unique()
    assert list(before) == ["a"] and list(after) == ["b"]
    assert out[out.FrameID < 45]["TrajectoryID"].nunique() == 1
    assert set(out[out.FrameID < 45]["TrajectoryID"]).isdisjoint(
        out[out.FrameID >= 55]["TrajectoryID"]
    )


def test_solver_contact_split_can_be_switched_off(tmp_path):
    out = _solve(tmp_path, lambda f: 0 if f < 50 else 1, contact_split=False)
    assert out["TrajectoryID"].nunique() == 1


def test_contact_settings_from_params():
    s = TrustSettings.from_params(
        {
            "IDENTITY_TRUST_CONTACT_SPLIT": False,
            "IDENTITY_TRUST_CONTACT_BL": 0.75,
            "IDENTITY_TRUST_CONTACT_WINDOW": 10,
            "IDENTITY_TRUST_CONTACT_MIN_FRAMES": 4,
            "IDENTITY_TRUST_CONTACT_FILL": 0,
        }
    )
    assert (s.contact_split, s.contact_bl, s.contact_window) == (False, 0.75, 10)
    assert (s.contact_min_frames, s.contact_fill) == (4, 0)
    d = TrustSettings.from_params({})
    assert d.contact_split is True and d.contact_bl == 0.5
