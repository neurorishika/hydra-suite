import logging

import numpy as np
import pandas as pd

from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence import IdentityEvidence
from hydra_suite.core.individual.identity.offline import (
    assess_evidence_quality,
    run_fragment_solver,
)


def _raw(n_traj, n_frames, maxp, same_label=False, seed=0):
    cat = 26
    out = {}
    for t in range(n_traj):
        seq = []
        for f in range(n_frames):
            p = np.full(cat, (1 - maxp) / (cat - 1))
            p[0] = 1e-12
            k = 5 if same_label else 1 + (t % (cat - 1))
            p[k] = maxp
            p /= p.sum()
            seq.append((f, np.log(p)))
        out[t] = seq
    return out


def test_diffuse_same_label_evidence_trips_breaker():
    cat = IdentityCatalog.from_labels([f"l{i}" for i in range(25)])
    raw = _raw(20, 100, 0.14, same_label=True)
    q = assess_evidence_quality(raw, cat)
    assert q.conf_frac == 0.0 and q.diversity < 0.3 and not q.ok


def test_confident_diverse_evidence_passes():
    cat = IdentityCatalog.from_labels([f"l{i}" for i in range(25)])
    q = assess_evidence_quality(_raw(20, 100, 0.9), cat)
    assert q.conf_frac > 0.9 and q.diversity > 0.5 and q.ok


_CATALOG_LABELS = ("unknown", "ant_a", "ant_b", "ant_c")


def _diffuse_evidence_cache(tmp_path, n_frames=40):
    """Two trajectories whose CNN evidence is diffuse and collapsed onto the
    SAME wrong label every frame -- the "broken run" shape from the brief
    (mis-preprocessed classifier): should trip the breaker.
    """
    path = tmp_path / "evidence_cache.npz"
    cache = IdentityEvidenceCache(path, catalog_labels=_CATALOG_LABELS, mode="w")
    probs = np.array([1e-12, 0.36, 0.32, 0.32])
    probs /= probs.sum()
    log_probs = np.log(probs)
    for f in range(n_frames):
        evidences = [
            IdentityEvidence.from_cnn(f, f, "cnn_identity", log_probs),
            IdentityEvidence.from_cnn(f, 1000 + f, "cnn_identity", log_probs),
        ]
        cache.save_frame(f, evidences)
    cache.flush()
    return IdentityEvidenceCache(path, mode="r")


def _diffuse_df(n_frames=40):
    rows = []
    for f in range(n_frames):
        rows.append(
            {"TrajectoryID": 1, "FrameID": f, "DetectionID": f, "X": 0.0, "Y": 0.0}
        )
    for f in range(n_frames):
        rows.append(
            {
                "TrajectoryID": 2,
                "FrameID": f,
                "DetectionID": 1000 + f,
                "X": 500.0,
                "Y": 500.0,
            }
        )
    return pd.DataFrame(rows)


def test_run_fragment_solver_refuses_to_act_on_uninformative_evidence(tmp_path, caplog):
    """End-to-end wiring: when the raw cache evidence fails the quality bar,
    run_fragment_solver must (a) skip PELT splitting even when requested,
    (b) leave IdentityFinalLabel at the "no evidence" default rather than
    committing to a (spurious) label, (c) still write the raw smoothed
    per-row decode for human inspection, and (d) log a loud ERROR naming
    both numbers.
    """
    n_frames = 40
    catalog = IdentityCatalog.from_labels(list(_CATALOG_LABELS[1:]))
    df = _diffuse_df(n_frames)
    cache = _diffuse_evidence_cache(tmp_path, n_frames)

    with caplog.at_level(logging.ERROR):
        out = run_fragment_solver(
            df,
            catalog,
            {"ENABLE_PELT_SPLITTING": True, "CHANGEPOINT_PENALTY": 0.5},
            cache=cache,
        )

    # (a) No split: same two TrajectoryIDs, PELT refused to run.
    assert sorted(out["TrajectoryID"].unique()) == [1, 2]

    # (b) No real identity committed -- solve_global_assignment got no
    # evidence (evidence_by_traj=None), which is EXACTLY the pre-existing
    # cache-absent "no evidence, no belief" degrade documented and tested
    # by test_honesty_fix.py: a uniform low-confidence guess (1 / n_known
    # labels == 1/3 here), not a real per-trajectory identification. The
    # breaker's job is only to make sure the solver sees this same
    # no-sidecar input -- not to itself special-case the output label.
    confidences = out.groupby("TrajectoryID")["IdentityFinalConfidence"].first()
    assert (confidences < 0.4).all(), (
        "expected the uniform no-evidence guess confidence (~1/3), got "
        f"{confidences.to_dict()} -- evidence may have leaked through "
        "despite the breaker tripping"
    )

    # (c) The raw smoothed decode column still exists (annotation still ran).
    assert "IdentityFinalSmoothedLabel" in out.columns

    # (d) A loud ERROR naming both numbers.
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("uninformative" in r.getMessage() for r in errors)
    assert any(
        "confident=" in r.getMessage() and "diversity=" in r.getMessage()
        for r in errors
    )


def test_run_fragment_solver_confident_evidence_is_not_blocked(tmp_path, caplog):
    """Contrast case: confident, per-trajectory-distinct cache evidence
    passes the breaker and the solver commits real, non-uniform identities
    -- proving the breaker only trips on genuinely uninformative evidence,
    not on any use of the (small) synthetic catalog/cache machinery.
    """
    n_frames = 40
    catalog = IdentityCatalog.from_labels(list(_CATALOG_LABELS[1:]))
    df = _diffuse_df(n_frames)

    path = tmp_path / "evidence_cache.npz"
    cache = IdentityEvidenceCache(path, catalog_labels=_CATALOG_LABELS, mode="w")
    probs_b = np.array([1e-12, 0.02, 0.96, 0.02])
    probs_b /= probs_b.sum()
    probs_c = np.array([1e-12, 0.02, 0.02, 0.96])
    probs_c /= probs_c.sum()
    for f in range(n_frames):
        cache.save_frame(
            f,
            [
                IdentityEvidence.from_cnn(f, f, "cnn_identity", np.log(probs_b)),
                IdentityEvidence.from_cnn(f, 1000 + f, "cnn_identity", np.log(probs_c)),
            ],
        )
    cache.flush()
    cache = IdentityEvidenceCache(path, mode="r")

    with caplog.at_level(logging.ERROR):
        out = run_fragment_solver(df, catalog, {}, cache=cache)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert not any("uninformative" in r.getMessage() for r in errors)

    confidences = out.groupby("TrajectoryID")["IdentityFinalConfidence"].first()
    assert (confidences > 0.4).all()
