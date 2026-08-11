"""Identity Phase 5 Task 1: offline evidence sourcing from the cache.

``load_trajectory_evidence`` is the seam that makes the post-hoc/offline
identity decoder self-sufficient: instead of reconstructing evidence from
the wide-CSV ``CNN_*_Prob``/``DetectedTag*`` columns (starved when
``ENABLE_IDENTITY_IN_TRACKING`` is off), it reads the always-written
Phase-3 ``IdentityEvidenceCache`` sidecar directly, joining trajectory rows
to cached evidence on ``(FrameID, DetectionID) -> IdentityEvidence.detection_id``.

These tests build a synthetic tracking-output ``DataFrame`` (the same shape
``postprocess_df.py`` hands to the offline solver: ``TrajectoryID``,
``FrameID``, ``DetectionID``) plus a real ``IdentityEvidenceCache`` (written
via ``mode="w"`` + ``save_frame``, exactly like the tracking worker does),
then assert the join, remap-to-global-catalog, and multi-source fusion
behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence import IdentityEvidence
from hydra_suite.core.individual.identity.smoothing import load_trajectory_evidence


def _write_cache(
    tmp_path, catalog_labels, evidences_by_frame, catalog_labels_by_source=None
):
    path = tmp_path / "evidence_cache.npz"
    cache = IdentityEvidenceCache(
        path,
        catalog_labels=catalog_labels,
        mode="w",
        catalog_labels_by_source=catalog_labels_by_source,
    )
    for frame_idx, evidences in evidences_by_frame.items():
        cache.save_frame(frame_idx, evidences)
    cache.flush()
    return IdentityEvidenceCache(path, mode="r")


def test_basic_join_frame_ordered(tmp_path):
    """A trajectory's evidence sequence is joined on (FrameID, DetectionID),
    frame-ordered, and returned as the passed catalog's log-probs."""
    catalog = IdentityCatalog.from_labels(["ant_a", "ant_b"])

    def lp(favor_idx, n=3, strength=5.0):
        v = np.zeros(n, dtype=np.float64)
        v[favor_idx] = strength
        return v - np.logaddexp.reduce(v)

    ev_f0 = IdentityEvidence.from_cnn(0, 7, "cnn", lp(1))  # favors ant_a (idx 1)
    ev_f1 = IdentityEvidence.from_cnn(1, 9, "cnn", lp(2))  # favors ant_b (idx 2)
    cache = _write_cache(
        tmp_path,
        catalog_labels=catalog.labels,
        evidences_by_frame={0: [ev_f0], 1: [ev_f1]},
    )

    df = pd.DataFrame(
        {
            "TrajectoryID": [1, 1],
            "FrameID": [0, 1],
            "DetectionID": [7, 9],
        }
    )

    result = load_trajectory_evidence(df, cache, catalog)

    assert set(result.keys()) == {1}
    seq = result[1]
    assert [f for f, _ in seq] == [0, 1]
    assert seq[0][1].shape == (catalog.size,)
    assert np.argmax(seq[0][1]) == catalog.index_of("ant_a")
    assert np.argmax(seq[1][1]) == catalog.index_of("ant_b")


def test_multiple_trajectories_ordered_independently(tmp_path):
    catalog = IdentityCatalog.from_labels(["ant_a", "ant_b"])

    def lp(favor_idx, n=3, strength=5.0):
        v = np.zeros(n, dtype=np.float64)
        v[favor_idx] = strength
        return v - np.logaddexp.reduce(v)

    evidences_by_frame = {
        0: [
            IdentityEvidence.from_cnn(0, 1, "cnn", lp(1)),
            IdentityEvidence.from_cnn(0, 2, "cnn", lp(2)),
        ],
        2: [
            IdentityEvidence.from_cnn(2, 1, "cnn", lp(2)),
        ],
        1: [
            IdentityEvidence.from_cnn(1, 2, "cnn", lp(1)),
        ],
    }
    cache = _write_cache(tmp_path, catalog.labels, evidences_by_frame)

    df = pd.DataFrame(
        {
            "TrajectoryID": [10, 20, 20],
            "FrameID": [0, 0, 2],
            "DetectionID": [1, 2, 1],
        }
    )
    # NOTE: traj 20 rows are given out of frame-order in the input df on
    # purpose (FrameID 0 then 2) to prove the function sorts by FrameID,
    # and traj 10 has only one row (frame 0, det 1).

    result = load_trajectory_evidence(df, cache, catalog)

    assert set(result.keys()) == {10, 20}
    assert [f for f, _ in result[10]] == [0]
    assert [f for f, _ in result[20]] == [0, 2]


def test_nan_or_absent_detection_id_rows_omitted(tmp_path):
    catalog = IdentityCatalog.from_labels(["ant_a", "ant_b"])
    v = np.array([0.0, 5.0, 0.0])
    v = v - np.logaddexp.reduce(v)
    cache = _write_cache(
        tmp_path,
        catalog.labels,
        {0: [IdentityEvidence.from_cnn(0, 3, "cnn", v)]},
    )

    df = pd.DataFrame(
        {
            "TrajectoryID": [1, 1, 1],
            "FrameID": [0, 1, 2],
            "DetectionID": [
                3,
                np.nan,
                3,
            ],  # frame 1: NaN det id; frame 2: no cached evidence
        }
    )

    result = load_trajectory_evidence(df, cache, catalog)

    # Only frame 0 has both a valid DetectionID AND cached evidence for it.
    assert [f for f, _ in result[1]] == [0]


def test_frame_with_no_cache_entry_omitted(tmp_path):
    catalog = IdentityCatalog.from_labels(["ant_a", "ant_b"])
    v = np.array([0.0, 5.0, 0.0])
    v = v - np.logaddexp.reduce(v)
    cache = _write_cache(
        tmp_path, catalog.labels, {0: [IdentityEvidence.from_cnn(0, 3, "cnn", v)]}
    )

    df = pd.DataFrame(
        {
            "TrajectoryID": [1, 1],
            "FrameID": [0, 5],  # frame 5 was never written to the cache
            "DetectionID": [3, 3],
        }
    )

    result = load_trajectory_evidence(df, cache, catalog)
    assert [f for f, _ in result[1]] == [0]


def test_remap_from_source_local_basis_to_global_catalog(tmp_path):
    """A source whose evidence was written against its own (smaller/reordered)
    phase-local basis is remapped into the passed global catalog, mirroring
    the tracking worker's `_remap_source_log_probs_to_catalog`."""
    global_catalog = IdentityCatalog.from_labels(["ant_a", "ant_b", "ant_c"])
    # This CNN phase only distinguishes ant_c vs ant_a (its own 2-class basis,
    # unknown-first), in a DIFFERENT order than the global catalog.
    source_labels = ("unknown", "ant_c", "ant_a")

    # Strong preference for source-local index 1 == "ant_c".
    src_v = np.array([0.0, 8.0, 0.0])
    src_v = src_v - np.logaddexp.reduce(src_v)
    ev = IdentityEvidence.from_cnn(0, 5, "phase_x", src_v)

    cache = _write_cache(
        tmp_path,
        catalog_labels=global_catalog.labels,
        evidences_by_frame={0: [ev]},
        catalog_labels_by_source={"phase_x": source_labels},
    )

    df = pd.DataFrame({"TrajectoryID": [1], "FrameID": [0], "DetectionID": [5]})

    result = load_trajectory_evidence(df, cache, global_catalog)

    assert result[1][0][1].shape == (global_catalog.size,)
    assert np.argmax(result[1][0][1]) == global_catalog.index_of("ant_c")


def test_two_sources_same_detection_are_fused(tmp_path):
    """A detection observed by two evidence sources in the same frame (e.g.
    a CNN phase + AprilTag) is fused into a single catalog log-vector."""
    catalog = IdentityCatalog.from_labels(["ant_a", "ant_b"])

    # CNN: mild preference for ant_a.
    cnn_v = np.array([0.0, 1.0, 0.0])
    cnn_v = cnn_v - np.logaddexp.reduce(cnn_v)
    cnn_ev = IdentityEvidence.from_cnn(0, 4, "cnn", cnn_v)

    # AprilTag: strong preference for ant_a too -- fusion should sharpen it
    # relative to either source alone.
    tag_v = np.array([0.0, 4.0, 0.0])
    tag_v = tag_v - np.logaddexp.reduce(tag_v)
    tag_ev = IdentityEvidence.from_apriltag(0, 4, tag_v)

    cache = _write_cache(tmp_path, catalog.labels, {0: [cnn_ev, tag_ev]})

    df = pd.DataFrame({"TrajectoryID": [1], "FrameID": [0], "DetectionID": [4]})

    result = load_trajectory_evidence(df, cache, catalog)

    fused = result[1][0][1]
    idx_a = catalog.index_of("ant_a")
    # Fused evidence must be at least as confident in ant_a as either lone source.
    assert fused[idx_a] > cnn_v[idx_a]
    assert fused[idx_a] > tag_v[idx_a]
    # And it is a valid normalized log-posterior.
    assert np.isclose(np.logaddexp.reduce(fused), 0.0, atol=1e-8)


def test_empty_df_returns_empty_dict(tmp_path):
    catalog = IdentityCatalog.from_labels(["ant_a", "ant_b"])
    cache = _write_cache(tmp_path, catalog.labels, {})
    df = pd.DataFrame({"TrajectoryID": [], "FrameID": [], "DetectionID": []})
    assert load_trajectory_evidence(df, cache, catalog) == {}


def test_missing_detectionid_column_raises():
    catalog = IdentityCatalog.from_labels(["ant_a"])
    df = pd.DataFrame({"TrajectoryID": [1], "FrameID": [0]})
    with pytest.raises(KeyError):
        load_trajectory_evidence(df, object(), catalog)
