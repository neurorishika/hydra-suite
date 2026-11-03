"""Honest identity evidence, the absolute support gate, and the trust layer.

Covers (2026-10 identity-evidence fixes):

1. ``phase_remap``: mass on non-catalog / non-identifying classes goes to
   ``unknown`` instead of being dropped and renormalized away.
2. ``FRAGMENT_MIN_ABSOLUTE_POSTERIOR`` + absolute seeding in the fragment solver.
3. ``FRAGMENT_LENGTH_WEIGHT`` clamp is loud.
4. ``evidence_trust``: classifier-independent trusted frames (each check
   switches itself off when the data can't support it), distinct-view
   collapse, tiered seeding.
5. Trusted-view label-switch splitting (the PELT replacement under trust).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from hydra_suite.core.individual.identity import offline
from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence import IdentityEvidence
from hydra_suite.core.individual.identity.evidence_trust import (
    DetectionGeometry,
    EvidenceTrust,
    FragmentViews,
    TrustSettings,
    build_evidence_trust,
    calibrate,
    collapse_views,
    find_label_switches,
    fragment_views,
    nearest_neighbour_distance,
)
from hydra_suite.core.individual.identity.phase_remap import (
    build_phase_label_map,
    noncatalog_to_unknown_enabled,
    remap_phase_log_probs,
)
from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec
from hydra_suite.core.individual.identity.smoothing import load_trajectory_evidence

# ---------------------------------------------------------------------------
# 1. honest remap
# ---------------------------------------------------------------------------

TAGS = {
    "label": "tags",
    "unique_identifier": True,
    "class_names_per_factor": [["red", "blue", "notag"]],
    "non_identifying_classes": ["notag"],
}


def _tag_catalog():
    spec = resolve_catalog_spec([TAGS], [])
    return spec, IdentityCatalog.from_spec(spec)


def test_noncatalog_mass_goes_to_unknown_and_mass_is_conserved():
    spec, catalog = _tag_catalog()
    assert not catalog.contains("notag")
    pmap = build_phase_label_map(spec, catalog, "tags")
    phase = ("unknown", "red", "blue", "notag")
    raw = np.array([0.0, 0.08, 0.02, 0.90])
    lp = remap_phase_log_probs(np.log(np.clip(raw, 1e-300, None)), phase, catalog, pmap)
    p = np.exp(lp)
    assert p.sum() == pytest.approx(1.0)
    assert p[catalog.unknown_index] == pytest.approx(0.90)
    assert p[catalog.index_of("red")] == pytest.approx(0.08)
    assert p[catalog.index_of("blue")] == pytest.approx(0.02)


def test_legacy_flag_reproduces_drop_and_renormalize():
    spec, catalog = _tag_catalog()
    pmap = build_phase_label_map(spec, catalog, "tags")
    phase = ("unknown", "red", "blue", "notag")
    raw = np.log(np.array([1e-12, 0.08, 0.02, 0.90]))
    p = np.exp(
        remap_phase_log_probs(raw, phase, catalog, pmap, noncatalog_to_unknown=False)
    )
    # "red" was 8 % of the crop and is now read as 80 % -- the fabrication.
    assert p[catalog.index_of("red")] == pytest.approx(0.8, rel=1e-6)


def test_multi_model_non_identifying_axis_value_goes_to_unknown():
    thorax = {
        "label": "thorax",
        "unique_identifier": True,
        "class_names_per_factor": [["red", "blue", "notag"]],
        "non_identifying_classes": ["notag"],
    }
    abdomen = {
        "label": "abdomen",
        "unique_identifier": True,
        "class_names_per_factor": [["square", "circle"]],
    }
    spec = resolve_catalog_spec([thorax, abdomen], [])
    catalog = IdentityCatalog.from_spec(spec)
    pmap = build_phase_label_map(spec, catalog, "thorax")
    assert "notag" not in pmap
    raw = np.log(np.array([1e-12, 0.3, 0.1, 0.6]))
    p = np.exp(
        remap_phase_log_probs(raw, ("unknown", "red", "blue", "notag"), catalog, pmap)
    )
    # Composite remap is a likelihood over global entries: every entry "red"
    # reaches carries red's 0.3, and the non-identifying 0.6 is the
    # likelihood of "no identity" -- twice any single red composite.
    sq = catalog.index_of("red_square")
    assert p[catalog.unknown_index] / p[sq] == pytest.approx(2.0)
    assert p[catalog.index_of("red_circle")] == pytest.approx(p[sq])


def test_noncatalog_flag_default_and_param():
    assert noncatalog_to_unknown_enabled(None) is True
    assert noncatalog_to_unknown_enabled({}) is True
    assert (
        noncatalog_to_unknown_enabled({"IDENTITY_NONCATALOG_TO_UNKNOWN": False})
        is False
    )


def _phase_cache(tmp_path, frames_probs, phase=("unknown", "red", "blue", "notag")):
    _, catalog = _tag_catalog()
    path = tmp_path / "ev.npz"
    cache = IdentityEvidenceCache(
        path,
        catalog_labels=catalog.labels,
        mode="w",
        catalog_labels_by_source={"tags": phase},
    )
    for f, probs in frames_probs.items():
        cache.save_frame(
            f,
            [
                IdentityEvidence.from_cnn(
                    f, f, "tags", np.log(np.clip(np.asarray(probs), 1e-300, None))
                )
            ],
        )
    cache.flush()
    return IdentityEvidenceCache(path, mode="r")


def test_load_trajectory_evidence_threads_the_flag(tmp_path):
    spec, catalog = _tag_catalog()
    cache = _phase_cache(tmp_path, {0: [0.0, 0.08, 0.02, 0.9]})
    df = pd.DataFrame({"TrajectoryID": [1], "FrameID": [0], "DetectionID": [0]})
    maps = {"tags": build_phase_label_map(spec, catalog, "tags")}
    honest = load_trajectory_evidence(df, cache, catalog, maps)[1][0][1]
    legacy = load_trajectory_evidence(
        df, cache, catalog, maps, noncatalog_to_unknown=False
    )[1][0][1]
    assert np.exp(honest)[catalog.index_of("red")] == pytest.approx(0.08)
    assert np.exp(legacy)[catalog.index_of("red")] == pytest.approx(0.8, rel=1e-6)


def _breaker_raw(catalog, unknown_mass, n_traj=6, n_frames=50):
    raw = {}
    for t in range(n_traj):
        seq = []
        for f in range(n_frames):
            p = np.full(catalog.size, 0.001)
            u = unknown_mass(t, f)
            p[0] = u
            p[1 + t] = 1.0 - u - 0.001 * (catalog.size - 2)
            seq.append((f, np.log(p / p.sum())))
        raw[t] = seq
    return raw


def test_breaker_ignores_confidently_non_identifying_rows():
    catalog = IdentityCatalog.from_labels([f"l{i}" for i in range(6)])
    # Every other frame is "notag" (honest unknown mass 0.9); the rest name
    # the right animal confidently. The notag frames must not count as an
    # uninformative source.
    raw = _breaker_raw(catalog, lambda t, f: 0.9 if f % 2 else 0.05)
    q = offline.assess_evidence_quality(raw, catalog)
    assert q.ok and q.conf_frac == 1.0


def test_breaker_trips_when_every_row_is_non_identifying():
    catalog = IdentityCatalog.from_labels([f"l{i}" for i in range(6)])
    q = offline.assess_evidence_quality(
        _breaker_raw(catalog, lambda t, f: 0.9), catalog
    )
    assert not q.ok


# ---------------------------------------------------------------------------
# 2. absolute gate + absolute seeding
# ---------------------------------------------------------------------------


ON = {"FRAGMENT_MIN_ABSOLUTE_POSTERIOR": 0.3, "FRAGMENT_ABSOLUTE_SEEDING": True}


def _row(tid, start, end, mean_probs, x=0.0, **extra):
    known = list(mean_probs)
    return {
        "TrajectoryID": tid,
        "StartFrame": start,
        "EndFrame": end,
        "StartX": x,
        "StartY": 0.0,
        "EndX": x,
        "EndY": 0.0,
        "MeanCNNProbs": dict(mean_probs),
        "MeanTagProbs": {},
        "CNNLogEvidence": {
            k: float(np.log(max(v, 1e-12))) for k, v in mean_probs.items()
        },
        "TagLogEvidence": {},
        "Stability": 0.5,
        "OnlineLabel": "unknown",
        "OnlineConfidence": 0.0,
        **extra,
        "_known": known,
    }


def _frags(rows):
    df = pd.DataFrame(rows)
    return df.drop(columns=["_known"])


def test_absolute_gate_drops_relatively_saturated_weak_support():
    # 0.05 vs 0.0001: relative support ~1.0, absolute 0.05.
    row = _row(1, 0, 100, {"a": 0.05, "b": 0.0001})
    rel = offline._combined_support(row, ["a", "b"], {})
    assert rel["a"] > 0.99
    gated = offline._fragment_support(row, ["a", "b"], ON)
    assert gated["a"] == 0.0
    off = offline._fragment_support(
        row, ["a", "b"], {"FRAGMENT_MIN_ABSOLUTE_POSTERIOR": 0.0}
    )
    assert off == rel


def test_absolute_gate_leaves_no_evidence_fragments_alone():
    row = _row(1, 0, 10, {})
    row["MeanCNNProbs"], row["CNNLogEvidence"] = {}, {}
    row["OnlineLabel"], row["OnlineConfidence"] = "a", 0.9
    s = offline._fragment_support(row, ["a", "b"], ON)
    assert s["a"] > 0.5


def test_weak_long_fragment_no_longer_blocks_a_strong_short_one():
    # Overlapping in time: only one can hold "a".
    long_weak = _row(1, 0, 1000, {"a": 0.06, "b": 0.0005}, x=0.0)
    short_strong = _row(2, 100, 200, {"a": 0.95, "b": 0.01}, x=500.0)
    frags = _frags([long_weak, short_strong])
    base = offline._iterative_assign(
        frags,
        ["a", "b"],
        {"FRAGMENT_MIN_ABSOLUTE_POSTERIOR": 0.0, "FRAGMENT_ABSOLUTE_SEEDING": False},
    )
    assert base[0] == "a" and base[1] is None  # the documented failure
    fixed = offline._iterative_assign(frags, ["a", "b"], ON)
    assert offline._iterative_assign(frags, ["a", "b"], {}) == base  # default off
    assert fixed[0] is None and fixed[1] == "a"


def test_absolute_seeding_orders_by_absolute_evidence():
    long_weak = _row(1, 0, 1000, {"a": 0.35, "b": 0.0005})
    short_strong = _row(2, 0, 400, {"a": 0.95, "b": 0.01})
    s1 = offline._fragment_support(long_weak, ["a", "b"], ON)
    s2 = offline._fragment_support(short_strong, ["a", "b"], ON)
    m1 = offline._seed_mass(long_weak, s1, 1001, ON)
    m2 = offline._seed_mass(short_strong, s2, 401, ON)
    assert m2 > m1
    r1 = offline._seed_mass(long_weak, s1, 1001, {"FRAGMENT_ABSOLUTE_SEEDING": False})
    r2 = offline._seed_mass(short_strong, s2, 401, {"FRAGMENT_ABSOLUTE_SEEDING": False})
    assert r1 > r2  # the historical relative key


def test_length_weight_clamp_warns_once(caplog):
    offline._LENGTH_WEIGHT_WARNED.discard(2.5)
    with caplog.at_level(logging.WARNING, logger=offline.log.name):
        assert offline._length_weight({"FRAGMENT_LENGTH_WEIGHT": 2.5}) == 1.0
        assert offline._length_weight({"FRAGMENT_LENGTH_WEIGHT": 2.5}) == 1.0
        assert offline._length_weight({"FRAGMENT_LENGTH_WEIGHT": 0.6}) == 0.6
    msgs = [r for r in caplog.records if "FRAGMENT_LENGTH_WEIGHT" in r.getMessage()]
    assert len(msgs) == 1


# ---------------------------------------------------------------------------
# 4. trust layer
# ---------------------------------------------------------------------------


def _box(cx, cy, length, width, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    hl, hw = length / 2.0, width / 2.0
    pts = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    rot = pts @ np.array([[c, s], [-s, c]])
    return rot + [cx, cy]


def _geometry(dets):
    """dets: list of (det_id, frame, cx, cy, length, width, angle)."""
    return DetectionGeometry.from_arrays(
        detection_ids=[d[0] for d in dets],
        frames=[d[1] for d in dets],
        centroids=[[d[2], d[3]] for d in dets],
        corners=[_box(d[2], d[3], d[4], d[5], d[6]) for d in dets],
    )


def test_nearest_neighbour_distance_per_frame():
    nn = nearest_neighbour_distance(
        np.array([0, 0, 0, 1]),
        np.array([0.0, 3.0, 10.0, 0.0]),
        np.array([0.0, 4.0, 0.0, 0.0]),
    )
    assert nn[0] == pytest.approx(5.0) and nn[1] == pytest.approx(5.0)
    assert np.isinf(nn[3])


def _isolated_population(n=300, length=20.0, width=5.0):
    # one animal per frame -> every detection is isolated
    return [(i, i, 100.0, 100.0, length, width, 0.0) for i in range(n)]


def test_calibration_uses_isolated_detections_and_disables_shape_when_too_few():
    settings = TrustSettings(enabled=True, min_prior_detections=200)
    cal, _ = calibrate(_geometry(_isolated_population(300)), settings)
    assert cal.body_length == pytest.approx(20.0)
    assert cal.prior_area == pytest.approx(100.0)
    assert cal.shape_active and cal.isolation_active
    cal2, _ = calibrate(_geometry(_isolated_population(50)), settings)
    assert not cal2.shape_active and cal2.isolation_active
    assert not cal2.keypoints_active  # no pose columns supplied


def test_prior_ignores_crowded_blobs():
    dets = _isolated_population(300)
    # 300 frames of a two-animal merge (big blob) touching a neighbour
    base = 10_000
    for i in range(300):
        dets.append((base + 2 * i, base + i, 500.0, 500.0, 40.0, 10.0, 0.0))
        dets.append((base + 2 * i + 1, base + i, 510.0, 500.0, 20.0, 5.0, 0.0))
    cal, _ = calibrate(_geometry(dets), TrustSettings(enabled=True))
    assert cal.body_length == pytest.approx(20.0)


def _trust_df(det_ids, frames, tids=None, **cols):
    df = pd.DataFrame(
        {
            "TrajectoryID": tids if tids is not None else [1] * len(det_ids),
            "FrameID": frames,
            "DetectionID": det_ids,
            "X": 0.0,
            "Y": 0.0,
        }
    )
    for k, v in cols.items():
        df[k] = v
    return df


def test_trust_checks_isolation_shape_and_keypoints():
    dets = _isolated_population(300)
    # frame 1000: a crowded pair (both 1 BL long) -> not isolated
    dets += [(5000, 1000, 0.0, 0.0, 20.0, 5.0, 0.0), (5001, 1000, 10.0, 0.0, 20, 5, 0)]
    # frame 1001: isolated but blob-shaped (area 4x) -> shape fails
    dets += [(5002, 1001, 0.0, 0.0, 40.0, 10.0, 0.0)]
    # frame 1002: isolated, good shape
    dets += [(5003, 1002, 0.0, 0.0, 20.0, 5.0, 0.0)]
    geom = _geometry(dets)
    df = _trust_df([5000, 5002, 5003], [1000, 1001, 1002])
    params = {"IDENTITY_EVIDENCE_TRUST": True}
    trust = build_evidence_trust(df, params, geometry=geom)
    assert trust.active
    assert set(trust.trusted) == {5003}

    # Pose present: keypoint completeness now also gates.
    df2 = _trust_df(
        [5003, 0],
        [1002, 0],
        PoseKpt_head_Conf=[0.9, 0.9],
        PoseKpt_tail_Conf=[0.1, 0.9],
    )
    trust2 = build_evidence_trust(df2, params, geometry=geom)
    assert trust2.calibration.keypoints_active
    assert set(trust2.trusted) == {0}


def test_trust_inactive_without_geometry_or_when_disabled(caplog):
    df = _trust_df([1], [1])
    assert not build_evidence_trust(df, {}).active
    with caplog.at_level(logging.WARNING):
        t = build_evidence_trust(df, {"IDENTITY_EVIDENCE_TRUST": True})
    assert not t.active and "geometry" in t.reason_inactive


def test_collapse_views_move_turn_and_axial():
    s = TrustSettings(view_move_bl=0.5, view_turn_deg=30.0)
    # stationary for 10 frames -> one view
    assert (
        len(collapse_views(range(10), [0.0] * 10, [0.0] * 10, [0.0] * 10, 20, s)) == 1
    )
    # each frame moves 0.6 BL -> every frame a view
    xs = [12.0 * i for i in range(5)]
    assert len(collapse_views(range(5), xs, [0.0] * 5, [0.0] * 5, 20.0, s)) == 5
    # 45 deg turn in place -> new view; a ~180 deg "turn" is an axial no-op
    angs = [0.0, np.radians(45.0), np.radians(45.0 + 179.0)]
    views = collapse_views(range(3), [0.0] * 3, [0.0] * 3, angs, 20.0, s)
    assert [len(v) for v in views] == [1, 2]


def _trust_obj(trusted, bl=20.0, **settings):
    from hydra_suite.core.individual.identity.evidence_trust import TrustCalibration

    cal = TrustCalibration(bl, bl, 100.0, 4.0, 1000, 1000, True, True, False)
    return EvidenceTrust(TrustSettings(enabled=True, **settings), cal, trusted)


def test_fragment_views_use_raw_absolute_trusted_evidence_only():
    catalog = IdentityCatalog.from_labels(["a", "b"])
    rows = _trust_df([10, 11, 12], [0, 1, 2])
    p_good = np.log(np.array([0.1, 0.8, 0.1]))
    p_bad = np.log(np.array([0.0001, 0.0001, 0.9998]))
    trust = _trust_obj({10: (0.0, 0.0, 0.0), 12: (100.0, 0.0, 0.0)})
    views = fragment_views(rows, [(0, p_good), (1, p_bad), (2, p_good)], trust)
    assert views.n_views == 2
    sup = views.support(list(catalog.labels[1:]))
    assert sup["a"] == pytest.approx(0.8) and sup["b"] == pytest.approx(0.1)


def test_tier_ordering_and_untrusted_cannot_displace_trusted():
    # Long untrusted fragment with strong-looking relative evidence vs a
    # shorter trusted one; they overlap in time.
    long_fallback = _row(1, 0, 2000, {"a": 0.6, "b": 0.01}, x=0.0)
    trusted = _row(
        2,
        100,
        300,
        {"a": 0.55, "b": 0.01},
        x=500.0,
        TrustedViews=4,
        TrustedSupport={"a": 0.9, "b": 0.01},
    )
    long_fallback["TrustedViews"] = 0
    long_fallback["TrustedSupport"] = {}
    frags = _frags([long_fallback, trusted])
    params = {"MAX_VELOCITY_BREAK": 1e9, **ON}
    out = offline._iterative_assign(frags, ["a", "b"], params)
    assert out[1] == "a" and out[0] is None
    # Without tiers the long fragment seeds first and keeps the label.
    untiered = _frags([long_fallback, trusted]).drop(
        columns=["TrustedViews", "TrustedSupport"]
    )
    out2 = offline._iterative_assign(untiered, ["a", "b"], params)
    assert out2[0] == "a"


# ---------------------------------------------------------------------------
# 5. trusted-view label-switch splitting
# ---------------------------------------------------------------------------


def _views(labels_conf, start=0, step=10):
    probs, frames = [], []
    for k, (lab, conf) in enumerate(labels_conf):
        p = np.full(2, (1 - conf) / 2)
        p[lab] = conf
        probs.append(p)
        frames.append((start + k * step, start + k * step + 2))
    return FragmentViews(frames, np.array(probs))


def test_switch_found_and_cut_lies_in_untrusted_gap():
    s = TrustSettings(split_min_views=3, split_min_log_odds=4.0)
    v = _views([(0, 0.9)] * 4 + [(1, 0.9)] * 4)
    cuts = find_label_switches(v, s)
    assert len(cuts) == 1
    left_end, right_start = v.view_frames[3][1], v.view_frames[4][0]
    assert left_end <= cuts[0] < right_start


def test_no_split_without_enough_views_confidence_or_a_real_switch():
    s = TrustSettings(split_min_views=3, split_min_log_odds=4.0)
    assert find_label_switches(_views([(0, 0.9)] * 2 + [(1, 0.9)] * 2), s) == []
    assert find_label_switches(_views([(0, 0.9)] * 8), s) == []
    assert find_label_switches(_views([(0, 0.4)] * 4 + [(1, 0.4)] * 4), s) == []
    strict = TrustSettings(split_min_views=3, split_min_log_odds=1e6)
    assert find_label_switches(_views([(0, 0.9)] * 4 + [(1, 0.9)] * 4), strict) == []


def test_two_switches_found_recursively():
    s = TrustSettings(split_min_views=3, split_min_log_odds=4.0)
    v = _views([(0, 0.9)] * 4 + [(1, 0.9)] * 4 + [(0, 0.9)] * 4)
    assert len(find_label_switches(v, s)) == 2


# ---------------------------------------------------------------------------
# end-to-end: run_fragment_solver with a trust object
# ---------------------------------------------------------------------------


def test_run_fragment_solver_trusted_split_replaces_pelt(tmp_path):
    catalog = IdentityCatalog.from_labels(["a", "b"])
    path = tmp_path / "ev.npz"
    cache = IdentityEvidenceCache(path, catalog_labels=catalog.labels, mode="w")
    rows, trusted = [], {}
    n = 80
    for f in range(n):
        lab = 1 if f < n // 2 else 2
        p = np.full(3, 0.05)
        p[lab] = 0.9
        cache.save_frame(f, [IdentityEvidence.from_cnn(f, f, "cnn", np.log(p))])
        rows.append(
            {"TrajectoryID": 1, "FrameID": f, "DetectionID": f, "X": 3.0 * f, "Y": 0.0}
        )
        trusted[f] = (3.0 * f, 0.0, 0.0)
    cache.flush()
    df = pd.DataFrame(rows)
    trust = _trust_obj(trusted, bl=10.0, split_min_views=3, split_min_log_odds=4.0)
    params = {"ENABLE_PELT_SPLITTING": True, "MIN_FRAGMENT_FRAMES": 5}
    out = offline.run_fragment_solver(
        df,
        catalog,
        params,
        cache=IdentityEvidenceCache(path, mode="r"),
        evidence_trust=trust,
    )
    first = out[out.FrameID < n // 2 - 5]["IdentityFinalLabel"].unique()
    last = out[out.FrameID > n // 2 + 5]["IdentityFinalLabel"].unique()
    assert list(first) == ["a"] and list(last) == ["b"]
    assert out["TrajectoryID"].nunique() == 2


def test_engine_param_defaults_and_config_round_trip():
    from hydra_suite.trackerkit.config.identity_schema import (
        EVIDENCE_ENGINE_KEYS,
        IdentityConfig,
    )

    def cfg_get(cfg, key, default=None):
        return cfg.get(key, default)

    d = IdentityConfig.from_engine_config({}, {}, cfg_get=cfg_get).posthoc
    assert d.noncatalog_to_unknown is True
    assert d.fragment_min_absolute_posterior == 0.0
    assert d.fragment_absolute_seeding is False
    assert d.evidence_trust_enabled is False
    cfg = {
        "identity_evidence_trust": True,
        "identity_trust_isolation_bl": 1.5,
        "fragment_min_absolute_posterior": 0.3,
    }
    p = IdentityConfig.from_engine_config(cfg, {}, cfg_get=cfg_get).posthoc
    params = {k: getattr(p, f) for f, k in EVIDENCE_ENGINE_KEYS.items()}
    s = TrustSettings.from_params(params)
    assert s.enabled and s.isolation_bl == 1.5
    assert params["FRAGMENT_MIN_ABSOLUTE_POSTERIOR"] == 0.3
    again = IdentityConfig.from_dict(
        IdentityConfig.from_engine_config(cfg, {}, cfg_get=cfg_get).to_dict()
    )
    assert again.posthoc == p


# ---------------------------------------------------------------------------
# review follow-ups
# ---------------------------------------------------------------------------


def test_trusted_tier_uses_an_absolute_candidate_floor():
    # Honest-but-moderate trusted evidence: red 0.45, rest on unknown.
    trusted = _row(
        1,
        0,
        100,
        {"red": 0.45, "blue": 0.05},
        TrustedViews=5,
        TrustedSupport={"red": 0.45, "blue": 0.05},
    )
    out = offline._iterative_assign(_frags([trusted]), ["red", "blue"], {})
    assert out[0] == "red"  # 0.45 >= identity_trust_min_support (0.3)
    strict = offline._iterative_assign(
        _frags([trusted]), ["red", "blue"], {"IDENTITY_TRUST_MIN_SUPPORT": 0.5}
    )
    assert strict[0] is None
    # FRAGMENT_MIN_SUPPORT stays the relative floor for untiered fragments.
    untiered = _row(2, 0, 100, {"red": 0.45, "blue": 0.05})
    rel = offline._iterative_assign(
        _frags([untiered]), ["red", "blue"], {"FRAGMENT_MIN_SUPPORT": 0.95}
    )
    assert rel[0] is None


def test_label_switch_search_is_fast_on_long_noisy_trajectories():
    import time

    rng = np.random.default_rng(0)
    V, L = 50_000, 8
    probs = rng.dirichlet(np.full(L, 0.3), size=V)
    frames = [(10 * k, 10 * k + 2) for k in range(V)]
    s = TrustSettings(split_min_views=3, split_min_log_odds=8.0)
    # Worst case for the span search: confident runs of exactly
    # split_min_views views alternating between two labels, so every run
    # boundary passes the local gate.
    runs = np.full((V, L), 0.01)
    runs[np.arange(V), (np.arange(V) // 3) % 2] = 0.93
    t0 = time.perf_counter()
    find_label_switches(FragmentViews(frames, probs), s)
    find_label_switches(FragmentViews(frames, runs), s)
    assert time.perf_counter() - t0 < 20.0


def test_isolation_is_per_arena():
    dets = _isolated_population(300)
    # Two animals 0.5 BL apart but in different arenas.
    dets += [(7000, 2000, 0.0, 0.0, 20.0, 5.0, 0.0), (7001, 2000, 10.0, 0.0, 20, 5, 0)]
    geom = _geometry(dets)
    df = _trust_df([7000, 7001], [2000, 2000], arena_id=[0, 1])
    trust = build_evidence_trust(df, {"IDENTITY_EVIDENCE_TRUST": True}, geometry=geom)
    assert {7000, 7001} <= set(trust.trusted)
    same = _trust_df([7000, 7001], [2000, 2000])
    trust2 = build_evidence_trust(
        same, {"IDENTITY_EVIDENCE_TRUST": True}, geometry=geom
    )
    assert not ({7000, 7001} & set(trust2.trusted))


def test_all_non_identifying_breaker_message(caplog):
    catalog = IdentityCatalog.from_labels([f"l{i}" for i in range(6)])
    raw = _breaker_raw(catalog, lambda t, f: 0.9)
    q = offline.assess_evidence_quality(raw, catalog)
    assert q.all_non_identifying and not q.ok
    broken = _breaker_raw(catalog, lambda t, f: 1e-6)
    for t in broken:  # every animal reads as the same label: collapsed source
        broken[t] = broken[0]
    assert not offline.assess_evidence_quality(broken, catalog).all_non_identifying


def test_split_ids_respect_a_global_floor():
    df = pd.DataFrame(
        {"TrajectoryID": [1] * 20, "FrameID": list(range(20)), "X": 0.0, "Y": 0.0}
    )
    out = offline.split_trajectories_at_changepoints(
        df, {1: [9]}, {"MIN_FRAGMENT_FRAMES": 5}, trajectory_id_floor=100
    )
    assert set(out["TrajectoryID"]) == {100, 101}


def test_trust_top_label_only_leaves_a_blocked_fragment_unknown():
    # A holds "a" with strong trusted evidence. B overlaps it, its own best
    # label is also "a", and "b" is a plausible runner-up (a lookalike tag).
    holder = _row(
        1,
        0,
        300,
        {"a": 0.9, "b": 0.05},
        x=0.0,
        TrustedViews=10,
        TrustedSupport={"a": 0.9, "b": 0.05},
    )
    blocked = _row(
        2,
        50,
        200,
        {"a": 0.5, "b": 0.4},
        x=500.0,
        TrustedViews=3,
        TrustedSupport={"a": 0.5, "b": 0.4},
    )
    frags = _frags([holder, blocked])
    params = {"MAX_VELOCITY_BREAK": 1e9, **ON, "IDENTITY_EVIDENCE_TRUST": True}
    out = offline._iterative_assign(frags, ["a", "b"], params)
    assert out[0] == "a" and out[1] is None
    # Fallback to the runner-up when the setting is off...
    fallback = offline._iterative_assign(
        frags, ["a", "b"], dict(params, IDENTITY_TRUST_TOP_LABEL_ONLY=False)
    )
    assert fallback[0] == "a" and fallback[1] == "b"
    # ...and the setting has no effect with the trust layer off.
    no_trust = offline._iterative_assign(
        frags, ["a", "b"], dict(params, IDENTITY_EVIDENCE_TRUST=False)
    )
    assert no_trust[1] == "b"


def test_trust_top_label_only_config_round_trip():
    from hydra_suite.trackerkit.config.identity_schema import (
        EVIDENCE_ENGINE_KEYS,
        IdentityConfig,
    )

    def cfg_get(cfg, key, default=None):
        return cfg.get(key, default)

    d = IdentityConfig.from_engine_config({}, {}, cfg_get=cfg_get).posthoc
    assert d.trust_top_label_only is True
    p = IdentityConfig.from_engine_config(
        {"identity_trust_top_label_only": False}, {}, cfg_get=cfg_get
    ).posthoc
    params = {k: getattr(p, f) for f, k in EVIDENCE_ENGINE_KEYS.items()}
    assert params["IDENTITY_TRUST_TOP_LABEL_ONLY"] is False
