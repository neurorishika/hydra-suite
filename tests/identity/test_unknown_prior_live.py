"""Task 5: the ``identity_unknown_prior`` knob is live in catalog evidence.

Spec R6: ``substrate._factor_log_prob`` floors each factor's probability at
1e-6, so after the composite product ``unknown`` was pinned at exactly
1e-12 on every detection -- the ``identity_unknown_prior`` config knob
existed but never touched this path. This test proves
``map_cnn_to_catalog``'s new ``unknown_prior`` parameter redistributes
fused probability mass onto the "unknown" catalog slot (index 0), leaving
the known labels' relative proportions unchanged, while defaulting to 0.0
(today's exact behavior) for backward compatibility.
"""

import numpy as np

from hydra_suite.core.individual.identity import substrate
from hydra_suite.core.individual.identity.catalog import IdentityCatalog


def _mapped(unknown_prior):
    catalog = IdentityCatalog.from_labels(["a", "b"])
    log_p, _ = substrate.map_cnn_to_catalog(
        [np.array([0.6, 0.4])],
        class_labels_per_factor=[["a", "b"]],
        factor_class_to_catalog={},
        is_composite=False,
        catalog_size=3,
        catalog=catalog,
        unknown_prior=unknown_prior,
    )
    return np.exp(log_p)


def test_default_zero_prior_is_backward_compatible():
    p = _mapped(0.0)
    assert p[0] < 1e-5


def test_unknown_prior_gets_exactly_that_mass():
    p = _mapped(0.05)
    assert np.isclose(p[0], 0.05) and np.isclose(p[1:].sum(), 0.95)
    assert np.isclose(p[1] / p[2], 0.6 / 0.4)


def test_cache_schema_version_is_3(tmp_path):
    from hydra_suite.core.individual.identity import cache

    assert cache._SCHEMA_VERSION == 3


def test_stale_schema_v2_sidecar_is_rejected_not_silently_reused(tmp_path, caplog):
    """A v2 sidecar predates the live `unknown_prior`, so its stored
    fused probabilities are not a faithful v3 reproduction. Loading it must
    behave as an empty cache (forcing a rebuild), not silently serve stale
    rows -- caught by writing a real v2-shaped sidecar by hand (the writer
    always stamps the CURRENT `_SCHEMA_VERSION`, so it cannot itself produce
    a stale one to load)."""
    import numpy as np

    from hydra_suite.core.individual.identity import cache as cache_mod
    from hydra_suite.core.individual.identity.evidence import EvidenceSource

    path = tmp_path / "stale_v2.npz"
    np.savez_compressed(
        str(path),
        evidence_schema_version=np.int64(2),
        catalog_labels=np.array(["unknown", "a", "b"], dtype="U255"),
        f0_source_order=np.array(["cnn0"], dtype="U255"),
        f0__cnn0_det_ids=np.array([0], dtype=np.int64),
        f0__cnn0_source_types=np.array([EvidenceSource.CNN.value], dtype="U32"),
        f0__cnn0_log_probs=np.log(np.array([[1e-12, 0.6, 0.4]])),
        f0__cnn0_catalog_size=np.int64(3),
        f0__cnn0_cal_sig=np.array([""], dtype="U255"),
        f0__cnn0_rt_sig=np.array([""], dtype="U64"),
    )

    with caplog.at_level("INFO"):
        reader = cache_mod.IdentityEvidenceCache(path, mode="r")

    assert reader.get_cached_frames() == []
    assert reader.load_frame(0) == []
    assert "schema" in caplog.text
