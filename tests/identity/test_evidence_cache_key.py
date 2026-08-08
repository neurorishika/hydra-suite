from hydra_suite.core.individual.identity.spec import CatalogEntry, IdentityCatalogSpec
from hydra_suite.core.inference.identity_evidence_key import identity_evidence_cache_key


def _spec(label="red_big"):
    return IdentityCatalogSpec(
        entries=(
            CatalogEntry(
                display_label=label,
                factors=(("color", "red"), ("size", "big")),
                source="cnn",
            ),
        )
    )


def test_stable_and_sensitive():
    a = identity_evidence_cache_key(_spec(), {"cnn0": (1.5,)}, "vidsig")
    b = identity_evidence_cache_key(_spec(), {"cnn0": (1.5,)}, "vidsig")
    assert a == b and isinstance(a, str) and len(a) >= 8
    assert identity_evidence_cache_key(_spec(), {"cnn0": (2.0,)}, "vidsig") != a
    assert (
        identity_evidence_cache_key(_spec("blue_big"), {"cnn0": (1.5,)}, "vidsig") != a
    )
    assert identity_evidence_cache_key(_spec(), {"cnn0": (1.5,)}, "other") != a
