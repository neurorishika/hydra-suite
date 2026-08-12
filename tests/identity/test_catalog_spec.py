import pytest

from hydra_suite.core.individual.identity.catalog import UNKNOWN_LABEL, IdentityCatalog
from hydra_suite.core.individual.identity.spec import CatalogEntry, IdentityCatalogSpec


def _spec():
    return IdentityCatalogSpec(
        entries=(
            CatalogEntry(
                display_label="red_big",
                factors=(("color", "red"), ("size", "big")),
                source="cnn",
            ),
            CatalogEntry(
                display_label="blue_small",
                factors=(("color", "blue"), ("size", "small")),
                source="cnn",
            ),
            CatalogEntry(display_label="ant7", factors=(), source="tag"),
        )
    )


def test_labels_are_display_labels_in_order():
    assert _spec().labels == ("red_big", "blue_small", "ant7")


def test_roundtrip_preserves_structure():
    spec = _spec()
    assert IdentityCatalogSpec.from_dict(spec.to_dict()) == spec


def test_underscore_in_class_name_survives_structurally():
    # A class name containing "_" would be mis-split by the legacy split("_") path.
    # The structured factors must round-trip exactly regardless of the display string.
    spec = IdentityCatalogSpec(
        entries=(
            CatalogEntry(
                display_label="dark_red_x_1",
                factors=(("color", "dark_red"), ("id", "x_1")),
                source="cnn",
            ),
        )
    )
    restored = IdentityCatalogSpec.from_dict(spec.to_dict())
    assert restored.entries[0].factors == (("color", "dark_red"), ("id", "x_1"))


def test_from_spec_matches_from_labels():
    spec = _spec()
    cat = IdentityCatalog.from_spec(spec)
    assert cat.labels == (UNKNOWN_LABEL, "red_big", "blue_small", "ant7")
    assert cat.labels == IdentityCatalog.from_labels(list(spec.labels)).labels


def test_from_spec_empty_raises():
    with pytest.raises(ValueError):
        IdentityCatalog.from_spec(IdentityCatalogSpec(entries=()))
