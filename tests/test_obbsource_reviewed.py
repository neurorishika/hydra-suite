from hydra_suite.detectkit.gui.models import OBBSource


def test_defaults_keep_legacy_sources_trusted():
    s = OBBSource(name="orig")
    assert s.reviewed is True and s.derived_from is None and s.sam2_variant is None


def test_roundtrip_preserves_new_fields():
    s = OBBSource(
        name="orig_seg",
        level="polygon",
        reviewed=False,
        derived_from="orig",
        sam2_variant="sam2.1-hiera-base_plus",
    )
    back = OBBSource.from_dict(s.to_dict())
    assert back.reviewed is False
    assert back.derived_from == "orig"
    assert back.sam2_variant == "sam2.1-hiera-base_plus"


def test_from_dict_missing_new_fields_defaults_reviewed_true():
    back = OBBSource.from_dict({"name": "legacy", "level": "obb"})
    assert back.reviewed is True and back.derived_from is None
