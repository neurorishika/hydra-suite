from hydra_suite.detectkit.gui.models import OBBSource


def test_obbsource_level_defaults_to_obb():
    src = OBBSource(path="/x", name="s")
    assert src.level == "obb"


def test_obbsource_level_roundtrips():
    src = OBBSource(path="/x", name="s", level="polygon")
    assert OBBSource.from_dict(src.to_dict()).level == "polygon"


def test_obbsource_from_dict_missing_level_is_obb():
    # Simulates a pre-migration project JSON with no "level" key.
    legacy = {"path": "/x", "name": "s", "validated": True, "source_kind": "detectkit"}
    assert OBBSource.from_dict(legacy).level == "obb"
