from hydra_suite.detectkit.gui.models import PendingEscalation


def test_legacy_dict_backfills_sam2_primer_fields():
    legacy = {
        "staged_path": "/tmp/staged",
        "target_level": "polygon",
        "sam2_variant": "sam2.1-hiera-large",
        "created_at": "2026-01-01T00:00:00",
    }
    p = PendingEscalation.from_dict(legacy)
    assert p.primer_kind == "sam2"
    assert p.primer_variant == "sam2.1-hiera-large"
    assert p.primer_prompt == ""
    assert p.primer_params == {}


def test_sam3_round_trip_preserves_prompt_and_params():
    p = PendingEscalation(
        staged_path="/tmp/s",
        target_level="polygon",
        created_at="2026-01-01T00:00:00",
        primer_kind="sam3",
        primer_variant="sam3",
        primer_prompt="black ant",
        primer_params={"confidence": 0.35, "tile_px": 1600},
    )
    restored = PendingEscalation.from_dict(p.to_dict())
    assert restored == p


def test_sam2_variant_stays_in_sync_for_legacy_readers():
    p = PendingEscalation.from_dict({"sam2_variant": "sam2.1-hiera-tiny"})
    assert p.sam2_variant == "sam2.1-hiera-tiny"
    assert p.to_dict()["sam2_variant"] == "sam2.1-hiera-tiny"
