import pytest

from hydra_suite.core.inference.slice_meta import (
    available_slice_profiles,
    remove_slice_profile,
    upsert_slice_profile,
)

BASE = {"geometry_mode": "auto_object", "imgsz": 640, "overlap": 0.2}
SETTINGS = {"enabled": True, "geometry_mode": "auto_object", "overlap": 0.2}


def test_saving_a_profile_does_not_silently_make_it_primary():
    meta = upsert_slice_profile(BASE, name="Balanced", settings=SETTINGS)
    assert meta["primary_profile_id"] == ""
    assert len(meta["profiles"]) == 1


def test_primary_is_set_only_when_explicitly_requested():
    meta = upsert_slice_profile(BASE, name="Balanced", settings=SETTINGS, primary=True)
    assert meta["primary_profile_id"] == meta["profiles"][0]["id"]


def test_removing_the_primary_requires_an_explicit_decision():
    meta = upsert_slice_profile(BASE, name="Balanced", settings=SETTINGS, primary=True)
    meta = upsert_slice_profile(meta, name="Fast scan", settings=SETTINGS)
    primary_id = meta["primary_profile_id"]
    other = next(p["id"] for p in meta["profiles"] if p["id"] != primary_id)
    with pytest.raises(ValueError, match="replacement"):
        remove_slice_profile(meta, primary_id)
    assert (
        remove_slice_profile(meta, primary_id, new_primary_id="")["primary_profile_id"]
        == ""
    )
    moved = remove_slice_profile(meta, primary_id, new_primary_id=other)
    assert moved["primary_profile_id"] == other
    assert len(available_slice_profiles(moved)) == 1


def test_unknown_replacement_is_rejected():
    meta = upsert_slice_profile(BASE, name="Balanced", settings=SETTINGS, primary=True)
    with pytest.raises(ValueError, match="Unknown"):
        remove_slice_profile(meta, meta["primary_profile_id"], new_primary_id="nope")


def test_removing_a_non_primary_profile_needs_no_decision():
    meta = upsert_slice_profile(BASE, name="Balanced", settings=SETTINGS, primary=True)
    meta = upsert_slice_profile(meta, name="Fast scan", settings=SETTINGS)
    victim = next(p for p in meta["profiles"] if p["id"] != meta["primary_profile_id"])
    result = remove_slice_profile(meta, victim["id"])
    assert result["primary_profile_id"] == meta["primary_profile_id"]


def test_unknown_future_keys_round_trip_untouched():
    meta = upsert_slice_profile(
        BASE, name="Balanced", settings=dict(SETTINGS, future_knob=7)
    )
    assert available_slice_profiles(meta)[0]["settings"]["future_knob"] == 7
