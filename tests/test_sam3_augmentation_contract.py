from dataclasses import asdict

from hydra_suite.training.contracts import AugmentationProfile, Sam3LoraParams


def test_rot90_field_defaults_off():
    assert AugmentationProfile().rot90 == 0.0


def test_sam3_params_default_augmentation_is_disabled():
    p = Sam3LoraParams()
    assert isinstance(p.augmentation, AugmentationProfile)
    assert p.augmentation.enabled is False


def test_default_instances_do_not_share_profile():
    a, b = Sam3LoraParams(), Sam3LoraParams()
    a.augmentation.fliplr = 0.9
    assert b.augmentation.fliplr == 0.0


def test_dict_round_trip_coerces_nested_profile():
    p = Sam3LoraParams(
        prompt="ant",
        augmentation=AugmentationProfile(enabled=True, fliplr=0.5, rot90=0.25),
    )
    q = Sam3LoraParams(**asdict(p))
    assert isinstance(q.augmentation, AugmentationProfile)
    assert q.augmentation == p.augmentation


def test_dict_without_augmentation_key_gets_disabled_profile():
    data = asdict(Sam3LoraParams(prompt="ant"))
    data.pop("augmentation")
    assert Sam3LoraParams(**data).augmentation.enabled is False
