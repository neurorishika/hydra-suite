"""canonical_aug_copies knob for the YOLO-classify offline prefit multiplier."""

from dataclasses import asdict

from hydra_suite.training.contracts import AugmentationProfile


def test_canonical_aug_copies_default_is_three():
    assert AugmentationProfile().canonical_aug_copies == 3


def test_canonical_aug_copies_is_overridable():
    assert AugmentationProfile(canonical_aug_copies=5).canonical_aug_copies == 5


def test_canonical_aug_copies_serializes_in_asdict():
    # TrainingRunSpec.to_dict() uses dataclasses.asdict; the new field must
    # round-trip so persisted run specs carry the knob.
    d = asdict(AugmentationProfile(canonical_aug_copies=4))
    assert d["canonical_aug_copies"] == 4
