import numpy as np

from hydra_suite.training.canonical_aug import CanonicalAug


def test_aug_is_deterministic_with_seed_and_shape_preserving():
    crop = np.random.default_rng(0).integers(0, 256, (56, 112, 3), np.uint8)
    a1 = CanonicalAug(seed=7)(crop.copy())
    a2 = CanonicalAug(seed=7)(crop.copy())
    np.testing.assert_array_equal(a1, a2)
    assert a1.shape == crop.shape and a1.dtype == np.uint8


def test_aug_changes_pixels():
    crop = np.random.default_rng(0).integers(0, 256, (56, 112, 3), np.uint8)
    assert not np.array_equal(CanonicalAug(seed=1)(crop.copy()), crop)


def test_aug_does_not_mutate_global_rng_state():
    # CanonicalAug must use its own seeded Generator, never the global
    # np.random state -- otherwise training runs become order-dependent.
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    crop = np.random.default_rng(0).integers(0, 256, (56, 112, 3), np.uint8)
    CanonicalAug(seed=2)(crop.copy())
    after = np.random.get_state()[1]
    np.testing.assert_array_equal(before, after)
