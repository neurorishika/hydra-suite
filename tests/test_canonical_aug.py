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


class _FakeWorkerInfo:
    def __init__(self, wid):
        self.id = wid


def test_aug_decorrelates_across_dataloader_workers(monkeypatch):
    # Two DataLoader workers fork the SAME seeded instance; without a
    # worker-aware reseed they would draw identical augmentation streams. Each
    # worker must instead get an independent-but-reproducible stream.

    crop = np.random.default_rng(0).integers(0, 256, (56, 112, 3), np.uint8)

    def _run_as_worker(wid):
        monkeypatch.setattr(
            "torch.utils.data.get_worker_info", lambda: _FakeWorkerInfo(wid)
        )
        return CanonicalAug(seed=7)(crop.copy())

    w0 = _run_as_worker(0)
    w1 = _run_as_worker(1)
    assert not np.array_equal(w0, w1), "workers must not share an aug stream"
    # ...yet each worker id is reproducible across runs (deterministic training).
    np.testing.assert_array_equal(w0, _run_as_worker(0))


def test_aug_no_worker_context_stays_deterministic(monkeypatch):
    # Outside a DataLoader worker, get_worker_info() is None and the same seed
    # must reproduce exactly (the determinism contract is untouched).
    monkeypatch.setattr("torch.utils.data.get_worker_info", lambda: None)
    crop = np.random.default_rng(0).integers(0, 256, (56, 112, 3), np.uint8)
    np.testing.assert_array_equal(
        CanonicalAug(seed=7)(crop.copy()), CanonicalAug(seed=7)(crop.copy())
    )
