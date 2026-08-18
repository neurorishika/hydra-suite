"""Shared-memory transport for SLEAP pose crops must not hold one fd per crop.

The transport creates one POSIX shared-memory segment per crop in a batch. Each
live ``SharedMemory`` object costs an open fd + an mmap for as long as it is
held, so keeping the whole batch mapped blew past the process fd limit on real
batches ("[Errno 24] Too many open files", and the resource_tracker KeyError its
failed-creation cleanup path raises). The segment outlives the local mapping, so
the writer must close its handle as soon as the pixels are written and only
unlink at the end.
"""

from __future__ import annotations

import resource

import numpy as np
import pytest

from hydra_suite.core.individual.pose.backends.sleap import share_crop_to_shm


def _cleanup(handles):
    for shm in handles:
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def test_share_crop_closes_local_mapping_but_segment_survives():
    from multiprocessing import shared_memory

    crop = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    image_id, payload, shm = share_crop_to_shm(0, crop)
    assert payload is not None and shm is not None
    try:
        # Local mapping released immediately: no fd, no mmap held for this crop.
        assert shm._fd == -1
        assert shm._buf is None
        # ...but the segment is still there for the service to attach to.
        attached = shared_memory.SharedMemory(name=payload["shm_name"])
        try:
            got = np.ndarray(
                tuple(payload["shape"]), dtype=np.uint8, buffer=attached.buf
            )
            assert np.array_equal(got, crop)
        finally:
            attached.close()
    finally:
        _cleanup([shm])
    assert image_id == "inmem_crop_000000"


def test_many_crops_do_not_exhaust_the_fd_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard))
    handles = []
    try:
        crop = np.zeros((32, 32, 3), dtype=np.uint8)
        for i in range(400):
            _id, payload, shm = share_crop_to_shm(i, crop)
            assert payload is not None, f"crop {i} failed to share"
            handles.append(shm)
    except OSError as exc:  # pragma: no cover - this is the regression
        pytest.fail(f"fd exhaustion sharing crops: {exc}")
    finally:
        _cleanup(handles)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
