import numpy as np

from hydra_suite.core.inference.cache.reuse import get_or_compute_raw
from hydra_suite.core.inference.result import OBBResult


def _make_raw_result(idx: int) -> OBBResult:
    return OBBResult(
        frame_idx=idx,
        centroids=np.zeros((1, 2)),
        angles=np.zeros(1),
        sizes=np.zeros(1),
        shapes=np.zeros((1, 2)),
        confidences=np.array([0.9]),
        corners=np.zeros((1, 4, 2)),
        detection_ids=OBBResult.make_detection_ids(idx, 1),
    )


class _FakeRunner:
    def __init__(self):
        self.calls = []

    def detect_batch_raw(self, frames, frame_indices=None, roi_mask=None):
        self.calls.append(list(frame_indices))
        return [_make_raw_result(idx) for idx in frame_indices]


def test_get_or_compute_raw_computes_on_empty_cache(tmp_path):
    runner = _FakeRunner()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    result = get_or_compute_raw(runner, tmp_path, frames, [0, 1, 2])
    assert set(result.keys()) == {0, 1, 2}
    assert runner.calls == [[0, 1, 2]]


def test_get_or_compute_raw_reads_fully_covered_cache_without_recompute(tmp_path):
    runner = _FakeRunner()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    get_or_compute_raw(runner, tmp_path, frames, [0, 1])  # populates cache
    runner.calls.clear()
    result = get_or_compute_raw(runner, tmp_path, frames, [0, 1])
    assert runner.calls == []  # no new compute — fully covered by existing cache
    assert set(result.keys()) == {0, 1}


def test_get_or_compute_raw_recomputes_whole_set_on_partial_miss(tmp_path):
    runner = _FakeRunner()
    frames2 = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    get_or_compute_raw(runner, tmp_path, frames2, [0, 1])  # populates cache for [0, 1]
    runner.calls.clear()
    frames3 = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    get_or_compute_raw(runner, tmp_path, frames3, [0, 1, 2])  # 2 is missing
    # Per the no-merge convention: the whole *requested* set is recomputed fresh,
    # not just the miss.
    assert runner.calls == [[0, 1, 2]]


def test_get_or_compute_raw_write_false_never_touches_the_cache_file(tmp_path):
    """`write=False` makes a MISS read-only: recompute, but do not persist.

    Callers that merely borrow a cache file another subsystem owns (TrackerKit's
    dataset export borrows tracking's `.inference_cache_<stem>/detection.npz`)
    must not have `DetectionCacheHandle.close()` rewrite that file from their
    own partial buffer.
    """
    runner = _FakeRunner()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

    # Populate a cache for [0, 1] the normal (owning) way.
    get_or_compute_raw(runner, tmp_path, frames, [0, 1])
    cache_path = tmp_path / "detection.npz"
    before = cache_path.read_bytes()
    runner.calls.clear()

    # A miss under write=False: recomputed, returned, nothing written.
    frames3 = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    result = get_or_compute_raw(runner, tmp_path, frames3, [0, 1, 2], write=False)
    assert set(result.keys()) == {0, 1, 2}
    assert runner.calls == [[0, 1, 2]]
    assert cache_path.read_bytes() == before


def test_get_or_compute_raw_write_false_creates_no_cache_file_at_all(tmp_path):
    runner = _FakeRunner()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    result = get_or_compute_raw(runner, tmp_path, frames, [0, 1], write=False)
    assert set(result.keys()) == {0, 1}
    assert not (tmp_path / "detection.npz").exists()


def test_get_or_compute_raw_write_false_still_serves_a_cache_hit(tmp_path):
    """The HIT path is unaffected by `write`: still a pure read, zero computes."""
    runner = _FakeRunner()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    get_or_compute_raw(runner, tmp_path, frames, [0, 1])
    runner.calls.clear()
    result = get_or_compute_raw(runner, tmp_path, frames, [0, 1], write=False)
    assert runner.calls == []
    assert set(result.keys()) == {0, 1}
