"""One click must serve both the cache-less first run and every cached run.

Closes the S2 estimated/measured two-record bridge (see ``store.py:243-300``)
immediately during ``session.calibrate``, instead of making the user pay a
whole extra tracking run before a cached run can find anything.
"""

from hydra_suite.core.inference.autotune import session
from tests.autotune_helpers import fake_store, make_calibration_context


def test_calibration_writes_both_an_estimated_and_a_measured_record(
    monkeypatch, tmp_path
):
    store = fake_store(monkeypatch, tmp_path / "store")
    ctx = make_calibration_context(monkeypatch, tmp_path, cache_dir=None)

    session.calibrate(ctx, budget_seconds=60.0)

    density_flags = {
        profile.key.workload.density_is_estimated for profile in store.saved
    }
    assert density_flags == {
        True,
        False,
    }, "expected an estimated-key AND a measured-key record"


def test_a_cacheless_run_hits_the_estimated_record(monkeypatch, tmp_path):
    fake_store(monkeypatch, tmp_path / "store")
    ctx = make_calibration_context(monkeypatch, tmp_path, cache_dir=None)
    session.calibrate(ctx, budget_seconds=60.0)

    lookup_ctx = make_calibration_context(
        monkeypatch, tmp_path, cache_dir=None, mode="lookup"
    )
    _, overlay, _ = session.lookup(lookup_ctx)

    assert overlay.status == "cache_hit"


def test_a_cached_run_hits_the_measured_record(monkeypatch, tmp_path):
    fake_store(monkeypatch, tmp_path / "store")
    session.calibrate(
        make_calibration_context(monkeypatch, tmp_path, cache_dir=None),
        budget_seconds=60.0,
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    lookup_ctx = make_calibration_context(
        monkeypatch,
        tmp_path,
        cache_dir=cache_dir,
        measured_counts=(2, 3, 2),
        mode="lookup",
    )
    _, overlay, _ = session.lookup(lookup_ctx)

    assert overlay.status == "cache_hit"


def test_zero_detection_frames_are_not_dropped_from_measured_density(
    monkeypatch, tmp_path
):
    """A frame the trial measured but which had no detections is a real 0,
    not a missing measurement -- ``sample_detection_workload`` zero-fills it
    from a production cache (``integration.py:124-142``), so the density
    ``session.calibrate`` writes into the bridge must include it too, or the
    measured-key record it writes lands on a different bucket than a later
    cache-based run would derive for the identical video.
    """

    store = fake_store(monkeypatch, tmp_path / "store")
    # A 3-frame window where the middle frame produced no detections at all:
    # the forward CSV the trial writes has NO row for that frame.
    ctx = make_calibration_context(
        monkeypatch, tmp_path, cache_dir=None, frame_counts=(2, 0, 3)
    )

    session.calibrate(ctx, budget_seconds=60.0)

    measured = next(
        profile
        for profile in store.saved
        if not profile.key.workload.density_is_estimated
    )
    candidate = next(c for c in measured.candidates if c.settings == measured.selected)
    assert 0 in candidate.detection_counts, (
        "the zero-detection frame must survive into the measured density, "
        f"got {candidate.detection_counts!r}"
    )
    # One measurement block is 3 frames (frame_counts=(2, 0, 3)); the search
    # runs several blocks, so the count is a multiple of 3, not literally 3.
    assert len(candidate.detection_counts) % 3 == 0
    assert len(candidate.detection_counts) >= 3
