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
