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


def test_calibration_works_on_a_project_with_no_slicing(monkeypatch, tmp_path):
    """Every fixture above is SLICED, which is exactly why the whole suite
    stayed green while ``session.calibrate`` raised ``TypeError`` on every
    non-SAHI project (``fly_obb``, ``worm_bgsub``, any bgsub or plain
    direct-OBB project with slicing off).

    ``InferenceTuningSettings.from_config`` leaves ``slice_tile_batch_size``
    at ``None`` there, ``candidate_space`` returns ``()`` for a ``None``
    current value, and ``static_max_for`` therefore falls back to ``None`` --
    which ``max()`` cannot compare to the ``detection_batch_size`` int.
    """

    from hydra_suite.core.inference.autotune.models import InferenceTuningSettings

    store = fake_store(monkeypatch, tmp_path / "store")
    ctx = make_calibration_context(monkeypatch, tmp_path, cache_dir=None, sliced=False)

    # The fixture really is the shape that broke: no tile batch size at all.
    baseline = InferenceTuningSettings.from_config(ctx.config)
    assert baseline.slice_tile_batch_size is None
    # ... and the derivation still produced a usable int rather than raising.
    assert isinstance(ctx.artifact_batch_size, int)
    assert ctx.artifact_batch_size >= ctx.config.detection_batch_size

    _effective, overlay, _result = session.calibrate(ctx, budget_seconds=60.0)

    assert overlay is not None
    assert overlay.status == "calibrated"
    assert store.saved, "a non-sliced project must still persist a profile"
    # And the digest helper -- the other copy of the same derivation -- works.
    assert session.calibration_key_digest(ctx)


def test_lookup_and_calibrate_agree_on_a_tensorrt_context(tmp_path):
    """The profile key ``calibrate`` writes under must be the key ``lookup``
    reads under, on the ONE backend where they used to diverge.

    ``_model_fingerprints`` folds ``tensorrt_profile_fingerprint(...)`` --
    derived from ``INFERENCE_AUTOTUNE_TENSORRT_PROFILE_BATCH_SIZE``, falling
    back to ``config.detection_batch_size`` -- into EVERY model fingerprint.
    While only ``calibrate`` injected that param, it keyed on the static
    candidate maximum and ``lookup`` keyed on the configured batch size, so
    no profile was ever findable on gpu_fast/CUDA.
    """

    from dataclasses import replace

    from hydra_suite.core.inference.autotune.integration import (
        build_tracking_autotune_request,
    )
    from tests.autotune_helpers import make_tensorrt_context

    ctx = make_tensorrt_context(tmp_path)

    def digest_of(context) -> str:
        return build_tracking_autotune_request(
            context.config,
            context.run_context,
            observation=context.probe.observation,
            backend=context.backend,
            device_identity=context.device_identity,
        ).key.digest

    # ``lookup`` builds exactly this request and injects nothing of its own.
    assert digest_of(ctx) == session.calibration_key_digest(ctx)

    # The param is genuinely load-bearing and genuinely the candidate
    # maximum -- not merely equal because both sides fell back to the
    # configured batch size.
    assert ctx.params[session.TENSORRT_PROFILE_BATCH_PARAM] == ctx.artifact_batch_size
    assert ctx.artifact_batch_size > ctx.config.detection_batch_size

    unkeyed_params = dict(ctx.params)
    unkeyed_params.pop(session.TENSORRT_PROFILE_BATCH_PARAM)
    unkeyed = replace(
        ctx,
        params=unkeyed_params,
        run_context=replace(ctx.run_context, params=unkeyed_params),
    )
    assert digest_of(unkeyed) != digest_of(ctx), (
        "the TensorRT profile batch size must change the key, else this "
        "test would pass even if the fix were reverted"
    )


def test_detection_cache_reuse_is_modelled_as_nothing_to_tune(monkeypatch, tmp_path):
    """Cache reuse is ALL-OR-NOTHING, so a reuse run has nothing to calibrate.

    ``worker.py`` only sets ``use_cached_detections`` after ``caches_all_valid()``
    -- ``cache_set_is_fully_reusable`` over the WHOLE set (detection.npz,
    headtail.npz, cnn_<label>.npz, pose.npz, apriltag.npz). Every stage is then
    served from disk and essentially no inference runs.

    This used to be modelled as ``RESULT_CACHE_STAGE_MASK=("detector",)`` with
    only the two detector fields frozen, i.e. it asserted pose/head-tail/identity
    still ran. They do not. That invented a middle state the pipeline never
    enters, and it cost twice: a calibration started with reuse active would
    "measure" stages that never execute, promoting a vector picked from
    cache-read noise; and the invented mask entered the profile key, so a reuse
    run could never match a profile calibrated without a cache.
    """

    fake_store(monkeypatch, tmp_path / "store")
    ctx = make_calibration_context(
        monkeypatch,
        tmp_path,
        cache_dir=tmp_path / "cache",
        measured_counts=(2, 3, 2),
        use_cached_detections=True,
    )

    assert ctx.run_context.execution_mode == "cache_replay"
    assert "RESULT_CACHE_STAGE_MASK" not in ctx.params

    from hydra_suite.core.inference.autotune.integration import (
        build_tracking_autotune_request,
    )

    request = build_tracking_autotune_request(
        ctx.config,
        ctx.run_context,
        observation=ctx.probe.observation,
        backend=ctx.backend,
        device_identity=ctx.device_identity,
    )

    # Nothing left to search, and measuring is declined outright -- exactly the
    # treatment cache_replay already received.
    assert request.eligible is False
    assert set(request.baseline.field_names()) <= set(
        request.planner.context.cached_fields
    )
    for field in request.baseline.field_names():
        assert request.planner.values_for(field, request.baseline) == ()
