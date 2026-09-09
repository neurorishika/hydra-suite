"""A replay pass must resolve the cache key the forward pass actually wrote.

Batch sizes are folded into the inference cache keys, and a tuned forward pass
writes its caches under a tuned key. Backward (and the read-only production
replay) used to rebuild that key from the project's *configured* batch sizes,
so a tuned forward pass made its own backward pass unreadable. These tests pin
the record-and-read-back contract that removes that mismatch.
"""

from __future__ import annotations

from hydra_suite.core.inference.autotune.models import (
    InferenceRuntimeOverlay,
    InferenceTuningSettings,
)
from hydra_suite.core.inference.autotune.replay_vector import (
    load_replay_vector,
    replay_vector_path,
    write_replay_vector,
)
from hydra_suite.core.inference.cache import keys as keys_mod
from hydra_suite.core.inference.cache.keys import detection_cache_key
from hydra_suite.core.inference.config import (
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
)


def _config(batch: int = 1) -> InferenceConfig:
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt", confidence_threshold=0.5),
        ),
        detection_batch_size=batch,
    )


def _key(config: InferenceConfig) -> str:
    return detection_cache_key(
        config.obb, None, config.detection_batch_size
    ).as_string()


def _overlay(*, requested: int, effective: int) -> InferenceRuntimeOverlay:
    base = InferenceTuningSettings(detection_batch_size=requested)
    return InferenceRuntimeOverlay(
        requested=base,
        admitted=InferenceTuningSettings(detection_batch_size=effective),
        effective=InferenceTuningSettings(detection_batch_size=effective),
        field_sources=(("detection_batch_size", "calibrated"),),
        status="calibrated",
        reason="measured",
    )


def test_backward_resolves_the_tuned_key_a_forward_pass_wrote(tmp_path):
    """A tuned forward pass at det=4, then a replay built from the untuned
    project config, must land on the SAME detection cache key."""
    configured = _config(batch=1)
    tuned_forward = _overlay(requested=1, effective=4).apply(configured)
    assert tuned_forward.detection_batch_size == 4
    assert _key(tuned_forward) != _key(configured)

    write_replay_vector(tmp_path, _overlay(requested=1, effective=4))

    # The replay pass starts from the project's own (untuned) config.
    record = load_replay_vector(tmp_path)
    assert record is not None
    replay_config = record.apply(_config(batch=1))

    assert replay_config.detection_batch_size == 4
    assert _key(replay_config) == _key(tuned_forward)


def test_untuned_project_keys_are_unchanged(tmp_path):
    """No record => resolve exactly as before: default batch, no ``|batch=``."""
    write_replay_vector(tmp_path, _overlay(requested=1, effective=1))
    assert not replay_vector_path(tmp_path).exists()
    assert load_replay_vector(tmp_path) is None

    write_replay_vector(tmp_path, None)
    assert load_replay_vector(tmp_path) is None

    # The configured key carries no ``|batch=`` term, and differs from any
    # tuned key -- i.e. an untuned project's keys are exactly what they were.
    configured = _config(batch=1)
    assert detection_cache_key(
        configured.obb, None, 1
    ).config_hash == keys_mod._direct_raw_config_hash(configured.obb)
    assert _key(configured) != _key(_config(batch=4))


def test_a_later_untuned_forward_pass_clears_a_stale_record(tmp_path):
    """A tuned run then an untuned run must not leave det=4 pointing at a
    cache written at the default batch."""
    write_replay_vector(tmp_path, _overlay(requested=1, effective=4))
    assert load_replay_vector(tmp_path) is not None

    write_replay_vector(tmp_path, _overlay(requested=1, effective=1))

    assert load_replay_vector(tmp_path) is None
    assert not replay_vector_path(tmp_path).exists()


def test_unreadable_record_degrades_to_configured_not_to_a_guess(tmp_path):
    replay_vector_path(tmp_path).write_text("{not json")
    assert load_replay_vector(tmp_path) is None

    replay_vector_path(tmp_path).write_text('{"schema": 999, "effective": {}}')
    assert load_replay_vector(tmp_path) is None


def test_record_round_trips_the_full_vector(tmp_path):
    requested = InferenceTuningSettings(
        detection_batch_size=1,
        slice_tile_batch_size=16,
        pose_batch_size=4,
        headtail_batch_size=8,
        identity_batch_sizes=(("cnn_identity", 8),),
        pipeline_depth=2,
    )
    effective = InferenceTuningSettings(
        detection_batch_size=4,
        slice_tile_batch_size=32,
        pose_batch_size=8,
        headtail_batch_size=16,
        identity_batch_sizes=(("cnn_identity", 16),),
        pipeline_depth=3,
    )
    overlay = InferenceRuntimeOverlay(
        requested=requested,
        admitted=effective,
        effective=effective,
        field_sources=(),
        status="cache_hit",
        reason="stored profile",
        profile_id="abc",
    )
    write_replay_vector(tmp_path, overlay)
    record = load_replay_vector(tmp_path)
    assert record is not None
    assert record.effective == effective
    assert record.requested == requested
    assert record.status == "cache_hit"
    assert record.profile_id == "abc"
