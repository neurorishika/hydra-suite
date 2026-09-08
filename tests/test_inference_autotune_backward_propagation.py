"""The backward pass must run at the forward pass's batch size."""

import json

from hydra_suite.core.inference.autotune.applied_vector import (
    read_applied_vector,
    write_applied_vector,
)
from hydra_suite.core.inference.autotune.models import InferenceTuningSettings
from hydra_suite.core.inference.cache.keys import detection_cache_key
from hydra_suite.core.inference.config import (
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
)


def test_roundtrip_preserves_every_field(tmp_path):
    settings = InferenceTuningSettings(
        detection_batch_size=4,
        slice_tile_batch_size=8,
        pose_batch_size=16,
        headtail_batch_size=None,
        identity_batch_sizes=(("ant", 32),),
        pipeline_depth=2,
    )
    write_applied_vector(tmp_path, settings)
    assert read_applied_vector(tmp_path) == settings


def test_missing_file_returns_none(tmp_path):
    """A cache written before this change must fall back to configured values,
    which is exactly today's behaviour for an untuned forward pass."""
    assert read_applied_vector(tmp_path) is None


def test_corrupt_file_returns_none_and_does_not_raise(tmp_path):
    (tmp_path / "applied_inference_vector.json").write_text(
        "{ not json", encoding="utf-8"
    )
    assert read_applied_vector(tmp_path) is None


def test_written_file_is_human_readable_json(tmp_path):
    settings = InferenceTuningSettings(
        detection_batch_size=4,
        slice_tile_batch_size=None,
        pose_batch_size=None,
        headtail_batch_size=None,
        identity_batch_sizes=(),
        pipeline_depth=1,
    )
    write_applied_vector(tmp_path, settings)
    data = json.loads((tmp_path / "applied_inference_vector.json").read_text())
    assert data["detection_batch_size"] == 4


def test_applying_the_forward_vector_reproduces_the_forward_cache_key(tmp_path):
    """Regression test for the courtship abort.

    The forward pass promotes detection_batch_size from 1 -> 4 and writes its
    detection cache under a key that folds in that batch size. If the
    backward pass re-resolves independently at the project's own (untuned)
    batch size, it computes a DIFFERENT key and misses the forward cache
    entirely -- that is the measured failure this task fixes.

    This test builds the forward pass's InferenceConfig at its tuned batch
    size, computes the detection cache key it would have written (the same
    call ``worker.py`` makes: ``detection_cache_key(config.obb, roi_mask,
    config.detection_batch_size)``), then simulates the backward pass:
    start from the project's baseline (untuned) config, apply the vector
    recovered from ``read_applied_vector``, and assert the resulting cache
    key is identical to the forward key.
    """
    baseline_config = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/nonexistent/model.pt"),
        ),
        detection_batch_size=1,
    )

    # Forward pass: promoted to batch size 4, writes its detection cache
    # under a key that folds that batch size in, and persists the applied
    # vector alongside it.
    forward_batch_size = 4
    forward_settings = InferenceTuningSettings(detection_batch_size=forward_batch_size)
    forward_config = forward_settings.apply(baseline_config)
    forward_key = detection_cache_key(
        forward_config.obb, batch_size=forward_config.detection_batch_size
    )
    write_applied_vector(tmp_path, forward_settings)

    # Backward pass: starts from the SAME untuned baseline config (as if it
    # had re-resolved independently and gotten batch size 1), then applies
    # the forward pass's recorded vector instead of trusting its own
    # resolution.
    recovered = read_applied_vector(tmp_path)
    assert recovered is not None
    backward_config = recovered.apply(baseline_config)
    backward_key = detection_cache_key(
        backward_config.obb, batch_size=backward_config.detection_batch_size
    )

    assert backward_key == forward_key
    # And show the failure mode this fixes: an independently re-resolved
    # (untuned) backward config would have missed the forward key.
    untuned_key = detection_cache_key(
        baseline_config.obb, batch_size=baseline_config.detection_batch_size
    )
    assert untuned_key != forward_key
