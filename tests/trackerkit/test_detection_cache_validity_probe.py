"""The TrackerKit optimizer-cache probe admits complete replay evidence only.

``detection_cache_dir_covers_range`` recognizes the modern
``InferenceRunner`` cache-directory layout, but must never mistake a valid
detection member for a complete production replay when configured downstream
evidence is missing or covers a different range. A non-directory/non-existent
path returns False gracefully rather than raising.
"""

import json
from pathlib import Path

import numpy as np

from hydra_suite.core.inference.cache.keys import video_signature
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runner import _open_caches
from hydra_suite.core.tracking.optimization.detection_config import (
    inference_config_for_optimizer_params,
)
from hydra_suite.trackerkit.gui.orchestrators.config import (
    detection_cache_dir_covers_range,
)


def _make_obb_result(frame_idx: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.zeros((1, 2), dtype=np.float32),
        angles=np.zeros((1,), dtype=np.float32),
        sizes=np.array([10.0], dtype=np.float32),
        shapes=np.array([[10.0, 4.0]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=np.zeros((1, 4, 2), dtype=np.float32),
        detection_ids=np.array([1], dtype=np.int64),
    )


def _write_modern_detection_cache_dir(
    cache_dir: Path,
    params: dict,
    frames: range,
    *,
    headtail_frames: range | None = None,
):
    """Populate the atomic cache-set layout written by ``InferenceRunner``."""
    cfg = inference_config_for_optimizer_params(params)
    caches = _open_caches(
        cfg,
        cache_dir,
        video_signature(""),
        params.get("ROI_MASK", None),
        write_mode="fresh",
    )
    handle = caches.detection
    assert handle is not None
    for frame_idx in frames:
        handle.write_frame(frame_idx, result=_make_obb_result(frame_idx))
    if caches.headtail is not None and headtail_frames is not None:
        # One frame per immutable chunk lets the mixed-coverage fixture remove
        # exactly the final valid chunk while retaining a structurally valid
        # downstream cache manifest.
        caches.headtail.chunk_size = 1
        for frame_idx in headtail_frames:
            # Downstream evidence may legitimately be empty, but must record
            # every replayed frame so its coverage matches detection exactly.
            caches.headtail.write_frame(
                frame_idx,
                det_indices=np.zeros(0, dtype=np.int32),
                heading_hints=np.zeros(0, dtype=np.float32),
                heading_confidences=np.zeros(0, dtype=np.float32),
                directed_mask=np.zeros(0, dtype=np.uint8),
            )
    caches.close()


def _headtail_params(tmp_path: Path) -> dict:
    model = tmp_path / "headtail.pt"
    model.write_bytes(b"cache-key fixture")
    return {
        "DETECTION_METHOD": "yolo_obb",
        "YOLO_HEADTAIL_MODEL_PATH": str(model),
    }


def _headtail_manifest_path(cache_dir: Path, params: dict) -> Path:
    """Return the active downstream manifest without constructing a runner."""

    cfg = inference_config_for_optimizer_params(params)
    caches = _open_caches(
        cfg,
        cache_dir,
        video_signature(""),
        params.get("ROI_MASK", None),
        read_only=True,
    )
    assert caches.headtail is not None
    return caches.headtail.path


def _drop_last_headtail_coverage(cache_dir: Path, params: dict) -> None:
    """Model a published downstream cache that is valid but one frame shorter."""

    manifest_path = _headtail_manifest_path(cache_dir, params)
    with np.load(manifest_path, allow_pickle=False) as raw:
        payload = {name: raw[name] for name in raw.files}
    entries = json.loads(str(payload["chunks_json"][0]))
    assert len(entries) > 1
    payload["chunks_json"] = np.asarray(
        [json.dumps(entries[:-1], sort_keys=True, separators=(",", ":"))]
    )
    np.savez(manifest_path, **payload)


def test_modern_cache_dir_covering_range_is_valid(tmp_path):
    params: dict = {}
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(cache_dir, params, frames=range(0, 5))

    active_before = (cache_dir / "cache_set.json").read_bytes()
    assert (
        detection_cache_dir_covers_range(
            str(cache_dir), "", params, start_frame=0, end_frame=4
        )
        is True
    )
    assert (cache_dir / "cache_set.json").read_bytes() == active_before


def test_optimizer_cache_probe_does_not_initialize_inference_backends(
    tmp_path, monkeypatch
):
    """Scanning candidate cache directories is metadata-only, never model I/O."""

    params: dict = {}
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(cache_dir, params, frames=range(0, 5))

    import hydra_suite.core.inference.runner as runner_module

    class _ForbiddenRunner:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("cache-admission probe must not construct a runner")

    monkeypatch.setattr(runner_module, "InferenceRunner", _ForbiddenRunner)
    assert detection_cache_dir_covers_range(
        str(cache_dir), "", params, start_frame=0, end_frame=4
    )


def test_modern_cache_file_path_covering_range_is_valid(tmp_path):
    """``current_detection_cache_path`` (and optimizer-reuse candidates) may be
    the ``detection.npz`` FILE itself, not just its containing directory --
    the probe must normalize to the containing dir before opening caches."""
    params: dict = {}
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(cache_dir, params, frames=range(0, 5))
    cache_file = cache_dir / "detection.npz"

    assert (
        detection_cache_dir_covers_range(
            str(cache_file), "", params, start_frame=0, end_frame=4
        )
        is True
    )


def test_modern_cache_dir_missing_frames_is_not_valid(tmp_path):
    params: dict = {}
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(cache_dir, params, frames=range(0, 3))

    assert (
        detection_cache_dir_covers_range(
            str(cache_dir), "", params, start_frame=0, end_frame=9
        )
        is False
    )


def test_modern_cache_with_corrupt_set_manifest_is_not_valid(tmp_path):
    params: dict = {}
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(cache_dir, params, frames=range(0, 5))
    (cache_dir / "cache_set.json").write_text("{}", encoding="utf-8")

    assert (
        detection_cache_dir_covers_range(
            str(cache_dir), "", params, start_frame=0, end_frame=4
        )
        is False
    )


def test_detection_only_cache_is_not_valid_with_configured_downstream_evidence(
    tmp_path,
):
    """The GUI must schedule full evidence preparation, not launch then fail."""

    params = _headtail_params(tmp_path)
    cache_dir = tmp_path / ".inference_cache_clip"
    # This is a normal cache from a tracking run before head-tail was enabled:
    # detection is healthy, but the configured replay member set differs.
    _write_modern_detection_cache_dir(cache_dir, {}, frames=range(0, 5))

    assert (
        detection_cache_dir_covers_range(
            str(cache_dir), "", params, start_frame=0, end_frame=4
        )
        is False
    )


def test_mixed_downstream_coverage_is_not_valid_for_optimizer_replay(tmp_path):
    params = _headtail_params(tmp_path)
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(
        cache_dir,
        params,
        frames=range(0, 5),
        headtail_frames=range(0, 5),
    )
    assert detection_cache_dir_covers_range(
        str(cache_dir), "", params, start_frame=0, end_frame=4
    )
    _drop_last_headtail_coverage(cache_dir, params)

    assert (
        detection_cache_dir_covers_range(
            str(cache_dir), "", params, start_frame=0, end_frame=4
        )
        is False
    )


def test_corrupt_downstream_evidence_is_not_valid_for_optimizer_replay(tmp_path):
    params = _headtail_params(tmp_path)
    cache_dir = tmp_path / ".inference_cache_clip"
    _write_modern_detection_cache_dir(
        cache_dir,
        params,
        frames=range(0, 5),
        headtail_frames=range(0, 5),
    )
    _headtail_manifest_path(cache_dir, params).write_bytes(b"not a cache manifest")

    assert (
        detection_cache_dir_covers_range(
            str(cache_dir), "", params, start_frame=0, end_frame=4
        )
        is False
    )


def test_nonexistent_path_returns_false_without_raising(tmp_path):
    missing = tmp_path / "does_not_exist.npz"

    assert (
        detection_cache_dir_covers_range(
            str(missing), "", {}, start_frame=0, end_frame=4
        )
        is False
    )


def test_empty_path_returns_false(tmp_path):
    assert (
        detection_cache_dir_covers_range("", "", {}, start_frame=0, end_frame=4)
        is False
    )
