"""Canonical crops reach pose backends as uint8 [0, 255], never float [0, 1].

The tracking pipeline builds canonical crops as float32 [0, 1]
(``stages/crops.py``: ``torch.from_numpy(...).float() / 255.0``).  Every pose
backend's preprocessing divides by 255 again, so a float [0, 1] crop handed
straight to ``predict_batch`` lands in [0, 1/255] and the model sees an
essentially constant image.

``ViTPoseBackend.predict_batch_cuda`` already guards this (``* 255`` then cast
to uint8), and both SLEAP paths guard it with an explicit comment
(``backends/sleap.py``).  ``ViTPoseBackend.predict_batch`` did not, so the CPU
and MPS pose paths were affected -- MPS tensors report ``is_cuda == False``, so
Apple hardware took the unguarded branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra_suite.core.individual.pose.crop_dtype import to_uint8_image


class TestToUint8Image:
    def test_scales_unit_range_float_to_full_range(self):
        crop = np.full((4, 4, 3), 1.0, dtype=np.float32)
        out = to_uint8_image(crop)
        assert out.dtype == np.uint8
        assert int(out.max()) == 255

    def test_unit_range_float_is_not_floored_to_black(self):
        # The failure mode being guarded: a straight cast floors every pixel
        # below 1.0 to zero, producing an all-black crop.
        crop = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = to_uint8_image(crop)
        assert int(out.min()) > 0
        assert 126 <= int(out[0, 0, 0]) <= 128

    def test_passes_uint8_through_unchanged(self):
        crop = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        out = to_uint8_image(crop)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, crop)

    def test_passes_full_range_float_through_without_rescaling(self):
        crop = np.full((4, 4, 3), 200.0, dtype=np.float32)
        out = to_uint8_image(crop)
        assert int(out[0, 0, 0]) == 200

    def test_clips_and_sanitises_out_of_range_values(self):
        crop = np.array(
            [[[np.nan, np.inf, -np.inf]], [[300.0, -20.0, 128.0]]],
            dtype=np.float32,
        )
        out = to_uint8_image(crop)
        assert out.dtype == np.uint8
        assert int(out.min()) >= 0
        assert int(out.max()) <= 255

    def test_preserves_shape(self):
        crop = np.zeros((7, 11, 3), dtype=np.float32)
        assert to_uint8_image(crop).shape == (7, 11, 3)


class _StubViTPoseBackend:
    """Minimal stand-in exercising the real ``predict_batch`` body."""

    def __init__(self, seen):
        from hydra_suite.core.individual.pose.vitpose.geometry import DEFAULT_GEOMETRY

        self._batch_size = 4
        self._geom = DEFAULT_GEOMETRY
        self._min_valid_conf = 0.2
        self._seen = seen

    def _forward(self, batch_chw):
        import torch

        b = batch_chw.shape[0]
        hm_w, hm_h = self._geom.heatmap_size_wh
        return torch.zeros((b, 2, hm_h, hm_w), dtype=torch.float32)


def test_predict_batch_hands_uint8_to_preprocess(monkeypatch):
    """A float [0, 1] crop must reach ``preprocess_crop`` as uint8 [0, 255]."""
    from hydra_suite.core.individual.pose.backends import vitpose as vitpose_backend

    seen: list[np.ndarray] = []
    real_preprocess = vitpose_backend.preprocess_crop

    def spy(crop, geom):
        seen.append(np.asarray(crop))
        return real_preprocess(crop, geom=geom)

    monkeypatch.setattr(vitpose_backend, "preprocess_crop", spy)

    backend = _StubViTPoseBackend(seen)
    crop = np.full((32, 32, 3), 0.5, dtype=np.float32)  # canonical crop range
    vitpose_backend.ViTPoseBackend.predict_batch(backend, [crop])

    assert len(seen) == 1
    assert seen[0].dtype == np.uint8, "float [0,1] crop reached the model unscaled"
    assert int(seen[0].max()) > 1, "crop was floored toward black"


def test_predict_batch_leaves_uint8_crops_untouched(monkeypatch):
    """The guard must be a no-op for the uint8 crops PoseKit already passes."""
    from hydra_suite.core.individual.pose.backends import vitpose as vitpose_backend

    seen: list[np.ndarray] = []
    real_preprocess = vitpose_backend.preprocess_crop

    def spy(crop, geom):
        seen.append(np.asarray(crop))
        return real_preprocess(crop, geom=geom)

    monkeypatch.setattr(vitpose_backend, "preprocess_crop", spy)

    backend = _StubViTPoseBackend(seen)
    crop = np.full((32, 32, 3), 200, dtype=np.uint8)
    vitpose_backend.ViTPoseBackend.predict_batch(backend, [crop])

    np.testing.assert_array_equal(seen[0], crop)


class _RecordingBackend:
    """Captures whatever the pose stage hands a backend."""

    def __init__(self):
        self.seen: list[np.ndarray] = []

    def predict_batch(self, crops):
        self.seen.extend(np.asarray(c) for c in crops)
        return [None] * len(crops)


def test_pose_stage_hands_backends_uint8(monkeypatch):
    """The stage boundary normalises, so every backend is covered — not just ViTPose.

    YOLO-pose has the same exposure: ultralytics' ``BasePredictor.preprocess``
    divides numpy input by 255 unconditionally.
    """
    import torch

    from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
    from hydra_suite.core.inference.result import OBBResult
    from hydra_suite.core.inference.stages import pose as pose_stage

    geometry = CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
    backend = _RecordingBackend()

    class _Model:
        n_keypoints = 2

    model = _Model()
    model.backend = backend

    corners = np.array(
        [[10.0, 10.0], [42.0, 10.0], [42.0, 26.0], [10.0, 26.0]], dtype=np.float32
    )
    obb = OBBResult(
        frame_idx=0,
        centroids=np.array([[26.0, 18.0]], dtype=np.float32),
        angles=np.array([0.0], dtype=np.float32),
        sizes=np.array([512.0], dtype=np.float32),
        shapes=np.array([[512.0, 2.0]], dtype=np.float32),
        confidences=np.array([0.9], dtype=np.float32),
        corners=np.stack([corners]),
        detection_ids=np.array([0], dtype=np.int64),
    )
    # Canonical crops as the pipeline builds them: float32 in [0, 1].
    crops = torch.full((1, 3, 64, 128), 0.5, dtype=torch.float32)

    pose_stage.run_pose(
        crops,
        obb,
        model,
        pose_stage.PoseConfig(),
        None,
        geometry=geometry,
    )

    assert backend.seen, "backend was never called"
    assert backend.seen[0].dtype == np.uint8
    assert int(backend.seen[0].max()) > 1


@pytest.mark.parametrize("value", [0.25, 0.5, 0.75, 1.0])
def test_cpu_and_cuda_entry_points_agree_on_scaling(value):
    """``predict_batch`` must reproduce the scaling ``predict_batch_cuda`` does.

    ``predict_batch_cuda`` converts device tensors with ``(c * 255).astype(uint8)``
    before delegating; the CPU path must land on the same pixel values, or the
    two devices silently disagree.
    """
    crop = np.full((4, 4, 3), value, dtype=np.float32)
    cuda_equivalent = (crop * 255).astype(np.uint8)
    np.testing.assert_array_equal(to_uint8_image(crop), cuda_equivalent)
