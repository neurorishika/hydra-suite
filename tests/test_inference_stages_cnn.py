from unittest.mock import MagicMock

import numpy as np
import torch

from hydra_suite.core.inference.config import CNNConfig
from hydra_suite.core.inference.result import OBBResult
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.cnn import CNNModel, run_cnn


def _cpu_rt():
    return RuntimeContext(
        cuda_mode=False,
        device="cpu",
        use_nvdec=False,
        default_runtime="cpu",
        tensor_on_cuda=False,
    )


def _obb(n: int) -> OBBResult:
    return OBBResult(
        frame_idx=0,
        centroids=np.zeros((n, 2), dtype=np.float32),
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.ones(n, dtype=np.float32) * 400.0,
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.ones(n, dtype=np.float32),
        corners=np.zeros((n, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(0, n),
    )


def _flat_model_backend(class_names: list[str]) -> MagicMock:
    backend = MagicMock()
    backend.metadata.class_names = [class_names]
    backend.metadata.input_size = (64, 64)
    n_classes = len(class_names)
    backend.predict_batch.side_effect = lambda crops: [
        [np.ones(n_classes) / n_classes] for _ in crops
    ]
    return backend


def _multihead_model_backend(factor_classes: list[list[str]]) -> MagicMock:
    backend = MagicMock()
    backend.metadata.class_names = factor_classes
    backend.metadata.input_size = (64, 64)
    backend.predict_batch.side_effect = lambda crops: [
        [np.ones(len(fc)) / len(fc) for fc in factor_classes] for _ in crops
    ]
    return backend


def test_run_cnn_flat_returns_one_factor_per_detection():
    config = CNNConfig(label="identity", model_path="/c.pt")
    backend = _flat_model_backend(["ant1", "ant2", "ant3"])
    model = CNNModel(
        backend=backend,
        input_size=(64, 64),
        factor_names=["identity"],
        factor_class_names=[["ant1", "ant2", "ant3"]],
    )
    crops = torch.zeros((2, 3, 64, 64))
    result = run_cnn(crops, _obb(2), model, config, _cpu_rt())
    assert result.label == "identity"
    assert len(result.predictions) == 2
    for pred in result.predictions:
        assert len(pred.factors) == 1
        assert pred.factors[0].factor_name == "identity"
        assert len(pred.factors[0].class_names) == 3
        assert pred.factors[0].raw_probabilities.shape == (3,)


def test_run_cnn_multihead_returns_k_factors():
    config = CNNConfig(label="behavior", model_path="/c.pt")
    backend = _multihead_model_backend([["a", "b"], ["x", "y", "z"]])
    model = CNNModel(
        backend=backend,
        input_size=(64, 64),
        factor_names=["color", "posture"],
        factor_class_names=[["a", "b"], ["x", "y", "z"]],
    )
    crops = torch.zeros((1, 3, 64, 64))
    result = run_cnn(crops, _obb(1), model, config, _cpu_rt())
    assert len(result.predictions[0].factors) == 2
    assert result.predictions[0].factors[0].factor_name == "color"
    assert result.predictions[0].factors[1].factor_name == "posture"
    assert result.predictions[0].factors[1].raw_probabilities.shape == (3,)


def test_run_cnn_empty_crops():
    config = CNNConfig(label="id", model_path="/c.pt")
    model = CNNModel(
        backend=MagicMock(),
        input_size=(64, 64),
        factor_names=["id"],
        factor_class_names=[["a", "b"]],
    )
    crops = torch.zeros((0, 3, 64, 64))
    result = run_cnn(crops, _obb(0), model, config, _cpu_rt())
    assert len(result.predictions) == 0


def test_run_cnn_raw_probabilities_not_calibrated():
    """Per spec: probabilities must be raw (pre-temperature) — calibration is
    tracking-time only inside IdentityEvidenceBuilder."""
    config = CNNConfig(label="id", model_path="/c.pt", calibration_temperature=0.5)
    raw_probs = np.array([0.7, 0.2, 0.1])
    backend = MagicMock()
    backend.metadata.class_names = [["a", "b", "c"]]
    backend.metadata.input_size = (64, 64)
    backend.predict_batch.return_value = [[raw_probs.copy()]]
    model = CNNModel(
        backend=backend,
        input_size=(64, 64),
        factor_names=["id"],
        factor_class_names=[["a", "b", "c"]],
    )
    crops = torch.zeros((1, 3, 64, 64))
    result = run_cnn(crops, _obb(1), model, config, _cpu_rt())
    np.testing.assert_array_almost_equal(
        result.predictions[0].factors[0].raw_probabilities, raw_probs
    )


def test_run_cnn_det_index_assigned_in_order():
    """Each prediction's det_index matches its position in the crop batch."""
    config = CNNConfig(label="id", model_path="/c.pt")
    backend = _flat_model_backend(["a", "b"])
    model = CNNModel(
        backend=backend,
        input_size=(64, 64),
        factor_names=["id"],
        factor_class_names=[["a", "b"]],
    )
    crops = torch.zeros((3, 3, 64, 64))
    result = run_cnn(crops, _obb(3), model, config, _cpu_rt())
    assert [p.det_index for p in result.predictions] == [0, 1, 2]
