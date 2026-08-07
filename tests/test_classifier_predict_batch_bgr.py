import inspect

from hydra_suite.core.identity.classification import backend


def test_predict_batch_accepts_input_is_bgr():
    sig = inspect.signature(
        backend.ClassifierBackend.predict_batch
    )  # actual class name
    assert "input_is_bgr" in sig.parameters
