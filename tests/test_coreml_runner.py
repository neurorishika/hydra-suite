import numpy as np
import pytest


def test_coreml_runner_signature_and_import():
    from hydra_suite.core.identity.pose.runtime.coreml_runner import CoreMLRunner

    assert hasattr(CoreMLRunner, "run")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("coremltools") is None,
    reason="coremltools not installed",
)
def test_coreml_runner_predicts(tmp_path):
    pytest.importorskip("coremltools")

    from hydra_suite.core.identity.pose.runtime.coreml_runner import CoreMLRunner
    from hydra_suite.core.identity.pose.vitpose.export import export_coreml
    from hydra_suite.core.identity.pose.vitpose.vitpose import build_vitpose

    model = build_vitpose("S", "classic", num_keypoints=3).eval()
    path = tmp_path / "m.mlpackage"
    export_coreml(model, path)
    runner = CoreMLRunner(path)
    out = runner.run(np.zeros((1, 3, 256, 192), np.float32))
    arr = next(iter(out.values())) if isinstance(out, dict) else out
    assert np.asarray(arr).shape[0] == 1
