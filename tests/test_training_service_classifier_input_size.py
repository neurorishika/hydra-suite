"""GAP 2 regression: a freshly trained YOLO-classify artifact must publish
its actual trained ``imgsz`` as ``input_size``, not the module's hardcoded
[224, 224] default.

``classifier_metadata_for_artifact`` accepts a ``fallback_input_size``
parameter for exactly this, but both publish call sites in
``training/service.py::_publish_training_artifacts`` omitted it -- the same
defect shape this branch has already fixed five times elsewhere (a parameter
accepted but never supplied). This test exercises the real call site
(``_publish_training_artifacts``), not just the reader
(``classifier_metadata_for_artifact``) in isolation, because a unit test of
the reader alone cannot catch a caller that never supplies the argument.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import hydra_suite.training.model_publish as mp
import hydra_suite.training.service as svc
from hydra_suite.training.contracts import (
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)


def _install_fake_ultralytics(monkeypatch) -> None:
    """Stub ``ultralytics.YOLO`` so classifier_metadata_for_artifact's .pt
    branch can run against a fake weights file without loading real weights.
    """

    class _FakeYOLO:
        def __init__(self, _path):
            self.names = {0: "queen", 1: "worker"}

    fake_module = types.SimpleNamespace(YOLO=_FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)


def test_publish_training_artifacts_stamps_trained_imgsz_not_hardcoded_224(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mp, "get_models_root", lambda: tmp_path)
    _install_fake_ultralytics(monkeypatch)

    src = tmp_path / "weights.pt"
    src.write_bytes(b"fake-weights")

    trained_imgsz = 320
    assert trained_imgsz != 224  # must differ from the old hardcoded fallback

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_YOLO,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path / "derived"),
        base_model="yolo26n-cls.pt",
        hyperparams=TrainingHyperParams(imgsz=trained_imgsz),
    )

    key, published_path = svc._publish_training_artifacts(
        spec=spec,
        artifact_paths=[str(src)],
        publish_metadata={},
        run_id="r1",
        dataset_fingerprint_value="fp",
    )

    assert key and published_path
    reg = mp.load_model_registry()
    assert reg["entries"][key]["input_size"] == [trained_imgsz, trained_imgsz]

    # Also verify the .v2meta.json sidecar TrackerKit actually reads at
    # inference time carries the same stamp, not just the registry mirror.
    sidecar_name = reg["entries"][key]["v2_sidecar"]
    sidecar = json.loads(
        (Path(published_path).parent / sidecar_name).read_text(encoding="utf-8")
    )
    assert sidecar["input_size"] == [trained_imgsz, trained_imgsz]
