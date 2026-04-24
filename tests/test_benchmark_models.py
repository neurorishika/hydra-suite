from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from tests.test_detectors_engine import _load_engine_module


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tools" / "benchmark_models.py"
    module_name = "benchmark_models_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_artifact_resolution_matches_realtime_obb_batch_clamp(
    monkeypatch,
    tmp_path: Path,
) -> None:
    engine_mod = _load_engine_module()
    monkeypatch.setitem(
        sys.modules,
        "hydra_suite.core.detectors",
        types.SimpleNamespace(YOLOOBBDetector=engine_mod.YOLOOBBDetector),
    )
    bench_mod = _load_benchmark_module()

    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")

    artifact_path = bench_mod._resolve_detector_runtime_artifact_path(
        str(model_path),
        "tensorrt",
        "obb",
        batch_size=8,
        tracking_realtime_mode=True,
    )

    assert artifact_path is not None
    assert artifact_path.name == "best_b1.engine"


def test_benchmark_artifact_resolution_matches_realtime_detect_batch_clamp(
    monkeypatch,
    tmp_path: Path,
) -> None:
    engine_mod = _load_engine_module()
    monkeypatch.setitem(
        sys.modules,
        "hydra_suite.core.detectors",
        types.SimpleNamespace(YOLOOBBDetector=engine_mod.YOLOOBBDetector),
    )
    bench_mod = _load_benchmark_module()

    model_path = tmp_path / "detect.pt"
    model_path.write_bytes(b"model")

    artifact_path = bench_mod._resolve_detector_runtime_artifact_path(
        str(model_path),
        "onnx_cuda",
        "detect",
        batch_size=8,
        tracking_realtime_mode=True,
    )

    assert artifact_path is not None
    assert artifact_path.name == "detect_detect_rawheadv1_b1.onnx"
