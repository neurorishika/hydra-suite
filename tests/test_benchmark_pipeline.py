import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_pipeline",
    Path(__file__).resolve().parents[1] / "tools" / "benchmark_pipeline.py",
)
bp = importlib.util.module_from_spec(_SPEC)
sys.modules["benchmark_pipeline"] = bp
_SPEC.loader.exec_module(bp)


def test_run_pipeline_benchmark_times_each_tier(monkeypatch):
    built = []

    class _FakeRunner:
        def __init__(self, cfg, cache_dir=None, video_path=None, cache_only=False):
            built.append(cfg.runtime_tier)

        def run_realtime(self, frame, frame_idx=0, roi_mask=None):
            return object()

        def close(self):
            pass

    monkeypatch.setattr(bp, "InferenceRunner", _FakeRunner)
    monkeypatch.setattr(
        bp, "build_inference_config_from_params", lambda p: bp._Cfg(runtime_tier="cpu")
    )

    rows = bp.run_pipeline_benchmark(
        {"DETECTION_METHOD": "yolo_obb"},
        tiers=["cpu", "gpu"],
        iterations=2,
        warmup=1,
        frame_size=(32, 32),
        compile_timing=False,
    )
    assert {r["tier"] for r in rows} == {"cpu", "gpu"}
    assert all("mean_ms" in r for r in rows)
    assert set(built) == {"cpu", "gpu"}  # a runner per tier, tier overridden
