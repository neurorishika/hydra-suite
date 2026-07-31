from hydra_suite.core.inference.config import (
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
)
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.runtime.resolver import ResolvedBackend


def test_from_config_populates_resolved():
    cfg = InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
        ),
        runtime_tier="cpu",
    )
    ctx = RuntimeContext.from_config(cfg)
    assert isinstance(ctx.resolved, ResolvedBackend)
    assert ctx.resolved.device == "cpu"
    # device is "mps" on Apple Silicon, "cpu" elsewhere — both are non-CUDA
    # (see tests/test_runtime_context_from_tier.py for the same pattern).
    assert ctx.device in ("mps", "cpu")
