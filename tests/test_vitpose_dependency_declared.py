import tomllib
from pathlib import Path


def test_huggingface_hub_is_declared():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    names = [d.lower().replace("_", "-") for d in deps]
    assert any(n.startswith("huggingface-hub") for n in names), (
        "huggingface_hub is imported at module top level in "
        "posekit/core/vitpose_checkpoints.py but not declared in pyproject.toml"
    )
