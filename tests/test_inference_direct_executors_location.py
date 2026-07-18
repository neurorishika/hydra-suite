"""Guards the Phase A relocation: direct executors live in core/inference,
and core/inference no longer imports from core/detectors."""

import ast
from pathlib import Path

import hydra_suite.core.inference.direct_executors as de


def test_direct_executor_factories_importable_from_inference():
    assert hasattr(de, "create_direct_obb_executor")
    assert hasattr(de, "create_direct_detect_executor")
    for name in (
        "DirectONNXOBBExecutor",
        "DirectTensorRTOBBExecutor",
        "DirectPyTorchCUDAOBBExecutor",
        "DirectONNXDetectExecutor",
        "DirectTensorRTDetectExecutor",
    ):
        assert hasattr(de, name), name


def test_inference_package_does_not_import_core_detectors():
    inference_dir = Path(de.__file__).parent
    offenders = []
    for py in inference_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = ",".join(a.name for a in node.names)
            if mod and "core.detectors" in mod:
                offenders.append(f"{py.name}:{node.lineno} -> {mod}")
    assert not offenders, "core/inference must not import core/detectors: " + "; ".join(
        offenders
    )
