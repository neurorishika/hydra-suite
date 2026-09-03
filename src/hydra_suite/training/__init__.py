"""MAT training framework for role-aware YOLO workflows."""

from .contracts import (
    AugmentationProfile,
    CustomCNNParams,
    DatasetBuildResult,
    PublishPolicy,
    SourceDataset,
    SplitConfig,
    TinyHeadTailParams,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
    ValidationIssue,
    ValidationReport,
)

# `.service` is NOT imported eagerly: it pulls in hydra_suite.core, which
# needs numba, sklearn, cv2 and ultralytics. The SAM3 training sidecar runs
# `python -m hydra_suite.training.sam3_lora.cli` in a deliberately minimal
# env that has none of them -- and `python -m` imports this package first,
# so an eager import here aborted every sidecar run before training began.
# `contracts` above stays eager: it is pure dataclasses and enums.
_LAZY = {
    "RoleRunConfig": ".service",
    "TrainingOrchestrator": ".service",
    "TrainingSessionResult": ".service",
}


def __getattr__(name: str):
    """PEP 562 lazy re-export, mirroring hydra_suite/__init__.py."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "AugmentationProfile",
    "CustomCNNParams",
    "DatasetBuildResult",
    "PublishPolicy",
    "RoleRunConfig",
    "SourceDataset",
    "SplitConfig",
    "TinyHeadTailParams",
    "TrainingHyperParams",
    "TrainingOrchestrator",
    "TrainingRole",
    "TrainingRunSpec",
    "TrainingSessionResult",
    "ValidationIssue",
    "ValidationReport",
]
