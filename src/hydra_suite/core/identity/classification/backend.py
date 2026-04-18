"""Shared classifier backend: checkpoint parsing, arch dispatch, runtime selection,
ONNX lazy-derivation, preprocessing, and batched forward pass.

Consumers wrap ``ClassifierBackend`` and apply their own semantics
(head-tail label validation and angle conversion; CNN identity class lookup
and scoring modes). The backend itself is semantics-free.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ClassifierMetadata", "ClassifierBackend"]


@dataclass(frozen=True)
class ClassifierMetadata:
    """Declarative description of a classifier artifact.

    All fields are populated eagerly from the checkpoint or manifest without
    loading weights, so import dialogs can preview the factor structure
    cheaply.

    Attributes:
        arch: Backbone identifier. One of ``"tinyclassifier"``, a torchvision
            backbone name (``"resnet50"``, ``"convnext_tiny"``, …), ``"yolo"``
            (flat single ultralytics model), or ``"yolo_multihead"`` (bundle).
        input_size: Canonical ``(H, W)`` shape expected by the model.
        is_multihead: True when ``factor_names`` has length > 1.
        factor_names: Per-factor names. ``["flat"]`` when flat.
        class_names_per_factor: Per-factor class lists; length matches
            ``factor_names``.
        monochrome: Whether the model was trained with monochrome augmentation,
            affecting preprocessing normalization.
        source_path: Absolute path the artifact was loaded from (for display
            and ONNX-derivation peer location).
    """

    arch: str
    input_size: tuple[int, int]
    is_multihead: bool
    factor_names: list[str]
    class_names_per_factor: list[list[str]]
    monochrome: bool
    source_path: str


class ClassifierBackend:
    """Placeholder — concrete loaders are added in subsequent tasks."""

    def __init__(self, model_path: str, compute_runtime: str = "cpu") -> None:
        self._model_path = str(model_path)
        self._compute_runtime = str(compute_runtime)
        self._metadata: ClassifierMetadata | None = None
        raise NotImplementedError("ClassifierBackend loaders land in Task 1.3/1.4")
