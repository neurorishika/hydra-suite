"""Overlay layer value objects and per-source providers for DetectKit."""

from .layer import ColourPolicy, Emphasis, LabelMode, LayerStyle, OverlayLayer
from .providers import (
    PROVIDERS,
    FrameContext,
    GroundTruthProvider,
    OverlayProvider,
    PredictionProvider,
    StagedEscalationProvider,
    resolve_pending_level,
    resolve_source_render_state,
)

__all__ = [
    "ColourPolicy",
    "Emphasis",
    "LabelMode",
    "LayerStyle",
    "OverlayLayer",
    "PROVIDERS",
    "FrameContext",
    "GroundTruthProvider",
    "OverlayProvider",
    "PredictionProvider",
    "StagedEscalationProvider",
    "resolve_pending_level",
    "resolve_source_render_state",
]
