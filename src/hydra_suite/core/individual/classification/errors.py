"""Shared error types for classifier loading and inference.

Consumers (head-tail, CNN identity) and import dialogs raise these concrete
types rather than logging-and-continuing, so upstream code can distinguish
format problems from runtime failures from config gaps.
"""

from __future__ import annotations


class ClassifierError(Exception):
    """Base class for all classifier-backend errors."""


class ClassifierFormatError(ClassifierError):
    """Checkpoint, manifest, or registry entry is malformed or unsupported."""


class ClassifierRuntimeError(ClassifierError):
    """Inference-time failure: bad runtime, missing provider, device error."""


class ClassifierConfigError(ClassifierError):
    """Configuration required for this model is missing or invalid."""


class HeadTailFormatError(ClassifierFormatError):
    """Model does not satisfy head-tail consumer constraints."""
