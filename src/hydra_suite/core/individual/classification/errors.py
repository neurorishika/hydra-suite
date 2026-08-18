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


class CalibrationRequiredError(ClassifierConfigError):
    """A ``unique_identifier`` CNN model is uncalibrated but calibration is
    mandatory for this config (``IDENTITY_CALIBRATION_REQUIRED``).

    Raised by ``build_inference_config_from_params``'s mandatory-calibration
    gate. Callers that want to proceed anyway should set the
    ``IDENTITY_CALIBRATION_OVERRIDE`` param, which downgrades this to a
    logged warning instead of raising.
    """


class PoseModelUnresolvedError(ClassifierConfigError):
    """Pose extraction is enabled but no usable pose model path resolved.

    Raised by ``build_inference_config_from_params`` instead of quietly
    returning ``pose=None``: a dropped stage is also dropped from the cache-key
    set, so ``InferenceRunner.caches_all_valid()`` would accept a cache written
    without pose and the run would skip pose inference entirely.
    """
