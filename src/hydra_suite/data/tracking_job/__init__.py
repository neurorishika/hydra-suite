"""Portable tracking jobs: pack anywhere, run anywhere, sync back."""

from .manifest import (
    JOB_MANIFEST_FILENAME,
    SUPPORTED_JOB_VERSION,
    JobManifest,
    JobModel,
    JobVideo,
    TrackingJobError,
    validate_job_relpath,
)
from .outputs import (
    PullDestination,
    discover_outputs,
    map_outputs_to_origins,
    plan_pull,
)
from .references import (
    PlannedModel,
    copy_model_reference,
    external_key_for,
    write_registry_subset,
)

# NOTE: pack.py and verify.py are deliberately NOT imported here. pack.py
# imports core.tracking.session_policy, and core/tracking/__init__.py does
# `from .worker import TrackingEngineCore`, which pulls in cv2/torch at
# IMPORT time. Re-exporting pack.py's names from this package's __init__
# would make every `import hydra_suite.data.tracking_job` (and even a direct
# `from hydra_suite.data.tracking_job.pack import ...`, since the package
# __init__ always runs first) trigger that heavy import chain -- exactly the
# failure mode fix Q6 (see tests/helpers/tracking_job.py) exists to avoid at
# collection time. Import pack/verify from their own submodules directly.

__all__ = [
    "JOB_MANIFEST_FILENAME",
    "SUPPORTED_JOB_VERSION",
    "JobManifest",
    "JobModel",
    "JobVideo",
    "TrackingJobError",
    "validate_job_relpath",
    "PlannedModel",
    "copy_model_reference",
    "external_key_for",
    "write_registry_subset",
    "PullDestination",
    "discover_outputs",
    "map_outputs_to_origins",
    "plan_pull",
]
