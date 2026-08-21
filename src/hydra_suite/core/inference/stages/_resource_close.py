"""Shared best-effort resource-closing helper for stage model wrappers.

``PoseModel``, ``CNNModel``, ``HeadTailModel``, and ``AprilTagModel`` are thin
dataclass wrappers around the real, closeable resource (``model.backend`` for
pose/CNN/head-tail, ``model.detector`` for AprilTag). The wrapper itself holds
no OS/process resources, but for the SLEAP service pose backend in
particular, ``model.backend.close()`` is what actually reaches
``shutdown_sleap_service()`` and terminates the subprocess -- so every
wrapper's ``close()`` must forward to it, not stay a no-op.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def close_backend_resource(resource: object) -> None:
    """Best-effort ``close()``/``release()`` on ``resource``, swallowing errors.

    Mirrors the underlying-resource lookup added for the interpolated-crop
    cleanup path: prefer ``release()`` (e.g. cv2 captures), else fall back to
    ``close()``. Errors are logged and swallowed -- cleanup must never raise
    on the way out of a pipeline run.
    """
    if resource is None:
        return
    try:
        if hasattr(resource, "release"):
            resource.release()
        elif hasattr(resource, "close"):
            resource.close()
    except Exception:
        logger.debug("Failed to close/release resource %r", resource, exc_info=True)
