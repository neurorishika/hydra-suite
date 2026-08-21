"""Shared best-effort resource-closing helper for stage model wrappers.

``PoseModel``, ``CNNModel``, ``HeadTailModel``, and ``AprilTagModel`` are thin
dataclass wrappers around the real, closeable resource (``model.backend`` for
pose/CNN/head-tail, ``model.detector`` for AprilTag). The wrapper itself holds
no OS/process resources, but for the SLEAP service pose backend in
particular, ``model.backend.close()`` is what actually reaches
``shutdown_sleap_service()`` and terminates the subprocess -- so every
wrapper's ``close()`` must forward to it, not stay a no-op.

Known, accepted tradeoff: closing the SLEAP service backend eagerly on every
stage ``close()`` means the service subprocess is torn down (and its ~8s
startup cost re-paid) on every preview click and every successive tracking
run that uses SLEAP pose, instead of staying warm across runs the way the
previous (leaking) no-op ``close()`` accidentally allowed. This is accepted
as the correct price of not leaking the subprocess -- do not try to
special-case SLEAP back into staying warm here; if warm-reuse is wanted
later, it should be an explicit, intentional caching layer above this
helper, not a side effect of a broken ``close()``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def close_backend_resource(resource: object) -> None:
    """Best-effort ``close()``/``release()`` on ``resource``, swallowing errors.

    Prefer ``close()`` -- the method every real backend/detector in this
    codebase (SLEAP service backend, CNN/head-tail/AprilTag backends) actually
    defines -- and fall back to ``release()`` only if ``close()`` isn't
    present. ``release()`` is checked second (not first) because it's an
    extremely common, semantically-unrelated method name (e.g.
    ``threading.Lock.release``); preferring it would risk silently calling
    the wrong method on some future resource that happens to expose both,
    reintroducing the leak this helper exists to close. Errors are logged
    and swallowed -- cleanup must never raise on the way out of a pipeline
    run, but a failed close (e.g. of the SLEAP service subprocess) is
    surfaced at ``warning`` so it isn't invisible at normal log levels.
    """
    if resource is None:
        return
    try:
        if hasattr(resource, "close"):
            resource.close()
        elif hasattr(resource, "release"):
            resource.release()
    except Exception:
        logger.warning("Failed to close/release resource %r", resource, exc_info=True)
