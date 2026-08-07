"""Error types for the Qt-free tracking session service.

Follows the ``core/individual/classification/errors.py`` precedent: the service
raises a concrete type on fatal failure so the caller (GUI or CLI) decides
presentation instead of the service reaching for a widget.
"""

from __future__ import annotations


class TrackingSessionError(Exception):
    """Fatal failure inside the post-tracking session pipeline."""
