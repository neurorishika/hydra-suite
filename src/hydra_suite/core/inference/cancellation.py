"""Explicit cooperative-cancellation control flow for inference."""


class InferenceCancelled(RuntimeError):
    """Raised when an admitted inference unit stops before it is complete."""
