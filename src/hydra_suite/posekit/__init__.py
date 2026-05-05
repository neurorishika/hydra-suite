"""Pose labeling subpackage."""


def main(*args, **kwargs):
    """Lazy entry point for the PoseKit labeler UI."""
    from .gui.main import main as _main  # deferred to avoid polluting sys.modules

    return _main(*args, **kwargs)


__all__ = ["main"]
