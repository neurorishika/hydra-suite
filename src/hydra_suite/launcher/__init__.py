"""HYDRA Suite launcher — central hub for all kit applications.

Kept import-light: ``hydra doctor`` routes through :mod:`.cli` and must not
pull in Qt.
"""


def main(*args, **kwargs):
    from .app import main as _main

    return _main(*args, **kwargs)


__all__ = ["main"]
