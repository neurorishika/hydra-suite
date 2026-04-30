from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


@contextmanager
def _patched_modules(stubs: Dict[str, Any] | None):
    if not stubs:
        yield
        return

    sentinel = object()
    original = {}
    # Snapshot the full set of loaded modules so we can detect (and evict)
    # any *project* modules that were transitively imported while the stubs
    # were active.  Without this, modules like
    # ``hydra_suite.utils.image_processing`` would permanently bind the
    # stub ``cv2`` into their module-level attribute, poisoning later tests
    # that import the real package.
    modules_before = set(sys.modules.keys())
    try:
        for name, stub in stubs.items():
            original[name] = sys.modules.get(name, sentinel)
            sys.modules[name] = stub
        yield
    finally:
        for name, old in original.items():
            if old is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        # Evict project modules that were loaded during the stub window so they
        # don't retain references to the now-removed stubs.  Only evict modules
        # from our own source tree — third-party and stdlib modules must not be
        # disturbed (e.g. torch cannot survive an import-evict-reimport cycle).
        _PROJECT_PREFIXES = ("hydra_suite.",)
        newly_loaded = set(sys.modules.keys()) - modules_before - set(original.keys())
        for name in newly_loaded:
            if any(name.startswith(pfx) for pfx in _PROJECT_PREFIXES):
                sys.modules.pop(name, None)


def load_src_module(
    src_relative_path: str,
    module_name: str,
    stubs: Dict[str, Any] | None = None,
):
    """
    Load a module from `src/` by file path, bypassing package side effects.
    """
    module_path = SRC_ROOT / src_relative_path
    if not module_path.exists():
        raise FileNotFoundError(f"Module path does not exist: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create spec for module: {module_path}")

    module = importlib.util.module_from_spec(spec)

    with _patched_modules(stubs):
        spec.loader.exec_module(module)

    return module


def make_cv2_stub() -> types.SimpleNamespace:
    """
    Minimal cv2 stub used by tests in environments without OpenCV.
    """
    import numpy as np

    def _identity(*args, **kwargs):
        return args[0]

    def _copy_make_border(img, top, bottom, left, right, border_type, value=(0, 0, 0)):
        return np.pad(
            img,
            (
                ((top, bottom), (left, right), (0, 0))
                if img.ndim == 3
                else ((top, bottom), (left, right))
            ),
            mode="constant",
            constant_values=0,
        )

    return types.SimpleNamespace(
        VideoCapture=object,
        RETR_EXTERNAL=0,
        CHAIN_APPROX_SIMPLE=0,
        MORPH_ELLIPSE=0,
        MORPH_OPEN=1,
        BORDER_CONSTANT=0,
        COLOR_BGR2RGB=0,
        COLOR_BGR2GRAY=1,
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_POS_FRAMES=8,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        INTER_AREA=1,
        INTER_NEAREST=2,
        INTER_LINEAR=3,
        convertScaleAbs=lambda img, alpha=1.0, beta=0: np.clip(
            img.astype(np.float32) * alpha + beta, 0, 255
        ).astype(np.uint8),
        LUT=lambda img, lut: lut[img],
        getStructuringElement=lambda _shape, k: np.ones(k, dtype=np.uint8),
        erode=lambda img, _kernel, iterations=1: img,
        morphologyEx=lambda img, _op, _kernel: img,
        findContours=lambda mask, _mode, _method: (mask, None),
        contourArea=lambda c: float(c["area"]),
        fitEllipse=lambda c: c["ellipse"],
        boundingRect=lambda c: c["rect"],
        copyMakeBorder=_copy_make_border,
        cvtColor=_identity,
        resize=lambda frame, _dsize, fx=1.0, fy=1.0, interpolation=0: frame,
    )
