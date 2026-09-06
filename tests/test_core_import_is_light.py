"""Importing a pure-geometry leaf under ``core`` must not drag in the world.

A SAM3 LoRA training run died at epoch 0 with ``ModuleNotFoundError: No
module named 'sklearn'`` -- not because it used sklearn, but because
per-epoch AP validation imported ``core.inference.semantic.calibration``,
which executed ``core/__init__``, which eagerly imported the tracking
worker, which reaches ``data.al`` -> ``filterkit.core`` -> ``sklearn``. The
slim ``sam3-lora`` sidecar environment has no sklearn, and correctly so.
"""

from __future__ import annotations

import subprocess
import sys

# Modules the SAM3 sidecar imports, and the heavy subtrees it must not pull.
_SIDECAR_IMPORTS = (
    "hydra_suite.core.inference.shape_prior",
    "hydra_suite.core.inference.match_geometry",
    "hydra_suite.core.inference.semantic.calibration",
    "hydra_suite.training.sam3_lora.detection_quality",
)
# NOT "sklearn" itself: coremltools imports sklearn on some dev machines, so
# it cannot discriminate. These hydra-side subtrees are the actual route --
# data.al -> filterkit.core -> sklearn -- and they are what must stay unloaded.
_FORBIDDEN_PREFIXES = (
    "hydra_suite.core.tracking",
    "hydra_suite.data.al",
    "hydra_suite.data.dataset_generation",
    "hydra_suite.filterkit",
)


def _leaked(module: str) -> list[str]:
    """Import *module* in a fresh interpreter; return forbidden modules loaded."""
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        "print('\\n'.join(sorted(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    return [
        name
        for name in out.splitlines()
        if any(name == p or name.startswith(p + ".") for p in _FORBIDDEN_PREFIXES)
    ]


def test_sidecar_imports_do_not_pull_the_tracking_stack() -> None:
    for module in _SIDECAR_IMPORTS:
        leaked = _leaked(module)
        assert not leaked, (
            f"importing {module} pulled in {leaked}. The SAM3 sidecar env has "
            "no sklearn; this is how a training run dies at epoch 0."
        )


def test_lazy_names_still_resolve() -> None:
    """Laziness must not remove the public API."""
    import hydra_suite.core as core

    for name in core.__all__:
        assert getattr(core, name) is not None
