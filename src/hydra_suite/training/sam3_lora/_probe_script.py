"""Availability probe that runs *inside* the SAM3 sidecar conda env.

Run directly as a script file (``conda run -n <env> python <this file>``)
from ``availability._run_probe`` -- deliberately NOT via ``python -m``. Using
``-m hydra_suite....`` would first import ``hydra_suite`` and
``hydra_suite.training`` as packages, whose ``__init__.py`` chains eagerly
import heavy submodules (e.g. numba-jitted core code) that a bare sidecar env
built from the documented recipe has no reason to have installed -- that
would make the probe fail on an unrelated import and misreport the cause.
This module is therefore standalone: it declares its own package checklist
and imports nothing from ``hydra_suite``.

``TRAINING_PACKAGES`` here must be kept in sync with
``availability.TRAINING_PACKAGES`` (the host-side checklist used for install
hints); ``tests/test_sam3_lora_availability.py`` asserts they match.

Prints exactly one line of JSON to stdout and exits 0 regardless of outcome;
failures are reported in the JSON payload, not via a non-zero exit or a
traceback, so the host side has a clean single-line result to parse even
when a package is missing.
"""

from __future__ import annotations

import importlib
import json
import sys

TRAINING_PACKAGES = (
    "sam3",
    "torch",
    "torchmetrics",
    "scipy",
    "einops",
    "decord",
    "iopath",
)


def _probe() -> dict:
    imported = {}
    for package in TRAINING_PACKAGES:
        try:
            imported[package] = importlib.import_module(package)
        except Exception as exc:  # noqa: BLE001 - report, never crash the probe
            return {
                "ok": False,
                "missing": [{"package": package, "error": str(exc)}],
            }
    torch = imported["torch"]
    cuda_available = bool(torch.cuda.is_available())
    capability = list(torch.cuda.get_device_capability()) if cuda_available else None
    return {
        "ok": True,
        "missing": [],
        "cuda_available": cuda_available,
        "cuda_compute_capability": capability,
    }


def main() -> int:
    result = _probe()
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
