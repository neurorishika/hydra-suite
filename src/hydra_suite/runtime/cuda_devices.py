"""Physical CUDA device enumeration for process-per-GPU fan-out.

Children are pinned with ``CUDA_VISIBLE_DEVICES=<uuid>``. UUIDs are used
because an ordinal in the child's mask names a PHYSICAL device regardless of
the parent's own mask, so an ordinal list computed under a parent mask would
be wrong. This module talks only to ``nvidia-smi`` (no torch/cupy import) so
it is cheap and safe on hosts without CUDA, where it simply returns ``[]``.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

_SMI_TIMEOUT_S = 5.0
_QUERY = "index,uuid,name,mig.mode.current"


@dataclass(frozen=True)
class CudaDevice:
    index: int
    uuid: str
    name: str


def _run_nvidia_smi(command: list[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=_SMI_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def list_cuda_devices(
    *, runner: Callable[[list[str]], Optional[str]] | None = None
) -> list[CudaDevice]:
    """Return every non-MIG physical GPU reported by nvidia-smi, index order."""
    run = runner or _run_nvidia_smi
    stdout = run(
        ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"]
    )
    if not stdout:
        return []
    devices: list[CudaDevice] = []
    for row in csv.reader(stdout.splitlines(), skipinitialspace=True):
        if len(row) != 4:
            continue
        index, uuid, name, mig = (item.strip() for item in row)
        if mig.lower() not in {"disabled", "n/a", "[n/a]", "not supported", ""}:
            continue
        try:
            devices.append(CudaDevice(index=int(index), uuid=uuid, name=name))
        except ValueError:
            continue
    devices.sort(key=lambda d: d.index)
    return devices


def parse_gpu_selectors(text: str) -> list[str]:
    """Split ``"0,2-4,GPU-abc"`` into ``["0","2","3","4","GPU-abc"]``."""
    out: list[str] = []
    for raw in str(text or "").split(","):
        token = raw.strip()
        if not token:
            continue
        if token.lower() == "auto":
            return ["auto"]
        lo, sep, hi = token.partition("-")
        if sep and lo.isdigit() and hi.isdigit():
            a, b = int(lo), int(hi)
            if b < a:
                raise ValueError(f"bad GPU range {token!r}")
            out.extend(str(i) for i in range(a, b + 1))
        else:
            out.append(token)
    return out


def resolve_gpu_selectors(
    selectors: Sequence[str], devices: Sequence[CudaDevice] | None = None
) -> list[CudaDevice]:
    """Map ordinals / UUID prefixes / ``auto`` onto physical devices.

    Raises ``ValueError`` when nothing is available, a selector is unknown or
    ambiguous, or the same physical device is named twice.
    """
    available = list(list_cuda_devices() if devices is None else devices)
    if not available:
        raise ValueError(
            "no CUDA devices visible to nvidia-smi; --gpus needs an NVIDIA host"
        )
    wanted = list(selectors)
    if wanted == ["auto"]:
        return available
    picked: list[CudaDevice] = []
    for selector in wanted:
        sel = str(selector).strip()
        if sel.isdigit():
            matches = [d for d in available if d.index == int(sel)]
        elif sel.upper().startswith("GPU-") and len(sel) > 4:
            matches = [d for d in available if d.uuid.upper().startswith(sel.upper())]
        else:
            raise ValueError(
                f"GPU selector {sel!r} is neither an ordinal nor a GPU- UUID prefix"
            )
        if len(matches) != 1:
            raise ValueError(
                f"GPU selector {sel!r} matched {len(matches)} devices "
                f"(available: {[d.index for d in available]})"
            )
        if matches[0] in picked:
            raise ValueError(f"GPU {matches[0].index} selected more than once")
        picked.append(matches[0])
    return picked
