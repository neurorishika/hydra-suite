"""One parser for the device-id spellings a training spec may carry.

Ultralytics writes GPUs as bare ordinals (``"0"``, ``"0,1"``); torch and the
SAM3 preflight write ``"cuda"`` / ``"cuda:N"``. The SAME plan schema feeds both
a YOLO role and a SAM3 role, so both paths must agree on what ``"0"`` means.
Before this module the YOLO supervisor understood bare ordinals and the SAM3
preflight refused them, and every ``device: "0"`` SAM3 run died at preflight
reporting "No CUDA device is available" on boxes with tens of GiB free.

This module holds nothing but string handling: it must import no other
training module (both `yolo_autobatch` and `sam3_lora.preflight` import it, so
anything heavier would be a cycle).
"""

from __future__ import annotations

from typing import Any

#: The message a caller shows when it rejects a device spelling. Naming the
#: accepted forms is the point: the old wording ("No CUDA device is
#: available") described a hardware fact that was not true and sent users
#: looking at their GPU instead of their device string.
CUDA_DEVICE_FORMS = "'auto', 'cuda', 'cuda:N', or a device ordinal such as '0'"

#: WARNING PERIOD (temporary), shared by EVERY supervisor that classifies a
#: device string. Both `hydra_suite.training.ultralytics_supervisor._accelerator`
#: and `hydra_suite.detectkit.sidecars.supervisor._accelerator_for` now
#: recognise Ultralytics' bare-ordinal GPU convention ("0", "0,1") as CUDA.
#: Those runs were classified CPU before, so they never faced the accelerator
#: admission gate. While this flag is True, a bare-ordinal run that the
#: accelerator gate would REFUSE is instead admitted with a loud warning and
#: the host-only accounting it had before the widening. Nothing else is
#: downgraded: a run the PRE-CHANGE evaluation would also have refused (host,
#: lease, dataset, prelaunch identity) is still refused.
#:
#: It lives HERE, next to the parser whose widening created the need, so that
#: `grep BARE_ORDINAL_ACCELERATOR_GATE_WARNING_PERIOD` reveals every site it
#: governs and ending the period stays a ONE-LINE change for the whole
#: codebase rather than two flags to find.
#:
#: TO END THE WARNING PERIOD: set this to False (then delete this constant and
#: the downgrade branches in `_run_ultralytics_once` and in
#: `ProtectedOperation.run`).
BARE_ORDINAL_ACCELERATOR_GATE_WARNING_PERIOD = True


def normalize_cuda_device(device: Any) -> str:
    """Map Ultralytics' bare-ordinal device convention onto a torch device.

    Ultralytics writes GPUs as ``"0"`` / ``"0,1"``. `_accelerator` in
    :mod:`hydra_suite.training.ultralytics_supervisor` matches ``"mps"``,
    ``"auto"`` and anything starting with ``"cuda"``, so a bare ``"0"`` falls
    through to `AcceleratorKind.CPU`.

    That misclassification -- host-only accounting and no CUDA UUID pin for
    every ``device: "0"`` run -- is now fixed: `_accelerator` recognises bare
    ordinals through `is_bare_ordinal_device` and resolves them with this same
    helper. Because widening the classification newly subjects always-worked
    runs to the accelerator admission gate, that gate is downgraded to a
    warning for exactly those runs while
    ``BARE_ORDINAL_ACCELERATOR_GATE_WARNING_PERIOD`` (defined below) is set.

    Normalisation changes only what WE reason about; it never mutates
    ``spec.device`` and never reaches the launch command, which must keep
    Ultralytics' own convention.

    Multi-GPU forms resolve against the FIRST device: Ultralytics' autobatch
    profiles one device, and spreading it across several would overstate
    capacity. SAM3 pins a single physical device by UUID and so does the same;
    both callers announce the narrowing rather than performing it silently.
    """

    value = str(device or "auto").strip()
    first = value.split(",")[0].strip()
    if first.isdigit():
        return f"cuda:{int(first)}"
    return value


def is_bare_ordinal_device(device: Any) -> bool:
    """True for Ultralytics' bare-ordinal GPU convention (``"0"``, ``"0,1"``).

    ``"cuda:0"``, ``"cpu"``, ``"mps"`` and ``"auto"`` are all False: those were
    already classified correctly before bare ordinals were recognised, so they
    are NOT part of the warning period.
    """

    first = str(device or "").strip().split(",")[0].strip()
    return first.isdigit()


def names_several_devices(device: Any) -> bool:
    """True when a bare-ordinal string names more than one GPU (``"0,1"``)."""

    return is_bare_ordinal_device(device) and "," in str(device or "")
