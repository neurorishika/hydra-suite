"""Resolve a positive YOLO training batch BEFORE the training child launches.

Ultralytics already ships its own profiling autobatch, so this module does not
probe anything itself. It runs Ultralytics' resolution in a short, contained
child, reads the number the child writes to a file, and clamps it. The number
is Ultralytics' ESTIMATE -- it profiles a single step and extrapolates from a
linear fit -- not a measured peak and not a guarantee. The bounded OOM-retry
ladder in :mod:`hydra_suite.training.ultralytics_supervisor` is what actually
protects the run.

Why a resolution phase instead of leaving ``batch=-1`` in the launch command:

* ``PressureSettings`` is an accounting type with a ``>= 1`` invariant, and
  ``_command_with_pressure`` rewrites ``batch=`` from it, so ``-1`` silently
  became ``1`` before the first launch and Ultralytics' autobatch never ran;
* with ``-1`` left in the command the parent never learns the selected value,
  so it can neither size containment for it nor persist its provenance;
* Ultralytics only measures on CUDA -- on CPU/MPS it returns the default
  without profiling.

The child WRITES ``run_dir/batch_resolution.json``; it does not print a marker
line. The supervisor drops child output under pressure (it reports
``dropped_output_lines``/``output_error``), so a dropped line would silently
degrade a resolved 48 into a fallback 16 with no trace. A file is non-lossy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from hydra_suite.runtime.resource_budget import AcceleratorKind
from hydra_suite.training.contracts import TrainingHyperParams

#: The largest batch Ultralytics 8.4.x actually PROFILES rather than
#: extrapolates to. Anything above it is a linear-fit extrapolation, so we
#: refuse to launch on it and record the clamp in the provenance.
YOLO_MAX_AUTO_BATCH = 64

#: The artifact the child writes and the parent reads and then rewrites.
BATCH_RESOLUTION_FILENAME = "batch_resolution.json"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ResolutionCanceled(Exception):
    """The user cancelled during resolution; nothing may launch."""


def default_batch() -> int:
    """The explicit fallback: the shared cross-kit hyperparameter default."""

    return int(TrainingHyperParams().batch)


def batch_resolution_block(
    requested: int,
    resolved: int,
    provenance: str,
    degraded_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """The SAM3 ``batch_resolution`` schema, with honest neutral values.

    The SAM3 side fills ``measured_reserved_bytes``/``free_bytes`` from a real
    probe. Nothing here is measured by us, so those stay 0 rather than
    borrowing an authority this path does not have.
    """

    return {
        "requested": int(requested),
        "resolved": int(resolved),
        "provenance": str(provenance),
        "fingerprint": "",
        "degraded_reasons": [str(item) for item in degraded_reasons],
        "measured_reserved_bytes": 0,
        "free_bytes": 0,
        "resolved_at_unix_ns": time.time_ns(),
    }


def _dataset_label_profile(dataset_dir: Path) -> tuple[int, int, list[str]]:
    """Mirror ``DetectionTrainer.auto_batch``'s ``max_num_obj`` and size.

    Ultralytics computes ``max(len(label["cls"]) for label in labels) * 4``
    (the 4 is mosaic augmentation) and passes ``len(dataset)``. Passing 1 here
    would make the assigner's memory invisible and inflate the estimate.

    Returns ``(max_num_obj, dataset_size, degraded_reasons)``. A missing label
    root is REPORTED, not swallowed: without it the estimate ignores assigner
    memory and skews optimistic, which shows up later as extra OOM-ladder hits.
    """

    dataset_dir = Path(dataset_dir).expanduser()
    roots = [dataset_dir / "train" / "labels", dataset_dir / "labels" / "train"]
    labels = [root for root in roots if root.is_dir()]
    if not labels:
        return (
            0,
            0,
            [f"no train labels found under {dataset_dir}; max_num_obj is unknown"],
        )
    most = 0
    count = 0
    for path in sorted(labels[0].rglob("*.txt")):
        try:
            lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError:
            continue
        count += 1
        most = max(most, len(lines))
    if count == 0:
        return 0, 0, [f"the train label root {labels[0]} is empty"]
    return most * 4, count, []


def child_command(spec: Any, run_dir: Path) -> tuple[str, ...]:
    """The contained resolution child's argv."""

    params = spec.hyperparams
    return (
        sys.executable,
        "-m",
        "hydra_suite.training.yolo_autobatch",
        "--model",
        str(spec.base_model),
        "--dataset-dir",
        str(Path(spec.derived_dataset_dir).expanduser()),
        "--imgsz",
        str(int(params.imgsz)),
        "--device",
        # Only CUDA reaches the child, and `_run_ultralytics_once` pins
        # CUDA_VISIBLE_DEVICES to the selected UUID, so the child always sees
        # exactly one visible device at ordinal 0. `spec.device` must NOT be
        # forwarded: it holds an Ultralytics token ("auto", "0") and
        # `torch.device()` raises on both, which would kill the child on its
        # first line and silently degrade every run to `fallback`.
        "cuda:0",
        "--default-batch",
        str(default_batch()),
        "--out",
        str(Path(run_dir) / BATCH_RESOLUTION_FILENAME),
    )


def _read_child_report(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("resolved")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def normalize_cuda_device(device: Any) -> str:
    """Map Ultralytics' bare-ordinal device convention onto a torch device.

    Ultralytics writes GPUs as ``"0"`` / ``"0,1"``. `_accelerator` in
    :mod:`hydra_suite.training.ultralytics_supervisor` matches ``"mps"``,
    ``"auto"`` and anything starting with ``"cuda"``, so a bare ``"0"`` falls
    through to `AcceleratorKind.CPU`.

    KNOWN, PRE-EXISTING, DELIBERATELY NOT FIXED HERE: that misclassification
    affects containment for EVERY ``device: "0"`` YOLO run, not just this one
    -- such runs get host-only accounting and no CUDA UUID pin today. Widening
    `_accelerator` would be more correct, but it would newly subject runs that
    have always worked to accelerator admission gates, so it is escalated as a
    separate decision. This normalisation is scoped to the batch-resolution
    path ONLY: it changes what WE reason about, never the global
    classification and never what is passed through to Ultralytics, which
    expects its own convention in the launch command.

    Multi-GPU forms resolve against the FIRST device: Ultralytics' autobatch
    profiles one device, and spreading it across several would overstate
    capacity.
    """

    value = str(device or "auto").strip()
    first = value.split(",")[0].strip()
    if first.isdigit():
        return f"cuda:{int(first)}"
    return value


def child_degraded_reasons(run_dir: Path | None) -> list[str]:
    """Whatever the child flagged about the workload it sized, or nothing."""

    if run_dir is None:
        return []
    try:
        payload = json.loads(
            (Path(run_dir) / BATCH_RESOLUTION_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    reasons = payload.get("degraded_reasons") if isinstance(payload, dict) else None
    if not isinstance(reasons, list):
        return []
    return [str(item) for item in reasons]


def resolve_yolo_batch(
    spec: Any,
    run_dir: Path | None,
    *,
    accelerator_kind: AcceleratorKind,
    run_child: Callable[[Sequence[str], Any], dict],
    log_cb: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[int, str]:
    """Return a POSITIVE ``(batch, provenance)``; never propagate ``-1``.

    ``provenance`` is one of ``explicit``, ``ultralytics_autobatch``,
    ``ultralytics_autobatch_clamped``, ``default_non_cuda``, ``fallback``.
    Raises :class:`ResolutionCanceled` when the user cancelled -- the caller
    must then launch nothing and write nothing.
    """

    def log(message: str) -> None:
        if log_cb is not None:
            log_cb(message)

    def cancelled() -> bool:
        return should_cancel is not None and should_cancel()

    requested = int(spec.hyperparams.batch)
    if requested > 0:
        return requested, "explicit"

    fallback = default_batch()
    if cancelled():
        raise ResolutionCanceled("YOLO batch resolution canceled before it started.")

    if accelerator_kind is not AcceleratorKind.CUDA:
        log(
            "auto batch: Ultralytics does not measure batch size off CUDA, so "
            f"there is nothing to resolve here -- using batch={fallback}."
        )
        return fallback, "default_non_cuda"

    if run_dir is None:
        log(
            "auto batch: no run directory was supplied, so the resolution "
            f"child has nowhere to report -- using batch={fallback}."
        )
        return fallback, "fallback"

    report = Path(run_dir) / BATCH_RESOLUTION_FILENAME
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    # A stale report from a previous run in the same (exist_ok) run dir would
    # fake a fresh resolution when this run's child fails.
    report.unlink(missing_ok=True)

    command = child_command(spec, Path(run_dir))
    # The child must run under limits sized for a real positive batch, not for
    # `max(1, -1)`, and it must be classified as CUDA so it gets the UUID pin.
    from dataclasses import replace

    normalized = normalize_cuda_device(spec.device)
    if str(spec.device or "").strip().count(",") >= 1:
        log(
            f"auto batch: {spec.device} names several devices, but Ultralytics "
            f"profiles one -- sizing against {normalized} alone rather than "
            "overstating the capacity of the set."
        )
    child_spec = replace(
        spec,
        device=normalized,
        hyperparams=replace(spec.hyperparams, batch=fallback, epochs=1),
    )
    log("auto batch: asking Ultralytics to size the batch before launch.")
    result = dict(run_child(command, child_spec) or {})
    if result.get("canceled"):
        raise ResolutionCanceled("YOLO batch resolution canceled.")
    if cancelled():
        raise ResolutionCanceled("YOLO batch resolution canceled.")

    if not result.get("success"):
        log(
            "auto batch: the resolution child did not finish "
            f"({result.get('failure_kind') or 'unknown'}), so nothing was "
            f"resolved -- falling back to batch={fallback}."
        )
        return fallback, "fallback"

    reported = _read_child_report(report)
    if reported is None:
        log(
            "auto batch: the resolution child left no readable report, so "
            f"nothing was resolved -- falling back to batch={fallback}."
        )
        return fallback, "fallback"

    resolved = min(YOLO_MAX_AUTO_BATCH, max(1, reported))
    if resolved != reported:
        log(
            f"auto batch: Ultralytics reported {reported}, which is outside "
            f"[1, {YOLO_MAX_AUTO_BATCH}] -- the range it actually profiles "
            f"rather than extrapolates. Clamped to {resolved}. This is "
            "Ultralytics' estimate, not a measured peak; the OOM-retry ladder "
            "protects the run."
        )
        return resolved, "ultralytics_autobatch_clamped"
    log(
        f"auto batch: Ultralytics estimated batch={resolved}. This is its "
        "estimate from a profiled step and a linear fit, not a measured peak "
        "and not a guarantee; the OOM-retry ladder protects the run."
    )
    return resolved, "ultralytics_autobatch"


# ---------------------------------------------------------------------------
# The contained resolution child.
# ---------------------------------------------------------------------------


def _ultralytics_estimate(
    *, model: str, imgsz: int, device: str, max_num_obj: int, dataset_size: int
) -> int:
    """Call Ultralytics' own autobatch. Imported lazily: torch is heavy."""

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.autobatch import check_train_batch_size

    loaded = YOLO(model)
    module = loaded.model.to(torch.device(device))
    module.train()
    return int(
        check_train_batch_size(
            model=module,
            imgsz=int(imgsz),
            amp=True,
            batch=-1,
            max_num_obj=int(max_num_obj),
            dataset_size=int(dataset_size),
        )
    )


def main(argv: Sequence[str] | None = None, *, estimate=None) -> int:
    """Child entry point: resolve, WRITE the report, exit."""

    parser = argparse.ArgumentParser(description="Resolve a YOLO training batch.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--default-batch", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    max_num_obj, dataset_size, degraded = _dataset_label_profile(Path(args.dataset_dir))
    resolver = estimate if estimate is not None else _ultralytics_estimate
    resolved = int(
        resolver(
            model=args.model,
            imgsz=args.imgsz,
            device=args.device,
            max_num_obj=max_num_obj,
            dataset_size=dataset_size,
        )
    )
    block = batch_resolution_block(-1, resolved, "ultralytics_autobatch")
    block["degraded_reasons"] = degraded
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(block, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
