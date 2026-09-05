"""Measure the real bf16 device peak of a SAM3 LoRA run, at N optimizer steps.

Runs the PRODUCTION training entry point (`cli.run_training`) against a real
prepared dataset and a real `spec.json`, stops it after a few optimizer steps
(peak VRAM is reached early and then stays flat -- the prior measurement was
steady across ~130 steps), and persists the observation as a
`MemoryMeasurement` in a `MemoryProfileStore`.

Why the production entry point rather than a hand-built step: the peak depends
on the loss, the matcher and the collated datapoints, not just the adapters. A
reimplemented step would measure a different program than the one admission
gates.

Why a store record and never a source constant: `preflight.py`'s
`_MEASURED_BF16_DEVICE_PEAK_BYTES` was once 29 GiB against a real 7.83 GiB and
excluded every card below ~32 GiB. A measurement that lives in the profile
store is dated, device-tagged and re-measurable; one baked into a constant is
a claim that silently rots the moment the adapter surface changes -- which is
exactly what happened when the surface grew from 206 modules to 312.

Must run INSIDE the SAM3 sidecar conda env (it imports `sam3`), with
`PYTHONPATH` pointed at the `src/` tree under test. CUDA only.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import sys
import time
from pathlib import Path


class _StopProbe(BaseException):
    """Not an Exception: the training loop must not be able to swallow it."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--optimizer-steps", type=int, default=5)
    parser.add_argument("--profile-store", required=True)
    parser.add_argument(
        "--force-scoring-head",
        action="store_true",
        help=(
            "Enable adapt_scoring_head and BYPASS the CLI scope refusal, to "
            "measure the full 314-module spike-parity surface. This "
            "configuration is NOT reachable in production; label it as such."
        ),
    )
    args = parser.parse_args(argv)

    import torch

    from hydra_suite.runtime.memory_profiles import (
        MemoryMeasurement,
        MemoryProfileStore,
        PressureSettings,
        ProfileIdentity,
    )
    from hydra_suite.runtime.resource_budget import AcceleratorKind
    from hydra_suite.training.sam3_lora import cli

    spec = cli._load_spec(Path(args.spec).expanduser().resolve())
    params = spec.sam3_params
    if args.force_scoring_head:
        params.adapt_scoring_head = True
        cli._lora_scope_refusal = lambda _params: None
        # The estimator has no coefficient for the scoring head, so the
        # trainable-count invariant would (correctly) refuse. Relax it for the
        # probe ONLY, and only in the direction of "more than budgeted".
        original = cli._validated_lora_trainables

        def _relaxed(model, *, adapted_modules, expected_parameters):
            return original(
                model,
                adapted_modules=adapted_modules,
                expected_parameters=expected_parameters
                + params.rank * 1_024,  # measured scoring-head coefficient
            )

        cli._validated_lora_trainables = _relaxed

    scopes = tuple(
        flag[len("adapt_") :]
        for flag in (
            "adapt_vision_encoder",
            "adapt_text_encoder",
            "adapt_geometry_encoder",
            "adapt_detr_encoder",
            "adapt_detr_decoder",
            "adapt_mask_decoder",
            "adapt_scoring_head",
        )
        if bool(getattr(params, flag, False))
    )

    steps = {"n": 0}
    original_step = torch.optim.AdamW.step

    def _counted_step(self, *a, **kw):
        out = original_step(self, *a, **kw)
        steps["n"] += 1
        if steps["n"] >= args.optimizer_steps:
            raise _StopProbe
        return out

    torch.optim.AdamW.step = _counted_step

    adapted: list[int] = []
    original_log = cli.emit_log

    def _log(message):
        match = re.search(r"Injected LoRA adapters into (\d+) Linear modules", message)
        if match:
            adapted.append(int(match.group(1)))
        return original_log(message)

    cli.emit_log = _log

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    stopped_early = False
    try:
        cli.run_training(spec, run_dir)
    except _StopProbe:
        stopped_early = True
    if not stopped_early:
        raise SystemExit(
            "training finished or refused before the probe's step budget; the "
            "peak below would not describe a steady-state step -- refusing to "
            "record it"
        )

    reserved = int(torch.cuda.max_memory_reserved())
    allocated = int(torch.cuda.max_memory_allocated())
    host = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

    identity = ProfileIdentity(
        operation="sam3_lora_train",
        model_identity="sam3",
        backend="torch",
        device_identity=torch.cuda.get_device_name(0),
        precision=str(params.mixed_precision),
        task="semantic_sam3",
        tiling_mode="sahi",
        adapter_scope="+".join(scopes) or "none",
        adapter_rank=int(params.rank),
    )
    settings = PressureSettings(
        input_width=1008, input_height=1008, batch_size=int(params.batch)
    )
    measurement = MemoryMeasurement(
        identity=identity,
        settings=settings,
        accelerator_kind=AcceleratorKind.CUDA,
        host_peak_bytes=host,
        accelerator_allocated_peak_bytes=allocated,
        accelerator_reserved_peak_bytes=reserved,
        observed_at_unix_ns=time.time_ns(),
    )
    store = MemoryProfileStore(Path(args.profile_store).expanduser().resolve())
    store.save(tuple(store.load()) + (measurement,))

    print(
        json.dumps(
            {
                "adapted_modules": adapted[-1] if adapted else None,
                "optimizer_steps": steps["n"],
                "seconds": round(time.time() - started, 1),
                "reserved_peak_bytes": reserved,
                "reserved_peak_gib": round(reserved / 1024**3, 2),
                "allocated_peak_gib": round(allocated / 1024**3, 2),
                "host_peak_gib": round(host / 1024**3, 2),
                "device": identity.device_identity,
                "adapter_scope": identity.adapter_scope,
                "rank": identity.adapter_rank,
                "batch": settings.batch_size,
                "production_reachable": not args.force_scoring_head,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
