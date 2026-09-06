"""SAM3 LoRA training entry point -- runs INSIDE the ``hydra-sam3`` sidecar
conda env, launched by ``train.py`` as
``python -m hydra_suite.training.sam3_lora.cli --spec ... --run-dir ...``.

This module (and only this module, in this package) may import ``sam3`` and
``torch`` -- but only inside function bodies, never at module scope, so a
syntax check or a stray import elsewhere in the package never requires
``sam3`` to be installed.

Progress and logging cross the process boundary as single-line JSON records
via ``protocol.emit_log``/``emit_progress``, printed to stdout with a
sentinel prefix; `train.py` parses them back out on the other end.

Cancellation is NOT plumbed into this process: the launcher cancels a run by
`terminate()`/`kill()`-ing this whole process, so there is no
`should_cancel` callback here. If this process is killed mid-epoch, it exits
without having written `adapters.pt`, which the launcher already treats as
a failed/canceled run -- see `train.py`'s artifact-existence check.

Checkpoint selection is always the LAST epoch's weights, never the epoch
with the best validation loss. The spike's own val-loss-vs-AP comparison is
not usable evidence either way: it came from a fold whose val split was
byte-identical to its train split, so the reported anti-correlation reflects
that overlap, not a real relationship between val loss and held-out AP.
There is no evidence for or against best-checkpoint selection here -- last-
epoch is kept because it is simple and matches the spike's own practice, not
because of the anti-correlation claim. Do not add best-checkpoint selection
or early stopping on val loss without first re-measuring on a fold with a
genuinely disjoint val split.

That re-measurement HAS since been done (2026-09-06, a real held-out fold, a
paired frame bootstrap over 16 frames), and it did not rescue val loss: every
per-query validation signal -- val_loss_mean, loss_ce, loss_bbox, loss_giou,
loss_mask, loss_dice, presence_loss -- ANTI-correlates with held-out AP
(Spearman -0.4 to -1.0). Selecting the minimum-loss checkpoint would be worse
than the always-last-epoch default the paragraph above describes. The ladder
spread itself is real (epoch_003 - FINAL = +0.0244 AP, Bonferroni CI
[+0.0079, +0.0437]; P(best) 0.991 vs 0.000), so there IS something to select
on -- just not the loss.

AP is therefore now ALSO recorded per epoch, in the same `val_series.jsonl`
row (see `detection_quality` and `ap_cadence`). It still selects nothing.
Read `detection_quality`'s module docstring before treating it as a rule: the
evidence is one run, one seed, one corpus, seed variance is unmeasured, and
the training-time number is tile-space and NMS-free, so it is not even on the
same scale as the study's. It is a within-run trend, recorded as evidence.

The training set is built and checked for emptiness BEFORE any `sam3` model
is loaded: an empty dataloader must exit nonzero, never silently train
nothing and exit 0 -- a zero-initialised LoRA `lora_B` makes an untrained
adapter a mathematical no-op, so a fake-success run would publish a
"finetuned" checkpoint byte-identical to stock SAM3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hydra_suite.training.contracts import Sam3LoraParams, sam3_prompt_text_error

from .artifacts import write_completion_marker
from .dataloader import (
    batch_count,
    build_descriptors,
    collate_batches,
    collate_epoch_batches,
    grouped_batch_count,
    query_count,
    scale_group_summary,
    try_build_descriptors,
    write_sam3_scale_grouping_stamp,
)
from .lora import (
    SUBMODULE_PATHS,
    SUBMODULE_PREFIXES,
    adapter_state_dict,
    inject_adapters,
    lora_config_from_params,
)
from .perflib_compat import install_grad_safe_addmm_act
from .protocol import emit_log, emit_progress
from .sizing import LORA_PARAMS_PER_RANK, expected_lora_trainable_params


class _SidecarSpec:
    """Minimal stand-in for `TrainingRunSpec` inside the sidecar.

    The launcher serialises the FULL spec to `spec.json` via
    `TrainingRunSpec.to_dict()`; this only reconstructs the fields the
    training loop below actually reads (`seed`, `derived_dataset_dir`,
    `sam3_params`), via the real `Sam3LoraParams` dataclass rather than a
    duplicated schema, so new params fields need no change here.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.seed: int = int(data.get("seed", 42))
        self.derived_dataset_dir: str = data["derived_dataset_dir"]
        self.device: str = str(data.get("device", "cuda"))
        sam3_data = data.get("sam3_params")
        self.sam3_params = Sam3LoraParams(**sam3_data) if sam3_data else None


def _load_spec(spec_path: Path) -> _SidecarSpec:
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    return _SidecarSpec(data)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_with_warmup(warmup_steps: int, total_steps: int):
    def _fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        span = max(1, total_steps - warmup_steps)
        progress = min(max(float(step - warmup_steps) / float(span), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _fn


def _forward_batch(
    batch: Any,
    model: Any,
    device: Any,
    *,
    copy_to_device: Any | None = None,
) -> tuple[Any, list[Any], Any]:
    """Move one collated batch to ``device``, convert targets, then forward.

    Meta's collator constructs a CPU ``BatchedDatapoint``. Its trainer uses
    ``copy_data_to_device`` on that unwrapped object before both
    ``back_convert`` and model forward; keep this sidecar on the same seam so
    every nested tensor (images, boxes, masks, and validity arrays) moves
    together.
    """
    if copy_to_device is None:
        from sam3.model.utils.misc import copy_data_to_device

        copy_to_device = copy_data_to_device

    model_input = batch["input"] if isinstance(batch, dict) else batch
    model_input = copy_to_device(model_input, device, non_blocking=True)
    targets = [model.back_convert(target) for target in model_input.find_targets]
    outputs = model(model_input)
    return model_input, targets, outputs


def _build_loss_wrapper(
    loss_wrapper_type: Any,
    *,
    loss_fns_find: list[Any],
    matcher: Any,
    o2m_matcher: Any,
) -> Any:
    """Build Meta's loss without distributed collectives in this sidecar."""
    return loss_wrapper_type(
        loss_fns_find=loss_fns_find,
        normalization="local",
        matcher=matcher,
        o2m_weight=2.0,
        o2m_matcher=o2m_matcher,
        use_o2m_matcher_on_o2m_aux=False,
        loss_fn_semantic_seg=None,
    )


def _attach_matcher_indices(outputs: Any, targets: list[Any], matcher: Any) -> None:
    """Attach matcher results when the model is in validation/eval mode.

    ``Sam3Image`` does this itself while ``model.training`` is true. Evaluation
    deliberately disables training behavior, so its SAM3Output dictionaries
    need the same indices before ``Sam3LossWrapper`` can score them.
    """
    for stage_outputs, target in zip(outputs.output, targets):
        for output in stage_outputs:
            output["indices"] = matcher(output, target)
            for auxiliary in output.get("aux_outputs", []):
                auxiliary["indices"] = matcher(auxiliary, target)


def _core_loss(loss_result: Any) -> Any:
    """Extract Meta's ``CORE_LOSS_KEY`` without importing SAM3 at module load."""
    return loss_result["core_loss"] if isinstance(loss_result, dict) else loss_result


def _validate_adapter_state(adapters: Any, torch_module: Any) -> None:
    """Reject corrupt, incomplete, non-finite, and mathematical no-op adapters."""

    if not isinstance(adapters, dict) or not adapters:
        raise ValueError("SAM3 adapter state must be a non-empty mapping")
    paths: dict[str, set[str]] = {}
    tensors: dict[str, Any] = {}
    for key, tensor in adapters.items():
        if not isinstance(key, str) or not torch_module.is_tensor(tensor):
            raise ValueError("SAM3 adapter state contains a non-tensor entry")
        path, separator, suffix = key.rpartition(".")
        if not separator or suffix not in {"lora_A", "lora_B"}:
            raise ValueError(f"unexpected SAM3 adapter key {key!r}")
        if tensor.ndim != 2 or tensor.numel() == 0:
            raise ValueError(f"SAM3 adapter tensor {key!r} must be non-empty 2-D")
        if not bool(torch_module.isfinite(tensor).all().item()):
            raise ValueError(f"SAM3 adapter tensor {key!r} contains non-finite values")
        paths.setdefault(path, set()).add(suffix)
        tensors[key] = tensor
    if any(suffixes != {"lora_A", "lora_B"} for suffixes in paths.values()):
        raise ValueError("SAM3 adapter state has an incomplete LoRA A/B pair")
    for path in paths:
        matrix_a = tensors[f"{path}.lora_A"]
        matrix_b = tensors[f"{path}.lora_B"]
        if matrix_a.shape[0] != matrix_b.shape[1]:
            raise ValueError(f"SAM3 adapter pair {path!r} has incompatible rank")
    has_nonzero_delta = False
    max_delta_elements = 1_048_576
    for path in paths:
        matrix_a = tensors[f"{path}.lora_A"].float()
        matrix_b = tensors[f"{path}.lora_B"]
        if not bool(torch_module.count_nonzero(matrix_a).item()) or not bool(
            torch_module.count_nonzero(matrix_b).item()
        ):
            continue
        rows_per_chunk = max(1, max_delta_elements // max(1, matrix_a.shape[1]))
        for start in range(0, matrix_b.shape[0], rows_per_chunk):
            delta = torch_module.matmul(
                matrix_b[start : start + rows_per_chunk].float(), matrix_a
            )
            if bool(torch_module.count_nonzero(delta).item()):
                has_nonzero_delta = True
                break
        if has_nonzero_delta:
            break
    if not has_nonzero_delta:
        raise ValueError(
            "SAM3 adapter is a mathematical no-op (all LoRA deltas are zero)"
        )


def _write_validated_adapter_artifact(
    adapters: Any, artifact_path: Path, torch_module: Any
) -> None:
    """Serialize, reload, validate, and atomically promote one adapter state."""

    _validate_adapter_state(adapters, torch_module)
    temporary = artifact_path.with_name(
        f".{artifact_path.name}.{os.getpid()}.validated.tmp"
    )
    try:
        with temporary.open("wb") as artifact_file:
            torch_module.save(adapters, artifact_file)
            artifact_file.flush()
            os.fsync(artifact_file.fileno())
        reloaded = torch_module.load(temporary, map_location="cpu", weights_only=True)
        _validate_adapter_state(reloaded, torch_module)
        os.replace(temporary, artifact_path)
        directory_fd = os.open(artifact_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        write_completion_marker(artifact_path)
    finally:
        temporary.unlink(missing_ok=True)


# Interrupted runs used to leave nothing at all: `adapters.pt` is written only
# after the final epoch, so a kill at hour six discarded every step. Epoch
# checkpoints are salvage -- `adapters.pt` remains the ONLY completion signal
# (the launcher treats its presence as "the run finished"), so these live in a
# subdirectory under their own names and can never be mistaken for it.
EPOCH_CHECKPOINT_DIRNAME = "checkpoints"

# Retention is a BUDGET, not a count.
#
# This used to be `KEEP_EPOCH_CHECKPOINTS = 3` plus a sliding window. That
# constant's stated rationale is sound and is preserved here: epoch counts are
# user-supplied, and disk exhaustion mid-run would destroy the very artifact
# this feature exists to preserve (a 200-epoch run on a full disk is a real
# failure mode). What was wrong was the MECHANISM, not the concern -- a
# newest-N window ALWAYS destroys the earliest epochs, which are exactly the
# ones a stall analysis needs. A 2026-09-06 run stopped learning after epoch 2
# and `epoch_001`/`epoch_002` had already been pruned, unmeasured.
#
# So: retain everything the disk can afford, measured at runtime (free space
# from `shutil.disk_usage`, adapter size from the files themselves -- never a
# hardcoded byte figure), and when the budget binds, thin the MIDDLE while
# always keeping the first and last checkpoint, and say so loudly.
CHECKPOINT_BUDGET_FRACTION_OF_FREE = 0.25


def checkpoint_budget_bytes(
    directory: Path, *, free_bytes: int | None = None, log: Any = emit_log
) -> int | None:
    """Measured retention budget for *directory*, in bytes.

    Derived, never hardcoded: a policy fraction of the space this run could
    actually use -- currently free space plus whatever the existing epoch
    checkpoints already occupy (they are reclaimable).

    Fails OPEN, and does so for the WHOLE measurement: any `OSError` from
    `disk_usage` OR from stat-ing the existing checkpoints returns `None`
    (retain everything, prune nothing) and says why. Two reasons this is not
    just belt-and-braces. First, a failed measurement must never masquerade
    as a binding budget -- deleting evidence because a stat raced with an
    unlink is the exact failure class this replaced. Second, this runs inside
    `_write_epoch_checkpoint`, so an escaping `OSError` would propagate into
    `run_training` and KILL the run; the count-based pruner it replaced made
    no stat calls at all and could not do that. Retention is best-effort
    housekeeping and must never be able to end a training run.
    """
    import shutil

    probe = directory if directory.is_dir() else directory.parent
    try:
        if free_bytes is None:
            free_bytes = int(shutil.disk_usage(probe).free)
        used = sum(path.stat().st_size for path in directory.glob("epoch_*.pt"))
    except OSError as exc:
        log(
            "checkpoint retention budget could not be measured "
            f"({exc}); retaining every epoch checkpoint."
        )
        return None
    return int((free_bytes + used) * CHECKPOINT_BUDGET_FRACTION_OF_FREE)


def plan_checkpoint_retention(
    paths: list[Path],
    *,
    free_bytes: int,
    adapter_bytes: int,
    budget_bytes: int | None = None,
) -> list[Path]:
    """Decide which epoch checkpoints must go, given a MEASURED budget.

    Pure: takes the measurements, returns the paths to delete, touches no
    disk. Policy when the budget binds: keep the FIRST and LAST checkpoints
    unconditionally, then repeatedly drop the middle-most survivor until the
    remainder fits. That preserves both ends of the loss curve -- the shape a
    stall analysis reads -- instead of a window that keeps only the tail.
    """
    ordered = sorted(paths, key=lambda path: path.name)
    if len(ordered) <= 2 or adapter_bytes <= 0:
        return []
    if budget_bytes is None:
        budget_bytes = int(
            (free_bytes + adapter_bytes * len(ordered))
            * CHECKPOINT_BUDGET_FRACTION_OF_FREE
        )
    max_keep = max(2, int(budget_bytes // adapter_bytes))
    if len(ordered) <= max_keep:
        return []
    removed: list[Path] = []
    kept = list(ordered)
    while len(kept) > max_keep:
        # Middle-most of the interior span; endpoints are never candidates.
        removed.append(kept.pop(len(kept) // 2))
    return sorted(removed, key=lambda path: path.name)


def enforce_checkpoint_budget(
    directory: Path,
    *,
    budget_bytes: int | None = None,
    log: Any = emit_log,
) -> list[Path]:
    """Apply `plan_checkpoint_retention` to *directory*, loudly.

    Returns the paths ACTUALLY removed (each with its `.complete.json`
    marker, so a marker never outlives the artifact it vouches for).

    Total fail-open contract: NO filesystem operation in this function --
    measurement or deletion -- may raise. It is called from
    `_write_epoch_checkpoint` on the training path, and retention is
    best-effort housekeeping that must never be able to end a run. The guards
    stay narrowly scoped to `OSError`: a `KeyError`/`TypeError` from
    `plan_checkpoint_retention` is a logic bug and must still surface.
    """
    if not directory.is_dir():
        return []
    try:
        paths = sorted(directory.glob("epoch_*.pt"), key=lambda path: path.name)
        if not paths:
            return []
        # Same fail-open contract as `checkpoint_budget_bytes`, for the same
        # reason: this call sits on the training path, so no measurement
        # failure here may prune, and none may raise.
        adapter_bytes = max(path.stat().st_size for path in paths)
    except OSError as exc:
        log(
            "checkpoint sizes could not be measured "
            f"({exc}); retaining every epoch checkpoint."
        )
        return []
    if budget_bytes is None:
        budget_bytes = checkpoint_budget_bytes(directory, log=log)
        if budget_bytes is None:
            return []
    stale = plan_checkpoint_retention(
        paths,
        free_bytes=0,
        adapter_bytes=adapter_bytes,
        budget_bytes=budget_bytes,
    )
    # Deletion is filesystem I/O on the training path, so it carries the SAME
    # fail-open contract as the measurement above. `missing_ok=True` only
    # suppresses FileNotFoundError; a PermissionError, a read-only mount, or a
    # race-losing OSError would otherwise propagate through
    # `_write_epoch_checkpoint` into `run_training` and kill a multi-hour run.
    #
    # Each deletion is INDEPENDENT: a failure logs and moves on rather than
    # aborting the rest. Aborting on the first failure would leave more disk
    # consumed with no compensating benefit -- a partially thinned directory is
    # a valid state for best-effort housekeeping, whereas a dead run is not.
    # The marker is only removed once its artifact is actually gone, so a
    # marker can still never outlive the artifact it vouches for.
    removed: list[Path] = []
    for path in stale:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log(f"could not delete epoch checkpoint {path.name} ({exc}); keeping it.")
            continue
        removed.append(path)
        marker = path.with_name(path.name + ".complete.json")
        try:
            marker.unlink(missing_ok=True)
        except OSError as exc:
            log(f"could not delete completion marker {marker.name} ({exc}).")
    if removed:
        log(
            "CHECKPOINT RETENTION BUDGET BINDING: disk budget "
            f"{budget_bytes / 1e6:.0f} MB holds only "
            f"{max(2, budget_bytes // max(1, adapter_bytes))} of {len(paths)} "
            f"epoch checkpoints (~{adapter_bytes / 1e6:.1f} MB each). Thinned "
            "the middle, kept first and last: deleted "
            + ", ".join(path.name for path in removed)
        )
    return removed


# --- Per-epoch validation series ------------------------------------------
#
# A single terminal validation number cannot distinguish "the model converged"
# from "the model stalled at epoch 2 and burned eight more epochs of GPU
# time". The series can, and it costs one forward-only pass per epoch. It is
# recorded UNCONDITIONALLY, as evidence, and `val_stats.json` keeps its
# existing shape so every current reader is unaffected.
VAL_SERIES_FILENAME = "val_series.jsonl"
VAL_CADENCE_ENV = "HYDRA_SAM3_VAL_EVERY"
# Detection quality (AP) rides along on the same pass and lands in the same
# row, but gets its own knob: the loss half is forward-only, while AP adds
# CPU-side contour extraction and a 19-point confidence sweep of the matcher.
# Whoever finds that too expensive must be able to back AP off WITHOUT losing
# the loss series. Defaults to the loss cadence, so the two stay in step
# unless someone deliberately separates them.
AP_CADENCE_ENV = "HYDRA_SAM3_AP_EVERY"


def val_cadence(default: int = 1) -> int:
    """Epochs between mid-run validation passes (1 = every epoch).

    A knob rather than a silent cost: the pass is forward-only but it is not
    free (order 10-20% of an epoch on the 2026-09-06 geometry), and epoch
    counts are user-supplied. The FINAL epoch is always recorded regardless,
    because the terminal evaluation already runs there.
    """
    raw = os.environ.get(VAL_CADENCE_ENV, "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 1 else default


def ap_cadence() -> int:
    """Epochs between per-epoch DETECTION-QUALITY (AP) evaluations.

    Defaults to `val_cadence()`. Set `HYDRA_SAM3_AP_EVERY` higher to keep the
    cheap loss series every epoch while sampling AP less often; the effective
    value is stamped into every row as `ap_cadence` for the same provenance
    reason `val_cadence` is.

    RECORDING ONLY. See `detection_quality`'s module docstring: AP is a
    candidate selection signal from ONE run, ONE seed and ONE corpus, and
    nothing in this file may select or stop on it.
    """
    default = val_cadence()
    raw = os.environ.get(AP_CADENCE_ENV, "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 1 else default


def append_val_record(run_dir_path: Path, record: dict[str, Any]) -> Path:
    """Append one epoch's validation record to the run's JSONL series.

    Append-per-epoch (not a dict rewritten at the end) so a killed or crashed
    run still leaves every epoch it actually reached.
    """
    path = run_dir_path / VAL_SERIES_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def _write_epoch_checkpoint(
    model: Any, run_dir_path: Path, epoch_number: int, torch_module: Any
) -> Path:
    """Persist this epoch's adapters as salvage, atomically.

    Writes into the run's `checkpoints` subdirectory, under its own epoch
    name (never the completion-signal artifact name), then applies the
    measured retention budget.
    """
    directory = run_dir_path / EPOCH_CHECKPOINT_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"epoch_{epoch_number:03d}.pt"
    _write_validated_adapter_artifact(adapter_state_dict(model), target, torch_module)
    enforce_checkpoint_budget(directory)
    return target


def _build_dataloader(spec: Any, params: Any, *, split: str) -> list:
    """Read the built COCO split into lightweight tile descriptors.

    No image is decoded, transformed, or rasterized until its epoch iterator
    reaches the corresponding batch.
    """
    return build_descriptors(spec.derived_dataset_dir, params, split, seed=spec.seed)


def _lora_scope_refusal(params: Any) -> str | None:
    """Refuse a LoRA scope selection that cannot be trained as configured.

    Split out of `_runtime_admission_refusal` so it is provable without CUDA.
    """
    # Derived from the injector's own scope tables rather than hardcoded, so a
    # new scope flag cannot be refused here as "disabled" while
    # `lora_config_from_params` happily accepts it.
    scopes = (*SUBMODULE_PREFIXES, *SUBMODULE_PATHS)
    if not any(bool(getattr(params, flag, False)) for flag in scopes):
        return (
            "SAM3 training requires at least one enabled adapter scope; all "
            "adapt_* flags are disabled."
        )
    # A scope with no measured LORA_PARAMS_PER_RANK coefficient would inject
    # adapters the estimator does not budget; `_validated_lora_trainables`
    # would then refuse with a confusing "estimator drift" message deep into
    # the run. Say what is actually wrong, here, before anything is built.
    unmeasured = [
        flag
        for flag in scopes
        if bool(getattr(params, flag, False)) and flag not in LORA_PARAMS_PER_RANK
    ]
    if unmeasured:
        return (
            "SAM3 adapter scope(s) "
            f"{', '.join(sorted(unmeasured))} have no measured trainable-parameter "
            "coefficient. Measure them against a live build_sam3_image_model and "
            "add them to LORA_PARAMS_PER_RANK before enabling."
        )
    return None


def _runtime_admission_refusal(torch_module: Any, params: Any) -> str | None:
    """Repeat the parent precision/hardware gate before importing SAM3."""
    prompt_error = sam3_prompt_text_error(getattr(params, "prompt", None))
    if prompt_error is not None:
        return f"SAM3 prompt {prompt_error}."
    for index, negative_prompt in enumerate(
        getattr(params, "negative_prompts", ()) or ()
    ):
        prompt_error = sam3_prompt_text_error(negative_prompt)
        if prompt_error is not None:
            return f"SAM3 negative prompt {index} {prompt_error}."
    if getattr(params, "mixed_precision", None) != "bf16":
        return (
            "SAM3 training supports only CUDA BF16; fp16/fp32 modes fail "
            "against SAM3's BF16 activation path and are disabled."
        )
    if not torch_module.cuda.is_available():
        return "SAM3 training requires a CUDA device; CPU and MPS are disabled."
    major, minor = torch_module.cuda.get_device_capability()
    if major < 8:
        return (
            "SAM3 training requires CUDA BF16 on compute capability >= 8.0; "
            f"the selected GPU reports {major}.{minor}. FP32 fallback is disabled."
        )
    if not bool(torch_module.cuda.is_bf16_supported()):
        return (
            "The selected CUDA runtime reports that BF16 operations are not "
            "supported; SAM3 training has no safe FP32 fallback."
        )
    return _lora_scope_refusal(params)


def _validated_lora_trainables(
    model: Any, *, adapted_modules: int, expected_parameters: int
) -> tuple[list[Any], int]:
    """Return LoRA-only tensors or refuse estimator/runtime shape drift."""

    trainable_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected_trainable = [
        name
        for name, _parameter in trainable_named
        if not name.endswith((".lora_A", ".lora_B"))
    ]
    actual_parameters = sum(
        int(parameter.numel()) for _name, parameter in trainable_named
    )
    if (
        adapted_modules < 1
        or len(trainable_named) != 2 * adapted_modules
        or unexpected_trainable
        or actual_parameters != expected_parameters
    ):
        unexpected = ", ".join(unexpected_trainable[:5]) or "none"
        raise RuntimeError(
            "SAM3 LoRA trainable-parameter invariant failed: "
            f"adapters={adapted_modules}, trainable={len(trainable_named)}, "
            f"parameters={actual_parameters}, "
            f"expected_parameters={expected_parameters}, "
            f"unexpected={unexpected}"
        )
    return [parameter for _name, parameter in trainable_named], actual_parameters


# Terms worth logging individually: a collapse shows up as one of these
# going to zero while the others stay put, which a single core scalar hides.
# A handful of skipped steps is normal on an overflow; a long run of them
# means the model is not recovering and the run should stop, not grind on.
MAX_CONSECUTIVE_SKIPPED_STEPS = 25

LOGGED_LOSS_TERMS = (
    "loss_ce",
    "loss_bbox",
    "loss_giou",
    "loss_mask",
    "loss_dice",
    "presence_loss",
)


def _loss_term_summary(loss_dict: Any) -> str:
    """Render the headline loss terms, or '' when they are unavailable."""
    if not isinstance(loss_dict, dict):
        return ""
    parts = []
    for term in LOGGED_LOSS_TERMS:
        value = loss_dict.get(term)
        if value is not None and getattr(value, "numel", lambda: 0)() == 1:
            parts.append(f"{term}={float(value):.4f}")
    return "  " + " ".join(parts) if parts else ""


def _assert_finite_loss(loss: Any, *, epoch: int, step: int, loss_dict: Any) -> None:
    """Abort the run the moment the loss stops being a number.

    A non-finite loss poisons every subsequent gradient, and SAM3's Hungarian
    matcher only notices much later -- it raised
    ``ValueError: matrix contains invalid numeric entries`` some 700 steps
    after the loss had already gone bad, by which point the run had burned
    GPU hours producing an unusable adapter. Failing here names the epoch,
    the step, and the per-term breakdown that led to it.
    """
    import torch

    if torch.isfinite(loss).all():
        return
    raise RuntimeError(
        f"SAM3 loss became non-finite at epoch {epoch}, step {step} "
        f"(value={float(loss)}).{_loss_term_summary(loss_dict)}  Training "
        "cannot recover from this -- refusing to keep stepping into a "
        "checkpoint that would be silently worthless."
    )


def _peak_vram_gib(device: Any) -> float:
    """Peak CUDA memory RESERVED by the allocator, in GiB.

    Reserved rather than allocated: reserved is what the device must actually
    have free for the run to survive, since the caching allocator does not
    return blocks to the driver between steps. This is the number a hardware
    requirement has to be written against.
    """
    import torch

    return torch.cuda.max_memory_reserved(device) / (1024**3)


class _LossWindow:
    """Mean core loss and per-term losses across one accumulation window.

    Logging the LAST micro-batch of a window is structurally misleading: the
    boundary lands on ``micro_idx = grad_accum - 1``, and with negatives
    interleaved one per tile that index always holds the same query TYPE.
    With ``num_negatives=1`` every logged line was a negative -- which
    legitimately has zero matched loss -- so loss_ce/bbox/giou/mask/dice read
    as 0.0000 forever and a normally-training run looked like a total
    collapse. Averaging the window reports what the optimizer stepped on.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._core = 0.0
        self._terms: dict[str, float] = {}
        self._n = 0

    def add(self, loss: Any, loss_dict: Any) -> None:
        self._n += 1
        # detach(): `float()` on a graph-attached tensor warns, and keeping a
        # reference to the graph across the whole window would pin every
        # micro-batch's activations in memory until the window is reset.
        self._core += float(loss.detach() if hasattr(loss, "detach") else loss)
        if not isinstance(loss_dict, dict):
            return
        for term in LOGGED_LOSS_TERMS:
            value = loss_dict.get(term)
            if value is not None and getattr(value, "numel", lambda: 0)() == 1:
                if hasattr(value, "detach"):
                    value = value.detach()
                self._terms[term] = self._terms.get(term, 0.0) + float(value)

    def summary(self) -> str:
        if not self._n:
            return "loss n/a"
        terms = " ".join(
            f"{term}={self._terms[term] / self._n:.4f}"
            for term in LOGGED_LOSS_TERMS
            if term in self._terms
        )
        return f"loss {self._core / self._n:.4f}" + (f"  {terms}" if terms else "")


def _build_model_and_loss(params: Any) -> tuple[Any, Any, Any, Any, list]:
    """Build the device, model, adapters, matcher, and loss ONE way.

    Shared verbatim by `run_training` and by the memory probe, so the
    probe measures the same construction ordering training uses --
    `requires_grad_(False)` -> inject adapters -> validate -> `.to(device)`.
    A probe that built a different stack would measure a different peak,
    which is exactly the class of bug measured auto-batching exists to end.

    Returns `(device, model, matcher, loss_fn, trainable_params)`. The
    optimizer is deliberately NOT built here: both callers need their own
    (training pairs it with a scheduler), and both must build one, because
    lazy Adam state is part of the peak.
    """
    import torch

    # NOTE: verified against the real Meta sam3 source on the CUDA box
    # (2026-08-31): sam3/build_sam.py does not exist. The builder lives in
    # sam3/model_builder.py. Do not "correct" this back.
    from sam3.model_builder import build_sam3_image_model
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

    # SAM3's vision trunk MLP is inference-only as shipped (it refuses to run
    # with grad enabled, and detaches its weights). Swap in an eager,
    # differentiable equivalent before the model is built. See perflib_compat.
    if install_grad_safe_addmm_act():
        emit_log("Patched vitdet.addmm_act with a grad-safe eager equivalent.")

    device = torch.device("cuda")
    model = build_sam3_image_model(eval_mode=False)

    lora_cfg = lora_config_from_params(params)
    # The preflight estimate intentionally budgets optimizer and gradient
    # state for LoRA parameters only.  SAM3 builders do not promise frozen
    # defaults, so establish that invariant here before adapters are created.
    model.requires_grad_(False)
    n_adapted = inject_adapters(model, lora_cfg)
    emit_log(f"Injected LoRA adapters into {n_adapted} Linear modules.")
    expected_trainable_params = expected_lora_trainable_params(params)
    trainable_params, actual_trainable_params = _validated_lora_trainables(
        model,
        adapted_modules=n_adapted,
        expected_parameters=expected_trainable_params,
    )
    # Validate exact estimator parity before moving any unexpectedly large
    # upstream model drift onto VRAM. `.to(device)` remains after injection so
    # the newly created adapter tensors move with the frozen base.
    model.to(device)
    emit_log(
        "Verified LoRA-only optimizer scope: "
        f"{actual_trainable_params:,} parameters in {len(trainable_params)} "
        f"tensors across {n_adapted} adapters."
    )

    # Verified against the real Meta sam3 source on the CUDA box (2026-08-31,
    # sam3-lora env): `inspect.signature` on `Sam3LossWrapper.__init__`,
    # `Boxes.__init__`, `IABCEMdetr.__init__`, `Masks.__init__`,
    # `BinaryHungarianMatcherV2.__init__`, `BinaryOneToManyMatcher.__init__`.
    # `Sam3LossWrapper` has no zero-arg form -- `loss_fns_find` is required
    # and positional-first. Values below are translated from Meta's own
    # reference finetuning config,
    # `sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml`
    # (segmentation variant, since this trains masks). ONE matcher instance
    # is built and shared between the loss wrapper and the training loop's
    # own `outputs["indices"] = matcher(...)` calls below, rather than two
    # diverging matchers.
    matcher = BinaryHungarianMatcherV2(
        focal=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2,
        stable=False,
    )
    # The image model computes matching internally in training mode. Use the
    # exact same matcher instance as the explicit validation path and loss.
    model.matcher = matcher
    loss_fn = _build_loss_wrapper(
        Sam3LossWrapper,
        loss_fns_find=[
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(
                weak_loss=False,
                weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                pos_weight=10.0,
                alpha=0.25,
                gamma=2,
                use_presence=True,
                pos_focal=False,
                pad_n_queries=200,
                pad_scale_pos=1.0,
            ),
            Masks(
                focal_alpha=0.25,
                focal_gamma=2.0,
                weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
                compute_aux=False,
            ),
        ],
        matcher=matcher,
        o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
    )
    return device, model, matcher, loss_fn, trainable_params


def run_training(spec: Any, run_dir_path: Path) -> bool:
    """Run the SAM3 LoRA training loop and write `adapters.pt`.

    Returns True on a completed run that wrote the artifact, False on the
    zero-datapoint refusal (the only failure mode this function itself
    reports -- everything past this point either succeeds or raises, and an
    uncaught exception is `main()`'s cue to exit nonzero).
    """
    params = spec.sam3_params

    train_descriptors = _build_dataloader(spec, params, split="train")
    if not train_descriptors:
        emit_log(
            "Training set produced zero datapoints; refusing to report "
            "success for a run that trained nothing."
        )
        return False

    # --- Lazy, training-only imports -----------------------------------
    import torch

    refusal = _runtime_admission_refusal(torch, params)
    if refusal:
        emit_log(refusal)
        return False
    _seed_everything(spec.seed)

    device, model, matcher, loss_fn, trainable_params = _build_model_and_loss(params)

    grad_accum = max(1, int(params.grad_accum))
    # NOT `max(1, ...)`. A silent floor of 1 is the exact bug measured auto
    # batch sizing exists to kill: the parent resolves `-1` to a positive
    # value before this process is ever launched, so a non-positive batch
    # here means someone hand-ran the child against an unresolved spec. Train
    # at a size nobody chose and the run is worthless and looks fine.
    batch_size = int(params.batch)
    if batch_size < 1:
        raise RuntimeError(
            f"SAM3 training received batch={batch_size}. A non-positive batch "
            "is a request to MEASURE one, which only the launcher can do "
            "(it probes this workload on this card first). Run through "
            "`train_sam3_lora`, or set a positive batch in the spec."
        )
    # Scale-grouped batching (plan Task 5). Engaged only when the built
    # dataset actually carries scale groups -- a single-scale build takes
    # literally today's path. `HYDRA_SAM3_SCALE_GROUPED_BATCHING=0` runs the
    # ungrouped arm, which is what Task 8's grouped-vs-ungrouped comparison
    # needs. Whichever arm runs, the run dir gets a requested-vs-applied stamp,
    # so the two are never confusable afterwards.
    #
    # SAM3 is single-process, single-GPU by construction, so there is no
    # distributed case to fall back FROM here (a committed tripwire test fails
    # the moment `sam3_lora/` gains one -- at which point this must RAISE, not
    # warn, exactly as the Ultralytics installer does).
    group_counts = scale_group_summary(train_descriptors)
    dataset_has_groups = bool(group_counts.keys() - {"ungrouped"})
    grouping_requested = str(
        os.environ.get("HYDRA_SAM3_SCALE_GROUPED_BATCHING", "1")
    ).strip().lower() not in {"0", "false", "no"}
    group_by_scale = bool(grouping_requested and dataset_has_groups)
    if grouping_requested and not dataset_has_groups:
        grouping_reason = "dataset carries no scale_group records (single-scale build)"
    elif not grouping_requested:
        grouping_reason = "disabled via HYDRA_SAM3_SCALE_GROUPED_BATCHING"
    else:
        grouping_reason = ""
    write_sam3_scale_grouping_stamp(
        run_dir_path,
        requested=grouping_requested,
        applied=group_by_scale,
        reason=grouping_reason,
        group_counts=group_counts,
    )
    if group_by_scale:
        n_batches = grouped_batch_count(train_descriptors, batch_size)
        emit_log(
            "scale-grouped batching ON: "
            + ", ".join(
                f"{name}={count}" for name, count in sorted(group_counts.items())
            )
        )
    else:
        n_batches = batch_count(query_count(train_descriptors), batch_size)
    steps_per_epoch = -(-n_batches // grad_accum)  # ceil division
    total_steps = max(1, steps_per_epoch * params.epochs)
    warmup_steps = min(50, total_steps // 4)

    optimizer = torch.optim.AdamW(trainable_params, lr=params.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_cosine_with_warmup(warmup_steps, total_steps)
    )

    autocast_dtype = torch.bfloat16

    global_step = 0
    logging_steps = 10
    loss_window = _LossWindow()
    skipped_steps = 0
    consecutive_skipped = 0
    # Announce the run's shape before the first step. Without this the only
    # output for the first several minutes is silence, and there is no way to
    # tell a slow run from a wedged one -- or to sanity-check the step budget
    # against the epochs actually requested.
    # Peak-memory telemetry is the empirical basis for this role's hardware
    # requirement. The admission gate's device estimate is an inherited
    # envelope constant, not a measurement of THIS configuration, so every run
    # reports what it actually used -- otherwise the requirement can never be
    # revised except by guesswork.
    torch.cuda.reset_peak_memory_stats(device)
    emit_log(
        f"training shape: {len(train_descriptors)} tiles, "
        f"{query_count(train_descriptors)} datapoints, "
        f"batch={batch_size} grad_accum={grad_accum}, "
        f"{n_batches} micro-batches/epoch, "
        f"~{max(1, n_batches // grad_accum)} steps/epoch x {params.epochs} epochs"
    )
    optimizer.zero_grad()

    for epoch in range(params.epochs):
        model.train()
        # Tile descriptors reshuffle every epoch (seeded from spec.seed +
        # epoch, so runs stay reproducible). Queries remain tile-grouped to
        # share one transformed image without a dataset-sized tensor cache.
        epoch_batches = collate_epoch_batches(
            train_descriptors,
            batch_size,
            seed=spec.seed + epoch,
            group_by_scale=group_by_scale,
        )
        n_epoch_batches = n_batches
        for micro_idx, batch in enumerate(epoch_batches):
            # Only the MODEL forward runs under autocast. The loss -- and with
            # it SAM3's Hungarian matcher -- is computed outside, in fp32.
            # `linear_sum_assignment` rejects any non-finite cost entry, and a
            # bf16 cost matrix built from bf16 logits reaches inf/NaN far more
            # readily than an fp32 one; a 10-epoch run died at epoch 5 with
            # "matrix contains invalid numeric entries". The spike this design
            # derives from matched in fp32 throughout (it used no autocast at
            # all), so this restores its matcher numerics without paying fp32
            # memory for the whole model -- a full fp32 run estimates ~58 GiB
            # and does not fit this class of card.
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
                model_input, targets, outputs = _forward_batch(batch, model, device)
            loss_dict = loss_fn(outputs, targets)
            loss = _core_loss(loss_dict)

            _assert_finite_loss(
                loss, epoch=epoch, step=global_step, loss_dict=loss_dict
            )
            loss_window.add(loss, loss_dict)
            (loss / grad_accum).backward()

            is_boundary = (micro_idx + 1) % grad_accum == 0
            is_final_micro_batch = (micro_idx + 1) == n_epoch_batches
            if is_boundary or is_final_micro_batch:
                # clip_grad_norm_ does NOT sanitise non-finite gradients -- it
                # scales by a norm that is itself inf/NaN, writing NaN straight
                # into every adapter weight. From then on the model emits NaN
                # logits, and the failure only surfaces later as a NaN loss (or,
                # before the loss guard existed, as the Hungarian matcher
                # rejecting the cost matrix hundreds of steps downstream).
                # Skipping the step is what torch's own GradScaler does on
                # overflow: drop the batch, keep the weights finite, continue.
                total_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=1.0
                )
                if torch.isfinite(total_norm):
                    optimizer.step()
                    consecutive_skipped = 0
                else:
                    consecutive_skipped += 1
                    skipped_steps += 1
                    if consecutive_skipped >= MAX_CONSECUTIVE_SKIPPED_STEPS:
                        raise RuntimeError(
                            f"SAM3 training skipped {consecutive_skipped} "
                            "consecutive optimizer steps on non-finite "
                            f"gradients (epoch {epoch}, step {global_step}). "
                            "The run is no longer learning -- refusing to "
                            "continue into a worthless checkpoint."
                        )
                # The schedule advances either way: it tracks intended
                # progress through the run, not the number of accepted steps.
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # Early steps log every time: waiting for step 10 hides both a
                # fast crash and a pathologically slow first epoch.
                if global_step <= 3 or global_step % logging_steps == 0:
                    emit_log(
                        f"epoch {epoch} step {global_step} "
                        f"{loss_window.summary()}"
                        + (f"  skipped={skipped_steps}" if skipped_steps else "")
                        + f"  vram_peak={_peak_vram_gib(device):.2f}GiB"
                    )
                loss_window.reset()
            del batch, model_input, targets, outputs, loss_dict, loss

        # Salvage checkpoint, skipped on the final epoch because the real
        # artifact is written moments later from the same weights.
        if epoch + 1 < params.epochs:
            saved = _write_epoch_checkpoint(model, run_dir_path, epoch + 1, torch)
            emit_log(f"epoch {epoch} checkpoint: {saved}")
            # Record the validation series as it happens. The final epoch is
            # deliberately excluded here: the terminal `_evaluate_and_write`
            # below appends it from the same computation, so no epoch is
            # evaluated twice.
            if (epoch + 1) % val_cadence() == 0:
                _record_epoch_validation(
                    model,
                    spec,
                    params,
                    matcher,
                    loss_fn,
                    device,
                    autocast_dtype,
                    True,
                    run_dir_path,
                    epoch + 1,
                )

        emit_progress(epoch + 1, params.epochs)

    # Keep adapters in memory until validation completes. A failed evaluation,
    # kill, or parent death must not expose a seemingly completed artifact.
    adapters = adapter_state_dict(model)
    _evaluate_and_write(
        model,
        spec,
        params,
        matcher,
        loss_fn,
        device,
        autocast_dtype,
        True,
        run_dir_path,
        params.epochs,
    )
    artifact_path = run_dir_path / "adapters.pt"
    _write_validated_adapter_artifact(adapters, artifact_path, torch)
    return True


def _evaluate_split(
    model: Any,
    spec: Any,
    params: Any,
    matcher: Any,
    loss_fn: Any,
    device: Any,
    autocast_dtype: Any,
    use_bf16: bool,
    *,
    with_detection_quality: bool = True,
) -> dict[str, Any] | None:
    """Run the validation split and return its loss decomposition.

    Returns `val_loss_mean` (the same number the terminal `val_stats.json`
    has always reported), the per-term breakdown averaged over batches, the
    batch count, and `elapsed_s` -- the pass's own measured wall clock, so the
    cost of this evidence is itself evidence rather than an estimate.

    When ``with_detection_quality`` is set it ALSO returns a detection-quality
    block (``ap``, ``ap_tiles``, ``ap_elapsed_s``, ``ap_sweep``, or
    ``ap_error``), collected from the very same forward pass -- no second
    inference, so nothing extra can perturb training -- and timed separately
    so the two costs stay attributable. See `detection_quality` for the four
    limitations that make AP a within-run trend and not a portable score.

    Reporting only. Nothing here may select a checkpoint: a 2026-09-06 study
    measured every per-query validation signal ANTI-correlating with held-out
    AP, so selecting on it would be actively wrong. That study also found AP
    the most stable signal it measured -- which makes AP a *candidate* for a
    future selection rule on ONE run, ONE seed and ONE corpus, with seed
    variance entirely unmeasured. It does not make it a rule, and this
    function still selects nothing.

    Returns `None` when there is no validation split (small datasets skip it
    -- see `dataset_build.py`'s `validation: "none"` case) rather than
    fabricating a placeholder.
    """
    import time

    import torch

    val_descriptors = try_build_descriptors(
        spec.derived_dataset_dir, params, "valid", seed=spec.seed
    )
    if not val_descriptors:
        return None
    n_val_batches = batch_count(query_count(val_descriptors), params.batch)
    val_batches = collate_batches(val_descriptors, params.batch)

    accumulator = None
    if with_detection_quality:
        from .detection_quality import DetectionQualityAccumulator

        accumulator = DetectionQualityAccumulator(val_descriptors)

    started = time.perf_counter()
    model.eval()
    total_loss = 0.0
    term_totals: dict[str, float] = {}
    with torch.no_grad():
        for batch in val_batches:
            # Same split as training: matcher and loss in fp32, forward in bf16.
            with torch.autocast(
                device_type="cuda", dtype=autocast_dtype, enabled=use_bf16
            ):
                model_input, targets, outputs = _forward_batch(batch, model, device)
            _attach_matcher_indices(outputs, targets, matcher)
            loss_dict = loss_fn(outputs, targets)
            loss = _core_loss(loss_dict)
            total_loss += float(loss)
            if isinstance(loss_dict, dict):
                for key, value in loss_dict.items():
                    try:
                        term_totals[key] = term_totals.get(key, 0.0) + float(value)
                    except (TypeError, ValueError):
                        continue
            # Read the predictions off the SAME forward, before it is freed.
            # `observe` is self-guarding: it can never raise into the loss
            # pass, it only records `ap_error`.
            if accumulator is not None:
                accumulator.observe(outputs)
            del batch, model_input, targets, outputs, loss_dict, loss

    # The loss pass's own clock EXCLUDES the AP work, which reports its own
    # `ap_elapsed_s`. Two metrics, two measured costs -- an aggregate number
    # could not tell anyone whether backing AP off would help.
    # `result()` runs the confidence sweep, which dominates the AP cost, so it
    # must finish BEFORE the clock is read -- otherwise subtracting
    # `ap_elapsed_s` would remove time `elapsed` never contained and the loss
    # pass would under-report itself by the whole sweep.
    detection_quality: dict[str, Any] = {}
    if accumulator is not None:
        detection_quality = accumulator.result()
    elapsed = time.perf_counter() - started
    if detection_quality:
        elapsed -= min(elapsed, float(detection_quality.get("ap_elapsed_s", 0.0)))

    return {
        "val_loss_mean": total_loss / n_val_batches,
        "val_batches": n_val_batches,
        "val_terms_mean": {
            key: total / n_val_batches for key, total in sorted(term_totals.items())
        },
        "elapsed_s": elapsed,
        **detection_quality,
    }


def _record_epoch_validation(
    model: Any,
    spec: Any,
    params: Any,
    matcher: Any,
    loss_fn: Any,
    device: Any,
    autocast_dtype: Any,
    use_bf16: bool,
    run_dir_path: Path,
    epoch_number: int,
) -> dict[str, Any] | None:
    """Evaluate the validation split mid-run and append it to the series.

    EVIDENCE ONLY -- if you are here to wire best-checkpoint selection or
    early stopping onto this series, don't. Selection stays last-epoch: a
    2026-09-06 study on a genuinely disjoint fold measured every per-query
    validation signal ANTI-correlating with held-out AP, so choosing the
    minimum of this curve would pick a worse detector. The series exists to
    distinguish "converged" from "stalled at epoch 2", nothing else.

    Provably behaviour-preserving: the model is returned to `train()` and
    every RNG stream is restored, so the next epoch draws exactly the numbers
    it would have drawn without this call.
    """
    import torch

    was_training = model.training
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    # The AP pass rides INSIDE this guard, on the loss pass's own forward --
    # deliberately not a second, unprotected inference pass.
    ap_every = ap_cadence()
    want_ap = epoch_number % ap_every == 0
    try:
        stats = _evaluate_split(
            model,
            spec,
            params,
            matcher,
            loss_fn,
            device,
            autocast_dtype,
            use_bf16,
            with_detection_quality=want_ap,
        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if was_training:
            model.train()
    if stats is None:
        return None
    # Stamp the effective cadence into every record. Without it the cadence
    # is only inferable from epoch gaps in the series -- and a cadence gap is
    # indistinguishable from a crash-restart gap. An unrecorded env knob that
    # changes what a run measured is a provenance defect this project has
    # already been bitten by (`object_tile_fraction` silently defaulting).
    record = {
        "epoch": epoch_number,
        "val_cadence": val_cadence(),
        "ap_cadence": ap_every,
        **stats,
    }
    append_val_record(run_dir_path, record)
    emit_log(
        f"epoch {epoch_number} val_loss_mean={stats['val_loss_mean']:.5f} "
        f"({stats['val_batches']} batches, {stats['elapsed_s']:.1f}s)"
        + _ap_log_suffix(stats)
        + " [recorded as evidence; selection is unchanged]"
    )
    return record


def _ap_log_suffix(stats: dict[str, Any]) -> str:
    """Log fragment for the detection-quality half of an epoch's record.

    A failed AP pass logs LOUDLY rather than vanishing: on the CUDA box the
    run log is the only place a vendor key rename would be noticed before the
    series is read weeks later.
    """
    if "ap_error" in stats:
        return f"  AP FAILED: {stats['ap_error']} (training unaffected)"
    if "ap" not in stats:
        return ""
    return (
        f"  ap={stats['ap']:.4f} ({stats.get('ap_tiles', 0)} tiles, "
        f"{stats.get('ap_elapsed_s', 0.0):.1f}s; within-run trend only)"
    )


def _evaluate_and_write(
    model: Any,
    spec: Any,
    params: Any,
    matcher: Any,
    loss_fn: Any,
    device: Any,
    autocast_dtype: Any,
    use_bf16: bool,
    run_dir_path: Path,
    epoch_number: int | None = None,
) -> Path | None:
    """Compute real validation-set loss, for reporting ONLY.

    Never influences checkpoint selection (see module docstring) -- this runs
    strictly after the `adapters.pt` save above. If there is no validation
    split (small datasets skip it -- see `dataset_build.py`'s `validation:
    "none"` case), no file is written and `None` is returned rather than
    fabricating a placeholder.

    Writes `val_stats.json` in its historical shape (`val_loss_mean`,
    `val_batches`, `note`) plus the additive per-term breakdown, and appends
    the SAME computation as the final entry of `val_series.jsonl`, so the
    series' last row and the terminal artifact can never disagree and the
    final epoch is never evaluated twice.
    """
    stats = _evaluate_split(
        model, spec, params, matcher, loss_fn, device, autocast_dtype, use_bf16
    )
    if stats is None:
        return None

    cadence = val_cadence()
    val_stats = {
        "val_loss_mean": stats["val_loss_mean"],
        "val_batches": stats["val_batches"],
        "val_terms_mean": stats["val_terms_mean"],
        "val_cadence": cadence,
        "note": "informational only; checkpoint selection is always 'last'",
    }
    metrics_path = run_dir_path / "val_stats.json"
    metrics_path.write_text(json.dumps(val_stats, indent=2), encoding="utf-8")
    if epoch_number is not None:
        # The FINAL row always carries AP, whatever the cadence: a series
        # whose last epoch has no detection-quality number cannot be read
        # against the always-last-epoch policy it exists to inform.
        append_val_record(
            run_dir_path,
            {
                "epoch": epoch_number,
                "val_cadence": cadence,
                "ap_cadence": ap_cadence(),
                **stats,
            },
        )
    return metrics_path


def _densest_first(descriptors: list) -> list:
    """Order tiles by active instance count, densest first.

    Mask memory scales with the number of active instances in a tile, so a
    probe fed median tiles understates the peak the run will actually hit --
    and a batch size chosen from an understated peak OOMs hours later. This
    is the same density the workload fingerprint records.
    """

    return sorted(descriptors, key=lambda d: len(d.instances), reverse=True)


def _host_peak_bytes() -> int:
    """Peak RSS of this process, in bytes."""

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes; macOS reports bytes. The child always runs on
    # the CUDA box (Linux), but keep the conversion honest either way.
    return int(usage) * (1 if sys.platform == "darwin" else 1024)


# How often the probe child reports in. Frequent enough that no plausible
# step time leaves the log silent for long, rare enough not to spam a 30-step
# run. Deliberately not expressed in seconds: step time varies by card and
# corpus, and quoting a duration nobody measured on this box would be a guess.
_PROBE_HEARTBEAT_STEPS = 10


def run_probe_measurement(spec: Any, run_dir_path: Path, batch_size: int) -> int:
    """Measure this configuration's device peak at ONE batch size.

    Runs inside a contained sidecar admitted at exactly `batch_size` (see
    `preflight.assess_probe_preflight`), builds the SAME model/loss stack
    training builds plus its own optimizer, and takes at least
    `PROBE_STEPS` full steps THROUGH `optimizer.step()` on the densest tiles.
    The first step is not enough on its own: Adam's `exp_avg`/`exp_avg_sq`
    are allocated lazily inside the first `step()`, so a one-step probe
    misses the whole optimizer state.

    Writes `run_dir/probe_records/batch_<N>.json` and returns a process exit
    code. An out-of-memory error is a RESULT, not a crash: it is recorded as
    `{"outcome": "oom"}` and reported with a non-zero exit, because the
    supervisor sees exit codes and cgroup kills, never a Python
    `OutOfMemoryError` raised inside this process.
    """

    from .autobatch import PROBE_RECORDS_DIRNAME, PROBE_STEPS, sidecar_alloc_conf_hash

    params = spec.sam3_params
    batch_size = max(1, int(batch_size))
    records_dir = run_dir_path / PROBE_RECORDS_DIRNAME
    records_dir.mkdir(parents=True, exist_ok=True)
    record_path = records_dir / f"batch_{batch_size}.json"

    def _write(payload: dict) -> None:
        record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    descriptors = _build_dataloader(spec, params, split="train")
    if not descriptors:
        emit_log("Probe found zero training datapoints; nothing to measure.")
        _write({"outcome": "empty_dataset", "batch_size": batch_size})
        return 1

    import torch

    refusal = _runtime_admission_refusal(torch, params)
    if refusal:
        emit_log(refusal)
        _write({"outcome": "refused", "batch_size": batch_size, "reason": refusal})
        return 1
    _seed_everything(spec.seed)

    device, model, matcher, loss_fn, trainable_params = _build_model_and_loss(params)
    del matcher
    # Built here and not in the shared helper on purpose: the optimizer IS
    # part of the peak this probe exists to measure.
    optimizer = torch.optim.AdamW(trainable_params, lr=params.lr)
    autocast_dtype = torch.bfloat16
    densest = _densest_first(descriptors)

    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        optimizer.zero_grad()
        steps = 0
        # The densest tiles are re-walked if the dataset is smaller than the
        # step budget; a probe must reach PROBE_STEPS, not run out of tiles.
        for _pass in range(PROBE_STEPS):
            for batch in collate_batches(densest, batch_size):
                with torch.autocast(
                    device_type="cuda", dtype=autocast_dtype, enabled=True
                ):
                    model_input, targets, outputs = _forward_batch(batch, model, device)
                loss_dict = loss_fn(outputs, targets)
                loss = _core_loss(loss_dict)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
                del batch, model_input, targets, outputs, loss_dict, loss
                if steps % _PROBE_HEARTBEAT_STEPS == 0 or steps >= PROBE_STEPS:
                    # The probe path was otherwise SILENT for its whole
                    # duration: nothing is written until the last step lands,
                    # and the launcher passes a no-op progress callback, so a
                    # user watching the log or the GUI saw nothing at all.
                    emit_log(
                        f"probe batch {batch_size}: step {steps}/{PROBE_STEPS}, "
                        f"reserved "
                        f"{torch.cuda.max_memory_reserved(device) / (1024 ** 3):.2f} GiB"
                    )
                if steps >= PROBE_STEPS:
                    break
            if steps >= PROBE_STEPS:
                break
        if steps < PROBE_STEPS:
            emit_log(
                f"Probe completed only {steps} of {PROBE_STEPS} steps at batch "
                f"{batch_size}; refusing to report an understated peak."
            )
            _write({"outcome": "incomplete", "batch_size": batch_size, "steps": steps})
            return 1
        reserved = int(torch.cuda.max_memory_reserved(device))
        allocated = int(torch.cuda.max_memory_allocated(device))
    except torch.cuda.OutOfMemoryError as exc:
        emit_log(f"Probe at batch {batch_size} ran out of device memory: {exc}")
        _write({"outcome": "oom", "batch_size": batch_size, "detail": str(exc)})
        return 2
    finally:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    _write(
        {
            "outcome": "ok",
            "batch_size": batch_size,
            "steps": steps,
            "accelerator_reserved_peak_bytes": reserved,
            "accelerator_allocated_peak_bytes": allocated,
            "host_peak_bytes": _host_peak_bytes(),
            "observed_at_unix_ns": time.time_ns(),
            # What THIS child actually ran under, hashed the same way the
            # parent hashes it for the fingerprint. The parent computes the
            # fingerprint from `sam3_env_environ()` -- its own idea of the
            # child's environment -- so without this a record could be filed
            # under an "expandable_segments" key by a child that never saw
            # the flag. Measured on courtship, same box, same 30 steps:
            # reserved 9.82 GiB without the flag vs 6.89 GiB with it, 42%
            # apart. In the product path the two always agree; this closes
            # the gap for any child launched outside `_child_environment`.
            "alloc_conf_hash": sidecar_alloc_conf_hash(dict(os.environ)),
        }
    )
    emit_log(
        f"Probe at batch {batch_size}: reserved {reserved / (1024 ** 3):.2f} GiB, "
        f"allocated {allocated / (1024 ** 3):.2f} GiB over {steps} steps."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", required=True, help="Path to the serialised spec.json"
    )
    parser.add_argument(
        "--run-dir", required=True, help="Run directory to write artifacts into"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Measure this configuration's device peak instead of training. "
            "The candidate batch size arrives on the command line, not in "
            "spec.json: the spec carries exactly one batch, and it is "
            "rewritten once, by the parent, with the RESOLVED value."
        ),
    )
    parser.add_argument(
        "--probe-batch",
        type=int,
        default=0,
        help="Batch size to measure; required with --probe",
    )
    args = parser.parse_args(argv)
    if args.probe and args.probe_batch < 1:
        parser.error("--probe requires a positive --probe-batch")

    run_dir_path = Path(args.run_dir).expanduser().resolve()
    run_dir_path.mkdir(parents=True, exist_ok=True)

    spec = _load_spec(Path(args.spec).expanduser().resolve())

    if args.probe:
        return run_probe_measurement(spec, run_dir_path, args.probe_batch)

    ok = run_training(spec, run_dir_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
