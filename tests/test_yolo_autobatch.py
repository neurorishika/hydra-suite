"""Pre-launch YOLO batch resolution: contained child, clamped, never `-1`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydra_suite.runtime.process_supervisor import ExitKind
from hydra_suite.runtime.resource_budget import AcceleratorKind
from hydra_suite.training.contracts import (
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)


def _spec(tmp_path, batch, *, imgsz=64, device="cpu"):
    return TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path),
        base_model="yolo.pt",
        hyperparams=TrainingHyperParams(batch=batch, imgsz=imgsz, workers=0),
        device=device,
    )


def _child(*, writes=None, fails=False, canceled=False, seen=None):
    """A fake resolution child that WRITES a file (it never prints a line)."""

    def run_child(command, spec):
        if seen is not None:
            seen.append(tuple(command))
        command = [str(item) for item in command]
        out = Path(command[command.index("--out") + 1])
        if writes is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"resolved": writes}), encoding="utf-8")
        if canceled:
            return {
                "success": False,
                "canceled": True,
                "failure_kind": ExitKind.CANCELED.value,
            }
        return {
            "success": not fails,
            "canceled": False,
            "failure_kind": (
                ExitKind.ORDINARY_FAILURE.value if fails else ExitKind.SUCCESS.value
            ),
            "error_message": "child died" if fails else "",
        }

    return run_child


def _resolve(
    tmp_path,
    *,
    batch=-1,
    device="cpu",
    accelerator=AcceleratorKind.CUDA,
    should_cancel=None,
    **child,
):
    from hydra_suite.training.yolo_autobatch import resolve_yolo_batch

    return resolve_yolo_batch(
        _spec(tmp_path, batch, device=device),
        tmp_path,
        accelerator_kind=accelerator,
        run_child=_child(**child),
        log_cb=lambda message: None,
        should_cancel=should_cancel,
    )


def test_an_explicit_batch_resolves_without_a_child(tmp_path):
    seen: list = []
    from hydra_suite.training.yolo_autobatch import resolve_yolo_batch

    batch, provenance = resolve_yolo_batch(
        _spec(tmp_path, 16),
        tmp_path,
        accelerator_kind=AcceleratorKind.CUDA,
        run_child=_child(writes=99, seen=seen),
    )
    assert (batch, provenance) == (16, "explicit")
    assert seen == []


def test_a_reported_batch_is_taken_as_ultralytics_estimate(tmp_path):
    assert _resolve(tmp_path, writes=24) == (24, "ultralytics_autobatch")


def test_an_absurd_report_is_clamped_not_trusted(tmp_path):
    from hydra_suite.training.yolo_autobatch import YOLO_MAX_AUTO_BATCH

    assert YOLO_MAX_AUTO_BATCH == 64
    assert _resolve(tmp_path, writes=1024) == (64, "ultralytics_autobatch_clamped")


def test_a_nonpositive_report_is_clamped_up_never_propagated(tmp_path):
    assert _resolve(tmp_path, writes=-1) == (1, "ultralytics_autobatch_clamped")


def test_non_cuda_returns_the_default_and_says_so(tmp_path):
    logged: list[str] = []
    from hydra_suite.training.yolo_autobatch import resolve_yolo_batch

    result = resolve_yolo_batch(
        _spec(tmp_path, -1, device="mps"),
        tmp_path,
        accelerator_kind=AcceleratorKind.MPS,
        run_child=_child(writes=48),
        log_cb=logged.append,
    )
    assert result == (TrainingHyperParams().batch, "default_non_cuda")
    assert any("does not measure" in line for line in logged)


def test_a_resolution_failure_falls_back_and_never_propagates_minus_one(tmp_path):
    batch, provenance = _resolve(tmp_path, fails=True)
    assert batch >= 1 and provenance == "fallback"


def test_a_missing_or_unparseable_report_falls_back(tmp_path):
    assert _resolve(tmp_path, writes=None)[1] == "fallback"
    (tmp_path / "batch_resolution.json").write_text("{", encoding="utf-8")
    assert _resolve(tmp_path, writes=None)[1] == "fallback"


def test_a_stale_report_from_a_previous_run_is_never_trusted(tmp_path):
    (tmp_path / "batch_resolution.json").write_text(
        json.dumps({"resolved": 512}), encoding="utf-8"
    )
    assert _resolve(tmp_path, fails=True) == (TrainingHyperParams().batch, "fallback")


def test_cancellation_during_resolution_raises_and_launches_nothing(tmp_path):
    from hydra_suite.training.yolo_autobatch import ResolutionCanceled

    with pytest.raises(ResolutionCanceled):
        _resolve(tmp_path, canceled=True)
    with pytest.raises(ResolutionCanceled):
        _resolve(tmp_path, writes=24, should_cancel=lambda: True)


def test_the_resolution_block_matches_the_sam3_schema(tmp_path):
    from hydra_suite.training.yolo_autobatch import batch_resolution_block

    block = batch_resolution_block(-1, 24, "ultralytics_autobatch")
    assert set(block) == {
        "requested",
        "resolved",
        "provenance",
        "fingerprint",
        "degraded_reasons",
        "measured_reserved_bytes",
        "free_bytes",
        "resolved_at_unix_ns",
    }
    assert block["requested"] == -1
    assert block["resolved"] == 24
    assert block["provenance"] == "ultralytics_autobatch"
    # Honesty: nothing here was measured, so nothing claims a measured peak.
    assert block["measured_reserved_bytes"] == 0
    assert block["free_bytes"] == 0
    assert block["fingerprint"] == ""
    assert block["degraded_reasons"] == []
    assert block["resolved_at_unix_ns"] > 0


def test_the_child_command_carries_the_run_geometry(tmp_path):
    seen: list = []
    _resolve(tmp_path, writes=8, seen=seen)
    command = [str(item) for item in seen[0]]
    assert "hydra_suite.training.yolo_autobatch" in command
    assert str(tmp_path / "batch_resolution.json") in command
    assert "64" in command  # imgsz
    assert command[command.index("--device") + 1] == "cuda:0"


def test_a_missing_label_root_is_reported_not_swallowed(tmp_path):
    """Without labels the estimate ignores assigner memory and skews
    optimistic, so it must be flagged rather than pass as a clean result."""
    from hydra_suite.training.yolo_autobatch import _dataset_label_profile

    max_num_obj, dataset_size, degraded = _dataset_label_profile(tmp_path)
    assert (max_num_obj, dataset_size) == (0, 0)
    assert degraded and "no train labels" in degraded[0]


def test_the_child_device_is_pinned_and_never_an_ultralytics_token(tmp_path):
    """`torch.device()` raises on "auto" and on "0", and the parent already
    pinned CUDA_VISIBLE_DEVICES, so the child must always be told `cuda:0`.
    Forwarding `spec.device` would kill the child on its first line and
    silently degrade every run to `fallback`."""
    from hydra_suite.training.yolo_autobatch import child_command

    for token in ("auto", "0", "cuda:3", ""):
        command = [
            str(item)
            for item in child_command(_spec(tmp_path, -1, device=token), tmp_path)
        ]
        assert command[command.index("--device") + 1] == "cuda:0"


def test_a_degraded_child_report_is_carried_through(tmp_path):
    from hydra_suite.training.yolo_autobatch import child_degraded_reasons

    assert child_degraded_reasons(tmp_path) == []
    (tmp_path / "batch_resolution.json").write_text(
        json.dumps({"resolved": 8, "degraded_reasons": ["no train labels"]}),
        encoding="utf-8",
    )
    assert child_degraded_reasons(tmp_path) == ["no train labels"]


def test_the_child_reports_ultralytics_estimate_to_the_file(tmp_path):
    """The child writes; it does not print. Dropped output cannot degrade it."""
    from hydra_suite.training import yolo_autobatch

    out = tmp_path / "batch_resolution.json"
    yolo_autobatch.main(
        [
            "--model",
            "yolo.pt",
            "--dataset-dir",
            str(tmp_path),
            "--imgsz",
            "640",
            "--device",
            "cuda:0",
            "--default-batch",
            "16",
            "--out",
            str(out),
        ],
        estimate=lambda **kwargs: 24,
    )
    assert json.loads(out.read_text())["resolved"] == 24


def test_max_num_obj_mirrors_the_ultralytics_mosaic_factor(tmp_path):
    """DetectionTrainer.auto_batch uses max(len(label.cls)) * 4."""
    from hydra_suite.training.yolo_autobatch import _dataset_label_profile

    labels = tmp_path / "train" / "labels"
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("0 1 1 1 1\n0 1 1 1 1\n", encoding="utf-8")
    (labels / "b.txt").write_text("0 1 1 1 1\n", encoding="utf-8")
    max_num_obj, dataset_size, degraded = _dataset_label_profile(tmp_path)
    assert max_num_obj == 8
    assert dataset_size == 2
    assert degraded == []


def test_ultralytics_device_ordinals_normalise_for_our_reasoning_only():
    """`_accelerator` does not recognise the bare-digit convention, so
    `device: "0"` -- what the runbook and every Ultralytics user writes --
    classified as CPU and never reached resolution."""
    from hydra_suite.training.yolo_autobatch import normalize_cuda_device

    assert normalize_cuda_device("0") == "cuda:0"
    assert normalize_cuda_device("1") == "cuda:1"
    assert normalize_cuda_device(" 2 ") == "cuda:2"
    # Multi-GPU resolves against the FIRST device: Ultralytics profiles one.
    assert normalize_cuda_device("0,1") == "cuda:0"
    assert normalize_cuda_device("1,0") == "cuda:1"
    # Everything else is passed through untouched.
    for value in ("auto", "mps", "cpu", "cuda:3"):
        assert normalize_cuda_device(value) == value
    assert normalize_cuda_device("") == "auto"
    assert normalize_cuda_device(None) == "auto"


def test_a_multi_gpu_device_says_it_sized_against_one(tmp_path):
    from hydra_suite.training.yolo_autobatch import resolve_yolo_batch

    logged: list[str] = []
    resolve_yolo_batch(
        _spec(tmp_path, -1, device="0,1"),
        tmp_path,
        accelerator_kind=AcceleratorKind.CUDA,
        run_child=_child(writes=24),
        log_cb=logged.append,
    )
    assert any("cuda:0" in line and "profiles one" in line for line in logged)


def test_the_resolution_child_is_classified_as_cuda_for_a_bare_ordinal(tmp_path):
    """The child needs the CUDA classification to get the UUID pin, so the
    spec it runs under carries the normalised device."""
    from hydra_suite.training.yolo_autobatch import resolve_yolo_batch

    seen: list = []

    def run_child(command, spec):
        seen.append(spec.device)
        out = Path(
            [str(i) for i in command][[str(i) for i in command].index("--out") + 1]
        )
        out.write_text(json.dumps({"resolved": 24}), encoding="utf-8")
        return {"success": True, "canceled": False}

    resolve_yolo_batch(
        _spec(tmp_path, -1, device="0"),
        tmp_path,
        accelerator_kind=AcceleratorKind.CUDA,
        run_child=run_child,
    )
    assert seen == ["cuda:0"]
