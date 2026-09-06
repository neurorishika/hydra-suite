"""SAM3 LoRA in-sidecar CLI tests (cli.py).

`cli.py` is the module that runs inside the `hydra-sam3` conda env; it is
never expected to import cleanly on this Mac if `sam3`/`torch` are absent
from the ambient env, EXCEPT for the zero-datapoint refusal path, which must
never reach the `import torch` line at all (mirrors the discipline the
in-process trainer used to test directly -- see the original
`test_sam3_train.py::test_zero_batches_reports_failure_not_success`, task-8
fix round 1, finding 2).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hydra_suite.training.contracts import Sam3LoraParams
from hydra_suite.training.sam3_lora import cli
from hydra_suite.training.sam3_lora.artifacts import completion_path, remove_artifact


def _write_spec(tmp_path, prompt="ant"):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "derived_dataset_dir": str(tmp_path / "dataset"),
                "sam3_params": {
                    "prompt": prompt,
                    "label_quality_acknowledged": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return spec_path


def test_load_spec_reconstructs_sam3_params(tmp_path):
    spec_path = _write_spec(tmp_path)
    spec = cli._load_spec(spec_path)

    assert spec.seed == 42
    assert spec.derived_dataset_dir == str(tmp_path / "dataset")
    assert spec.sam3_params.prompt == "ant"


def test_child_runtime_refuses_oversized_prompt_before_sam3_import():
    from hydra_suite.training.contracts import SAM3_MAX_PROMPT_CODEPOINTS

    params = Sam3LoraParams(
        prompt="x" * (SAM3_MAX_PROMPT_CODEPOINTS + 1), mixed_precision="bf16"
    )
    torch_module = SimpleNamespace(cuda=SimpleNamespace())

    refusal = cli._runtime_admission_refusal(torch_module, params)

    assert refusal is not None
    assert "per-prompt cap" in refusal


def test_run_training_zero_datapoints_returns_false_without_importing_sam3(
    tmp_path, monkeypatch
):
    """Zero datapoints must refuse before touching `sam3`/`torch` at all."""
    spec = cli._load_spec(_write_spec(tmp_path))
    monkeypatch.setattr(cli, "_build_dataloader", lambda spec, params, split: [])

    logs = []
    monkeypatch.setattr(cli, "emit_log", logs.append)

    ok = cli.run_training(spec, tmp_path / "run")

    assert ok is False
    assert not (tmp_path / "run" / "adapters.pt").exists()
    assert any("zero datapoints" in msg for msg in logs)


def test_main_zero_datapoints_exits_nonzero(tmp_path, monkeypatch, capsys):
    spec_path = _write_spec(tmp_path)
    run_dir = tmp_path / "run"
    monkeypatch.setattr(cli, "_build_dataloader", lambda spec, params, split: [])

    rc = cli.main(["--spec", str(spec_path), "--run-dir", str(run_dir)])

    assert rc != 0
    assert not (run_dir / "adapters.pt").exists()


class _FakeCuda:
    def __init__(self, available=True, capability=(8, 0)):
        self._available = available
        self._capability = capability

    def is_available(self):
        return self._available

    def get_device_capability(self):
        return self._capability

    def is_bf16_supported(self):
        return self._available and self._capability[0] >= 8


def test_runtime_precision_matrix_fails_closed():
    torch = SimpleNamespace(cuda=_FakeCuda())
    assert cli._runtime_admission_refusal(torch, Sam3LoraParams()) is None
    assert "only CUDA BF16" in cli._runtime_admission_refusal(
        torch, Sam3LoraParams(mixed_precision="fp16")
    )
    assert "only CUDA BF16" in cli._runtime_admission_refusal(
        torch, Sam3LoraParams(mixed_precision="fp32")
    )
    assert "CUDA device" in cli._runtime_admission_refusal(
        SimpleNamespace(cuda=_FakeCuda(available=False)), Sam3LoraParams()
    )
    assert "8.0" in cli._runtime_admission_refusal(
        SimpleNamespace(cuda=_FakeCuda(capability=(7, 5))), Sam3LoraParams()
    )

    cuda = _FakeCuda()
    cuda.is_bf16_supported = lambda: False
    assert "BF16" in cli._runtime_admission_refusal(
        SimpleNamespace(cuda=cuda), Sam3LoraParams()
    )


def test_runtime_refuses_empty_adapter_scope():
    params = Sam3LoraParams(
        adapt_vision_encoder=False,
        adapt_text_encoder=False,
        adapt_geometry_encoder=False,
        adapt_detr_encoder=False,
        adapt_detr_decoder=False,
        adapt_mask_decoder=False,
    )

    assert (
        "adapter"
        in cli._runtime_admission_refusal(
            SimpleNamespace(cuda=_FakeCuda()), params
        ).lower()
    )


def test_atomic_adapter_writer_rejects_noop_and_promotes_valid_pairs(tmp_path):
    torch = pytest.importorskip("torch")
    artifact = tmp_path / "adapters.pt"
    noop = {
        "block.lora_A": torch.ones((2, 3)),
        "block.lora_B": torch.zeros((4, 2)),
    }

    with pytest.raises(ValueError, match="no-op"):
        cli._write_validated_adapter_artifact(noop, artifact, torch)
    assert not artifact.exists()
    assert not completion_path(artifact).exists()

    valid = {**noop, "block.lora_B": torch.ones((4, 2))}
    cli._write_validated_adapter_artifact(valid, artifact, torch)

    assert artifact.exists()
    assert completion_path(artifact).exists()
    loaded = torch.load(artifact, map_location="cpu", weights_only=True)
    assert set(loaded) == set(valid)


@pytest.mark.parametrize(
    "matrix_a,matrix_b",
    [
        ([[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0]]),
        ([[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0]]),
    ],
)
def test_atomic_adapter_writer_rejects_zero_product_pairs(tmp_path, matrix_a, matrix_b):
    torch = pytest.importorskip("torch")
    adapters = {
        "block.lora_A": torch.tensor(matrix_a),
        "block.lora_B": torch.tensor(matrix_b),
    }

    with pytest.raises(ValueError, match="no-op"):
        cli._write_validated_adapter_artifact(adapters, tmp_path / "adapters.pt", torch)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"block.lora_A": object()},
    ],
)
def test_atomic_adapter_writer_rejects_incomplete_schema(tmp_path, payload):
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError):
        cli._write_validated_adapter_artifact(payload, tmp_path / "adapters.pt", torch)


def test_parent_cleanup_removes_only_exact_private_artifact_staging(tmp_path):
    artifact = tmp_path / "adapters.pt"
    staging = tmp_path / ".adapters.pt.123.validated.tmp"
    marker_staging = tmp_path / ".adapters.pt.complete.json.123.tmp"
    unrelated = tmp_path / ".other.pt.123.validated.tmp"
    for path in (artifact, staging, marker_staging, unrelated):
        path.write_bytes(b"partial")

    remove_artifact(artifact, remove_staging=True)

    assert not artifact.exists()
    assert not staging.exists()
    assert not marker_staging.exists()
    assert unrelated.exists()


def test_collated_batch_moves_to_device_before_target_conversion_and_forward():
    cpu_input = SimpleNamespace(find_targets=["cpu-target-a", "cpu-target-b"])
    gpu_input = SimpleNamespace(find_targets=["gpu-target-a", "gpu-target-b"])
    events = []

    def copy_to_device(value, device, *, non_blocking):
        events.append(("copy", value, device, non_blocking))
        return gpu_input

    class Model:
        def back_convert(self, target):
            events.append(("back_convert", target))
            return f"converted-{target}"

        def __call__(self, model_input):
            events.append(("forward", model_input))
            return "outputs"

    got_input, got_targets, got_outputs = cli._forward_batch(
        {"input": cpu_input},
        Model(),
        "cuda:0",
        copy_to_device=copy_to_device,
    )

    assert got_input is gpu_input
    assert got_targets == ["converted-gpu-target-a", "converted-gpu-target-b"]
    assert got_outputs == "outputs"
    assert events == [
        ("copy", cpu_input, "cuda:0", True),
        ("back_convert", "gpu-target-a"),
        ("back_convert", "gpu-target-b"),
        ("forward", gpu_input),
    ]


def test_loss_wrapper_uses_local_normalization_for_single_process_sidecar():
    captured = {}

    class LossWrapper:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    matcher = object()
    o2m_matcher = object()
    losses = [object()]

    cli._build_loss_wrapper(
        LossWrapper,
        loss_fns_find=losses,
        matcher=matcher,
        o2m_matcher=o2m_matcher,
    )

    assert captured["normalization"] == "local"
    assert captured["loss_fns_find"] is losses
    assert captured["matcher"] is matcher
    assert captured["o2m_matcher"] is o2m_matcher


def test_validation_matching_is_attached_to_every_main_and_aux_output():
    main = {"name": "main", "aux_outputs": [{"name": "aux"}]}
    outputs = SimpleNamespace(output=[[main]])
    calls = []

    def matcher(output, target):
        calls.append((output["name"], target))
        return f"indices-{output['name']}"

    cli._attach_matcher_indices(outputs, ["target"], matcher)

    assert calls == [("main", "target"), ("aux", "target")]
    assert main["indices"] == "indices-main"
    assert main["aux_outputs"][0]["indices"] == "indices-aux"


def test_core_loss_uses_metas_actual_loss_key():
    marker = object()

    assert cli._core_loss({"core_loss": marker}) is marker


def test_adapters_are_injected_before_the_model_moves_to_device():
    """LoRA params must be created before `.to(device)`, not after.

    `inject_adapters` builds fresh lora_A/lora_B Parameters on the device of
    the module it wraps; it does not replay an earlier `.to()`. Injecting
    after the move left every adapter on CPU while the frozen base was on
    CUDA, and the first forward died with "Expected all tensors to be on the
    same device ... mat2 is on cpu".
    """
    import inspect

    from hydra_suite.training.sam3_lora import cli

    source = inspect.getsource(cli.run_training)
    freeze_at = source.index("model.requires_grad_(False)")
    inject_at = source.index("inject_adapters(model")
    move_at = source.index("model.to(device)")
    assert (
        freeze_at < inject_at
    ), "the complete SAM3 base must be frozen before LoRA adapters are injected"
    assert inject_at < move_at, (
        "model.to(device) runs before inject_adapters; the adapters would be "
        "created on CPU and never moved"
    )
    assert "_validated_lora_trainables" in source


def test_lora_trainable_validation_rejects_estimator_shape_drift():
    torch = pytest.importorskip("torch")

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.zeros(2, 3))
            self.lora_B = torch.nn.Parameter(torch.zeros(4, 2))

    model = torch.nn.Module()
    model.adapter = Adapter()
    tensors, count = cli._validated_lora_trainables(
        model, adapted_modules=1, expected_parameters=14
    )
    assert len(tensors) == 2
    assert count == 14

    with pytest.raises(RuntimeError, match="expected_parameters=13"):
        cli._validated_lora_trainables(model, adapted_modules=1, expected_parameters=13)


def test_loss_term_summary_renders_headline_terms():
    torch = pytest.importorskip("torch")
    from hydra_suite.training.sam3_lora.cli import _loss_term_summary

    summary = _loss_term_summary(
        {
            "loss_ce": torch.tensor(0.45),
            "loss_mask": torch.tensor(0.0),
            "loss_ce_aux_0": torch.tensor(9.9),  # not a headline term
            "indices": [1, 2, 3],  # not a scalar tensor
        }
    )
    assert "loss_ce=0.4500" in summary
    assert "loss_mask=0.0000" in summary
    assert "aux" not in summary


def test_loss_term_summary_tolerates_a_non_dict():
    torch = pytest.importorskip("torch")
    from hydra_suite.training.sam3_lora.cli import _loss_term_summary

    assert _loss_term_summary(torch.tensor(1.0)) == ""


def test_finite_loss_passes_and_non_finite_aborts():
    """A NaN loss must stop the run where it happens, not 700 steps later.

    SAM3's Hungarian matcher only reports the damage downstream, as
    "matrix contains invalid numeric entries", long after the adapter has
    become worthless.
    """
    torch = pytest.importorskip("torch")
    from hydra_suite.training.sam3_lora.cli import _assert_finite_loss

    terms = {"loss_ce": torch.tensor(0.5)}
    _assert_finite_loss(torch.tensor(1.5), epoch=0, step=1, loss_dict=terms)

    for bad in (float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="non-finite"):
            _assert_finite_loss(torch.tensor(bad), epoch=2, step=30, loss_dict=terms)


def test_non_finite_abort_names_the_step_and_terms():
    torch = pytest.importorskip("torch")
    from hydra_suite.training.sam3_lora.cli import _assert_finite_loss

    with pytest.raises(RuntimeError) as excinfo:
        _assert_finite_loss(
            torch.tensor(float("nan")),
            epoch=4,
            step=730,
            loss_dict={"loss_mask": torch.tensor(0.0)},
        )
    message = str(excinfo.value)
    assert "epoch 4" in message and "step 730" in message
    assert "loss_mask=0.0000" in message


def test_loss_window_averages_positives_and_negatives_together():
    """The bug this replaces: the logged micro-batch was always a negative.

    The accumulation boundary lands on a fixed micro_idx, so with one
    negative interleaved per tile every logged line had zero matched loss
    and the run looked collapsed while training normally.
    """
    torch = pytest.importorskip("torch")
    from hydra_suite.training.sam3_lora.cli import _LossWindow

    window = _LossWindow()
    positive = {"loss_ce": torch.tensor(0.40), "presence_loss": torch.tensor(0.10)}
    negative = {"loss_ce": torch.tensor(0.00), "presence_loss": torch.tensor(0.02)}
    window.add(torch.tensor(300.0), positive)
    window.add(torch.tensor(20.0), negative)

    summary = window.summary()
    assert "loss 160.0000" in summary
    assert "loss_ce=0.2000" in summary, "a positive batch must not be averaged away"
    assert "presence_loss=0.0600" in summary


def test_loss_window_reset_clears_the_previous_window():
    torch = pytest.importorskip("torch")
    from hydra_suite.training.sam3_lora.cli import _LossWindow

    window = _LossWindow()
    window.add(torch.tensor(9.0), {"loss_ce": torch.tensor(9.0)})
    window.reset()
    assert window.summary() == "loss n/a"
    window.add(torch.tensor(2.0), {"loss_ce": torch.tensor(1.0)})
    assert "loss 2.0000" in window.summary()


def test_training_loop_feeds_every_micro_batch_into_the_loss_window():
    """Regression: the accumulate call was silently missing.

    `_LossWindow` was created and logged, but nothing ever called `.add()`,
    so every line in a 2000-step run read "loss n/a" and the run produced no
    usable loss trace at all. Source-level because the loop needs CUDA.
    """
    import inspect

    from hydra_suite.training.sam3_lora import cli

    source = inspect.getsource(cli.run_training)
    add_at = source.index("loss_window.add(")
    backward_at = source.index("grad_accum).backward()")
    reset_at = source.index("loss_window.reset()")

    assert add_at < backward_at, "the window must see the loss before backward"
    assert add_at < reset_at, "accumulate must precede the window reset"
    assert source.count("loss_window.add(") == 1


def test_optimizer_step_is_skipped_on_non_finite_gradients():
    """clip_grad_norm_ propagates inf/NaN into the weights rather than
    blocking it, so the step must be gated on the returned norm.

    Once adapter weights go NaN the model emits NaN logits forever; the
    failure then surfaces far downstream (a NaN loss, or the Hungarian
    matcher rejecting the cost matrix hundreds of steps later).
    """
    import inspect

    from hydra_suite.training.sam3_lora import cli

    source = inspect.getsource(cli.run_training)
    assert "total_norm = torch.nn.utils.clip_grad_norm_(" in source
    guard_at = source.index("if torch.isfinite(total_norm):")
    step_at = source.index("optimizer.step()")
    assert guard_at < step_at, "optimizer.step() must sit behind the finite check"
    # The schedule must advance regardless, or a skipped step stalls the LR.
    assert source.index("scheduler.step()") > source.index("consecutive_skipped += 1")


def test_a_long_run_of_skipped_steps_aborts():
    from hydra_suite.training.sam3_lora import cli

    assert cli.MAX_CONSECUTIVE_SKIPPED_STEPS > 0
    source = __import__("inspect").getsource(cli.run_training)
    assert "MAX_CONSECUTIVE_SKIPPED_STEPS" in source
    assert "no longer learning" in source


def _make_checkpoints(directory, count, size=1024):
    for n in range(1, count + 1):
        (directory / f"epoch_{n:03d}.pt").write_bytes(b"x" * size)
        (directory / f"epoch_{n:03d}.pt.complete.json").write_text("{}")
    return sorted(directory.glob("epoch_*.pt"))


def test_retention_keeps_every_checkpoint_when_the_budget_allows(tmp_path):
    """A count cap ALWAYS destroys the earliest epochs; a budget only prunes
    when disk actually demands it."""
    from hydra_suite.training.sam3_lora.cli import plan_checkpoint_retention

    paths = _make_checkpoints(tmp_path, 10)
    removed = plan_checkpoint_retention(
        paths, free_bytes=100 * 1024 * 1024, adapter_bytes=1024
    )
    assert removed == []


def test_retention_thins_the_middle_and_keeps_first_and_last(tmp_path):
    from hydra_suite.training.sam3_lora.cli import plan_checkpoint_retention

    paths = _make_checkpoints(tmp_path, 9)
    # Budget deliberately tiny: only a few adapters fit.
    removed = plan_checkpoint_retention(paths, free_bytes=0, adapter_bytes=1024)
    kept = [p for p in paths if p not in removed]
    assert kept, "retention must never delete everything"
    assert kept[0].name == "epoch_001.pt", "the earliest epoch is the stall evidence"
    assert kept[-1].name == "epoch_009.pt", "the newest epoch is the salvage artifact"
    assert len(kept) < len(paths)


def test_retention_never_deletes_the_checkpoint_just_written(tmp_path):
    from hydra_suite.training.sam3_lora.cli import plan_checkpoint_retention

    paths = _make_checkpoints(tmp_path, 4)
    removed = plan_checkpoint_retention(paths, free_bytes=0, adapter_bytes=10**9)
    assert paths[-1] not in removed


def test_retention_budget_is_derived_from_measured_free_space_and_size(tmp_path):
    """No hardcoded byte budget and no hardcoded count may appear."""
    import inspect

    from hydra_suite.training.sam3_lora import cli

    assert not hasattr(cli, "KEEP_EPOCH_CHECKPOINTS")
    source = inspect.getsource(cli.enforce_checkpoint_budget) + inspect.getsource(
        cli.checkpoint_budget_bytes
    )
    assert "disk_usage" in source
    assert "stat()" in source or "st_size" in source


def test_enforce_budget_removes_markers_and_logs_loudly(tmp_path):
    from hydra_suite.training.sam3_lora import cli

    _make_checkpoints(tmp_path, 6)
    messages: list[str] = []
    removed = cli.enforce_checkpoint_budget(
        tmp_path, budget_bytes=2048, log=messages.append
    )
    assert removed
    for stale in removed:
        assert not stale.exists()
        assert not stale.with_name(stale.name + ".complete.json").exists()
    assert any("budget" in m.lower() for m in messages)


def test_an_unmeasurable_budget_retains_everything(tmp_path):
    """A failed measurement must never masquerade as a binding budget."""
    import shutil

    from hydra_suite.training.sam3_lora import cli

    paths = _make_checkpoints(tmp_path, 6)

    def _boom(_path):
        raise OSError("no such device")

    messages: list[str] = []
    original = shutil.disk_usage
    shutil.disk_usage = _boom
    try:
        removed = cli.enforce_checkpoint_budget(tmp_path, log=messages.append)
    finally:
        shutil.disk_usage = original

    assert removed == []
    assert all(path.exists() for path in paths)
    assert any("could not be measured" in m for m in messages)


def test_enforce_budget_is_safe_on_a_missing_directory(tmp_path):
    from hydra_suite.training.sam3_lora import cli

    assert cli.enforce_checkpoint_budget(tmp_path / "absent") == []


def test_val_series_is_appended_per_epoch_as_jsonl(tmp_path):
    from hydra_suite.training.sam3_lora import cli

    cli.append_val_record(tmp_path, {"epoch": 1, "val_loss_mean": 2.0})
    cli.append_val_record(tmp_path, {"epoch": 2, "val_loss_mean": 1.5})

    lines = (tmp_path / cli.VAL_SERIES_FILENAME).read_text().strip().splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2]


def test_val_cadence_defaults_to_every_epoch_and_is_env_configurable(monkeypatch):
    from hydra_suite.training.sam3_lora import cli

    monkeypatch.delenv(cli.VAL_CADENCE_ENV, raising=False)
    assert cli.val_cadence() == 1
    monkeypatch.setenv(cli.VAL_CADENCE_ENV, "3")
    assert cli.val_cadence() == 3
    monkeypatch.setenv(cli.VAL_CADENCE_ENV, "garbage")
    assert cli.val_cadence() == 1


def test_per_epoch_validation_cannot_perturb_training():
    """Evidence recording must be provably behaviour-preserving: RNG state is
    saved and restored around the mid-run validation pass."""
    import inspect

    from hydra_suite.training.sam3_lora import cli

    source = inspect.getsource(cli._record_epoch_validation)
    assert "get_rng_state" in source and "set_rng_state" in source
    assert "model.train()" in source


class _RngBurningModel:
    """Stub standing in for the SAM3 image model on the no-`sam3` box.

    Records mode transitions; the evaluation stub burns every RNG stream so a
    missing restore is observable rather than merely un-greppable.
    """

    def __init__(self) -> None:
        self.training = True
        self.mode_calls: list[str] = []

    def train(self) -> None:
        self.training = True
        self.mode_calls.append("train")

    def eval(self) -> None:
        self.training = False
        self.mode_calls.append("eval")


def _rng_fingerprint():
    import random as _random

    import numpy as _np
    import torch as _torch

    return (
        _random.getstate(),
        _np.random.get_state()[1].tobytes(),
        _torch.get_rng_state().clone(),
    )


def _assert_same_rng(before, after):
    import torch as _torch

    assert before[0] == after[0], "python RNG stream was not restored"
    assert before[1] == after[1], "numpy RNG stream was not restored"
    assert _torch.equal(before[2], after[2]), "torch RNG stream was not restored"


def test_mid_run_validation_restores_every_rng_stream(tmp_path, monkeypatch):
    """The load-bearing property of the whole branch: recording evidence must
    not move a single number the next epoch draws."""
    pytest.importorskip("torch")
    import random as _random

    import numpy as _np
    import torch as _torch

    from hydra_suite.training.sam3_lora import cli

    def _burn(*_args, **_kwargs):
        _random.random()
        _np.random.rand()
        _torch.rand(4)
        return {
            "val_loss_mean": 1.25,
            "val_batches": 3,
            "val_terms_mean": {"loss_ce": 1.0},
            "elapsed_s": 0.5,
        }

    monkeypatch.setattr(cli, "_evaluate_split", _burn)
    model = _RngBurningModel()

    _random.seed(7)
    _np.random.seed(7)
    _torch.manual_seed(7)
    before = _rng_fingerprint()

    record = cli._record_epoch_validation(
        model, None, None, None, None, None, None, True, tmp_path, 2
    )

    _assert_same_rng(before, _rng_fingerprint())
    assert model.training, "the model must be returned to train mode"
    assert record["epoch"] == 2
    assert record["val_cadence"] == 1, "effective cadence must be stamped"


def test_rng_is_restored_even_when_validation_raises(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import random as _random

    import numpy as _np
    import torch as _torch

    from hydra_suite.training.sam3_lora import cli

    def _burn_then_fail(*_args, **_kwargs):
        _random.random()
        _np.random.rand()
        _torch.rand(4)
        raise RuntimeError("CUDA OOM during validation")

    monkeypatch.setattr(cli, "_evaluate_split", _burn_then_fail)
    model = _RngBurningModel()

    _random.seed(11)
    _np.random.seed(11)
    _torch.manual_seed(11)
    before = _rng_fingerprint()

    with pytest.raises(RuntimeError):
        cli._record_epoch_validation(
            model, None, None, None, None, None, None, True, tmp_path, 3
        )

    _assert_same_rng(before, _rng_fingerprint())
    assert model.training


def test_cadence_is_stamped_so_a_gap_is_not_mistaken_for_a_crash(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from hydra_suite.training.sam3_lora import cli

    monkeypatch.setenv(cli.VAL_CADENCE_ENV, "2")
    monkeypatch.setattr(
        cli,
        "_evaluate_split",
        lambda *a, **k: {
            "val_loss_mean": 1.0,
            "val_batches": 1,
            "val_terms_mean": {},
            "elapsed_s": 0.1,
        },
    )
    cli._record_epoch_validation(
        _RngBurningModel(), None, None, None, None, None, None, True, tmp_path, 4
    )
    row = json.loads((tmp_path / cli.VAL_SERIES_FILENAME).read_text().strip())
    assert row["val_cadence"] == 2


def test_a_stat_failure_skips_pruning_instead_of_killing_the_run(tmp_path):
    """Retention is best-effort housekeeping; it runs on the training path and
    must never raise into `run_training`."""
    from pathlib import Path as _Path

    from hydra_suite.training.sam3_lora import cli

    paths = _make_checkpoints(tmp_path, 6)
    original = _Path.stat

    def _boom(self, *args, **kwargs):
        if self.name.startswith("epoch_"):
            raise OSError("stale NFS file handle")
        return original(self, *args, **kwargs)

    messages: list[str] = []
    _Path.stat = _boom
    try:
        removed = cli.enforce_checkpoint_budget(tmp_path, log=messages.append)
    finally:
        _Path.stat = original

    assert removed == []
    assert all(path.exists() for path in paths)
    assert any("could not be measured" in m for m in messages)


def test_val_record_carries_the_full_loss_decomposition_and_its_cost(tmp_path):
    import inspect

    from hydra_suite.training.sam3_lora import cli

    source = inspect.getsource(cli._evaluate_split)
    assert "val_loss_mean" in source
    assert "elapsed_s" in source
    # The terminal artifact keeps its existing shape; the series adds to it.
    assert "informational only" in inspect.getsource(cli._evaluate_and_write)


def test_nothing_selects_a_checkpoint_on_the_recorded_series():
    """Selection stays last-epoch; a study measured every per-query validation
    signal ANTI-correlating with held-out AP."""
    import inspect

    from hydra_suite.training.sam3_lora import cli

    source = inspect.getsource(cli._record_epoch_validation)
    assert "anti-correlat" in source.lower()
    assert "selection" in source.lower()


def test_epoch_checkpoints_never_shadow_the_completion_signal():
    """`adapters.pt` is the launcher's 'run finished' marker.

    A per-epoch write to that name would make a killed run look complete, so
    salvage lives in a subdirectory under its own names.
    """
    import inspect

    from hydra_suite.training.sam3_lora import cli

    source = inspect.getsource(cli._write_epoch_checkpoint)
    assert cli.EPOCH_CHECKPOINT_DIRNAME in source
    assert "adapters.pt" not in source

    loop = inspect.getsource(cli.run_training)
    # Salvage is skipped on the final epoch; the real artifact follows.
    assert "epoch + 1 < params.epochs" in loop
