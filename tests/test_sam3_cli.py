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
