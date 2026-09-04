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
from hydra_suite.training.sam3_lora.artifacts import completion_path


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
    inject_at = source.index("inject_adapters(model")
    move_at = source.index("model.to(device)")
    assert inject_at < move_at, (
        "model.to(device) runs before inject_adapters; the adapters would be "
        "created on CPU and never moved"
    )
