"""Typed, metadata-only SAM3 resource admission tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from hydra_suite.runtime.resource_budget import AcceleratorKind, ResourceObservation
from hydra_suite.training.contracts import (
    PublishPolicy,
    Sam3LoraParams,
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.sam3_lora import preflight as pf


def _spec(tmp_path, **param_overrides):
    params = Sam3LoraParams(
        prompt=param_overrides.pop("prompt", "ant"),
        label_quality_acknowledged=param_overrides.pop("ack", True),
        negative_prompts=param_overrides.pop("negative_prompts", ["background"]),
        **param_overrides,
    )
    return TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[SourceDataset(path="/tmp/x", level="polygon")],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
        sam3_params=params,
    )


def _cuda(*, free_gib=48, total_gib=48, major=8):
    return pf.CudaDeviceObservation(
        index=0,
        uuid="GPU-physical-0",
        pci_bus_id="00000000:01:00.0",
        name="Test CUDA",
        compute_capability=(major, 0),
        free_bytes=free_gib * pf.GiB,
        total_bytes=total_gib * pf.GiB,
    )


def _host(*, total_gib=64, available_gib=56):
    return ResourceObservation(
        total_host_bytes=total_gib * pf.GiB,
        available_host_bytes=available_gib * pf.GiB,
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_name="Test CUDA",
        total_accelerator_bytes=48 * pf.GiB,
        available_accelerator_bytes=48 * pf.GiB,
    )


def _write_coco(tmp_path, *, tiles=4, instances_per_tile=7, split="train"):
    split_dir = tmp_path / "dataset" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    images = [
        {
            "id": index + 1,
            "file_name": f"tile-{index}.png",
            "width": 1008,
            "height": 1008,
        }
        for index in range(tiles)
    ]
    annotations = []
    ann_id = 1
    for image in images:
        for _ in range(instances_per_tile):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image["id"],
                    "iscrowd": 0,
                    "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                }
            )
            ann_id += 1
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations}), encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def _disk(monkeypatch):
    monkeypatch.setattr(pf, "_free_disk_bytes", lambda _path: 100 * pf.GiB)


def _decision(spec, *, cuda=None, host=None):
    cuda = cuda if cuda is not None else _cuda()
    host = host if host is not None else _host()
    return pf.assess_preflight(spec, cuda_device=cuda, observation=host)


def _allocation(phase, name):
    return dict(phase.dominant_allocations)[name]


def test_healthy_bf16_ampere_spec_is_admitted_without_heavy_imports(tmp_path):
    _write_coco(tmp_path)
    sys.modules.pop("ultralytics", None)

    decision = _decision(_spec(tmp_path))

    assert decision.admitted
    assert decision.refusals == ()
    assert decision.cuda_device.uuid == "GPU-physical-0"
    assert decision.budget.estimator_version == pf.SAM3_ESTIMATOR_VERSION
    assert decision.budget.limits.batch_size == 1
    assert decision.budget.limits.workers == 0
    assert "ultralytics" not in sys.modules


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_unimplemented_precision_modes_fail_closed(tmp_path, precision):
    _write_coco(tmp_path)

    decision = _decision(_spec(tmp_path, mixed_precision=precision))

    assert not decision.admitted
    assert any("bf16" in reason.lower() for reason in decision.refusals)


def test_pre_ampere_bf16_is_refused_instead_of_falling_back(tmp_path):
    _write_coco(tmp_path)

    decision = _decision(_spec(tmp_path), cuda=_cuda(major=7))

    assert not decision.admitted
    assert any("compute capability" in reason.lower() for reason in decision.refusals)
    assert any("8.0" in reason for reason in decision.refusals)


def test_no_cuda_is_refused(tmp_path):
    _write_coco(tmp_path)

    decision = pf.assess_preflight(
        _spec(tmp_path),
        cuda_device=None,
        observation=ResourceObservation(
            total_host_bytes=64 * pf.GiB,
            available_host_bytes=56 * pf.GiB,
        ),
    )

    assert not decision.admitted
    assert any("cuda" in reason.lower() for reason in decision.refusals)


def test_absolute_host_reserve_refuses_at_boundary(tmp_path):
    _write_coco(tmp_path)
    spec = _spec(tmp_path, host_reserve_gb=12.0, host_reserve_fraction=0.1)
    observation = _host(total_gib=64, available_gib=20)

    decision = _decision(spec, host=observation)

    assert decision.budget.reserved_host_bytes == 12 * pf.GiB
    assert not decision.admitted
    assert any("host memory" in reason.lower() for reason in decision.refusals)


def test_proportional_host_reserve_can_dominate_absolute_floor(tmp_path):
    _write_coco(tmp_path)
    spec = _spec(tmp_path, host_reserve_gb=4.0, host_reserve_fraction=0.25)

    decision = _decision(spec)

    assert decision.budget.reserved_host_bytes == 16 * pf.GiB


def test_hard_limit_headroom_cannot_consume_reserved_host_memory(tmp_path):
    _write_coco(tmp_path)
    spec = _spec(
        tmp_path,
        host_reserve_gb=8.0,
        host_reserve_fraction=0.0,
        host_limit_headroom_fraction=1.25,
    )
    roomy = _decision(spec, host=_host(total_gib=128, available_gib=120))
    raw_peak = roomy.budget.host_peak_bytes
    observation = ResourceObservation(
        total_host_bytes=128 * pf.GiB,
        available_host_bytes=raw_peak
        + int(128 * pf.GiB * pf._MINIMUM_HOST_RESERVE_FRACTION),
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_name="Test CUDA",
        total_accelerator_bytes=48 * pf.GiB,
        available_accelerator_bytes=48 * pf.GiB,
    )

    decision = _decision(spec, host=observation)

    assert decision.budget.host_peak_bytes <= decision.budget.usable_host_bytes
    assert decision.containment_hard_host_bytes > decision.budget.usable_host_bytes
    assert not decision.admitted
    assert any("hard containment limit" in reason for reason in decision.refusals)


def test_admitted_hard_limit_plus_reserve_never_exceeds_available_host(tmp_path):
    _write_coco(tmp_path)

    decision = _decision(_spec(tmp_path))

    assert decision.admitted
    assert (
        decision.containment_hard_host_bytes + decision.budget.reserved_host_bytes
        <= decision.budget.available_host_bytes
    )


def test_zero_configured_reserve_cannot_disable_machine_survival_floor(tmp_path):
    _write_coco(tmp_path)
    spec = _spec(tmp_path, host_reserve_gb=0.0, host_reserve_fraction=0.0)

    decision = _decision(spec, host=_host(total_gib=128, available_gib=120))

    assert decision.admitted
    assert decision.policy.reserve_host_bytes == pf._MINIMUM_HOST_RESERVE_BYTES
    assert decision.policy.reserve_host_fraction == pf._MINIMUM_HOST_RESERVE_FRACTION
    assert decision.budget.reserved_host_bytes >= pf._MINIMUM_HOST_RESERVE_BYTES
    assert decision.budget.reserved_host_bytes >= int(
        128 * pf.GiB * pf._MINIMUM_HOST_RESERVE_FRACTION
    )
    assert (
        decision.containment_hard_host_bytes + decision.budget.reserved_host_bytes
        <= decision.budget.available_host_bytes
    )


def test_cuda_safety_fraction_cannot_expose_last_ten_percent(tmp_path):
    _write_coco(tmp_path)

    decision = _decision(_spec(tmp_path, cuda_safety_fraction=1.0))

    assert decision.policy.accelerator_safety_fraction == pytest.approx(
        pf._MAXIMUM_CUDA_SAFETY_FRACTION
    )


def test_crowded_tile_increases_dense_mask_peak(tmp_path):
    _write_coco(tmp_path, instances_per_tile=1)
    sparse = _decision(_spec(tmp_path))
    _write_coco(tmp_path, instances_per_tile=80)
    crowded = _decision(_spec(tmp_path))

    sparse_training = next(p for p in sparse.request.phases if p.name == "training")
    crowded_training = next(p for p in crowded.request.phases if p.name == "training")
    assert crowded.dataset.max_active_instances_per_tile == 80
    assert _allocation(crowded_training, "dense masks") > _allocation(
        sparse_training, "dense masks"
    )
    assert crowded.budget.accelerator_peak_bytes > sparse.budget.accelerator_peak_bytes


def test_validation_crowding_contributes_to_peak(tmp_path):
    _write_coco(tmp_path, instances_per_tile=20)
    _write_coco(tmp_path, tiles=1, instances_per_tile=90, split="valid")

    decision = _decision(_spec(tmp_path))

    assert decision.dataset.max_active_instances_per_tile == 90
    validation = next(p for p in decision.request.phases if p.name == "validation")
    assert _allocation(validation, "dense masks") == 90 * 1008**2 * 5


def test_multi_polygon_annotation_counts_one_example_and_multiple_masks(tmp_path):
    _write_coco(tmp_path, tiles=1, instances_per_tile=20)
    path = tmp_path / "dataset" / "train" / "_annotations.coco.json"
    coco = json.loads(path.read_text(encoding="utf-8"))
    coco["annotations"][0]["segmentation"].append([0, 0, 2, 0, 2, 2])
    path.write_text(json.dumps(coco), encoding="utf-8")

    profile = pf._dataset_profile(str(tmp_path / "dataset"))

    assert profile.train_instances == 20
    assert profile.max_active_instances_per_tile == 21


@pytest.mark.parametrize(
    "segmentation",
    [
        [0, 0, 2, 0, 2, 2],
        [[0, 0], [2, 0], [2, 2]],
    ],
)
def test_preflight_uses_loader_polygon_semantics(tmp_path, segmentation):
    _write_coco(tmp_path, tiles=1, instances_per_tile=20)
    path = tmp_path / "dataset" / "train" / "_annotations.coco.json"
    coco = json.loads(path.read_text(encoding="utf-8"))
    coco["annotations"][0]["segmentation"] = segmentation
    path.write_text(json.dumps(coco), encoding="utf-8")

    profile = pf._dataset_profile(str(tmp_path / "dataset"))

    assert profile.train_instances == 20
    assert profile.max_active_instances_per_tile == 20
    assert profile.polygon_count == 20


def test_compact_high_cardinality_metadata_is_rejected_before_json_load(
    tmp_path, monkeypatch
):
    path = tmp_path / "many-values.json"
    path.write_text('{"images":[0,1,2,3,4,5]}', encoding="utf-8")
    monkeypatch.setattr(pf, "_MAX_JSON_VALUES", 6)
    monkeypatch.setattr(
        pf.json,
        "loads",
        lambda _raw: pytest.fail("unbounded JSON materialization was reached"),
    )

    with pytest.raises(ValueError, match="cardinality"):
        pf._load_coco(path)


def test_raw_metadata_read_is_hard_capped_with_compact_fixture(tmp_path, monkeypatch):
    path = tmp_path / "raw-cap.json"
    path.write_text('{"images":[]}', encoding="utf-8")
    monkeypatch.setattr(pf, "_MAX_COCO_METADATA_BYTES", 8)
    monkeypatch.setattr(
        pf.json,
        "loads",
        lambda _raw: pytest.fail("unbounded JSON materialization was reached"),
    )

    with pytest.raises(ValueError, match="metadata-only preflight cap"):
        pf._load_coco(path)


def test_compact_deep_metadata_is_rejected_before_json_load(tmp_path, monkeypatch):
    path = tmp_path / "deep.json"
    path.write_text('{"images":[[[[[]]]]]}', encoding="utf-8")
    monkeypatch.setattr(pf, "_MAX_JSON_DEPTH", 4)
    monkeypatch.setattr(
        pf.json,
        "loads",
        lambda _raw: pytest.fail("unbounded JSON materialization was reached"),
    )

    with pytest.raises(ValueError, match="nesting"):
        pf._load_coco(path)


def test_estimated_parsed_metadata_is_capped_before_json_load(tmp_path, monkeypatch):
    path = tmp_path / "expanded.json"
    path.write_text('{"images":[{"file_name":"abcdefghij"}]}', encoding="utf-8")
    monkeypatch.setattr(pf, "_MAX_ESTIMATED_PARSED_BYTES", 64)
    monkeypatch.setattr(
        pf.json,
        "loads",
        lambda _raw: pytest.fail("unbounded JSON materialization was reached"),
    )

    with pytest.raises(ValueError, match="parsed-memory"):
        pf._load_coco(path)


def test_invalid_or_crowd_polygons_do_not_satisfy_example_floor(tmp_path):
    _write_coco(tmp_path, tiles=1, instances_per_tile=20)
    path = tmp_path / "dataset" / "train" / "_annotations.coco.json"
    coco = json.loads(path.read_text(encoding="utf-8"))
    for index, annotation in enumerate(coco["annotations"]):
        annotation["iscrowd"] = 1 if index < 10 else 0
        if index >= 10:
            annotation["segmentation"] = [[0, 0, 1, 1]]
    path.write_text(json.dumps(coco), encoding="utf-8")

    decision = _decision(_spec(tmp_path))

    assert decision.dataset.train_instances == 0
    assert not decision.admitted


def test_batch_size_increases_collation_and_accelerator_peak(tmp_path):
    _write_coco(tmp_path)

    batch_one = _decision(_spec(tmp_path, batch=1))
    batch_two = _decision(_spec(tmp_path, batch=2))

    assert (
        batch_two.budget.accelerator_peak_bytes
        > batch_one.budget.accelerator_peak_bytes
    )
    assert not batch_two.admitted


def test_tile_count_does_not_scale_streamed_tensor_or_device_peak(tmp_path):
    _write_coco(tmp_path, tiles=1, instances_per_tile=20)
    one_tile = _decision(_spec(tmp_path))
    _write_coco(tmp_path, tiles=1000, instances_per_tile=20)
    many_tiles = _decision(_spec(tmp_path))

    one_training = next(p for p in one_tile.request.phases if p.name == "training")
    many_training = next(p for p in many_tiles.request.phases if p.name == "training")
    for allocation in (
        "decoded tiles",
        "transformed tile tensors",
        "collated image tensors",
        "dense masks",
    ):
        assert _allocation(one_training, allocation) == _allocation(
            many_training, allocation
        )
    assert (
        one_tile.budget.accelerator_peak_bytes
        == many_tiles.budget.accelerator_peak_bytes
    )
    assert many_tiles.budget.host_peak_bytes > one_tile.budget.host_peak_bytes


def test_negative_query_descriptor_memory_scales_with_tiles_and_query_count(tmp_path):
    _write_coco(tmp_path, tiles=1000, instances_per_tile=1)
    prompts = [f"background-{index}" for index in range(100)]

    no_negatives = _decision(_spec(tmp_path, num_negatives=0, negative_prompts=prompts))
    many_negatives = _decision(
        _spec(tmp_path, num_negatives=100, negative_prompts=prompts)
    )
    no_negative_training = next(
        phase for phase in no_negatives.request.phases if phase.name == "training"
    )
    many_negative_training = next(
        phase for phase in many_negatives.request.phases if phase.name == "training"
    )

    assert _allocation(
        many_negative_training, "negative query descriptors"
    ) > _allocation(no_negative_training, "negative query descriptors")
    assert many_negatives.budget.host_peak_bytes > no_negatives.budget.host_peak_bytes


@pytest.mark.parametrize("num_negatives", [-1, 101])
def test_negative_query_count_outside_typed_limit_is_refused(tmp_path, num_negatives):
    _write_coco(tmp_path)

    decision = _decision(_spec(tmp_path, num_negatives=num_negatives))

    assert not decision.admitted
    assert any("num_negatives" in reason for reason in decision.refusals)


def test_negative_prompt_pool_cardinality_and_bytes_are_bounded(tmp_path):
    _write_coco(tmp_path)

    too_many = _decision(
        _spec(
            tmp_path,
            num_negatives=1,
            negative_prompts=[
                f"background-{index}"
                for index in range(pf._MAX_NEGATIVE_PROMPT_COUNT + 1)
            ],
        )
    )
    too_large = _decision(
        _spec(
            tmp_path,
            num_negatives=1,
            negative_prompts=["x" * (pf._MAX_NEGATIVE_PROMPT_BYTES + 1)],
        )
    )

    assert not too_many.admitted
    assert any("entries" in reason for reason in too_many.refusals)
    assert not too_large.admitted
    assert any("UTF-8 bytes" in reason for reason in too_large.refusals)


def test_configured_prompt_bytes_are_capped_even_when_manifest_takes_precedence(
    tmp_path,
):
    _write_coco(tmp_path)
    manifest = tmp_path / "dataset" / "build_manifest.json"
    manifest.write_text(json.dumps({"negative_prompts": ["small"]}), encoding="utf-8")
    spec = _spec(
        tmp_path,
        num_negatives=1,
        negative_prompts=["x" * pf.SAM3_MAX_CONFIGURED_PROMPT_BYTES],
    )

    decision = _decision(spec)

    assert not decision.admitted
    assert any("per-prompt cap" in reason.lower() for reason in decision.refusals)


def test_over_cardinality_configured_pool_is_not_iterated_even_with_manifest(
    tmp_path,
):
    class BombList(list):
        def __iter__(self):
            pytest.fail("unsafe configured prompt pool must not be iterated")

    _write_coco(tmp_path)
    manifest = tmp_path / "dataset" / "build_manifest.json"
    manifest.write_text(json.dumps({"negative_prompts": ["small"]}), encoding="utf-8")
    spec = _spec(tmp_path, num_negatives=1)
    spec.sam3_params.negative_prompts = BombList(
        ["x"] * (pf._MAX_NEGATIVE_PROMPT_COUNT + 1)
    )

    decision = _decision(spec)

    assert not decision.admitted
    assert any(
        "configured negative prompts" in reason.lower() for reason in decision.refusals
    )


def test_resolved_manifest_prompt_obeys_per_prompt_cap(tmp_path):
    from hydra_suite.training.contracts import SAM3_MAX_PROMPT_CODEPOINTS

    _write_coco(tmp_path)
    manifest = tmp_path / "dataset" / "build_manifest.json"
    manifest.write_text(
        json.dumps({"negative_prompts": ["x" * (SAM3_MAX_PROMPT_CODEPOINTS + 1)]}),
        encoding="utf-8",
    )

    decision = _decision(_spec(tmp_path, num_negatives=1))

    assert not decision.admitted
    assert any("per-prompt cap" in reason.lower() for reason in decision.refusals)


def test_non_utf8_encodable_prompt_is_refused(tmp_path):
    _write_coco(tmp_path)

    decision = _decision(_spec(tmp_path, prompt="\ud800"))

    assert not decision.admitted
    assert any("prompt" in reason.lower() for reason in decision.refusals)


def test_absent_validation_and_disabled_publish_remove_inactive_phases(tmp_path):
    _write_coco(tmp_path)
    spec = _spec(tmp_path)
    spec.publish_policy = PublishPolicy(auto_import=False)

    decision = _decision(spec)

    assert {phase.name for phase in decision.request.phases} == {
        "model_load",
        "training",
    }
    training = next(
        phase for phase in decision.request.phases if phase.name == "training"
    )
    assert training.disk_transient_bytes > 0


def test_present_but_empty_validation_split_is_refused(tmp_path):
    _write_coco(tmp_path)
    _write_coco(tmp_path, tiles=0, instances_per_tile=0, split="valid")

    decision = _decision(_spec(tmp_path))

    assert decision.dataset.validation_present
    assert decision.dataset.validation_tiles == 0
    assert not decision.admitted
    assert any("validation" in reason.lower() for reason in decision.refusals)


def test_missing_resolved_negative_prompts_is_refused_before_loading(tmp_path):
    _write_coco(tmp_path)

    decision = _decision(_spec(tmp_path, num_negatives=1, negative_prompts=[]))

    assert not decision.admitted
    assert any("negative prompt" in reason.lower() for reason in decision.refusals)


def test_manifest_resolved_negative_prompts_match_loader_precedence(tmp_path):
    _write_coco(tmp_path)
    (tmp_path / "dataset" / "build_manifest.json").write_text(
        json.dumps({"negative_prompts": ["manifest-background"]}), encoding="utf-8"
    )

    decision = _decision(_spec(tmp_path, num_negatives=1, negative_prompts=[]))

    assert decision.admitted


def test_lora_scope_changes_optimizer_state_and_adapter_terms(tmp_path):
    _write_coco(tmp_path)
    _write_coco(tmp_path, split="valid")
    default = _decision(_spec(tmp_path))
    with_text = _decision(_spec(tmp_path, adapt_text_encoder=True))

    default_train = next(p for p in default.request.phases if p.name == "training")
    text_train = next(p for p in with_text.request.phases if p.name == "training")
    default_validation = next(
        p for p in default.request.phases if p.name == "validation"
    )
    text_validation = next(
        p for p in with_text.request.phases if p.name == "validation"
    )
    default_publish = next(p for p in default.request.phases if p.name == "publish")
    text_publish = next(p for p in with_text.request.phases if p.name == "publish")
    assert _allocation(text_train, "LoRA and optimizer state") > _allocation(
        default_train, "LoRA and optimizer state"
    )
    assert _allocation(text_train, "CPU LoRA training state") > _allocation(
        default_train, "CPU LoRA training state"
    )
    assert _allocation(text_validation, "LoRA reload copy") > _allocation(
        default_validation, "LoRA reload copy"
    )
    assert _allocation(text_publish, "LoRA adapter") > _allocation(
        default_publish, "LoRA adapter"
    )
    assert _allocation(text_publish, "LoRA serialization copies") > _allocation(
        default_publish, "LoRA serialization copies"
    )


def test_rank_and_scope_scale_cpu_state_reload_and_serialization_copies(tmp_path):
    _write_coco(tmp_path)
    _write_coco(tmp_path, split="valid")
    rank_16 = _decision(_spec(tmp_path, rank=16))
    rank_32 = _decision(_spec(tmp_path, rank=32))

    phases_16 = {phase.name: phase for phase in rank_16.request.phases}
    phases_32 = {phase.name: phase for phase in rank_32.request.phases}
    for phase_name, allocation_name in (
        ("training", "CPU LoRA training state"),
        ("validation", "LoRA reload copy"),
        ("publish", "LoRA serialization copies"),
    ):
        assert _allocation(phases_32[phase_name], allocation_name) == 2 * _allocation(
            phases_16[phase_name], allocation_name
        )
        assert (
            phases_32[phase_name].host_peak_bytes
            > phases_16[phase_name].host_peak_bytes
        )


@pytest.mark.parametrize("rank", [-1, 0, pf._MAX_LORA_RANK + 1])
def test_unsafe_lora_ranks_are_refused_with_bounded_estimates(tmp_path, rank):
    _write_coco(tmp_path)

    decision = _decision(_spec(tmp_path, rank=rank))

    assert not decision.admitted
    assert any("rank" in reason.lower() for reason in decision.refusals)
    training = next(p for p in decision.request.phases if p.name == "training")
    assert _allocation(training, "CPU LoRA training state") <= (
        pf._MAX_LORA_RANK
        * sum(pf._LORA_PARAMS_PER_RANK.values())
        * pf._LORA_CPU_TRAINING_BYTES_PER_PARAM
    )


def test_unsafe_combined_rank_and_scope_size_is_refused(tmp_path):
    _write_coco(tmp_path)

    decision = _decision(
        _spec(
            tmp_path,
            rank=pf._MAX_LORA_RANK,
            adapt_text_encoder=True,
        )
    )

    assert not decision.admitted
    assert any("parameters" in reason.lower() for reason in decision.refusals)


def test_scope_with_no_estimated_trainable_parameters_is_refused(tmp_path):
    _write_coco(tmp_path)

    decision = _decision(
        _spec(
            tmp_path,
            adapt_vision_encoder=False,
            adapt_text_encoder=False,
            adapt_geometry_encoder=False,
            adapt_detr_encoder=False,
            adapt_detr_decoder=False,
            adapt_mask_decoder=True,
        )
    )

    assert not decision.admitted
    assert any("scope" in reason.lower() for reason in decision.refusals)


def test_artifact_and_publish_disk_targets_are_observed_separately(
    tmp_path, monkeypatch
):
    _write_coco(tmp_path)
    run_dir = tmp_path / "run-filesystem" / "run"
    models_root = tmp_path / "models-filesystem"
    observed = []

    def fake_free_disk(path):
        observed.append(str(path))
        if str(path) == str(run_dir):
            return 1
        if str(path) == str(models_root):
            return 2
        raise AssertionError(f"unexpected disk target: {path}")

    monkeypatch.setattr(pf, "_free_disk_bytes", fake_free_disk)

    decision = pf.assess_preflight(
        _spec(tmp_path),
        cuda_device=_cuda(),
        observation=_host(),
        run_dir=run_dir,
        models_root=models_root,
    )

    assert observed == [str(run_dir), str(models_root)]
    assert decision.artifact_free_disk_bytes == 1
    assert decision.publish_free_disk_bytes == 2
    assert any("run artifact" in reason.lower() for reason in decision.refusals)
    assert any("publish" in reason.lower() for reason in decision.refusals)


def test_empty_lora_scope_is_refused_instead_of_adapting_everything(tmp_path):
    _write_coco(tmp_path)
    decision = _decision(
        _spec(
            tmp_path,
            adapt_vision_encoder=False,
            adapt_text_encoder=False,
            adapt_geometry_encoder=False,
            adapt_detr_encoder=False,
            adapt_detr_decoder=False,
            adapt_mask_decoder=False,
        )
    )

    assert not decision.admitted
    assert any("adapter" in reason.lower() for reason in decision.refusals)


def test_explicit_empty_cuda_visible_devices_hides_all_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    with pytest.raises(ValueError, match="hides all"):
        pf._visible_device_selector("auto")


@pytest.mark.parametrize("device", ["cuda:-1", "cuda:-20"])
def test_negative_cuda_indices_are_rejected(device):
    with pytest.raises(ValueError, match="non-negative"):
        pf._visible_device_selector(device)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_non_cuda_device_selection_is_rejected_before_gpu_probe(device):
    with pytest.raises(ValueError, match="CUDA"):
        pf._visible_device_selector(device)


def test_cuda_visible_remapping_resolves_the_selected_physical_uuid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb,GPU-aaaa")
    probe = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "0, GPU-aaaa, 0000:01:00.0, First, 8.9, 49140, 48000, Disabled\n"
            "1, GPU-bbbb, 0000:02:00.0, Second, 8.9, 49140, 47000, Disabled\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(pf.subprocess, "run", lambda *_args, **_kwargs: probe)

    selected = pf._probe_cuda_device("cuda:1")

    assert selected is not None
    assert selected.uuid == "GPU-aaaa"
    assert selected.index == 0


def test_unique_cuda_uuid_prefix_resolves_but_ambiguous_prefix_refuses(monkeypatch):
    probe = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "0, GPU-abcd1111, 0000:01:00.0, First, 8.9, 49140, 48000, Disabled\n"
            "1, GPU-abcd2222, 0000:02:00.0, Second, 8.9, 49140, 47000, Disabled\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(pf.subprocess, "run", lambda *_args, **_kwargs: probe)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd1")

    selected = pf._probe_cuda_device("cuda:0")

    assert selected is not None
    assert selected.uuid == "GPU-abcd1111"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd")
    assert pf._probe_cuda_device("cuda:0") is None


def test_mig_device_is_refused_until_slice_telemetry_and_leasing_are_supported(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-012345")
    monkeypatch.setattr(
        pf.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("MIG must fail before GPU probing"),
    )

    assert pf._probe_cuda_device("cuda:0") is None


def test_numeric_cuda_selection_uses_pci_bus_order(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    call = {}
    probe = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "0, GPU-later, 0000:02:00.0, Later, 8.9, 49140, 48000, Disabled\n"
            "1, GPU-first, 0000:01:00.0, First, 8.9, 49140, 47000, Disabled\n"
        ),
        stderr="",
    )

    def fake_run(*_args, **kwargs):
        call.update(kwargs)
        return probe

    monkeypatch.setattr(pf.subprocess, "run", fake_run)

    selected = pf._probe_cuda_device("cuda:0")

    assert selected is not None
    assert selected.uuid == "GPU-first"
    assert "CUDA_DEVICE_ORDER" not in os.environ
    assert call["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_numeric_visible_device_token_maps_to_physical_nvidia_index(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    probe = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "0, GPU-zero, 0000:01:00.0, Zero, 8.9, 49140, 48000, Disabled\n"
            "1, GPU-one, 0000:02:00.0, One, 8.9, 49140, 46000, Disabled\n"
            "2, GPU-two, 0000:03:00.0, Two, 8.9, 49140, 47000, Disabled\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(pf.subprocess, "run", lambda *_args, **_kwargs: probe)

    selected = pf._probe_cuda_device("cuda:0")

    assert selected is not None
    assert selected.uuid == "GPU-two"
    assert selected.index == 2


def test_numeric_visible_device_without_explicit_pci_order_fails_closed(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    probe = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "0, GPU-zero, 0000:01:00.0, Zero, 8.9, 49140, 48000, Disabled\n"
            "2, GPU-two, 0000:03:00.0, Two, 8.9, 49140, 47000, Disabled\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(pf.subprocess, "run", lambda *_args, **_kwargs: probe)

    assert pf._probe_cuda_device("cuda:0") is None


def test_numeric_or_parent_uuid_selection_refuses_mig_enabled_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    probe = subprocess.CompletedProcess(
        [],
        0,
        stdout=("0, GPU-parent, 0000:01:00.0, Parent, 8.9, 49140, 48000, Enabled\n"),
        stderr="",
    )
    monkeypatch.setattr(pf.subprocess, "run", lambda *_args, **_kwargs: probe)

    assert pf._probe_cuda_device("cuda:0") is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-parent")
    assert pf._probe_cuda_device("cuda:0") is None


def test_prompt_instances_disk_ack_and_resume_refusals_are_preserved(
    tmp_path, monkeypatch
):
    _write_coco(tmp_path, tiles=1, instances_per_tile=2)
    spec = _spec(tmp_path, prompt="", ack=False)
    spec.resume_from = "/tmp/last.pt"
    monkeypatch.setattr(pf, "_free_disk_bytes", lambda _path: 1)

    decision = _decision(spec)

    combined = " ".join(decision.refusals).lower()
    assert "prompt" in combined
    assert "instance" in combined
    assert "disk" in combined
    assert "acknowledge" in combined
    assert "resum" in combined


def test_legacy_preflight_wrappers_expose_decision_reasons_and_warnings(
    tmp_path, monkeypatch
):
    _write_coco(tmp_path)
    decision = _decision(_spec(tmp_path))
    monkeypatch.setattr(pf, "assess_preflight", lambda _spec: decision)

    assert pf.preflight(_spec(tmp_path)) == []
    assert pf.preflight_warnings(_spec(tmp_path)) == list(decision.warnings)


def test_diagnostics_report_phases_reserves_observations_and_effective_limits(tmp_path):
    _write_coco(tmp_path)
    _write_coco(tmp_path, split="valid")

    payload = _decision(_spec(tmp_path)).to_dict()

    assert payload["budget"]["dominant_host_phase"]
    assert payload["budget"]["dominant_accelerator_phase"]
    assert payload["budget"]["available_host_bytes"] == 56 * pf.GiB
    assert payload["budget"]["available_accelerator_bytes"] == 48 * pf.GiB
    assert payload["policy"]["reserve_host_bytes"] == 8 * pf.GiB
    assert payload["policy"]["reserve_host_fraction"] == pytest.approx(0.15)
    assert payload["policy"]["accelerator_safety_fraction"] == pytest.approx(0.85)
    assert payload["budget"]["limits"]["tiles"] == 1
    assert {phase["name"] for phase in payload["request"]["phases"]} == {
        "model_load",
        "training",
        "validation",
        "publish",
    }
