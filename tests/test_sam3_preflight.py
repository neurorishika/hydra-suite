"""Typed, metadata-only SAM3 resource admission tests."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys

import pytest

from hydra_suite.runtime.memory_profiles import (
    MemoryMeasurement,
    PressureSettings,
    ProfileIdentity,
)
from hydra_suite.runtime.resource_budget import AcceleratorKind, ResourceObservation
from hydra_suite.training.contracts import (
    PublishPolicy,
    Sam3LoraParams,
    SourceDataset,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.sam3_lora import autobatch as ab
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
        free_bytes=int(free_gib * pf.GiB),
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


def _training_phase(decision):
    """The training phase estimate.

    `budget.accelerator_peak_bytes` is a MAX over phases, and the model-load
    phase carries its own `_DEVICE_STEADY_BYTES` floor, so it is the wrong
    place to read a training-phase requirement that a measurement lowered
    below that constant.
    """

    return next(phase for phase in decision.request.phases if phase.name == "training")


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


@pytest.mark.parametrize("precision", ["fp16", "int8", "tf32"])
def test_unimplemented_precision_modes_fail_closed(tmp_path, precision):
    """FP16 stays refused: its range overflows SAM3's loss scales and there
    is no GradScaler anywhere in this path."""
    _write_coco(tmp_path)

    decision = _decision(_spec(tmp_path, mixed_precision=precision))

    assert not decision.admitted
    assert any(precision in reason for reason in decision.refusals)


def test_fp32_is_admitted_with_a_doubled_device_estimate(tmp_path):
    """FP32 was refused for depending on "SAM3's BF16 activation path" --
    `perflib.fused.addmm_act`, which `perflib_compat` now replaces with a
    dtype-neutral eager equivalent. Nothing in the training path needs bf16.

    The device estimate must scale, or the gate admits a run that OOMs the
    card after minutes of setup.
    """
    _write_coco(tmp_path)

    bf16 = _decision(_spec(tmp_path, mixed_precision="bf16"))
    fp32 = _decision(_spec(tmp_path, mixed_precision="fp32"))

    # No longer refused on PRECISION grounds...
    assert not any("not available" in reason for reason in fp32.refusals)
    # ...but the device estimate must scale, so a card that cannot hold the
    # fp32 envelope is still refused -- on honest capacity grounds.
    assert fp32.budget.accelerator_peak_bytes > bf16.budget.accelerator_peak_bytes


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
    # The growth must carry the full per-extra-batch allowance. batch > 1 has
    # never been measured for this role, so that allowance stays deliberately
    # conservative; this pins the scaling, not an admission verdict.
    # (It previously asserted `not admitted`, which only held because the
    # batch-1 constant was 3.7x the measured figure -- the test was pinning a
    # miscalibration rather than a requirement.)
    assert (
        batch_two.budget.accelerator_peak_bytes
        - batch_one.budget.accelerator_peak_bytes
    ) >= pf._EXTRA_BATCH_DEVICE_BYTES


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
        def __len__(self):
            return 0

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
    assert any("list or tuple" in reason.lower() for reason in decision.refusals)


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


def test_publish_phase_budgets_one_base_and_one_active_tensor_not_a_full_clone(
    tmp_path,
):
    _write_coco(tmp_path)

    publish = next(
        phase
        for phase in _decision(_spec(tmp_path)).request.phases
        if phase.name == "publish"
    )
    allocations = dict(publish.dominant_allocations)

    assert allocations["base checkpoint"] == pf._CHECKPOINT_BYTES
    assert allocations["largest possible active tensor"] == pf._CHECKPOINT_BYTES
    assert "merged checkpoint" not in allocations
    assert "serialization temporary" not in allocations


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
    """Every enabled scope contributing nothing must be refused, not sized to 0.

    This used to be expressed as "adapt_mask_decoder alone", which was a
    zero-coefficient scope while `segmentation_head` held no adaptable Linear.
    Splitting SAM3's own attention clone gave it four (2_048 params/rank), so
    the only remaining way to enable a scope-set with no trainable parameters
    is to enable none -- which is exactly the condition this refusal names.
    """
    _write_coco(tmp_path)

    decision = _decision(
        _spec(
            tmp_path,
            **{flag: False for flag in pf._LORA_PARAMS_PER_RANK},
        )
    )

    assert not decision.admitted
    assert any("scope" in reason.lower() for reason in decision.refusals)


def test_mask_decoder_scope_alone_is_now_a_real_scope(tmp_path):
    """Regression companion: `segmentation_head.cross_attend_prompt` is one
    SAM3 attention clone, so the mask-decoder scope now budgets four wrapped
    projections instead of being an admissible no-op."""
    _write_coco(tmp_path)

    decision = _decision(
        _spec(
            tmp_path,
            **{flag: flag == "adapt_mask_decoder" for flag in pf._LORA_PARAMS_PER_RANK},
        )
    )

    assert decision.admitted
    assert pf._LORA_PARAMS_PER_RANK["adapt_mask_decoder"] == 2_048


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


@pytest.mark.parametrize("device", ["cpu", "mps", "tpu", "gpu0"])
def test_non_cuda_device_selection_is_rejected_before_gpu_probe(device):
    """SAM3 LoRA training is CUDA-only; only the WORDING of the refusal moved.

    The message must name the real cause and the accepted spellings: the old
    "No CUDA device is available" sent users to inspect a GPU that was fine.
    """

    with pytest.raises(ValueError, match="requires a CUDA device"):
        pf._visible_device_selector(device)
    message = pf.sam3_device_form_error(device)
    assert message is not None
    assert "Accepted:" in message and "'0'" in message


@pytest.mark.parametrize("device", ["auto", "cuda", "cuda:0", "0", "1", "0,1"])
def test_accepted_device_forms_report_no_form_error(device):
    assert pf.sam3_device_form_error(device) is None


def _two_gpu_probe():
    return subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "0, GPU-first, 0000:01:00.0, First, 8.9, 49140, 48000, Disabled\n"
            "1, GPU-second, 0000:02:00.0, Second, 8.9, 49140, 47000, Disabled\n"
        ),
        stderr="",
    )


@pytest.mark.parametrize(
    "ordinal, equivalent", [("0", "cuda:0"), ("1", "cuda:1"), ("0,1", "cuda:0")]
)
def test_bare_ordinal_devices_resolve_like_their_cuda_spelling(
    monkeypatch, ordinal, equivalent
):
    """Ultralytics' bare ordinals are what DetectKit plans carry.

    Refusing them killed four real SAM3 runs at preflight on a box with 47 GiB
    free. A multi-GPU form resolves against the FIRST device.
    """

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setattr(pf.subprocess, "run", lambda *_a, **_k: _two_gpu_probe())

    observed = pf._probe_cuda_device(ordinal)

    assert observed is not None
    assert observed == pf._probe_cuda_device(equivalent)


def test_multi_gpu_device_string_warns_that_only_the_first_gpu_is_used(tmp_path):
    _write_coco(tmp_path)
    spec = _spec(tmp_path)
    spec.device = "0,1"

    decision = _decision(spec)

    assert any(
        "names several GPUs" in warning and "cuda:0" in warning
        for warning in decision.warnings
    )


def test_bare_ordinal_refusal_names_the_accepted_forms(tmp_path):
    """A refused SPELLING must not masquerade as absent hardware."""

    _write_coco(tmp_path)
    spec = _spec(tmp_path)
    spec.device = "cpu"

    decision = pf.assess_preflight(spec, cuda_device=None, observation=_host())

    assert any("Accepted:" in reason for reason in decision.refusals)
    assert not any("No CUDA device is available" in r for r in decision.refusals)


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


def test_missing_huggingface_credential_is_refused_at_preflight(tmp_path, monkeypatch):
    """A local checkpoint does not replace HF auth.

    `build_sam3_image_model` fetches the config from the GATED facebook/sam3
    repo at build time. Without this gate the run dies minutes in, inside a
    sidecar subprocess, as a bare `GatedRepoError: 401` with no hint about
    what to do.
    """
    _write_coco(tmp_path)
    monkeypatch.setattr(pf, "_huggingface_token_present", lambda: False)

    decision = _decision(_spec(tmp_path))

    assert not decision.admitted
    reason = next(r for r in decision.refusals if "Hugging Face" in r)
    assert "hf auth login" in reason
    assert "huggingface.co/facebook/sam3" in reason


def test_present_huggingface_credential_does_not_refuse(tmp_path, monkeypatch):
    _write_coco(tmp_path)
    monkeypatch.setattr(pf, "_huggingface_token_present", lambda: True)

    decision = _decision(_spec(tmp_path))

    assert not any("Hugging Face" in reason for reason in decision.refusals)


def test_token_probe_reads_env_vars_then_the_token_file(tmp_path, monkeypatch):
    """Local-only probe: no network call belongs in a millisecond gate."""
    for name in pf._HF_TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert pf._huggingface_token_present() is False

    # An empty token file is not a credential.
    (tmp_path / "token").write_text("")
    assert pf._huggingface_token_present() is False

    (tmp_path / "token").write_text("hf_xxx")
    assert pf._huggingface_token_present() is True

    # An env var alone is enough, with no file at all.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty"))
    assert pf._huggingface_token_present() is False
    monkeypatch.setenv("HF_TOKEN", "hf_yyy")
    assert pf._huggingface_token_present() is True


# ---------------------------------------------------------------------------
# Probe admission (Task 3a): measure first, gate on the measurement later.
# ---------------------------------------------------------------------------


def test_probe_admission_does_not_use_the_fallback_device_envelope(
    tmp_path, monkeypatch
):
    """The inherited device envelope is exactly what the probe exists to
    replace. Gating the probe on it would refuse hardware whose real batch-1
    peak fits, before it was ever measured."""

    _write_coco(tmp_path)
    monkeypatch.setattr(pf, "_MEASURED_BF16_DEVICE_PEAK_BYTES", 10_000 * pf.GiB)
    monkeypatch.setattr(pf, "_EXTRA_BATCH_DEVICE_BYTES", 10_000 * pf.GiB)
    monkeypatch.setattr(pf, "_DEVICE_STEADY_BYTES", 10_000 * pf.GiB)

    decision = pf.assess_probe_preflight(
        _spec(tmp_path, batch=-1),
        batch=1,
        cuda_device=_cuda(),
        observation=_host(),
    )

    assert decision.admitted, decision.refusals
    assert decision.budget.limits.batch_size == 1


def test_probe_admission_still_refuses_below_the_hard_floor(tmp_path):
    """The floor is model + LoRA state + one tile -- an under-estimate by
    construction. A card that cannot hold even that is refused before a probe
    child is ever launched."""

    _write_coco(tmp_path)

    decision = pf.assess_probe_preflight(
        _spec(tmp_path, batch=-1),
        batch=1,
        cuda_device=_cuda(free_gib=2, total_gib=2),
    )

    assert not decision.admitted


def test_probe_admission_is_evaluated_per_candidate_on_the_host_side(tmp_path):
    """Host demand scales with batch (decoded tiles, transformed tiles,
    collated images, dense masks), so each candidate gets its own estimate."""

    _write_coco(tmp_path)
    spec = _spec(tmp_path, batch=-1)

    small = pf.assess_probe_preflight(
        spec, batch=1, cuda_device=_cuda(), observation=_host()
    )
    large = pf.assess_probe_preflight(
        spec, batch=8, cuda_device=_cuda(), observation=_host()
    )

    def _training_host_peak(decision):
        return next(
            phase.host_peak_bytes
            for phase in decision.request.phases
            if phase.name == "training"
        )

    assert _training_host_peak(large) > _training_host_peak(small)
    assert large.budget.limits.batch_size == 8


def test_probe_admission_shares_every_other_refusal_with_normal_admission(tmp_path):
    """Leases, dataset validity, prompts, and publish policy are shared
    concerns -- factored, not forked."""

    _write_coco(tmp_path)

    decision = pf.assess_probe_preflight(
        _spec(tmp_path, batch=-1, ack=False),
        batch=1,
        cuda_device=_cuda(),
        observation=_host(),
    )

    assert not decision.admitted
    assert any("Label quality" in reason for reason in decision.refusals)


def test_dataset_profile_reports_the_p95_instance_density(tmp_path):
    """Non-uniform on purpose: with every tile the same density a `return
    max(...)` stub would pass and the p95 would be untested. One outlier tile
    of 50 must NOT drag the p95 up with it -- distinguishing the two is the
    whole reason the fingerprint records both."""

    split_dir = tmp_path / "dataset" / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    densities = [1] * 15 + [2] * 4 + [50]
    images = [
        {"id": i + 1, "file_name": f"t{i}.png", "width": 1008, "height": 1008}
        for i in range(len(densities))
    ]
    annotations = []
    ann_id = 1
    for image, count in zip(images, densities):
        for _ in range(count):
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

    profile = pf._dataset_profile(str(tmp_path / "dataset"))

    assert profile.max_active_instances_per_tile == 50
    assert profile.p95_active_instances_per_tile == 2


# ---------------------------------------------------------------------------
# Task 4: measurement may RAISE the device estimate, never lower it.
# ---------------------------------------------------------------------------


def _identity(precision="bf16"):
    return ProfileIdentity(
        operation="sam3_lora_training",
        model_identity="model",
        backend="env",
        device_identity="Test CUDA|48",
        precision=precision,
        task="task",
    )


def _records(peaks, *, precision="bf16"):
    return tuple(
        MemoryMeasurement(
            identity=_identity(precision),
            settings=PressureSettings(
                input_width=1008, input_height=1008, batch_size=batch
            ),
            accelerator_kind=AcceleratorKind.CUDA,
            host_peak_bytes=pf.GiB,
            accelerator_reserved_peak_bytes=int(peak),
            observed_at_unix_ns=1,
        )
        for batch, peak in sorted(peaks.items())
    )


def _install_records(monkeypatch, records, *, identity=None):
    monkeypatch.setattr(pf, "_profile_records", lambda: tuple(records))
    monkeypatch.setattr(
        pf,
        "_workload_identity",
        lambda *_a, **_k: identity if identity is not None else _identity(),
    )


def _legacy_expression(tmp_path, *, batch, precision="bf16"):
    """The pre-Task-4 analytic expression, written out literally.

    Deliberately NOT a call into `preflight`: this is the drift guard, so it
    must fail if the production expression changes shape.
    """

    dataset = pf._dataset_profile(str(tmp_path / "dataset"))
    image_pixels = 1008 * 1008
    inflight_tiles = min(batch, max(1, dataset.tile_count))
    active_instances = inflight_tiles * dataset.max_active_instances_per_tile
    dense_masks_device = active_instances * image_pixels * 16
    # Sam3LoraParams defaults also adapt the mask decoder, which the
    # "default" LoRA state the old expression subtracts does NOT include.
    lora_training_state = 16 * (565_248 + 28_680 + 52_224 + 64_512 + 2_048) * 24
    default_lora_training_state = 16 * (565_248 + 28_680 + 52_224 + 64_512) * 24
    multiplier = 2.0 if precision == "fp32" else 1.0
    return int(
        pf._MEASURED_BF16_DEVICE_PEAK_BYTES * multiplier
        + max(0, batch - 1) * pf._EXTRA_BATCH_DEVICE_BYTES * multiplier
        + max(0, lora_training_state - default_lora_training_state)
        + dense_masks_device
    )


def test_a_measurement_at_an_observed_rung_decides_below_the_analytic_estimate(
    tmp_path, monkeypatch
):
    """The rule this file used to encode, deliberately INVERTED.

    It used to read "a 2-step probe measured 7.34 GiB on a surface whose real
    peak reached 12.99 GiB, so a measurement may only RAISE the analytic
    estimate". That growth was later isolated to allocator fragmentation:
    under the `expandable_segments:True` that `sam3_env_environ()` now
    hardcodes for every probe and training child it does not happen, and a
    30-step probe under-reads a full run by 2.8% rather than 41%. Since the
    allocator config is part of the fingerprint, a record taken under the old
    allocator cannot reach this path at all.

    So at an OBSERVED rung the measurement now decides, downward included.
    """

    _write_coco(tmp_path)
    _install_records(monkeypatch, _records({1: int(7.34 * pf.GiB)}))

    decision = _decision(_spec(tmp_path, batch=1))

    analytic = _legacy_expression(tmp_path, batch=1)
    assert analytic > 12 * pf.GiB > 7.34 * pf.GiB
    assert _training_phase(decision).accelerator_peak_bytes == int(7.34 * pf.GiB)
    assert decision.device_peak_measured_bytes == int(7.34 * pf.GiB)
    assert decision.device_peak_provenance == "measured"
    assert not decision.device_peak_measured_extrapolated
    assert decision.device_peak_analytic_bytes == analytic


def test_a_between_rung_batch_is_charged_the_observed_peak_above_it(
    tmp_path, monkeypatch
):
    """A batch inside the observed range but not ON a rung still decides
    without the analytic cap -- but not on the fit.

    With rungs 1, 2 and 4, batch 3's fitted envelope is a guess, so the
    requirement is raised to the OBSERVED peak at batch 4: a guaranteed-safe
    upper bound by monotonicity, `true_need(3) <= peak@4`. Charging the fit
    instead would admit batch 3 on a card that provably cannot hold batch 4.
    """

    records = _records({1: 4 * pf.GiB, 2: 5 * pf.GiB, 4: 7 * pf.GiB})
    _write_coco(tmp_path)
    _install_records(monkeypatch, records)

    fitted = pf.measured_envelope_bytes(records, 3)
    assert fitted < 7 * pf.GiB, "the fit must be the LOWER number here"

    decision = _decision(_spec(tmp_path, batch=3))

    assert decision.device_peak_provenance == "measured"
    assert not decision.device_peak_measured_extrapolated
    assert _training_phase(decision).accelerator_peak_bytes == 7 * pf.GiB
    assert 7 * pf.GiB < _legacy_expression(tmp_path, batch=3)


def test_a_batch_on_a_rung_is_charged_that_rung_not_the_one_above(
    tmp_path, monkeypatch
):
    """The upper-rung rule must not leak onto observed rungs: batch 2 is an
    observation and is charged its own envelope, never batch 4's peak."""

    records = _records({1: 4 * pf.GiB, 2: 5 * pf.GiB, 4: 7 * pf.GiB})
    _write_coco(tmp_path)
    _install_records(monkeypatch, records)

    decision = _decision(_spec(tmp_path, batch=2))

    assert decision.device_peak_provenance == "measured"
    assert _training_phase(decision).accelerator_peak_bytes == (
        pf.measured_envelope_bytes(records, 2)
    )
    assert _training_phase(decision).accelerator_peak_bytes < 7 * pf.GiB


def test_no_records_leaves_the_analytic_estimate_in_charge(tmp_path, monkeypatch):
    """The fallback is retained, not deleted: an unprobed workload is still
    admitted purely on the analytic estimate."""

    _write_coco(tmp_path)
    _install_records(monkeypatch, ())

    decision = _decision(_spec(tmp_path, batch=2))

    assert decision.device_peak_provenance == "analytic"
    assert decision.device_peak_measured_bytes == 0
    assert not decision.device_peak_measured_extrapolated
    assert decision.budget.accelerator_peak_bytes == _legacy_expression(
        tmp_path, batch=2
    )


def test_a_sub_steady_measurement_admits_a_dataset_with_a_validation_split(
    tmp_path, monkeypatch
):
    """A measurement below `_DEVICE_STEADY_BYTES` must not raise out of preflight.

    `PhaseEstimate` rejects a peak below its own steady floor. Now that a
    measurement DECIDES, the ordinary case (real runs reserve 6.9-7.3 GiB,
    under the 8 GiB resident-weights guess) puts the training AND validation
    phases' peaks below that analytic steady constant. Clamping only the
    training phase left `validation` raising `ValueError` -- an unhandled
    traceback out of the training launch rather than a refusal -- for every
    project with a valid split, which is most of them.

    Every other measured-record test uses a train-only dataset, so the
    validation phase is not even constructed there. This one has both splits
    on purpose.
    """

    _write_coco(tmp_path)
    _write_coco(tmp_path, split="valid")
    _install_records(monkeypatch, _records({1: 6 * pf.GiB}))

    decision = _decision(_spec(tmp_path, batch=1))

    assert decision.dataset.validation_tiles > 0, "the guarded phase must exist"
    validation = next(
        phase for phase in decision.request.phases if phase.name == "validation"
    )
    assert validation.accelerator_peak_bytes == 6 * pf.GiB
    assert validation.accelerator_steady_bytes <= validation.accelerator_peak_bytes
    assert decision.device_peak_provenance == "measured"


def test_the_measured_headroom_gate_reads_the_requirement_not_the_budget_peak(
    tmp_path, monkeypatch
):
    """The 0.80 measured headroom must gate the MEASURED requirement.

    `budget.accelerator_peak_bytes` is a max over phases and the `model_load`
    phase carries the analytic 8 GiB `_DEVICE_STEADY_BYTES` floor, which a
    measurement can legitimately sit below. Gating on the budget peak would
    apply the measured fraction to an analytic constant and then report that
    constant as "the measured device requirement".

    Here the measurement is 6 GiB and free is 9.5 GiB: 0.80 x 9.5 = 7.6 GiB,
    which clears the 6 GiB requirement but NOT the 8 GiB model-load floor,
    while the policy's own 0.85 x 9.5 = 8.075 GiB does clear that floor. The
    run must be admitted.
    """

    _write_coco(tmp_path)
    _install_records(monkeypatch, _records({1: 6 * pf.GiB}))

    host = ResourceObservation(
        total_host_bytes=64 * pf.GiB,
        available_host_bytes=56 * pf.GiB,
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_name="Test CUDA",
        total_accelerator_bytes=48 * pf.GiB,
        available_accelerator_bytes=int(9.5 * pf.GiB),
    )
    decision = _decision(_spec(tmp_path, batch=1), host=host)

    assert decision.device_peak_provenance == "measured"
    assert _training_phase(decision).accelerator_peak_bytes == 6 * pf.GiB
    assert decision.budget.accelerator_peak_bytes == 8 * pf.GiB
    assert not [
        refusal for refusal in decision.refusals if "measured device" in refusal
    ], decision.refusals
    assert decision.admitted, decision.refusals


def test_a_default_allocator_record_cannot_decide_for_an_expandable_run(
    tmp_path, monkeypatch
):
    """Guard (b), exercised through the REAL identity matching.

    `_install_records` bypasses `records_for`, which would make this
    tautological. Here `pf._profile_records` returns two records that differ
    only in the allocator sub-hash inside `backend`, and the real
    `_stored_records` path (fingerprint -> `records_for` ->
    `validate_probe_records`) must surface only the expandable one -- so a
    stale default-allocator measurement can never decide.
    """

    from hydra_suite.training.sam3_lora import autobatch as ab

    _write_coco(tmp_path)
    live_hash = ab.sidecar_alloc_conf_hash()
    stale_hash = ab.sidecar_alloc_conf_hash({})
    assert live_hash != stale_hash
    assert live_hash == ab.sidecar_alloc_conf_hash(
        {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    ), "the sidecar environment must hash as expandable_segments"

    live_identity = dataclasses.replace(_identity(), backend=f"env|{live_hash}")
    stale_identity = dataclasses.replace(_identity(), backend=f"env|{stale_hash}")

    def _record(identity, batch, peak):
        return MemoryMeasurement(
            identity=identity,
            settings=PressureSettings(
                input_width=1008, input_height=1008, batch_size=batch
            ),
            accelerator_kind=AcceleratorKind.CUDA,
            host_peak_bytes=pf.GiB,
            accelerator_reserved_peak_bytes=int(peak),
            observed_at_unix_ns=1,
        )

    stored = (
        _record(stale_identity, 1, 3 * pf.GiB),
        _record(live_identity, 1, 9 * pf.GiB),
    )
    monkeypatch.setattr(pf, "_profile_records", lambda: stored)
    # Only the fingerprint is faked; `records_for` and `validate_probe_records`
    # are the real ones.
    monkeypatch.setattr(pf, "_workload_identity", lambda *_a, **_k: live_identity)

    matched = pf._stored_records(_spec(tmp_path, batch=1), _cuda())
    assert [record.accelerator_reserved_peak_bytes for record in matched] == [
        9 * pf.GiB
    ], "the default-allocator record must not survive identity matching"

    decision = _decision(_spec(tmp_path, batch=1))
    assert decision.device_peak_provenance == "measured"
    assert _training_phase(decision).accelerator_peak_bytes == 9 * pf.GiB


def test_measured_extra_batch_bytes_admits_batch_4_on_a_24gb_card(tmp_path):
    """Task 8 pin: with `_EXTRA_BATCH_DEVICE_BYTES` measured at 2 GiB, a
    24 GB card's analytic-only admission (no probe records yet) clears
    batch 4, not just batch 1.

    Base at b1 is ~12.1 GiB; +2 GiB/item puts b2 at ~14.2 and b4 at ~18.4
    GiB, both under a 24 GB card's usable budget (0.85 safety -> ~20.4 GiB).
    b8 (~26.4 GiB) does not fit. This is a regression guard: reverting the
    constant to its old 18 GiB value pushes b2 alone to ~30 GiB, which
    would refuse everything past batch 1 on the very same card.
    """

    _write_coco(tmp_path)
    cuda = _cuda(free_gib=24, total_gib=24)
    host = ResourceObservation(
        total_host_bytes=64 * pf.GiB,
        available_host_bytes=56 * pf.GiB,
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_name="Test CUDA",
        total_accelerator_bytes=24 * pf.GiB,
        available_accelerator_bytes=24 * pf.GiB,
    )

    decision_b4 = _decision(_spec(tmp_path, batch=4), cuda=cuda, host=host)
    assert decision_b4.admitted, decision_b4.refusals

    decision_b8 = _decision(_spec(tmp_path, batch=8), cuda=cuda, host=host)
    assert not decision_b8.admitted


def test_reverting_extra_batch_bytes_to_18gib_would_refuse_batch_2(
    tmp_path, monkeypatch
):
    """Companion regression guard: prove the OLD 18 GiB value actually
    breaks the batch-4 story this branch fixes, so a silent revert of
    `_EXTRA_BATCH_DEVICE_BYTES` is caught here rather than only in a
    provenance comment nobody re-reads."""

    _write_coco(tmp_path)
    monkeypatch.setattr(pf, "_EXTRA_BATCH_DEVICE_BYTES", 18 * pf.GiB)
    cuda = _cuda(free_gib=24, total_gib=24)
    host = ResourceObservation(
        total_host_bytes=64 * pf.GiB,
        available_host_bytes=56 * pf.GiB,
        accelerator_kind=AcceleratorKind.CUDA,
        accelerator_name="Test CUDA",
        total_accelerator_bytes=24 * pf.GiB,
        available_accelerator_bytes=24 * pf.GiB,
    )

    decision_b2 = _decision(_spec(tmp_path, batch=2), cuda=cuda, host=host)
    assert not decision_b2.admitted


def test_a_measurement_above_the_analytic_estimate_raises_the_requirement(
    tmp_path, monkeypatch
):
    _write_coco(tmp_path)
    _install_records(monkeypatch, _records({1: 20 * pf.GiB}))

    decision = _decision(_spec(tmp_path, batch=1))

    assert decision.budget.accelerator_peak_bytes == 20 * pf.GiB
    assert decision.device_peak_provenance == "measured"
    assert decision.device_peak_analytic_bytes == _legacy_expression(tmp_path, batch=1)


def test_the_measured_envelope_is_never_below_an_observation(tmp_path, monkeypatch):
    """The envelope is `max(fit, every observation at batch <= n)`; the fit's
    base is clamped at zero, so a superlinear jump cannot be smoothed away."""

    records = _records({1: 16 * pf.GiB, 2: 40 * pf.GiB})
    for batch in (1, 2):
        observed = max(
            record.accelerator_reserved_peak_bytes
            for record in records
            if record.settings.batch_size <= batch
        )
        assert pf.measured_envelope_bytes(records, batch) >= observed

    _write_coco(tmp_path)
    _install_records(monkeypatch, records)

    decision = _decision(_spec(tmp_path, batch=2))

    # base clamps to 0, slope 24 GiB/item -> 48 GiB, above both observations.
    assert decision.budget.accelerator_peak_bytes == 48 * pf.GiB
    assert decision.device_peak_provenance == "measured"


def test_a_batch_beyond_the_observations_is_flagged_extrapolated(tmp_path, monkeypatch):
    """Guard (a): past the top rung the measured side is a FIT, not an
    observation, so `max(analytic, measured)` still applies and provenance
    says so (`max_extrapolated`) rather than naming a winner. That string is
    what distinguishes this case from "analytic decided because there were no
    records at all"."""

    _write_coco(tmp_path)
    _install_records(monkeypatch, _records({1: 40 * pf.GiB}))

    decision = _decision(_spec(tmp_path, batch=2))

    # fitted: slope 40 GiB/item from a single point -> 80 GiB at batch 2.
    assert decision.budget.accelerator_peak_bytes == 80 * pf.GiB
    assert decision.device_peak_provenance == "max_extrapolated"
    assert decision.device_peak_measured_extrapolated

    # The analytic wins here, yet the measured side is still a guess.
    # `_EXTRA_BATCH_DEVICE_BYTES` is now measured at 2 GiB/item (was an
    # unmeasured 18 GiB), so the analytic side at batch 2 is only ~14.2 GiB
    # for this dataset -- a single b1=5 GiB point extrapolates to 10 GiB,
    # comfortably below it.
    _install_records(monkeypatch, _records({1: 5 * pf.GiB}))
    small = _decision(_spec(tmp_path, batch=2))
    assert small.device_peak_provenance == "max_extrapolated"
    # The decision-level proof that the ANALYTIC side actually won: provenance
    # and the flag alone would pass even if the fitted guess had decided.
    assert small.budget.accelerator_peak_bytes == _legacy_expression(tmp_path, batch=2)
    assert small.device_peak_measured_extrapolated
    assert not _decision(_spec(tmp_path, batch=1)).device_peak_measured_extrapolated


def test_a_precision_change_misses_the_cache_and_falls_back(tmp_path, monkeypatch):
    _write_coco(tmp_path)
    _install_records(
        monkeypatch,
        _records({1: 40 * pf.GiB}, precision="bf16"),
        identity=_identity("fp32"),
    )

    decision = _decision(_spec(tmp_path, batch=1, mixed_precision="fp32"))

    assert decision.budget.accelerator_peak_bytes == _legacy_expression(
        tmp_path, batch=1, precision="fp32"
    )
    assert decision.device_peak_provenance == "analytic"


def test_the_fallback_matches_the_previous_expression_exactly(tmp_path, monkeypatch):
    _write_coco(tmp_path)
    _install_records(monkeypatch, ())

    for batch in (1, 2, 4):
        decision = _decision(_spec(tmp_path, batch=batch))
        assert decision.budget.accelerator_peak_bytes == _legacy_expression(
            tmp_path, batch=batch
        )
        assert decision.device_peak_provenance == "analytic"


def test_the_measured_path_applies_the_safety_fraction_exactly_once(
    tmp_path, monkeypatch
):
    """20 GiB free admits a 16 GiB measured requirement at 0.8 once
    (16 <= 16.0) and refuses it if 0.8 were applied twice (16 > 12.8)."""

    _write_coco(tmp_path)
    _install_records(monkeypatch, _records({1: 16 * pf.GiB}))

    def decide(free_gib):
        host = ResourceObservation(
            total_host_bytes=64 * pf.GiB,
            available_host_bytes=56 * pf.GiB,
            accelerator_kind=AcceleratorKind.CUDA,
            accelerator_name="Test CUDA",
            total_accelerator_bytes=48 * pf.GiB,
            available_accelerator_bytes=int(free_gib * pf.GiB),
        )
        return _decision(
            _spec(tmp_path, batch=1), cuda=_cuda(free_gib=free_gib), host=host
        )

    assert decide(20).admitted
    refused = decide(19.9)
    assert not refused.admitted
    assert any("measured" in text for text in refused.refusals)


def test_preflight_never_probes(tmp_path, monkeypatch):
    _write_coco(tmp_path)

    def _explode(*_a, **_k):
        raise AssertionError("preflight must never launch a probe")

    monkeypatch.setattr(ab, "run_probe", _explode)
    _install_records(monkeypatch, _records({1: 16 * pf.GiB}))

    assert _decision(_spec(tmp_path, batch=1)).budget.accelerator_peak_bytes


def test_the_probe_floor_never_consults_a_measurement(tmp_path, monkeypatch):
    """The floor is a deliberate under-estimate; gating the probe on the
    measurement it exists to produce would refuse hardware that fits."""

    _write_coco(tmp_path)
    _install_records(monkeypatch, _records({1: 40 * pf.GiB}))

    decision = pf.assess_probe_preflight(
        _spec(tmp_path, batch=1), batch=1, cuda_device=_cuda(), observation=_host()
    )

    assert decision.budget.accelerator_peak_bytes < 40 * pf.GiB
    assert decision.device_peak_provenance == "probe_floor"


def test_the_stamped_analytic_estimate_is_the_one_the_budget_used(
    tmp_path, monkeypatch
):
    """One derivation, not two that happen to agree. A non-trivial dataset so
    the dense-mask term is non-zero and a drifting duplicate would show."""

    _write_coco(tmp_path, tiles=6, instances_per_tile=9)
    _install_records(monkeypatch, ())

    for batch in (1, 3):
        decision = _decision(_spec(tmp_path, batch=batch))
        assert decision.device_peak_analytic_bytes > 0
        assert (
            decision.budget.accelerator_peak_bytes
            == decision.device_peak_analytic_bytes
        )
