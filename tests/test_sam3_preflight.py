"""Typed, metadata-only SAM3 resource admission tests."""

from __future__ import annotations

import json
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
    assert (
        many_tiles.budget.host_peak_bytes - one_tile.budget.host_peak_bytes
        == many_tiles.dataset.metadata_bytes - one_tile.dataset.metadata_bytes
    )


def test_absent_validation_and_disabled_publish_remove_inactive_phases(tmp_path):
    _write_coco(tmp_path)
    spec = _spec(tmp_path)
    spec.publish_policy = PublishPolicy(auto_import=False)

    decision = _decision(spec)

    assert {phase.name for phase in decision.request.phases} == {
        "model_load",
        "training",
    }
    assert all(phase.disk_transient_bytes == 0 for phase in decision.request.phases)


def test_lora_scope_changes_optimizer_state_and_adapter_terms(tmp_path):
    _write_coco(tmp_path)
    default = _decision(_spec(tmp_path))
    with_text = _decision(_spec(tmp_path, adapt_text_encoder=True))

    default_train = next(p for p in default.request.phases if p.name == "training")
    text_train = next(p for p in with_text.request.phases if p.name == "training")
    default_publish = next(p for p in default.request.phases if p.name == "publish")
    text_publish = next(p for p in with_text.request.phases if p.name == "publish")
    assert _allocation(text_train, "LoRA and optimizer state") > _allocation(
        default_train, "LoRA and optimizer state"
    )
    assert _allocation(text_publish, "LoRA adapter") > _allocation(
        default_publish, "LoRA adapter"
    )


def test_explicit_empty_cuda_visible_devices_hides_all_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    with pytest.raises(ValueError, match="hides all"):
        pf._visible_device_selector("auto")


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
