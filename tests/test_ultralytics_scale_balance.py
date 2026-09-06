"""Tests for no-data-reduction SAHI multi-scale loss balancing."""

import json

import pytest

from hydra_suite.training.ultralytics_scale_balance import (
    ScaleGroupedBatchSampler,
    _balance_settings_from_argv,
    _grouped_loader,
    install_sahi_multiscale_loss_balance,
    scale_group_for_path,
    scale_group_weights,
)


def test_scale_group_for_path_reads_emitted_tile_and_full_frame_names():
    assert scale_group_for_path("frame_t2000x2000_0001.jpg") == "tile:2000x2000"
    assert scale_group_for_path("frame_full.jpg") == "full"
    assert scale_group_for_path("ordinary_frame.jpg") == "other"


def test_scale_group_weights_are_inverse_frequency_without_full_frame_reweighting():
    paths = (
        ["a_t100x100_0000.jpg"]
        + [f"b_t50x50_{index:04d}.jpg" for index in range(4)]
        + ["a_full.jpg"]
    )

    groups, weights = scale_group_weights(paths, power=1.0)

    assert groups == ["tile:100x100"] + ["tile:50x50"] * 4 + ["full"]
    assert weights["tile:100x100"] == pytest.approx(2.5)
    assert weights["tile:50x50"] == pytest.approx(0.625)
    assert weights["full"] == 1.0


def test_grouped_sampler_keeps_every_tile_once_and_never_mixes_scales():
    groups = ["tile:100x100", "tile:100x100", "tile:50x50", "full", "full"]
    sampler = ScaleGroupedBatchSampler(groups, batch_size=2, seed=7)

    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(
        range(len(groups))
    )
    assert all(len({groups[index] for index in batch}) == 1 for batch in batches)


def test_grouped_loader_retains_the_batch_size_contract_expected_by_ultralytics():
    import torch

    class _Dataset(torch.utils.data.Dataset):
        im_files = ["a_t100x100_0000.jpg", "b_t50x50_0000.jpg"]

        @staticmethod
        def collate_fn(batch):
            return batch

        def __len__(self):
            return len(self.im_files)

        def __getitem__(self, index):
            return index

    loader = _grouped_loader(
        _Dataset(), batch_size=2, workers=0, device="cpu", rank=-1, seed=7, power=0.5
    )
    try:
        assert loader.batch_size == 2
        assert len(loader) == 2
    finally:
        loader.close()


def test_manifest_settings_enable_balance_only_for_sliced_training_dataset(tmp_path):
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("path: .\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "type": "sliced_obb",
                "slice_geometry": {
                    "multiscale_loss_balance": {"enabled": True, "power": 0.5}
                },
            }
        ),
        encoding="utf-8",
    )

    assert _balance_settings_from_argv([f"data={dataset_yaml}"]) == {"power": 0.5}


def test_installer_wires_the_ultralytics_trainer_only_for_enabled_sliced_data(tmp_path):
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("path: .\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "type": "sliced_obb",
                "slice_geometry": {
                    "multiscale_loss_balance": {"enabled": True, "power": 0.5}
                },
            }
        ),
        encoding="utf-8",
    )

    assert install_sahi_multiscale_loss_balance([f"data={dataset_yaml}"])
    assert hasattr(
        DetectionTrainer.get_dataloader, "_hydra_sahi_scale_balance_original"
    )
