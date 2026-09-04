"""Admission and isolation coverage for DetectKit dataset preparation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from hydra_suite.detectkit.config.training import SliceTrainingConfig
from hydra_suite.detectkit.jobs.dataset_preparation_sidecar import (
    MAX_SOURCES,
    _bounded_request_payload,
    _decode_result,
    _write_request,
    decode_request,
)
from hydra_suite.detectkit.jobs.training import DatasetPreparationRequest
from hydra_suite.training.contracts import SourceDataset, SplitConfig, TrainingRole
from hydra_suite.training.dataset_io import DatasetLimitError


def _request(source, *, sources=None):
    return DatasetPreparationRequest(
        sources=tuple(sources or [SourceDataset(path=str(source), level="aabb")]),
        roles=(TrainingRole.DETECT_DIRECT,),
        class_names=("ant",),
        split=SplitConfig(),
        seed=7,
        dedup=True,
        crop_pad_ratio=0.15,
        min_crop_size_px=32,
        enforce_square=True,
        imgsz_by_role=((TrainingRole.DETECT_DIRECT.value, 64),),
        slice_settings=SliceTrainingConfig(enabled=False),
    )


def _source(root):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("ant\n", encoding="utf-8")
    for index in range(3):
        cv2.imwrite(
            str(root / "images" / f"frame{index}.jpg"),
            np.full((16, 16, 3), index * 32, dtype=np.uint8),
        )
        (root / "labels" / f"frame{index}.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
    return root


def test_request_cardinality_is_checked_before_serialization(tmp_path):
    sources = [
        SourceDataset(path=str(tmp_path / str(i))) for i in range(MAX_SOURCES + 1)
    ]
    with pytest.raises(DatasetLimitError, match="sources"):
        _bounded_request_payload(_request(tmp_path, sources=sources))


def test_bounded_request_round_trip(tmp_path):
    request = _request(tmp_path)
    decoded = decode_request(_bounded_request_payload(request))
    assert decoded.sources == request.sources
    assert decoded.roles == request.roles
    assert decoded.slice_settings.enabled is False


def test_child_promotes_only_completed_workspace(tmp_path):
    from hydra_suite.detectkit.jobs.dataset_preparation_child import main

    source = _source(tmp_path / "source")
    workspace = tmp_path / "workspace"
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    staging = workspace / ".dataset-preparation-test.staging"
    final = workspace / "prepared" / "dataset-preparation-test"
    _write_request(request_path, _bounded_request_payload(_request(source)))
    assert (
        main(
            [
                "--request",
                str(request_path),
                "--result",
                str(result_path),
                "--staging-root",
                str(staging),
                "--final-root",
                str(final),
                "--disk-required-bytes",
                "1",
            ]
        )
        == 0
    )
    result = _decode_result(result_path)
    dataset = result.role_dataset_dirs[TrainingRole.DETECT_DIRECT.value]
    assert "/prepared/dataset-preparation-" in dataset
    assert result.preflight is not None and result.preflight.valid
    assert not staging.exists()
