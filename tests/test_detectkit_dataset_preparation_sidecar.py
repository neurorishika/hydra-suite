"""Admission and isolation coverage for DetectKit dataset preparation."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

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
    prepare_role_datasets_contained,
)
from hydra_suite.detectkit.jobs.training import (
    DatasetPreparationCancelled,
    DatasetPreparationRequest,
)
from hydra_suite.runtime.process_supervisor import ExitKind
from hydra_suite.training import TrainingOrchestrator
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


def test_request_text_is_bounded_before_json_materialization(tmp_path):
    request = _request(tmp_path)
    request = DatasetPreparationRequest(
        sources=request.sources,
        roles=request.roles,
        class_names=("x" * 16_385,),
        split=request.split,
        seed=request.seed,
        dedup=request.dedup,
        crop_pad_ratio=request.crop_pad_ratio,
        min_crop_size_px=request.min_crop_size_px,
        enforce_square=request.enforce_square,
        imgsz_by_role=request.imgsz_by_role,
        slice_settings=request.slice_settings,
    )
    with pytest.raises(DatasetLimitError, match="before|exceeds 16384"):
        _bounded_request_payload(request)


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


@pytest.mark.skipif(
    os.environ.get("HYDRA_RUN_CONTAINMENT_SMOKE") != "1",
    reason="requires an unrestricted POSIX guardian launch boundary",
)
def test_real_supervised_preparation_sidecar_smoke(tmp_path, monkeypatch):
    source = _source(tmp_path / "source")
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path / "hydra-data"))
    result = prepare_role_datasets_contained(
        TrainingOrchestrator(workspace),
        _request(source),
        log=lambda _message: None,
        status=lambda _message: None,
        should_cancel=lambda: False,
    )
    assert result.preflight is not None and result.preflight.valid
    assert Path(result.role_dataset_dirs["detect_direct"]).is_dir()


def test_child_failure_leaves_no_partial_dataset(tmp_path):
    from hydra_suite.detectkit.jobs.dataset_preparation_child import main

    source = _source(tmp_path / "source")
    # Deduplication collapses these to one item, making the final validation
    # fail after files have already been copied into the private workspace.
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    for path in (source / "images").glob("*.jpg"):
        cv2.imwrite(str(path), image)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    staging = tmp_path / "workspace" / ".dataset-preparation-fail.staging"
    final = tmp_path / "workspace" / "prepared" / "dataset-preparation-fail"
    _write_request(request_path, _bounded_request_payload(_request(source)))
    code = main(
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
    assert code == 1
    assert not staging.exists()
    assert not final.exists()


def test_parent_cancellation_terminates_sidecar_and_cleans_private_paths(
    tmp_path, monkeypatch
):
    from hydra_suite.detectkit.jobs import dataset_preparation_sidecar as module

    source = _source(tmp_path / "source")
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path / "hydra-data"))
    canceled = []

    class Output:
        def drain(self, timeout=0.0):
            return [], False, None

    class Sidecar:
        def __init__(self, *_args, **_kwargs):
            self.output = Output()
            self.process = SimpleNamespace(poll=lambda: None)

        def cancel(self, grace):
            canceled.append(grace)

    monkeypatch.setattr(module, "SupervisedSidecar", Sidecar)
    with pytest.raises(DatasetPreparationCancelled):
        module.prepare_role_datasets_contained(
            TrainingOrchestrator(tmp_path / "workspace"),
            _request(source),
            log=lambda _message: None,
            status=lambda _message: None,
            should_cancel=lambda: True,
        )
    assert canceled
    assert not list((tmp_path / "workspace").glob(".dataset-preparation-*.staging"))


def test_hard_limit_classification_is_preserved_and_output_removed(
    tmp_path, monkeypatch
):
    from hydra_suite.detectkit.jobs import dataset_preparation_sidecar as module

    source = _source(tmp_path / "source")
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path / "hydra-data"))

    class Output:
        def drain(self, timeout=0.0):
            return [], True, None

    class Sidecar:
        def __init__(self, *_args, **_kwargs):
            self.output = Output()
            self.process = SimpleNamespace(poll=lambda: 137)

        def wait(self, **_kwargs):
            return SimpleNamespace(
                classified_exit=SimpleNamespace(
                    kind=ExitKind.HOST_HARD_LIMIT,
                    message="host hard memory limit reached",
                )
            )

        def cancel(self, _grace):
            return None

    monkeypatch.setattr(module, "SupervisedSidecar", Sidecar)
    with pytest.raises(RuntimeError, match="host hard memory limit") as raised:
        module.prepare_role_datasets_contained(
            TrainingOrchestrator(tmp_path / "workspace"),
            _request(source),
            log=lambda _message: None,
            status=lambda _message: None,
            should_cancel=lambda: False,
        )
    assert raised.value.failure_kind is ExitKind.HOST_HARD_LIMIT
    assert not list((tmp_path / "workspace").glob(".dataset-preparation-*.staging"))
