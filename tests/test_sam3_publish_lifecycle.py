"""Publish provenance and stuck-run reaping in the SAM3 training lifecycle.

Defect A: the published-model registry and the training-run registry are two
separate files. A publish that never had a run row leaves an artifact whose
provenance cannot be reconstructed, and a run whose owner process vanished
stays ``running`` forever.
"""

import json
import os
from pathlib import Path

import pytest

from hydra_suite.training.contracts import Sam3LoraParams


@pytest.fixture()
def isolated_data_dir(tmp_path, monkeypatch):
    """Point every hydra data path at tmp_path; never touch the real registry."""

    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def _runs_registry() -> Path:
    from hydra_suite.training.registry import get_registry_path

    return get_registry_path()


def _write_runs(records) -> None:
    path = _runs_registry()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": list(records)}), encoding="utf-8")


def test_publish_refuses_a_run_that_was_never_registered(isolated_data_dir, tmp_path):
    """A publish with no run row must be refused before any work happens."""

    from hydra_suite.training.sam3_lora.publish import (
        Sam3PublishError,
        publish_sam3_model,
    )

    _write_runs([])
    with pytest.raises(Sam3PublishError, match="not registered"):
        publish_sam3_model(
            run_id="20260904-143950_semantic_sam3_ca1bd031",
            adapters_path=tmp_path / "adapters.pt",
            base_checkpoint=tmp_path / "base.pt",
            build_manifest={},
            params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
            source_fingerprint="courtship:whatever",
            models_root=tmp_path / "models",
        )
    # Refused before admission, so nothing was staged on disk.
    assert not (tmp_path / "models").exists()


def test_publish_admits_a_registered_run(isolated_data_dir, tmp_path):
    """The gate keys on the run row, not on the artifact."""

    from hydra_suite.training.sam3_lora import publish as pub

    _write_runs([{"run_id": "run-1", "status": "completed"}])
    pub._require_registered_run("run-1")


def test_provably_dead_owner_is_reaped(isolated_data_dir):
    """A ``running`` row whose owner pid is gone becomes terminal."""

    import socket

    from hydra_suite.training.registry import (
        STALE_RUN_STATUS,
        load_registry,
        reap_stale_running_runs,
    )

    # A pid that has certainly exited: fork a child and reap it.
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child
        os._exit(0)
    os.waitpid(pid, 0)

    _write_runs(
        [
            {
                "run_id": "dead",
                "status": "running",
                "owner_pid": pid,
                "owner_hostname": socket.gethostname(),
                "owner_process_start_time": 1.0,
            },
            {
                "run_id": "live",
                "status": "running",
                "owner_pid": os.getpid(),
                "owner_hostname": socket.gethostname(),
                "owner_process_start_time": __import__("psutil")
                .Process(os.getpid())
                .create_time(),
            },
            {"run_id": "legacy", "status": "running"},
            {
                "run_id": "owned",
                "status": "recovery-required",
                "owner_pid": pid,
                "owner_hostname": socket.gethostname(),
                "owner_process_start_time": 1.0,
            },
        ]
    )
    assert reap_stale_running_runs() == ["dead"]
    by_id = {rec["run_id"]: rec for rec in load_registry()["runs"]}
    assert by_id["dead"]["status"] == STALE_RUN_STATUS
    assert by_id["dead"]["finished_at"]
    # Never reaped: still live, no owner recorded, and Defect-B owned state.
    assert by_id["live"]["status"] == "running"
    assert by_id["legacy"]["status"] == "running"
    assert by_id["owned"]["status"] == "recovery-required"


def test_create_run_record_stamps_its_owner(isolated_data_dir, tmp_path):
    """Without an owner stamp no future reaper could ever prove the run dead."""

    import socket

    from hydra_suite.training.contracts import (
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )
    from hydra_suite.training.registry import create_run_record

    spec = TrainingRunSpec(
        role=TrainingRole.SEMANTIC_SAM3,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path),
        base_model="sam3",
        hyperparams=TrainingHyperParams(),
    )
    rec = create_run_record(spec, run_id="r1", run_dir=tmp_path, dataset_fp="fp")
    assert rec["owner_pid"] == os.getpid()
    assert rec["owner_hostname"] == socket.gethostname()
    assert rec["owner_process_start_time"] > 0
