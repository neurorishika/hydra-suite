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

from hydra_suite.training.sam3_lora import cli


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
