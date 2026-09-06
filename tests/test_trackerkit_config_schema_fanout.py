"""Session-state fields for the batch GPU fan-out (Task 6)."""

from hydra_suite.trackerkit.config.schemas import TrackerConfig


def test_defaults():
    c = TrackerConfig()
    assert c.batch_parallel is False
    assert c.batch_parallel_jobs == 0
    assert c.batch_parallel_gpus == "auto"


def test_round_trip():
    c = TrackerConfig(
        batch_parallel=True, batch_parallel_jobs=4, batch_parallel_gpus="0-3"
    )
    d = c.to_dict()
    assert (
        d["batch_parallel"] is True
        and d["batch_parallel_jobs"] == 4
        and d["batch_parallel_gpus"] == "0-3"
    )
    back = TrackerConfig.from_dict(d)
    assert (
        back.batch_parallel,
        back.batch_parallel_jobs,
        back.batch_parallel_gpus,
    ) == (
        True,
        4,
        "0-3",
    )


def test_from_dict_tolerates_missing_keys():
    back = TrackerConfig.from_dict({})
    assert back.batch_parallel is False
    assert back.batch_parallel_jobs == 0
    assert back.batch_parallel_gpus == "auto"


def test_fanout_fields_never_reach_engine_config():
    """The GUI's sidecar config builder must not emit these keys."""
    from pathlib import Path

    import hydra_suite.trackerkit.gui.orchestrators.config as mod

    src = Path(mod.__file__).read_text()
    assert "batch_parallel" not in src
