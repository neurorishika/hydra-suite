"""Session-state fields for the batch GPU fan-out (Task 6)."""

from hydra_suite.trackerkit.config.schemas import TrackerConfig


def test_defaults():
    c = TrackerConfig()
    assert c.batch_parallel is False
    assert c.batch_parallel_jobs == 0
    assert c.batch_parallel_gpus == "auto"
    assert c.batch_parallel_jobs_per_gpu == 1
    assert c.batch_parallel_longest_first is False


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
    assert back.batch_parallel_jobs_per_gpu == 1


def test_fanout_fields_never_reach_engine_config():
    """The GUI's sidecar config builder must not emit these keys."""
    from pathlib import Path

    import hydra_suite.trackerkit.gui.orchestrators.config as mod

    src = Path(mod.__file__).read_text()
    assert "batch_parallel" not in src


def test_jobs_per_gpu_round_trips():
    c = TrackerConfig(batch_parallel_jobs_per_gpu=3)
    d = c.to_dict()
    assert d["batch_parallel_jobs_per_gpu"] == 3
    assert TrackerConfig.from_dict(d).batch_parallel_jobs_per_gpu == 3


def test_longest_first_round_trips():
    d = TrackerConfig(batch_parallel_longest_first=True).to_dict()
    assert d["batch_parallel_longest_first"] is True
    assert TrackerConfig.from_dict(d).batch_parallel_longest_first is True
    assert TrackerConfig.from_dict({}).batch_parallel_longest_first is False


def test_cpu_budget_round_trips():
    cfg = TrackerConfig()
    assert cfg.cpu_budget == 0
    cfg.cpu_budget = 6
    assert TrackerConfig.from_dict(cfg.to_dict()).cpu_budget == 6
    assert TrackerConfig.from_dict({}).cpu_budget == 0
    assert TrackerConfig.from_dict({"cpu_budget": -3}).cpu_budget == 0


def test_cpu_budget_never_reaches_engine_config():
    """Session state like batch_parallel_*: the sidecar builder never emits it."""
    from pathlib import Path

    import hydra_suite.trackerkit.engine_params as engine_mod
    import hydra_suite.trackerkit.gui.orchestrators.config as mod

    assert "cpu_budget" not in Path(mod.__file__).read_text()
    assert "cpu_budget" not in Path(engine_mod.__file__).read_text()
