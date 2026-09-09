"""HYDRA_MODELS_DIR relocates the models root without moving the data dir.

A portable job supplies its own models root (``HYDRA_MODELS_DIR=<job>/models``)
while engine artifacts, calibration profiles and training runs must stay on the
HOST's data dir -- that separation is the whole reason this override exists
independently of ``HYDRA_DATA_DIR``.
"""

from hydra_suite import paths


def test_models_dir_follows_env_override(tmp_path, monkeypatch):
    models = tmp_path / "job" / "models"
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(models))
    assert paths.get_models_dir() == models
    assert models.is_dir(), "get_models_dir must create the directory"


def test_models_dir_override_does_not_move_data_dir(tmp_path, monkeypatch):
    data = tmp_path / "hostdata"
    models = tmp_path / "job" / "models"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data))
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(models))
    assert paths.get_models_dir() == models
    assert paths.get_data_dir() == data
    # Engine artifacts and calibration profiles must stay host-scoped.
    assert paths.get_training_runs_dir().is_relative_to(data)


def test_models_dir_without_override_is_data_dir_models(tmp_path, monkeypatch):
    data = tmp_path / "hostdata"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data))
    monkeypatch.delenv("HYDRA_MODELS_DIR", raising=False)
    assert paths.get_models_dir() == data / "models"


def test_override_is_read_per_call_not_cached(tmp_path, monkeypatch):
    first = tmp_path / "a"
    second = tmp_path / "b"
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(first))
    assert paths.get_models_dir() == first
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(second))
    assert paths.get_models_dir() == second


def test_expanduser_is_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HYDRA_MODELS_DIR", "~/jobmodels")
    assert paths.get_models_dir() == tmp_path / "jobmodels"


def test_empty_override_is_ignored(tmp_path, monkeypatch):
    """An empty string must not relocate the root to the CWD."""
    data = tmp_path / "hostdata"
    monkeypatch.setenv("HYDRA_DATA_DIR", str(data))
    monkeypatch.setenv("HYDRA_MODELS_DIR", "")
    assert paths.get_models_dir() == data / "models"


def test_registry_path_follows_the_override(tmp_path, monkeypatch):
    models = tmp_path / "job" / "models"
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(models))
    from hydra_suite.core.inference import model_paths
    from hydra_suite.training import model_publish

    assert model_publish._registry_path() == models / "model_registry.json"
    assert model_paths.get_models_root_directory() == str(models)


def test_print_paths_reports_the_override(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HYDRA_MODELS_DIR", str(tmp_path / "m"))
    paths.print_paths()
    assert "HYDRA_MODELS_DIR" in capsys.readouterr().out
