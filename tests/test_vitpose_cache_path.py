import importlib


def test_vitpose_cache_honors_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path))
    import hydra_suite.paths as paths

    importlib.reload(paths)
    d = paths.get_vitpose_cache_dir()
    assert str(d).startswith(str(tmp_path))
    assert "vitpose" in str(d).lower()
