"""get_platform_config_dir ignores HYDRA_CONFIG_DIR.

run.sh redirects HYDRA_CONFIG_DIR at the job snapshot, so preflight cannot use
it to find the HOST's shared-root mount table. This helper is how preflight
reaches the real platformdirs location regardless of that redirection.
"""

from platformdirs import user_config_dir

from hydra_suite import paths


def test_ignores_the_hydra_config_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "job" / "config"))
    resolved = paths.get_platform_config_dir()
    assert resolved == paths.Path(user_config_dir(paths.APP_NAME, paths.APP_AUTHOR))
    assert not str(resolved).startswith(str(tmp_path))


def test_matches_user_config_dir_when_no_override_is_set(monkeypatch):
    monkeypatch.delenv("HYDRA_CONFIG_DIR", raising=False)
    assert paths.get_platform_config_dir() == paths.get_config_dir()


def test_is_stable_across_an_override_change(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "a"))
    first = paths.get_platform_config_dir()
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "b"))
    assert paths.get_platform_config_dir() == first


def test_empty_override_is_also_ignored(monkeypatch):
    """run.sh exports HYDRA_HOST_CONFIG_DIR='' -- the empty case must not leak here."""
    monkeypatch.setenv("HYDRA_CONFIG_DIR", "")
    assert paths.get_platform_config_dir() == paths.Path(
        user_config_dir(paths.APP_NAME, paths.APP_AUTHOR)
    )
