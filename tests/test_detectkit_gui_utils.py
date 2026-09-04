from __future__ import annotations


def test_ui_settings_read_is_bounded_before_json_decode(tmp_path, monkeypatch):
    from hydra_suite.detectkit.gui import utils

    settings = tmp_path / "ui_settings.json"
    settings.write_bytes(b" " * (utils.MAX_UI_JSON_BYTES + 1))
    monkeypatch.setattr(utils, "get_ui_settings_path", lambda: settings)

    assert utils.load_ui_settings() == {}


def test_gui_json_loader_rejects_deep_or_non_mapping_input(tmp_path):
    import pytest

    from hydra_suite.detectkit.gui import utils

    deep = tmp_path / "deep.json"
    deep.write_text("[" * 100 + "0" + "]" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="nesting"):
        utils.load_bounded_json_mapping(deep)

    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        utils.load_bounded_json_mapping(array)
