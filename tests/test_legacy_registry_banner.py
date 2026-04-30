def test_count_legacy_entries_all_pre_v2(tmp_path, monkeypatch):
    import json as _json

    from hydra_suite.training import model_publish

    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text(_json.dumps({"a.pth": {"arch": "x"}, "b.pth": {"arch": "y"}}))
    monkeypatch.setattr(model_publish, "_registry_path", lambda: reg_path)

    assert model_publish.count_legacy_registry_entries() == 2


def test_count_legacy_entries_v2_with_partial_entries(tmp_path, monkeypatch):
    import json as _json

    from hydra_suite.training import model_publish

    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text(
        _json.dumps(
            {
                "schema_version": 2,
                "entries": {
                    "good.pth": {
                        "schema_version": 2,
                        "factor_names": ["flat"],
                        "class_names_per_factor": [["a"]],
                        "input_size": [64, 64],
                    },
                    "partial.pth": {
                        "arch": "y",
                    },
                },
            }
        )
    )
    monkeypatch.setattr(model_publish, "_registry_path", lambda: reg_path)

    assert model_publish.count_legacy_registry_entries() == 1
