from pathlib import Path

import pytest

from hydra_suite.classkit.config.presets import (
    age_preset,
    color_tag_preset,
    get_custom_scheme_preset_key,
    head_tail_preset,
    list_saved_scheme_presets,
    save_scheme_preset,
)
from hydra_suite.classkit.config.schemas import Factor, LabelingScheme, ProjectConfig


def test_factor_has_name_and_labels():
    f = Factor(name="tag_1", labels=["red", "blue", "green"])
    assert f.name == "tag_1"
    assert f.labels == ["red", "blue", "green"]
    assert f.shortcut_keys == []


def test_scheme_single_factor():
    scheme = LabelingScheme(
        name="age",
        factors=[Factor(name="age", labels=["young", "old"])],
        training_modes=["flat_tiny", "flat_yolo"],
    )
    assert len(scheme.factors) == 1
    assert scheme.total_classes == 2


def test_scheme_two_factor_cartesian():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red", "blue", "green"]),
            Factor(name="tag_2", labels=["red", "blue", "green"]),
        ],
        training_modes=["flat_yolo", "multihead_yolo"],
    )
    assert scheme.total_classes == 16
    assert scheme.factors[0].labels == ["red", "blue", "green", "unknown"]
    assert scheme.factors[1].labels == ["red", "blue", "green", "unknown"]


def test_scheme_composite_label_round_trip():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red", "blue"]),
            Factor(name="tag_2", labels=["green", "yellow"]),
        ],
        training_modes=["flat_yolo"],
    )
    composite = scheme.encode_label(["red", "green"])
    assert composite == "red_green"
    decoded = scheme.decode_label(composite)
    assert decoded == ["red", "green"]
    # Legacy pipe-encoded composites must still decode for old project files.
    assert scheme.decode_label("red|green") == ["red", "green"]


def test_project_config_accepts_scheme():
    scheme = LabelingScheme(
        name="test",
        factors=[Factor(name="f", labels=["a", "b"])],
        training_modes=["flat_tiny"],
    )
    cfg = ProjectConfig(name="proj", classes=[], root_dir=Path("/tmp"), scheme=scheme)
    assert cfg.scheme is not None


def test_project_config_scheme_defaults_none():
    cfg = ProjectConfig(name="proj", classes=[], root_dir=Path("/tmp"))
    assert cfg.scheme is None


def test_encode_label_wrong_length_raises():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red"]),
            Factor(name="tag_2", labels=["blue"]),
        ],
        training_modes=["flat_yolo"],
    )
    with pytest.raises(ValueError, match="Expected 2 factor values"):
        scheme.encode_label(["red"])  # missing second factor


def test_decode_label_wrong_parts_raises():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red"]),
            Factor(name="tag_2", labels=["blue"]),
        ],
        training_modes=["flat_yolo"],
    )
    with pytest.raises(ValueError, match="Expected 2 parts"):
        scheme.decode_label("red_blue_green")  # too many parts


def test_head_tail_preset():
    scheme = head_tail_preset()
    assert scheme.name == "head_tail"
    assert len(scheme.factors) == 1
    assert set(scheme.factors[0].labels) == {"left", "right", "up", "down"}
    assert scheme.total_classes == 4
    assert scheme.training_modes == ["flat_yolo", "flat_custom"]


def test_color_tag_preset_1factor():
    colors = ["red", "blue", "green", "yellow", "white"]
    scheme = color_tag_preset(n_factors=1, colors=colors)
    assert scheme.total_classes == 5
    assert len(scheme.factors) == 1
    assert scheme.training_modes == ["flat_yolo", "flat_custom"]


def test_color_tag_preset_2factor():
    colors = ["red", "blue", "green", "yellow", "white"]
    scheme = color_tag_preset(n_factors=2, colors=colors)
    assert scheme.total_classes == 36
    assert len(scheme.factors) == 2
    assert scheme.factors[0].name == "tag_1"
    assert scheme.factors[1].name == "tag_2"
    assert scheme.factors[0].labels[-1] == "unknown"
    assert scheme.factors[1].labels[-1] == "unknown"
    assert scheme.training_modes == [
        "flat_yolo",
        "flat_custom",
        "multihead_yolo",
        "multihead_custom",
    ]


def test_color_tag_preset_3factor():
    colors = ["red", "blue", "green", "yellow", "white"]
    scheme = color_tag_preset(n_factors=3, colors=colors)
    assert scheme.total_classes == 216
    assert scheme.training_modes == [
        "flat_yolo",
        "flat_custom",
        "multihead_yolo",
        "multihead_custom",
    ]


def test_age_preset_default():
    scheme = age_preset()
    assert scheme.total_classes == 2
    assert "young" in scheme.factors[0].labels
    assert "old" in scheme.factors[0].labels
    assert scheme.training_modes == ["flat_yolo", "flat_custom"]


def test_age_preset_extra_classes():
    scheme = age_preset(extra_classes=["juvenile"])
    assert scheme.total_classes == 3
    assert "juvenile" in scheme.factors[0].labels


def test_color_tag_preset_custom_colors():
    scheme = color_tag_preset(n_factors=2, colors=["a", "b", "c"])
    assert scheme.total_classes == 16


def test_multi_factor_valid_labels_include_partial_unknown_composites():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red", "blue"]),
            Factor(name="tag_2", labels=["left", "right"]),
        ],
        training_modes=["multihead_custom"],
    )

    valid_labels = scheme.valid_encoded_labels()

    assert "red_unknown" in valid_labels
    assert "unknown_left" in valid_labels
    assert "unknown_unknown" in valid_labels


def test_color_tag_preset_invalid_n_factors_raises():
    with pytest.raises(ValueError):
        color_tag_preset(n_factors=0, colors=["red"])


def test_color_tag_preset_empty_colors_raises():
    with pytest.raises(ValueError):
        color_tag_preset(n_factors=1, colors=[])


def test_factor_round_trip():
    f = Factor(name="tag_1", labels=["red", "blue"], shortcut_keys=["r", "b"])
    assert Factor.from_dict(f.to_dict()) == f


def test_labeling_scheme_round_trip():
    scheme = LabelingScheme(
        name="test",
        factors=[Factor(name="color", labels=["red", "blue"])],
        training_modes=["flat_tiny"],
        description="A test scheme",
    )
    assert LabelingScheme.from_dict(scheme.to_dict()) == scheme


def test_labeling_scheme_from_dict_defaults():
    d = {"name": "minimal", "factors": [{"name": "f", "labels": ["a"]}]}
    scheme = LabelingScheme.from_dict(d)
    assert scheme.training_modes == []
    assert scheme.description == ""
    assert scheme.factors[0].shortcut_keys == []


def test_save_scheme_preset_persists_in_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "config"))
    scheme = LabelingScheme(
        name="tag_stack",
        factors=[
            Factor(name="tag_1", labels=["red", "blue"]),
            Factor(name="tag_2", labels=["left", "right"]),
        ],
        training_modes=["flat_tiny", "multihead_tiny"],
    )

    preset_path = save_scheme_preset("Colony Tags", scheme)
    presets = list_saved_scheme_presets()

    assert preset_path.exists()
    assert [preset.key for preset in presets] == [
        get_custom_scheme_preset_key("Colony Tags")
    ]
    assert presets[0].is_custom is True
    assert presets[0].scheme == scheme


def test_normalize_legacy_composite_label_converts_pipe_to_underscore():
    from hydra_suite.classkit.config.schemas import normalize_legacy_composite_label

    assert normalize_legacy_composite_label("red|green") == "red_green"
    assert normalize_legacy_composite_label("red_green") == "red_green"
    assert normalize_legacy_composite_label(None) == ""
    assert normalize_legacy_composite_label("") == ""


def test_db_migration_rewrites_pipe_encoded_labels(tmp_path):
    """Legacy pipe-encoded composite labels should be migrated on demand."""
    from hydra_suite.classkit.core.store.db import ClassKitDB

    db_path = tmp_path / "classkit.db"
    db = ClassKitDB(db_path)

    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"x")
    db.add_images([image_path], hashes=["dummy"])
    db.update_labels_batch({str(image_path): "red|left"})

    assert db.get_all_labels() == ["red|left"]

    migrated = db.migrate_legacy_composite_labels()

    assert migrated == 1
    assert db.get_all_labels() == ["red_left"]


def test_save_scheme_preset_requires_overwrite_for_duplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "config"))
    scheme = LabelingScheme(
        name="age",
        factors=[Factor(name="age", labels=["young", "old"])],
        training_modes=["flat_tiny"],
    )

    save_scheme_preset("Age Labels", scheme)

    with pytest.raises(FileExistsError):
        save_scheme_preset("Age Labels", scheme)
