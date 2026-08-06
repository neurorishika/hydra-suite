import os

import pandas as pd

from hydra_suite.core.post.rich_export import (
    drop_empty_rich_export_columns,
    rich_export_path,
    write_rich_export_csv,
)


def test_rich_export_path_suffixes():
    assert rich_export_path("/a/b_final.csv") == "/a/b_final_with_individual.csv"
    assert rich_export_path("/a/b_final.csv", legacy=True) == "/a/b_final_with_pose.csv"


def test_drop_empty_columns_removes_all_nan():
    df = pd.DataFrame({"keep": [1, 2], "drop": [None, None]})
    out = drop_empty_rich_export_columns(df)
    assert list(out.columns) == ["keep"]


def test_write_rich_export_removes_legacy_alias(tmp_path):
    final = tmp_path / "clip_final.csv"
    legacy = tmp_path / "clip_final_with_pose.csv"
    legacy.write_text("stale\n")
    out = write_rich_export_csv(pd.DataFrame({"X": [1, 2]}), str(final))
    assert out == str(tmp_path / "clip_final_with_individual.csv")
    assert not os.path.exists(str(legacy))
