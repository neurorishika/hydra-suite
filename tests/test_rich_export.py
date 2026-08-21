import pandas as pd

from hydra_suite.core.post.rich_export import count_by_source


def test_count_by_source_counts_real_and_interp():
    df = pd.DataFrame(
        {
            "TagSource": ["real", "real", "interp", None, "interp"],
        }
    )
    assert count_by_source(df, "TagSource") == {"real": 2, "interp": 2}


def test_count_by_source_missing_column_returns_zeros():
    df = pd.DataFrame({"Other": [1, 2, 3]})
    assert count_by_source(df, "TagSource") == {"real": 0, "interp": 0}


def test_count_by_source_empty_df():
    df = pd.DataFrame({"TagSource": []})
    assert count_by_source(df, "TagSource") == {"real": 0, "interp": 0}
