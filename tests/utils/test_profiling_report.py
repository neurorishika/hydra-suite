from hydra_suite.utils.profiling_report import render_tree_lines

SNAP = {
    "name": "root",
    "total_s": 10.0,
    "self_s": 1.0,
    "n_calls": 1,
    "units": 0.0,
    "max_s": 10.0,
    "first_call_s": 10.0,
    "thread": "MainThread",
    "children": [
        {
            "name": "window",
            "total_s": 9.0,
            "self_s": 1.0,
            "n_calls": 100,
            "units": 100.0,
            "max_s": 0.5,
            "first_call_s": 0.5,
            "thread": "MainThread",
            "children": [
                {
                    "name": "cnn",
                    "total_s": 6.0,
                    "self_s": 6.0,
                    "n_calls": 100,
                    "units": 4000.0,
                    "max_s": 0.1,
                    "first_call_s": 0.1,
                    "thread": "MainThread",
                    "children": [],
                },
                {
                    "name": "decode",
                    "total_s": 2.0,
                    "self_s": 2.0,
                    "n_calls": 100,
                    "units": 0.0,
                    "max_s": 0.1,
                    "first_call_s": 0.1,
                    "thread": "pipeline-obb-producer",
                    "children": [],
                },
            ],
        }
    ],
}


def test_children_sort_by_total_descending():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    body = [ln for ln in lines if "cnn" in ln or "decode" in ln]
    assert "cnn" in body[0] and "decode" in body[1]


def test_percentages_are_of_the_parent():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    cnn = next(ln for ln in lines if "cnn" in ln)
    assert "66.7%" in cnn  # 6.0 / 9.0


def test_ms_per_unit_is_shown_when_units_present():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    cnn = next(ln for ln in lines if "cnn" in ln)
    assert "1.50 ms/u" in cnn  # 6.0 s / 4000 units


def test_ms_per_unit_omitted_without_units():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    decode = next(ln for ln in lines if "decode" in ln)
    assert "ms/u" not in decode


def test_off_thread_nodes_are_marked_concurrent():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    decode = next(ln for ln in lines if "decode" in ln)
    assert "concurrent" in decode
    cnn = next(ln for ln in lines if "cnn" in ln)
    assert "concurrent" not in cnn


def test_depth_is_indented():
    lines = render_tree_lines(SNAP, main_thread="MainThread")
    window = next(ln for ln in lines if "window" in ln)
    cnn = next(ln for ln in lines if "cnn" in ln)
    assert len(cnn) - len(cnn.lstrip()) > len(window) - len(window.lstrip())
