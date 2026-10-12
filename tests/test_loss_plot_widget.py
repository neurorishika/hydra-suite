from __future__ import annotations

from hydra_suite.trackerkit.gui.widgets.loss_plot_widget import (
    parse_ultralytics_log_line,
)


def test_parse_epoch_line():
    line = "      1/100      0.987      1.234      0.567        40       640   100%"
    result = parse_ultralytics_log_line(line)
    assert result is not None
    assert result["epoch"] == 1
    assert result["total_epochs"] == 100
    assert abs(result["box_loss"] - 0.987) < 0.01


def test_parse_non_epoch_line():
    line = "Ultralytics YOLO v8.3.0 - training started"
    result = parse_ultralytics_log_line(line)
    assert result is None


def test_parse_val_metrics_line():
    line = "                 all        120        200      0.912      0.887      0.902      0.678"
    result = parse_ultralytics_log_line(line)
    assert result is None


def test_parse_supervised_line_with_ansi_and_carriage_returns():
    line = (
        "[segment_direct] \x1b[K    179/300      11.3G      0.837     0.8421"
        "     0.3406    0.00156     0.2713        190        640:  71% ━━╸─ 185/260\r"
        "\x1b[K    179/300      11.3G     0.8301     0.8402     0.3398   0.001552"
        "      0.270        150        640: 100% ━━━━━━━━━━━━ 260/260 6.9it/s"
    )
    result = parse_ultralytics_log_line(line)
    assert result == {
        "epoch": 179,
        "total_epochs": 300,
        "box_loss": 0.8301,
        "cls_loss": 0.3398,  # seg models: box, seg, cls, dfl, sem
        "dfl_loss": 0.001552,
    }
