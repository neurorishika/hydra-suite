from hydra_suite.posekit.gui.dialogs.training import parse_loss_components


def test_parses_vitpose_metrics_csv(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "epoch,train_loss,val_loss,pck@0.05,pck@0.1\n"
        "0,1.5,1.8,0.10,0.20\n"
        "1,0.9,1.1,0.30,0.45\n",
        encoding="utf-8",
    )
    train_vals, val_vals = parse_loss_components(tmp_path)
    assert train_vals == {"loss": [1.5, 0.9]}
    assert val_vals == {"loss": [1.8, 1.1]}


def test_returns_empty_when_no_csv(tmp_path):
    assert parse_loss_components(tmp_path) == ({}, {})


def test_results_csv_preserved_as_multi_component(tmp_path):
    # Mirror the REAL column/key format observed in Step 1. Adjust the header
    # and the expected component key(s) to match the transplanted logic exactly.
    (tmp_path / "results.csv").write_text(
        "epoch,train/box_loss,val/box_loss\n" "0,1.0,1.2\n" "1,0.5,0.7\n",
        encoding="utf-8",
    )
    train_vals, val_vals = parse_loss_components(tmp_path)
    # component key derived by the existing logic (e.g. "box_loss")
    assert list(train_vals.values())[0] == [1.0, 0.5]
    assert list(val_vals.values())[0] == [1.2, 0.7]
    assert train_vals.keys() == val_vals.keys()
