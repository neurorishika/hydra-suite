import torch
from torch.utils.data import DataLoader, TensorDataset

from hydra_suite.training.calibration_fit import (
    CalibrationResult,
    fit_calibration_from_val,
)
from hydra_suite.training.runner import _calibrate_and_pack


class _FlatNet(torch.nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.lin = torch.nn.Linear(4, k)

    def forward(self, x):
        return self.lin(x) * 3.0  # inflate ⇒ overconfident


def _flat_loader(n=800, k=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = torch.randint(0, k, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=64)


def test_flat_returns_single_temperature_and_ece_drop():
    model = _FlatNet()
    res = fit_calibration_from_val(model, _flat_loader(), "cpu", num_factors=1)
    assert isinstance(res, CalibrationResult)
    assert len(res.temperatures) == 1
    assert len(res.ece_before) == 1 and len(res.ece_after) == 1
    assert res.ece_after[0] <= res.ece_before[0] + 1e-6
    assert isinstance(res.signature, str) and res.signature


def test_multihead_returns_per_factor_temperatures():
    # 2 factors of width 3 and 2; labels (N,2)
    class _MH(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 5)  # 3 + 2 concat

        def forward(self, x):
            return self.lin(x) * 3.0

    g = torch.Generator().manual_seed(2)
    x = torch.randn(400, 4, generator=g)
    y = torch.stack(
        [
            torch.randint(0, 3, (400,), generator=g),
            torch.randint(0, 2, (400,), generator=g),
        ],
        dim=1,
    )
    loader = DataLoader(TensorDataset(x, y), batch_size=50)

    def split(logits):  # (N,5) -> [(N,3),(N,2)]
        return [logits[:, :3], logits[:, 3:]]

    res = fit_calibration_from_val(
        _MH(), loader, "cpu", split_logits=split, num_factors=2
    )
    assert len(res.temperatures) == 2
    assert len(res.ece_before) == 2 and len(res.ece_after) == 2


def test_calibrate_and_pack_empty_loader_does_not_raise():
    # A val_loader that yields zero batches must degrade to "uncalibrated but
    # saved" ({}), not raise -- an empty val split must never sink a training
    # run that produced a fully-trained checkpoint.
    model = _FlatNet()
    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long))
    )
    out = _calibrate_and_pack(model, empty_loader, "cpu", num_factors=1)
    assert out == {}


def test_calibrate_and_pack_none_loader_returns_empty():
    model = _FlatNet()
    assert _calibrate_and_pack(model, None, "cpu", num_factors=1) == {}


def test_calibrate_and_pack_normal_path_returns_three_keys():
    model = _FlatNet()
    out = _calibrate_and_pack(model, _flat_loader(), "cpu", num_factors=1)
    assert set(out.keys()) == {
        "calibration_temperature",
        "calibration_signature",
        "calibration_ece",
    }
