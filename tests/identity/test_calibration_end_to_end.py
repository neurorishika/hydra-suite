import torch
from torch.utils.data import DataLoader, TensorDataset

from hydra_suite.training.runner import _calibrate_and_pack


class _FlatNet(torch.nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.lin = torch.nn.Linear(4, k)

    def forward(self, x):
        return self.lin(x) * 3.0  # inflate logits ⇒ overconfident, calibration helps


def _flat_loader(n=400, k=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = torch.randint(0, k, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=50)


def test_calibrate_and_pack_returns_expected_keys_for_flat_model():
    model = _FlatNet()
    cal = _calibrate_and_pack(model, _flat_loader(), "cpu", num_factors=1)

    assert set(cal.keys()) == {
        "calibration_temperature",
        "calibration_signature",
        "calibration_ece",
    }
    assert isinstance(cal["calibration_temperature"], list)
    assert len(cal["calibration_temperature"]) == 1
    assert isinstance(cal["calibration_temperature"][0], float)
    assert (
        isinstance(cal["calibration_signature"], str) and cal["calibration_signature"]
    )
    assert isinstance(cal["calibration_ece"], list)
    assert len(cal["calibration_ece"]) == 1


def test_calibrate_and_pack_returns_per_factor_temperatures():
    class _MH(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 5)  # factors of width 3 and 2

        def forward(self, x):
            return self.lin(x) * 3.0

    g = torch.Generator().manual_seed(4)
    x = torch.randn(300, 4, generator=g)
    y = torch.stack(
        [
            torch.randint(0, 3, (300,), generator=g),
            torch.randint(0, 2, (300,), generator=g),
        ],
        dim=1,
    )
    loader = DataLoader(TensorDataset(x, y), batch_size=50)

    def split(logits):
        return [logits[:, :3], logits[:, 3:]]

    cal = _calibrate_and_pack(_MH(), loader, "cpu", num_factors=2, split_logits=split)
    assert len(cal["calibration_temperature"]) == 2
    assert len(cal["calibration_ece"]) == 2


def test_calibrate_and_pack_returns_empty_dict_without_val_loader():
    model = _FlatNet()
    cal = _calibrate_and_pack(model, None, "cpu", num_factors=1)
    assert cal == {}
