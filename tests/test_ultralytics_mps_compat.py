"""Regression coverage for the Ultralytics MPS target-assignment workaround."""

from __future__ import annotations

from hydra_suite.training.ultralytics_entrypoint import (
    install_mps_task_aligned_assigner_fallback,
)


class _Device:
    def __init__(self, name: str) -> None:
        self.type = name


class _Tensor:
    def __init__(self, device: str, value: str) -> None:
        self.device = _Device(device)
        self.value = value

    def cpu(self) -> "_Tensor":
        return _Tensor("cpu", self.value)

    def to(self, device: _Device) -> "_Tensor":
        return _Tensor(device.type, self.value)


class _Assigner:
    calls: list[tuple[str, ...]] = []

    def forward(self, *args: _Tensor, **kwargs: _Tensor) -> tuple[_Tensor, ...]:
        self.calls.append(tuple(item.device.type for item in (*args, *kwargs.values())))
        return args[:2]


def test_mps_target_assignment_runs_on_cpu_and_returns_to_mps() -> None:
    assert install_mps_task_aligned_assigner_fallback(_Assigner) is True
    result = _Assigner().forward(
        _Tensor("mps", "scores"), _Tensor("mps", "boxes"), mask=_Tensor("mps", "mask")
    )

    assert _Assigner.calls == [("cpu", "cpu", "cpu")]
    assert [item.device.type for item in result] == ["mps", "mps"]
    assert [item.value for item in result] == ["scores", "boxes"]


def test_cpu_target_assignment_does_not_copy_and_install_is_idempotent() -> None:
    assert install_mps_task_aligned_assigner_fallback(_Assigner) is False
    result = _Assigner().forward(_Tensor("cpu", "scores"), _Tensor("cpu", "boxes"))

    assert _Assigner.calls[-1] == ("cpu", "cpu")
    assert [item.device.type for item in result] == ["cpu", "cpu"]


class _Props:
    def __init__(self, uuid: str) -> None:
        self.uuid = uuid


class _Cuda:
    def __init__(self, uuids: list[str]) -> None:
        self.uuids = uuids
        self.initialised = False

    def is_available(self) -> bool:
        return True

    def init(self) -> None:
        self.initialised = True

    def device_count(self) -> int:
        return len(self.uuids)

    def get_device_properties(self, index: int) -> _Props:
        return _Props(self.uuids[index])


class _Torch:
    def __init__(self, uuids: list[str]) -> None:
        self.cuda = _Cuda(uuids)


def test_gpu_pin_is_latched_and_verified(monkeypatch) -> None:
    from hydra_suite.training.ultralytics_entrypoint import latch_cuda_visible_devices

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-43681f19-6815-bbbc")
    torch = _Torch(["43681f19-6815-bbbc-1e38-10214f54799d"])
    assert latch_cuda_visible_devices(torch) is True
    assert torch.cuda.initialised is True


def test_a_pin_that_torch_does_not_honour_fails_loudly(monkeypatch) -> None:
    import pytest

    from hydra_suite.training.ultralytics_entrypoint import latch_cuda_visible_devices

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa")
    with pytest.raises(RuntimeError, match="not honoured"):
        latch_cuda_visible_devices(_Torch(["bbbb-cccc"]))


def test_no_mask_means_no_cuda_initialisation(monkeypatch) -> None:
    from hydra_suite.training.ultralytics_entrypoint import latch_cuda_visible_devices

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    torch = _Torch(["bbbb"])
    assert latch_cuda_visible_devices(torch) is False
    assert torch.cuda.initialised is False


def test_a_pinned_gpu_that_torch_cannot_see_fails_loudly(monkeypatch) -> None:
    import pytest

    from hydra_suite.training.ultralytics_entrypoint import latch_cuda_visible_devices

    torch = _Torch([])
    torch.cuda.is_available = lambda: False
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa")
    with pytest.raises(RuntimeError, match="sees none"):
        latch_cuda_visible_devices(torch)
