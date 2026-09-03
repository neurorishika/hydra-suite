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
