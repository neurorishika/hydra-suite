"""Torch device picker shared by every kit's inference dialogs."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox


class DeviceCombo(QComboBox):
    """Auto / CUDA / MPS / CPU, listing only the accelerators this host has."""

    def __init__(self, current: str = "auto", parent=None) -> None:
        super().__init__(parent)
        from hydra_suite.core.inference.torch_device import device_choices

        for label, value in device_choices():
            self.addItem(label, value)
        self.set_device(current)
        self.setToolTip(
            "Where the model runs. Auto picks CUDA, then Apple MPS, then CPU. "
            "CPU works everywhere but is much slower."
        )

    def device(self) -> str:
        return str(self.currentData() or "auto")

    def set_device(self, value: str) -> None:
        index = self.findData(str(value or "auto"))
        self.setCurrentIndex(index if index >= 0 else 0)
