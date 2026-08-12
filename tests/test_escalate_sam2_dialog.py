import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from hydra_suite.core.inference.sam2.checkpoints import (
    DEFAULT_VARIANT,
    available_variants,
)
from hydra_suite.detectkit.gui.dialogs.escalate_sam2_dialog import EscalateSam2Dialog
from hydra_suite.detectkit.gui.models import OBBSource

_app = QApplication.instance() or QApplication([])


def test_dialog_lists_variants_and_eligible_sources():
    sources = [
        OBBSource(name="a", level="obb"),
        OBBSource(name="b_seg", level="polygon"),
    ]  # already polygon -> disabled
    dlg = EscalateSam2Dialog(sources)
    assert dlg.selected_variant() == DEFAULT_VARIANT
    assert set(dlg._variant_combo_items()) == set(available_variants())
    # 'a' selectable, 'b_seg' disabled
    assert "a" in dlg.selectable_source_names()
    assert "b_seg" not in dlg.selectable_source_names()
