"""FilterKit application entry point."""

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from hydra_suite.filterkit.gui.main_window import FilterKitWindow


def parse_args():
    ap = argparse.ArgumentParser(description="FilterKit dataset subsampling UI")
    ap.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Optional dataset root (flat folder, COCO JSON, YOLO OBB/detect, or class-folder dataset)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("FilterKit")
    app.setApplicationDisplayName("FilterKit")
    app.setOrganizationName("NeuroRishika")

    try:
        from hydra_suite.paths import get_brand_qicon

        icon = get_brand_qicon("filterkit.svg")
        if icon and not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass

    window = FilterKitWindow()
    if args.dataset:
        window.load_dataset_root(Path(args.dataset), show_errors=True)
    window.show()
    sys.exit(app.exec())
