"""DetectKit application entry point."""

from __future__ import annotations

import logging
import sys


def _launch_gui(argv: list[str] | None = None) -> int:
    """Launch the DetectKit GUI, importing Qt only on the GUI path."""

    from PySide6.QtWidgets import QApplication

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    qt_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication(qt_argv)
    app.setApplicationName("DetectKit")
    app.setApplicationDisplayName("DetectKit")
    app.setOrganizationName("NeuroRishika")
    app.setDesktopFileName("detectkit")

    try:
        from hydra_suite.paths import get_brand_qicon

        icon = get_brand_qicon("detectkit.svg")
        if icon and not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass

    from hydra_suite.detectkit.gui.main_window import MainWindow

    window = MainWindow()
    window.resize(1600, 1000)
    window.showMaximized()
    return int(app.exec())


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI or dispatch ``detectkit train`` to the headless CLI."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "train":
        from hydra_suite.detectkit.cli import main as training_main

        return training_main(args[1:])
    return _launch_gui(args)


if __name__ == "__main__":
    raise SystemExit(main())
