"""HYDRA Suite Launcher — central hub for all kit applications.

Provides a GUI launcher with icons and descriptions for every app in the
suite, plus controls for the ``HYDRA_DATA_DIR`` and ``HYDRA_CONFIG_DIR``
environment variable overrides used by *platformdirs*.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

_DARK_STYLESHEET = """
QWidget {
    background-color: #1e1e1e;
    color: #cccccc;
    font-family: "SF Pro Text", "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 11px;
}
QGroupBox {
    background-color: transparent;
    border: none;
    border-top: 1px solid #3e3e42;
    border-radius: 0px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: 600;
    color: #9cdcfe;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    background-color: #1e1e1e;
    color: #9cdcfe;
}
QPushButton {
    background-color: #0e639c;
    color: #ffffff;
    border: none;
    border-radius: 3px;
    padding: 5px 14px;
    font-weight: 500;
    min-height: 22px;
}
QPushButton:hover { background-color: #1177bb; }
QPushButton:pressed { background-color: #0d5a8f; }
QPushButton:disabled { background-color: #3e3e42; color: #777777; }
QLineEdit {
    background-color: transparent;
    color: #cccccc;
    border: none;
    border-bottom: 1px solid #3e3e42;
    border-radius: 0px;
    padding: 4px 2px;
    min-height: 22px;
}
QLineEdit[readOnly="true"] {
    background-color: transparent;
    color: #aaaaaa;
    border-bottom: 1px solid #3e3e42;
}
QLineEdit[readOnly="true"]:hover { border-bottom-color: #3e3e42; }
QLabel {
    color: #cccccc;
    background-color: transparent;
}
QToolButton {
    background-color: transparent;
    border: none;
    border-radius: 12px;
    padding: 8px;
}
QToolButton:hover {
    background-color: #2a2d2e;
}
QToolButton:pressed {
    background-color: #094771;
}
QScrollBar:vertical {
    background-color: transparent; width: 6px; border-radius: 3px; margin: 0px;
}
QScrollBar::handle:vertical {
    background-color: #3e3e42; border-radius: 3px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background-color: #007acc; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
"""


# ---------------------------------------------------------------------------
# App catalogue
# ---------------------------------------------------------------------------

APP_CATALOG = [
    {
        "name": "TrackerKit",
        "icon": "trackerkit.svg",
        "entry": "hydra_suite.trackerkit.app:main",
        "description": (
            "Real-time multi-animal tracker with Kalman filtering, "
            "Hungarian assignment, and post-processing."
        ),
    },
    {
        "name": "PoseKit",
        "icon": "posekit.svg",
        "entry": "hydra_suite.posekit.gui.main:main",
        "description": (
            "Pose-labeling tool for creating and editing keypoint "
            "annotations on image datasets."
        ),
    },
    {
        "name": "DetectKit",
        "icon": "detectkit.svg",
        "entry": "hydra_suite.detectkit.app:main",
        "description": (
            "Detection dataset builder for training and evaluating "
            "YOLO-based object detectors."
        ),
    },
    {
        "name": "ClassKit",
        "icon": "classkit.svg",
        "entry": "hydra_suite.classkit.app:main",
        "description": (
            "Classification & embedding toolkit for identity models, "
            "label management, and training."
        ),
    },
    {
        "name": "RefineKit",
        "icon": "refinekit.svg",
        "entry": "hydra_suite.refinekit.app:main",
        "description": (
            "Interactive identity proofreading tool for reviewing "
            "and correcting tracking results."
        ),
    },
    {
        "name": "FilterKit",
        "icon": "filterkit.svg",
        "entry": "hydra_suite.filterkit.app:main",
        "description": (
            "Dataset filtering and curation tool for managing "
            "image datasets and annotations."
        ),
    },
]


# ---------------------------------------------------------------------------
# Icon helper
# ---------------------------------------------------------------------------


def _load_icon(name: str, size: int = 64) -> QIcon:
    """Load a brand SVG as a QIcon at the given logical size."""
    try:
        from hydra_suite.paths import get_brand_icon_bytes

        data = get_brand_icon_bytes(name)
        if data is None:
            return QIcon()

        from PySide6.QtCore import QByteArray
        from PySide6.QtGui import QImage, QPainter
        from PySide6.QtSvg import QSvgRenderer

        renderer = QSvgRenderer(QByteArray(data))
        image = QImage(QSize(size, size), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        return QIcon(QPixmap.fromImage(image))
    except Exception:
        return QIcon()


# ---------------------------------------------------------------------------
# App card widget
# ---------------------------------------------------------------------------


class AppCard(QWidget):
    """A clickable card — icon-only (name is embedded in the SVG logo)."""

    _ICON_SIZE_INITIAL = 160
    _ICON_FRACTION = 0.85  # icon occupies this fraction of the card width
    _ICON_MIN = 80
    _ICON_MAX = 260

    def __init__(self, info: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._info = info

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._btn = QToolButton()
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._btn.setToolTip(info["name"])
        # Load at 512 px so Qt can scale down to any size cleanly
        icon = _load_icon(info["icon"], size=512)
        self._btn.setIcon(icon)
        self._btn.setIconSize(QSize(self._ICON_SIZE_INITIAL, self._ICON_SIZE_INITIAL))
        self._btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._btn.setMinimumSize(120, 110)
        self._btn.clicked.connect(self._launch)
        layout.addWidget(self._btn)

        desc = QLabel(info["description"])
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setStyleSheet("color: #6a9eb5; padding: 0px 4px; font-size: 10px;")
        layout.addWidget(desc)

    def _launch(self) -> None:
        """Launch the app in a new process."""
        entry = self._info["entry"]  # e.g. "hydra_suite.trackerkit.app:main"
        module, _func = entry.rsplit(":", 1)
        env = os.environ.copy()
        try:
            subprocess.Popen(
                [sys.executable, "-c", f"from {module} import {_func}; {_func}()"],
                start_new_session=True,
                env=env,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                f"Failed to launch {self._info['name']}",
                str(exc),
            )

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        """Scale the tool icon proportionally to the card width on resize."""
        side = max(
            self._ICON_MIN,
            min(int(self.width() * self._ICON_FRACTION), self._ICON_MAX),
        )
        self._btn.setIconSize(QSize(side, side))
        super().resizeEvent(event)


# ---------------------------------------------------------------------------
# Directory settings panel
# ---------------------------------------------------------------------------


class DirSettingsPanel(QGroupBox):
    """Panel for viewing/editing HYDRA_DATA_DIR, HYDRA_CONFIG_DIR, and HYDRA_PROJECTS_DIR."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Directories", parent)

        form = QVBoxLayout(self)

        # --- Data directory ---
        form.addWidget(QLabel("Data directory (models, training runs):"))
        row_data = QHBoxLayout()
        self._data_edit = QLineEdit()
        self._data_edit.setReadOnly(True)
        self._data_edit.setPlaceholderText(self._default_data_dir())
        self._data_edit.setText(os.environ.get("HYDRA_DATA_DIR", ""))
        self._data_edit.setToolTip(
            "Use Browse to select a custom HYDRA_DATA_DIR override."
        )
        row_data.addWidget(self._data_edit)

        browse_data = QPushButton("Browse…")
        browse_data.clicked.connect(lambda: self._browse(self._data_edit))
        row_data.addWidget(browse_data)

        clear_data = QPushButton("Clear")
        clear_data.setToolTip("Remove the override and use the platform default")
        clear_data.clicked.connect(lambda: self._data_edit.clear())
        row_data.addWidget(clear_data)

        open_data = QPushButton("Open")
        open_data.setToolTip("Open the resolved data directory in the file manager")
        open_data.clicked.connect(lambda: self._open_dir(self._resolved_data_dir()))
        row_data.addWidget(open_data)

        form.addLayout(row_data)

        # --- Config directory ---
        form.addWidget(QLabel("Config directory (presets, skeletons):"))
        row_cfg = QHBoxLayout()
        self._config_edit = QLineEdit()
        self._config_edit.setReadOnly(True)
        self._config_edit.setPlaceholderText(self._default_config_dir())
        self._config_edit.setText(os.environ.get("HYDRA_CONFIG_DIR", ""))
        self._config_edit.setToolTip(
            "Use Browse to select a custom HYDRA_CONFIG_DIR override."
        )
        row_cfg.addWidget(self._config_edit)

        browse_cfg = QPushButton("Browse…")
        browse_cfg.clicked.connect(lambda: self._browse(self._config_edit))
        row_cfg.addWidget(browse_cfg)

        clear_cfg = QPushButton("Clear")
        clear_cfg.setToolTip("Remove the override and use the platform default")
        clear_cfg.clicked.connect(lambda: self._config_edit.clear())
        row_cfg.addWidget(clear_cfg)

        open_cfg = QPushButton("Open")
        open_cfg.setToolTip("Open the resolved config directory in the file manager")
        open_cfg.clicked.connect(lambda: self._open_dir(self._resolved_config_dir()))
        row_cfg.addWidget(open_cfg)

        form.addLayout(row_cfg)

        # --- Projects directory ---
        form.addWidget(QLabel("Projects directory (default browse location):"))
        row_proj = QHBoxLayout()
        self._projects_edit = QLineEdit()
        self._projects_edit.setReadOnly(True)
        self._projects_edit.setPlaceholderText(self._default_projects_dir())
        self._projects_edit.setText(os.environ.get("HYDRA_PROJECTS_DIR", ""))
        self._projects_edit.setToolTip(
            "Use Browse to set the default folder for opening/browsing projects."
        )
        row_proj.addWidget(self._projects_edit)

        browse_proj = QPushButton("Browse\u2026")
        browse_proj.clicked.connect(lambda: self._browse(self._projects_edit))
        row_proj.addWidget(browse_proj)

        clear_proj = QPushButton("Clear")
        clear_proj.setToolTip("Remove the override and use the platform default")
        clear_proj.clicked.connect(lambda: self._projects_edit.clear())
        row_proj.addWidget(clear_proj)

        open_proj = QPushButton("Open")
        open_proj.setToolTip("Open the projects directory in the file manager")
        open_proj.clicked.connect(lambda: self._open_dir(self._resolved_projects_dir()))
        row_proj.addWidget(open_proj)

        form.addLayout(row_proj)

        # --- Apply button ---
        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip(
            "Set the environment variables for this session. "
            "Apps launched from this launcher will inherit them."
        )
        apply_btn.clicked.connect(self._apply)
        form.addWidget(apply_btn, alignment=Qt.AlignmentFlag.AlignRight)

    # -- helpers --

    @staticmethod
    def _default_data_dir() -> str:
        from platformdirs import user_data_dir

        return user_data_dir("hydra-suite", "Rishika Mohanta")

    @staticmethod
    def _default_config_dir() -> str:
        from platformdirs import user_config_dir

        return user_config_dir("hydra-suite", "Rishika Mohanta")

    @staticmethod
    def _default_projects_dir() -> str:
        from platformdirs import user_documents_dir

        return str(Path(user_documents_dir()) / "hydra-projects")

    def _resolved_data_dir(self) -> str:
        val = self._data_edit.text().strip()
        return val if val else self._default_data_dir()

    def _resolved_config_dir(self) -> str:
        val = self._config_edit.text().strip()
        return val if val else self._default_config_dir()

    def _resolved_projects_dir(self) -> str:
        val = self._projects_edit.text().strip()
        return val if val else self._default_projects_dir()

    def _browse(self, line_edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            line_edit.setText(d)

    def _apply(self) -> None:
        data = self._data_edit.text().strip()
        cfg = self._config_edit.text().strip()
        proj = self._projects_edit.text().strip()
        if data:
            os.environ["HYDRA_DATA_DIR"] = data
        elif "HYDRA_DATA_DIR" in os.environ:
            del os.environ["HYDRA_DATA_DIR"]
        if cfg:
            os.environ["HYDRA_CONFIG_DIR"] = cfg
        elif "HYDRA_CONFIG_DIR" in os.environ:
            del os.environ["HYDRA_CONFIG_DIR"]
        if proj:
            os.environ["HYDRA_PROJECTS_DIR"] = proj
        elif "HYDRA_PROJECTS_DIR" in os.environ:
            del os.environ["HYDRA_PROJECTS_DIR"]

        QMessageBox.information(
            self,
            "Directories updated",
            f"Data directory:     {self._resolved_data_dir()}\n"
            f"Config directory:   {self._resolved_config_dir()}\n"
            f"Projects directory: {self._resolved_projects_dir()}\n\n"
            "Apps launched from this session will use these paths.",
        )

    @staticmethod
    def _open_dir(path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        elif sys.platform == "win32":
            os.startfile(str(p))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(p)])


# ---------------------------------------------------------------------------
# Main launcher window
# ---------------------------------------------------------------------------


class LauncherWindow(QWidget):
    """HYDRA Suite launcher hub."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HYDRA Suite")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(_DARK_STYLESHEET)

        root = QVBoxLayout(self)
        root.setSpacing(20)
        root.setContentsMargins(32, 28, 32, 24)

        # -- Header: logo + title, flat on background --
        header = QHBoxLayout()
        header.setSpacing(16)
        header.setContentsMargins(0, 0, 0, 0)

        logo_label = QLabel()
        logo_label.setStyleSheet("background: transparent;")
        icon = _load_icon("hydra.svg", size=72)
        if not icon.isNull():
            logo_label.setPixmap(icon.pixmap(QSize(56, 56)))
        header.addWidget(logo_label)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)

        title = QLabel("HYDRA Suite")
        title_font = QFont()
        title_font.setPointSize(26)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #ffffff; background: transparent;")
        text_col.addWidget(title)

        subtitle = QLabel(
            "Holistic YOLO-based Detection Recognition and Analysis Suite"
        )
        subtitle.setStyleSheet(
            "color: #9cdcfe; font-size: 11px; background: transparent;"
        )
        text_col.addWidget(subtitle)

        header.addLayout(text_col)
        header.addStretch()
        root.addLayout(header)

        # -- App grid (2 rows × 3 cols) --
        grid = QGridLayout()
        grid.setSpacing(16)
        cols = 3
        for idx, info in enumerate(APP_CATALOG):
            card = AppCard(info)
            grid.addWidget(card, idx // cols, idx % cols)
        root.addLayout(grid, stretch=1)

        # -- Directory settings --
        self._dirs = DirSettingsPanel()
        root.addWidget(self._dirs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the HYDRA Suite launcher hub."""
    app = QApplication(sys.argv)
    app.setApplicationName("HYDRA Suite")
    app.setApplicationDisplayName("HYDRA Suite")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("NeuroRishika")
    app.setDesktopFileName("hydra-suite")

    try:
        from hydra_suite.paths import get_brand_qicon

        icon = get_brand_qicon("hydra.svg")
        if icon and not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass

    window = LauncherWindow()
    try:
        window.setWindowIcon(app.windowIcon())
    except Exception:
        pass
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
