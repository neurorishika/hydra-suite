# src/hydra_suite/widgets/dialogs.py
"""BaseDialog — standard QDialog base class for all kit dialogs."""

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

HYDRA_DIALOG_TEXT_COLOR = "#ffffff"
HYDRA_DIALOG_MUTED_TEXT_COLOR = "#ffffff"

HYDRA_DIALOG_STYLE = f"""
QDialog {{
    background-color: #1e1e1e;
    color: {HYDRA_DIALOG_TEXT_COLOR};
}}
QLabel {{
    color: {HYDRA_DIALOG_TEXT_COLOR};
    background-color: transparent;
}}
QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {{
    background-color: #3c3c3c;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 4px 8px;
    color: {HYDRA_DIALOG_TEXT_COLOR};
    selection-background-color: #094771;
    min-height: 22px;
}}
QPlainTextEdit,
QTextEdit,
QListWidget,
QTableWidget {{
    background-color: #252526;
    alternate-background-color: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 4px;
    color: {HYDRA_DIALOG_TEXT_COLOR};
    selection-background-color: #094771;
}}
QLineEdit:hover,
QComboBox:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QPlainTextEdit:hover,
QTextEdit:hover,
QListWidget:hover,
QTableWidget:hover {{
    border-color: #0e639c;
}}
QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QPlainTextEdit:focus,
QTextEdit:focus,
QListWidget:focus,
QTableWidget:focus {{
    border-color: #007acc;
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #3e3e42;
    background-color: #4a4a4a;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}}
QComboBox::drop-down:hover {{
    background-color: #5a5a5a;
}}
QComboBox QAbstractItemView {{
    background-color: #252526;
    border: 1px solid #3e3e42;
    selection-background-color: #094771;
    selection-color: #ffffff;
    color: {HYDRA_DIALOG_TEXT_COLOR};
    outline: none;
}}
QComboBox QAbstractItemView::item {{
    padding: 6px 10px;
    min-height: 22px;
}}
QComboBox QAbstractItemView::item:hover {{
    background-color: #2a2d2e;
}}
QComboBox QAbstractItemView::item:selected {{
    background-color: #094771;
    color: #ffffff;
}}
QSpinBox::up-button,
QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #3e3e42;
    background-color: #4a4a4a;
    border-top-right-radius: 4px;
}}
QSpinBox::down-button,
QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid #3e3e42;
    background-color: #4a4a4a;
    border-bottom-right-radius: 4px;
}}
QSpinBox::up-button:hover,
QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover,
QDoubleSpinBox::down-button:hover {{
    background-color: #0e639c;
}}
QGroupBox {{
    font-weight: 600;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    margin-top: 8px;
    padding: 6px;
    background-color: #252526;
    color: {HYDRA_DIALOG_TEXT_COLOR};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 1px 6px;
    background-color: #1e1e1e;
    color: #9cdcfe;
    border-radius: 3px;
}}
QPushButton {{
    background-color: #0e639c;
    border: none;
    color: #ffffff;
    padding: 5px 12px;
    border-radius: 4px;
    min-height: 22px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: #1177bb;
}}
QPushButton:pressed {{
    background-color: #0d5a8f;
}}
QPushButton:checked {{
    background-color: #094771;
    border: 1px solid #007acc;
}}
QPushButton:disabled {{
    background-color: #3e3e42;
    color: #d8d8d8;
}}
QCheckBox,
QRadioButton {{
    color: {HYDRA_DIALOG_TEXT_COLOR};
    spacing: 8px;
}}
QCheckBox::indicator,
QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid #3e3e42;
    background-color: #3c3c3c;
}}
QCheckBox::indicator {{
    border-radius: 3px;
}}
QRadioButton::indicator {{
    border-radius: 7px;
}}
QCheckBox::indicator:checked,
QRadioButton::indicator:checked {{
    background-color: #0e639c;
    border-color: #007acc;
}}
QCheckBox::indicator:hover,
QRadioButton::indicator:hover {{
    border-color: #007acc;
}}
QListWidget::item,
QTableWidget::item {{
    padding: 4px 8px;
}}
QListWidget::item:alternate,
QTableWidget::item:alternate {{
    background-color: #2d2d30;
}}
QListWidget::item:selected,
QTableWidget::item:selected {{
    background-color: #094771;
    color: #ffffff;
}}
QListWidget::item:hover:!selected,
QTableWidget::item:hover:!selected {{
    background-color: #2a2d2e;
}}
QHeaderView::section {{
    background-color: #2d2d30;
    color: {HYDRA_DIALOG_TEXT_COLOR};
    border: none;
    border-right: 1px solid #3e3e42;
    border-bottom: 1px solid #3e3e42;
    padding: 4px 8px;
    font-weight: 600;
}}
QProgressBar {{
    border: 1px solid #3e3e42;
    border-radius: 4px;
    text-align: center;
    background-color: #252526;
    color: {HYDRA_DIALOG_TEXT_COLOR};
    font-size: 11px;
}}
QProgressBar::chunk {{
    background-color: #0e639c;
    border-radius: 3px;
}}
QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    color: #3e3e42;
}}
QScrollBar:vertical {{
    background-color: #252526;
    width: 10px;
    border-radius: 5px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background-color: #5a5a5a;
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: #007acc;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background-color: #252526;
    height: 10px;
    border-radius: 5px;
    margin: 0px;
}}
QScrollBar::handle:horizontal {{
    background-color: #5a5a5a;
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background-color: #007acc;
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
"""


class BaseDialog(QDialog):
    """Base class for all kit dialogs.

    Subclasses call ``add_content(widget)`` to insert their UI above the
    button box.  The button box is always created automatically in the
    correct position.

    Parameters
    ----------
    title:
        Window title shown in the title bar.
    parent:
        Parent widget (passed to QDialog).
    buttons:
        ``QDialogButtonBox.StandardButtons`` flags.  Defaults to
        ``Ok | Cancel``.
    apply_dark_style:
        Whether to apply the shared dark stylesheet.  Default ``True``.
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        buttons: QDialogButtonBox.StandardButtons = (
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        ),
        apply_dark_style: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        if apply_dark_style:
            self.setStyleSheet(HYDRA_DIALOG_STYLE)

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(16, 16, 16, 16)
        self._root_layout.setSpacing(16)
        self._content_layout = QVBoxLayout()
        self._content_layout.setSpacing(12)
        self._root_layout.addLayout(self._content_layout)

        self._buttons = QDialogButtonBox(buttons)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._root_layout.addWidget(self._buttons)

    def add_content(self, widget: QWidget) -> None:
        """Insert *widget* above the button box."""
        self._content_layout.addWidget(widget)
