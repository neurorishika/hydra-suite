"""DetectKit dialog exports."""

from .evaluation_dialog import EvaluationDialog
from .history_dialog import HistoryDialog
from .new_project import NewProjectDialog
from .source_manager import SourceManagerDialog
from .source_validation import DetectKitSourceValidationDialog
from .training_dialog import TrainingDialog

__all__ = [
    "EvaluationDialog",
    "HistoryDialog",
    "NewProjectDialog",
    "SourceManagerDialog",
    "DetectKitSourceValidationDialog",
    "TrainingDialog",
]
