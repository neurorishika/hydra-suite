"""GUI dialogs."""

from .cnn_identity_import_dialog import CNNIdentityImportDialog
from .headtail_import_dialog import HeadTailImportDialog, describe_headtail_candidate
from .train_yolo_dialog import TrainYoloDialog

__all__ = [
    "CNNIdentityImportDialog",
    "HeadTailImportDialog",
    "TrainYoloDialog",
    "describe_headtail_candidate",
]
