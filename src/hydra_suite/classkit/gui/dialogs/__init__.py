"""classkit GUI dialogs package."""

from importlib import import_module

_DIALOG_MODULES = {
    "AddSourceDialog": "add_source",
    "SourceManagerDialog": "source_manager",
    "ClassEditorDialog": "class_editor",
    "ShortcutEditorDialog": "shortcut_editor",
    "NewProjectDialog": "new_project",
    "EmbeddingDialog": "embedding",
    "ClusterDialog": "cluster",
    "ClassKitTrainingDialog": "training",
    "ExportDialog": "export",
    "ModelHistoryDialog": "model_history",
    "AprilTagAutoLabelDialog": "apriltag_autolabel",
    "MachineLabelingDialog": "machine_labeling",
    "ReviewRelabelDialog": "review_relabel",
}

__all__ = [
    "AddSourceDialog",
    "SourceManagerDialog",
    "ClassEditorDialog",
    "ShortcutEditorDialog",
    "NewProjectDialog",
    "EmbeddingDialog",
    "ClusterDialog",
    "ClassKitTrainingDialog",
    "ExportDialog",
    "ModelHistoryDialog",
    "AprilTagAutoLabelDialog",
    "MachineLabelingDialog",
    "ReviewRelabelDialog",
]


def __getattr__(name: str):
    module_name = _DIALOG_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
