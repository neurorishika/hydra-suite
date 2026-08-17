import pandas as pd

from hydra_suite.core.post import dataset_export


def test_generate_dataset_reports_error_on_empty_selection(tmp_path, monkeypatch):
    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1], "State": ["active", "active"]}).to_csv(
        csv, index=False
    )

    class _Scorer:
        def __init__(self, params):
            pass

        def score_frame(self, frame_id, detection_data=None, tracking_data=None):
            pass

        def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
            return []  # nothing meets criteria

    monkeypatch.setattr(dataset_export, "FrameQualityScorer", _Scorer)
    monkeypatch.setattr(dataset_export, "export_dataset", lambda **k: "unused")

    result = dataset_export.generate_active_learning_dataset(
        video_path=str(tmp_path / "in.mp4"),
        csv_path=str(csv),
        detection_cache_path=None,
        output_dir=str(tmp_path / "out"),
        dataset_name="",
        class_name="object",
        params={},
        max_frames=5,
        diversity_window=30,
        include_context=True,
        probabilistic=False,
    )
    assert result["success"] is False
    assert "error" in result


def test_generate_dataset_success(tmp_path, monkeypatch):
    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1, 2], "State": ["active"] * 3}).to_csv(
        csv, index=False
    )

    class _Scorer:
        def __init__(self, params):
            pass

        def score_frame(self, frame_id, detection_data=None, tracking_data=None):
            pass

        def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
            return [0, 2]

    monkeypatch.setattr(dataset_export, "FrameQualityScorer", _Scorer)
    fake_manifest = {"round_dir": str(tmp_path / "dataset_dir"), "roots": []}
    monkeypatch.setattr(dataset_export, "export_dataset", lambda **k: fake_manifest)

    result = dataset_export.generate_active_learning_dataset(
        video_path=str(tmp_path / "in.mp4"),
        csv_path=str(csv),
        detection_cache_path=None,
        output_dir=str(tmp_path / "out"),
        dataset_name="",
        class_name="object",
        params={},
        max_frames=5,
        diversity_window=30,
        include_context=True,
        probabilistic=False,
    )
    assert result == {
        "success": True,
        "num_frames": 2,
        "dir": str(tmp_path / "dataset_dir"),
        "manifest": fake_manifest,
    }


def test_generate_dataset_cancelled_after_export(tmp_path, monkeypatch):
    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1, 2], "State": ["active"] * 3}).to_csv(
        csv, index=False
    )

    class _Scorer:
        def __init__(self, params):
            pass

        def score_frame(self, frame_id, detection_data=None, tracking_data=None):
            pass

        def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
            return [0, 2]

    monkeypatch.setattr(dataset_export, "FrameQualityScorer", _Scorer)

    exported = {"done": False}
    fake_manifest = {"round_dir": str(tmp_path / "dataset_dir"), "roots": []}

    def _fake_export_dataset(**k):
        exported["done"] = True
        return fake_manifest

    monkeypatch.setattr(dataset_export, "export_dataset", _fake_export_dataset)

    # should_stop flips True only once export_dataset has actually run, so
    # every pre-export check must pass and only the post-export guard trips.
    def _should_stop():
        return exported["done"]

    result = dataset_export.generate_active_learning_dataset(
        video_path=str(tmp_path / "in.mp4"),
        csv_path=str(csv),
        detection_cache_path=None,
        output_dir=str(tmp_path / "out"),
        dataset_name="",
        class_name="object",
        params={},
        max_frames=5,
        diversity_window=30,
        include_context=True,
        probabilistic=False,
        should_stop=_should_stop,
    )
    assert result["success"] is False
    assert result["cancelled"] is True
    assert result["dir"] == str(tmp_path / "dataset_dir")
    assert result["num_frames"] == 2
    assert result["manifest"] == fake_manifest
