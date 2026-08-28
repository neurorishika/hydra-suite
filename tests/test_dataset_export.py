import numpy as np
import pandas as pd

from hydra_suite.core.post import dataset_export


def test_frame_lookup_matches_per_frame_rows():
    """Verify groupby replacement semantics match per-frame filtering."""
    df = pd.DataFrame(
        {
            "FrameID": [1, 1, 2, 3, 3, 3],
            "X": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "Y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    # Verify the groupby-based approach matches the original per-frame scan
    grouped = {int(fid): sub for fid, sub in df.groupby("FrameID")}
    assert len(grouped[1]) == 2
    assert len(grouped[2]) == 1
    assert len(grouped[3]) == 3
    assert list(grouped[3]["X"]) == [40.0, 50.0, 60.0]

    # Also verify the semantics match the original filtering approach
    for frame_id in df["FrameID"].unique():
        frame_data_old = df[df["FrameID"] == frame_id]
        frame_data_new = grouped.get(int(frame_id))
        assert len(frame_data_old) == len(frame_data_new)
        assert (
            (
                frame_data_old.reset_index(drop=True)
                == frame_data_new.reset_index(drop=True)
            )
            .all()
            .all()
        )


class _StubScorer:
    """Minimal FrameQualityScorer stand-in returning a fixed selection."""

    def __init__(self, params, frame_shape=None):
        pass

    def score_frame(self, frame_id, detection_data=None, tracking_data=None):
        pass

    def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
        return [0, 1, 2]


def test_generate_dataset_reports_error_on_empty_selection(tmp_path, monkeypatch):
    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1], "State": ["active", "active"]}).to_csv(
        csv, index=False
    )

    class _Scorer:
        def __init__(self, params, frame_shape=None):
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
        def __init__(self, params, frame_shape=None):
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
        # No real video backs `in.mp4` here; dedup is exercised separately
        # below (test_dedup_*). Disable it so this test stays focused on the
        # selection/export/return-value contract.
        dedup_method="none",
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
        def __init__(self, params, frame_shape=None):
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
        # No real video backs `in.mp4` here; dedup is exercised separately
        # below (test_dedup_*). Disable it so this test stays focused on the
        # cancellation-after-export contract.
        dedup_method="none",
    )
    assert result["success"] is False
    assert result["cancelled"] is True
    assert result["dir"] == str(tmp_path / "dataset_dir")
    assert result["num_frames"] == 2
    assert result["manifest"] == fake_manifest


def test_dedup_runs_over_selected_frames_only(monkeypatch, tmp_path):
    """pHash over a whole video is prohibitive; only the picks get deduped."""
    import hydra_suite.core.post.dataset_export as de

    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1, 2], "State": ["active"] * 3}).to_csv(
        csv, index=False
    )

    seen = {}

    def fake_pool(source, cfg):
        seen["n_candidates"] = source.length()
        seen["method"] = cfg.dedup_method
        return [ref for ref in source][:1]

    monkeypatch.setattr(de, "build_candidate_pool", fake_pool)
    monkeypatch.setattr(de, "FrameQualityScorer", _StubScorer)

    fake_manifest = {"round_dir": str(tmp_path / "dataset_dir"), "roots": []}
    exported = {}

    def _fake_export_dataset(**kwargs):
        exported["frame_ids"] = kwargs["frame_ids"]
        return fake_manifest

    monkeypatch.setattr(de, "export_dataset", _fake_export_dataset)

    messages = []

    def _progress(pct, msg):
        messages.append(msg)

    result = de.generate_active_learning_dataset(
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
        progress=_progress,
    )

    assert seen["n_candidates"] == 3
    assert seen["method"] == "phash"
    # fake_pool kept only the first candidate -- confirm that survives to export.
    assert exported["frame_ids"] == [0]
    assert result["success"] is True
    assert result["num_frames"] == 1
    # The drop-count message must be accurate for the normal partial-drop
    # case, and must never claim frames were unreadable when they weren't.
    drop_messages = [m for m in messages if "dropped" in m]
    assert drop_messages == ["Perceptual dedup dropped 2 near-duplicate frames."]
    assert not any("unreadable" in m.lower() for m in messages)


def test_dedup_none_skips_the_pool_entirely(monkeypatch, tmp_path):
    import hydra_suite.core.post.dataset_export as de

    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1, 2], "State": ["active"] * 3}).to_csv(
        csv, index=False
    )

    def boom(*args, **kwargs):
        raise AssertionError("build_candidate_pool must not run when method='none'")

    monkeypatch.setattr(de, "build_candidate_pool", boom)
    monkeypatch.setattr(de, "FrameQualityScorer", _StubScorer)

    fake_manifest = {"round_dir": str(tmp_path / "dataset_dir"), "roots": []}
    exported = {}

    def _fake_export_dataset(**kwargs):
        exported["frame_ids"] = kwargs["frame_ids"]
        return fake_manifest

    monkeypatch.setattr(de, "export_dataset", _fake_export_dataset)

    result = de.generate_active_learning_dataset(
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
        dedup_method="none",
    )

    assert result["success"] is True
    # All 3 selected frames pass through untouched -- no dedup work happened.
    assert exported["frame_ids"] == [0, 1, 2]


def test_dedup_unreadable_video_returns_diagnostic_error(monkeypatch, tmp_path):
    """An unreadable video must be reported as unreadable, not as a dedup drop."""
    import hydra_suite.core.post.dataset_export as de

    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1, 2], "State": ["active"] * 3}).to_csv(
        csv, index=False
    )

    monkeypatch.setattr(de, "FrameQualityScorer", _StubScorer)

    def _boom_export(**kwargs):
        raise AssertionError("export_dataset must not run when dedup left nothing")

    monkeypatch.setattr(de, "export_dataset", _boom_export)

    # "in.mp4" is never created, so every read through the real
    # build_candidate_pool -> _SelectedFrameSource -> VideoFrameSource chain
    # fails, exercising the real (non-monkeypatched) dedup path end to end.
    result = de.generate_active_learning_dataset(
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
    assert "cancelled" not in result or not result.get("cancelled")
    assert "error" in result
    assert (
        "unreadable" in result["error"].lower()
        or "could not read" in result["error"].lower()
    )
    assert "near-duplicate" not in result["error"].lower()


def test_dedup_genuine_collapse_reports_duplicates_not_unreadable(
    monkeypatch, tmp_path
):
    """All-readable frames that all dedup away must not be blamed on I/O."""
    import hydra_suite.core.post.dataset_export as de

    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1, 2], "State": ["active"] * 3}).to_csv(
        csv, index=False
    )

    monkeypatch.setattr(de, "FrameQualityScorer", _StubScorer)

    def _fake_pool_collapses_everything(source, cfg):
        # Simulates every candidate being read successfully but judged a
        # near-duplicate of an earlier one -- never touches source.read(),
        # so unreadable_count stays 0.
        return []

    monkeypatch.setattr(de, "build_candidate_pool", _fake_pool_collapses_everything)

    def _boom_export(**kwargs):
        raise AssertionError("export_dataset must not run when dedup left nothing")

    monkeypatch.setattr(de, "export_dataset", _boom_export)

    result = de.generate_active_learning_dataset(
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
    assert "near-duplicate" in result["error"].lower()
    assert "unreadable" not in result["error"].lower()
    assert "could not read" not in result["error"].lower()


class _FakeVideoFrameSourceForMix:
    """Stand-in for VideoFrameSource: per-frame-id canned reads.

    Used to exercise the REAL `_SelectedFrameSource.read()` counting logic
    (and the real `build_candidate_pool`) end to end, rather than
    monkeypatching `build_candidate_pool` itself -- the mixed-cause scenario
    only exists inside that real counting path.
    """

    def __init__(self, video_path, stride=1):
        self._frames = {
            0: np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8),
            1: np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8),
            2: None,  # unreadable
        }

    def read(self, ref):
        return self._frames.get(ref.frame_id)

    def length(self):
        return len(self._frames)

    def __iter__(self):
        return iter(())


def test_dedup_mixed_unreadable_and_duplicate_reports_both_counts(
    monkeypatch, tmp_path
):
    """Unreadable frames and genuine duplicates must be reported separately."""
    import hydra_suite.core.post.dataset_export as de

    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1, 2], "State": ["active"] * 3}).to_csv(
        csv, index=False
    )

    class _Scorer(_StubScorer):
        def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
            # frame 0 and frame 1 are identical (RandomState(0) reused) --
            # a genuine duplicate. Frame 2 is unreadable. Frame 0 survives.
            return [0, 1, 2]

    monkeypatch.setattr(de, "FrameQualityScorer", _Scorer)
    # Real build_candidate_pool + real _SelectedFrameSource; only the inner
    # VideoFrameSource is swapped for one with canned per-id reads.
    monkeypatch.setattr(de, "VideoFrameSource", _FakeVideoFrameSourceForMix)

    fake_manifest = {"round_dir": str(tmp_path / "dataset_dir"), "roots": []}
    exported = {}

    def _fake_export_dataset(**kwargs):
        exported["frame_ids"] = kwargs["frame_ids"]
        return fake_manifest

    monkeypatch.setattr(de, "export_dataset", _fake_export_dataset)

    messages = []

    def _progress(pct, msg):
        messages.append(msg)

    result = de.generate_active_learning_dataset(
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
        progress=_progress,
    )

    assert result["success"] is True
    # Frame 0 survives (unique); frame 1 dropped as a duplicate of frame 0;
    # frame 2 dropped because it could not be read.
    assert exported["frame_ids"] == [0]

    drop_messages = [m for m in messages if "dropped" in m or "could not be read" in m]
    assert len(drop_messages) == 1
    message = drop_messages[0]
    assert "dropped 1 near-duplicate frame" in message
    assert "1 frame could not be read" in message
    # The unreadable frame must not be folded into the duplicate count.
    assert "dropped 2" not in message


def test_dedup_all_readable_partial_drop_message_unchanged(monkeypatch, tmp_path):
    """With nothing unreadable, the message must read exactly as before --
    no stray '0 frames could not be read' clause."""
    import hydra_suite.core.post.dataset_export as de

    csv = tmp_path / "track.csv"
    pd.DataFrame({"FrameID": [0, 1, 2], "State": ["active"] * 3}).to_csv(
        csv, index=False
    )

    monkeypatch.setattr(de, "FrameQualityScorer", _StubScorer)

    seen = {}

    def fake_pool(source, cfg):
        seen["n_candidates"] = source.length()
        return [ref for ref in source][:1]

    monkeypatch.setattr(de, "build_candidate_pool", fake_pool)

    fake_manifest = {"round_dir": str(tmp_path / "dataset_dir"), "roots": []}
    monkeypatch.setattr(de, "export_dataset", lambda **kwargs: fake_manifest)

    messages = []

    def _progress(pct, msg):
        messages.append(msg)

    result = de.generate_active_learning_dataset(
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
        progress=_progress,
    )

    assert result["success"] is True
    drop_messages = [m for m in messages if "dropped" in m]
    assert drop_messages == ["Perceptual dedup dropped 2 near-duplicate frames."]
    assert not any("could not be read" in m for m in messages)
