import json
from pathlib import Path

import pytest

from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs import staged_review as sr


def _source(tmp_path, labels: dict[str, str], level="obb", classes="object\n"):
    root = tmp_path / "src"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for rel, text in labels.items():
        p = root / "labels" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        (root / "images" / rel).with_suffix(".png").parent.mkdir(
            parents=True, exist_ok=True
        )
    (root / "classes.txt").write_text(classes)
    return OBBSource(path=str(root), name="src", level=level)


def _staging(tmp_path, labels: dict[str, str], classes="object\n"):
    # MUST live under artifacts/pending_escalations/: with project_dir=None,
    # `_is_safe_to_delete` (sam2_escalation.py:35) accepts a path only if its
    # parent is "pending_escalations" and its grandparent is "artifacts".
    # A staging dir anywhere else is silently NOT deleted, and
    # test_finish_review_removes_staging would fail with nothing to show why.
    root = tmp_path / "artifacts" / "pending_escalations" / "staging"
    (root / "labels").mkdir(parents=True)
    for rel, text in labels.items():
        p = root / "labels" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (root / "classes.txt").write_text(classes)
    return root


def test_decisions_round_trip(tmp_path):
    staged = _staging(tmp_path, {"a.txt": "", "b.txt": ""})

    sr.write_decisions(staged, {"a.txt": sr.ACCEPTED_ADD_NEW})

    assert sr.read_decisions(staged) == {"a.txt": sr.ACCEPTED_ADD_NEW}


def test_decisions_read_as_empty_when_absent(tmp_path):
    assert sr.read_decisions(_staging(tmp_path, {})) == {}


def test_staged_frames_are_sorted_posix_relative_paths(tmp_path):
    staged = _staging(tmp_path, {"b.txt": "", "sub/a.txt": ""})

    assert sr.staged_frames(staged) == ["b.txt", "sub/a.txt"]


def test_review_progress_counts_decided_over_total(tmp_path):
    staged = _staging(tmp_path, {"a.txt": "", "b.txt": "", "c.txt": ""})
    sr.write_decisions(staged, {"a.txt": sr.REJECTED})

    assert sr.review_progress(staged) == (1, 3)


def test_snapshot_captures_labels_level_and_classes(tmp_path):
    source = _source(tmp_path, {"a.txt": "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"})
    staged = _staging(tmp_path, {"a.txt": ""})

    sr.ensure_snapshot(source, staged)

    assert (staged / "labels_before" / "a.txt").read_text().startswith("0 ")
    state = json.loads((staged / "state_before.json").read_text())
    assert state["level"] == "obb"
    assert state["classes_txt"] == "object\n"


def test_snapshot_is_taken_once_and_never_overwritten(tmp_path):
    source = _source(tmp_path, {"a.txt": "original\n"})
    staged = _staging(tmp_path, {"a.txt": ""})

    sr.ensure_snapshot(source, staged)
    (Path(source.path) / "labels" / "a.txt").write_text("changed\n")
    sr.ensure_snapshot(source, staged)

    assert (staged / "labels_before" / "a.txt").read_text() == "original\n"


def test_revert_restores_labels_level_and_classes_and_clears_decisions(tmp_path):
    source = _source(tmp_path, {"a.txt": "original\n"}, level="obb")
    staged = _staging(tmp_path, {"a.txt": ""})
    sr.ensure_snapshot(source, staged)

    (Path(source.path) / "labels" / "a.txt").write_text("accepted\n")
    (Path(source.path) / "labels" / "new.txt").write_text("appeared\n")
    (Path(source.path) / "classes.txt").write_text("object\nant\n")
    source.level = "polygon"
    sr.write_decisions(staged, {"a.txt": sr.ACCEPTED_OVERWRITE})

    sr.revert_review(source, staged)

    assert (Path(source.path) / "labels" / "a.txt").read_text() == "original\n"
    assert not (Path(source.path) / "labels" / "new.txt").exists()
    assert (Path(source.path) / "classes.txt").read_text() == "object\n"
    assert source.level == "obb"
    assert sr.read_decisions(staged) == {}


def test_revert_clears_classes_appended_to_a_source_that_had_none(tmp_path):
    """Restoring the class list must mean restoring it, including to empty."""
    source = _source(tmp_path, {"a.txt": "original\n"}, classes="")
    staged = _staging(tmp_path, {"a.txt": ""})
    sr.ensure_snapshot(source, staged)
    (Path(source.path) / "classes.txt").write_text("larva\n")

    sr.revert_review(source, staged)

    assert (Path(source.path) / "classes.txt").read_text() == ""


def test_revert_without_a_snapshot_is_refused(tmp_path):
    source = _source(tmp_path, {"a.txt": "original\n"})
    staged = _staging(tmp_path, {"a.txt": ""})

    with pytest.raises(RuntimeError, match="no snapshot"):
        sr.revert_review(source, staged)


def test_matching_class_names_map_by_name(tmp_path):
    source = _source(tmp_path, {}, classes="ant\nbeetle\n")
    staged = _staging(tmp_path, {}, classes="beetle\n")

    assert sr.resolve_staged_class_ids(source, staged) == {0: 1}


def test_an_unknown_staged_class_is_appended_to_the_source(tmp_path):
    source = _source(tmp_path, {}, classes="ant\n")
    staged = _staging(tmp_path, {}, classes="ant\nlarva\n")

    mapping = sr.resolve_staged_class_ids(source, staged)

    assert mapping == {0: 0, 1: 1}
    assert (Path(source.path) / "classes.txt").read_text() == "ant\nlarva\n"


def test_appending_never_renumbers_an_existing_class(tmp_path):
    source = _source(tmp_path, {}, classes="ant\nbeetle\n")
    staged = _staging(tmp_path, {}, classes="larva\n")

    mapping = sr.resolve_staged_class_ids(source, staged)

    assert mapping == {0: 2}
    assert (Path(source.path) / "classes.txt").read_text().splitlines()[:2] == [
        "ant",
        "beetle",
    ]


def test_resolution_is_idempotent(tmp_path):
    source = _source(tmp_path, {}, classes="ant\n")
    staged = _staging(tmp_path, {}, classes="larva\n")

    first = sr.resolve_staged_class_ids(source, staged)
    second = sr.resolve_staged_class_ids(source, staged)

    assert first == second
    assert (Path(source.path) / "classes.txt").read_text() == "ant\nlarva\n"
