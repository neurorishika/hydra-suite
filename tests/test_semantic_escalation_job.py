import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.core.inference.semantic.base import SemanticInstance
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.semantic_escalation import (
    SemanticEscalationRequest,
    is_prompt_failure,
    rethreshold_staged,
    run_semantic_escalation,
)


class ScriptedLabeler:
    """Returns the same tile-local instances for every tile it is given."""

    def __init__(self, instances):
        self._instances = instances

    @property
    def name(self):
        return "fake"

    def label_image(
        self, image_bgr, prompt, *, confidence_threshold=0.0, max_instances=0
    ):
        return [i for i in self._instances if i.confidence >= confidence_threshold]


class _Project:
    def __init__(self, project_dir, sources):
        self.project_dir = str(project_dir)
        self.sources = sources


def _make_source(tmp_path, name="src", n_images=2, level="polygon"):
    root = tmp_path / "sources" / name
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for i in range(n_images):
        cv2.imwrite(
            str(root / "images" / f"f{i}.png"), np.zeros((400, 400, 3), dtype=np.uint8)
        )
        (root / "labels" / f"f{i}.txt").write_text("")
    (root / "classes.txt").write_text("object\n")
    return OBBSource(path=str(root), name=name, level=level)


def _blob(cx, cy, side=20.0):
    h = side / 2.0
    return np.array(
        [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]],
        dtype=np.float32,
    )


def _request(tmp_path, src, **kw):
    defaults = dict(
        project=_Project(tmp_path, [src]),
        source_names=[src.name],
        variant="sam3",
        prompt="ant",
        confidence=0.1,
        max_instances=0,
        reference_body_px=20.0,
        overlap=0.0,
        seam_margin_px=2.0,
        merge_iou=0.5,
        tile_fraction=None,
        tile_px=None,
        overwrite=False,
    )
    defaults.update(kw)
    return SemanticEscalationRequest(**defaults)


def test_polygon_level_sources_are_not_filtered_out(tmp_path):
    src = _make_source(tmp_path, level="polygon")
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    result = run_semantic_escalation(_request(tmp_path, src), labeler)
    assert result.staged == [src.name]
    assert result.labelled > 0


def test_original_labels_are_never_touched(tmp_path):
    src = _make_source(tmp_path)
    (Path(src.path) / "labels" / "f0.txt").write_text("0 0.1 0.1 0.2 0.2\n")
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src), labeler)
    assert (Path(src.path) / "labels" / "f0.txt").read_text() == "0 0.1 0.1 0.2 0.2\n"


def test_two_prompts_stage_into_different_directories(tmp_path):
    src = _make_source(tmp_path)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, prompt="ant"), labeler)
    first = src.pending_escalation.staged_path
    run_semantic_escalation(
        _request(tmp_path, src, prompt="beetle", overwrite=True),
        labeler,
        overwrite=True,
    )
    assert src.pending_escalation.staged_path != first


def test_empty_images_are_counted_and_flagged_as_a_prompt_failure(tmp_path):
    src = _make_source(tmp_path, n_images=3)
    result = run_semantic_escalation(_request(tmp_path, src), ScriptedLabeler([]))
    assert result.empty_images == 3
    assert result.labelled == 0
    assert is_prompt_failure(result, frames_processed=3) is True


def test_degenerate_contours_are_dropped_not_fatal(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    two_points = np.array([[10, 10], [20, 20]], dtype=np.float32)
    labeler = ScriptedLabeler(
        [
            SemanticInstance(two_points, 0.9),
            SemanticInstance(_blob(200, 200), 0.9),
        ]
    )
    result = run_semantic_escalation(_request(tmp_path, src), labeler)
    assert result.degenerate >= 1
    assert result.labelled == 1


def test_candidates_cache_is_written_into_the_staging_dir(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src), labeler)
    cache = Path(src.pending_escalation.staged_path) / "candidates.json"
    data = json.loads(cache.read_text())
    assert data["version"] == 1
    assert "f0.png" in data["images"]


def test_rethreshold_rewrites_labels_without_inference(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler(
        [
            SemanticInstance(_blob(100, 100), 0.9),
            SemanticInstance(_blob(300, 300), 0.2),
        ]
    )
    run_semantic_escalation(_request(tmp_path, src, confidence=0.1), labeler)
    staged = Path(src.pending_escalation.staged_path) / "labels" / "f0.txt"
    assert len(staged.read_text().strip().splitlines()) == 2
    kept = rethreshold_staged(src, confidence=0.5, merge_iou=0.5)
    assert kept == 1
    assert len(staged.read_text().strip().splitlines()) == 1


def test_resume_skips_images_already_in_the_cache(tmp_path):
    src = _make_source(tmp_path, n_images=2)

    class Counting(ScriptedLabeler):
        calls = 0

        def label_image(self, *a, **k):
            Counting.calls += 1
            return super().label_image(*a, **k)

    labeler = Counting([SemanticInstance(_blob(200, 200), 0.9)])
    req = _request(tmp_path, src)
    run_semantic_escalation(req, labeler, should_stop=lambda: Counting.calls >= 1)
    first_calls = Counting.calls
    run_semantic_escalation(req, labeler, overwrite=True)
    # The already-cached image is not re-inferred.
    assert Counting.calls < first_calls + 2


def test_fingerprint_mismatch_wipes_the_cache(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, prompt="ant"), labeler)
    staged = Path(src.pending_escalation.staged_path)
    run_json = json.loads((staged / "run.json").read_text())
    assert run_json["prompt"] == "ant"


def test_already_pending_source_is_skipped_without_overwrite(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src), labeler)
    result = run_semantic_escalation(_request(tmp_path, src), labeler)
    assert result.staged == []
    assert result.skipped and result.skipped[0][0] == src.name


from hydra_suite.detectkit.jobs.semantic_escalation import (
    accept_pending_semantic_escalation,
)


def test_accept_creates_a_sibling_and_leaves_the_origin_untouched(tmp_path):
    src = _make_source(tmp_path, n_images=2)
    (Path(src.path) / "labels" / "f0.txt").write_text("0 0.1 0.1 0.2 0.2\n")
    original = (Path(src.path) / "labels" / "f0.txt").read_bytes()
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, project=project), labeler)

    sibling = accept_pending_semantic_escalation(src, project, tmp_path)

    assert sibling is not src
    assert sibling in project.sources
    assert sibling.level == "polygon"
    assert sibling.reviewed is False
    assert sibling.derived_from == src.name
    assert (Path(src.path) / "labels" / "f0.txt").read_bytes() == original
    assert src.pending_escalation is None


def test_sibling_carries_images_and_the_prompt_as_its_class_name(tmp_path):
    src = _make_source(tmp_path, n_images=2)
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(
        _request(tmp_path, src, project=project, prompt="black ant"), labeler
    )
    sibling = accept_pending_semantic_escalation(src, project, tmp_path)
    root = Path(sibling.path)
    assert len(list((root / "images").rglob("*.png"))) == 2
    assert len(list((root / "labels").rglob("*.txt"))) == 2
    assert (root / "classes.txt").read_text().strip() == "black ant"


def test_the_candidate_cache_never_reaches_the_sibling(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, project=project), labeler)
    sibling = accept_pending_semantic_escalation(src, project, tmp_path)
    assert not (Path(sibling.path) / "candidates.json").exists()
    assert not (Path(sibling.path) / "run.json").exists()


def test_accept_refuses_a_sam2_pending_record(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    from hydra_suite.detectkit.gui.models import PendingEscalation

    src.pending_escalation = PendingEscalation(
        staged_path=str(tmp_path), primer_kind="sam2"
    )
    with pytest.raises(ValueError, match="not a SAM3"):
        accept_pending_semantic_escalation(src, _Project(tmp_path, [src]), tmp_path)


def test_accept_refuses_when_the_staging_dir_is_gone(tmp_path):
    import shutil

    src = _make_source(tmp_path, n_images=1)
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, project=project), labeler)
    shutil.rmtree(src.pending_escalation.staged_path)
    with pytest.raises(RuntimeError, match="missing on disk"):
        accept_pending_semantic_escalation(src, project, tmp_path)
