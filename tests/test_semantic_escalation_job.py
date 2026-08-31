import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hydra_suite.core.inference.semantic.base import SemanticInstance
from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.detectkit.jobs.semantic_escalation import (
    SemanticEscalationRequest,
    accept_pending_semantic_escalation,
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
    first = src.staged_review.staged_path
    run_semantic_escalation(
        _request(tmp_path, src, prompt="beetle", overwrite=True),
        labeler,
        overwrite=True,
    )
    assert src.staged_review.staged_path != first


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
    cache = Path(src.staged_review.staged_path) / "candidates.json"
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
    staged = Path(src.staged_review.staged_path) / "labels" / "f0.txt"
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
    """I11: the old version of this test never tested the wipe.

    It ran ONE escalation and asserted ``run.json["prompt"] == "ant"`` -- an
    assertion about the fingerprint's contents, not about what a MISMATCHED
    fingerprint does. Resuming a cache built with incompatible tile geometry
    is exactly the corruption the fingerprint exists to prevent, so the wipe
    itself has to be pinned.
    """
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, prompt="ant"), labeler)
    staged = Path(src.staged_review.staged_path)
    assert json.loads((staged / "run.json").read_text())["prompt"] == "ant"

    # Poison the cache with a candidate that no labeler produced, and
    # invalidate the fingerprint by changing a tiled-geometry parameter.
    cache_path = staged / "candidates.json"
    cache = json.loads(cache_path.read_text())
    cache["images"]["f0.png"]["candidates"].append(
        {"p": [[1.0, 1.0], [9.0, 1.0], [9.0, 9.0]], "c": 0.99, "t": 0}
    )
    cache_path.write_text(json.dumps(cache))
    assert (
        len(json.loads(cache_path.read_text())["images"]["f0.png"]["candidates"]) == 2
    )

    run_semantic_escalation(
        _request(tmp_path, src, prompt="ant", seam_margin_px=17.0, overwrite=True),
        labeler,
    )
    staged2 = Path(src.staged_review.staged_path)
    assert staged2 == staged, "same prompt+variant must reuse the same staging dir"
    after = json.loads((staged2 / "candidates.json").read_text())
    # The poisoned entry is gone: the cache was WIPED and rebuilt, not resumed.
    assert len(after["images"]["f0.png"]["candidates"]) == 1
    assert json.loads((staged2 / "run.json").read_text())["seam_margin_px"] == 17.0


def test_a_matching_fingerprint_resumes_instead_of_wiping(tmp_path):
    """The other half of the same invariant: an unchanged run resumes."""
    src = _make_source(tmp_path, n_images=2)

    class Counting(ScriptedLabeler):
        calls = 0

        def label_image(self, *a, **k):
            Counting.calls += 1
            return super().label_image(*a, **k)

    labeler = Counting([SemanticInstance(_blob(200, 200), 0.9)])
    req = _request(tmp_path, src)
    run_semantic_escalation(req, labeler, should_stop=lambda: Counting.calls >= 1)
    cached_after_cancel = set(
        json.loads(
            (Path(src.staged_review.staged_path) / "candidates.json").read_text()
        )["images"]
    )
    assert len(cached_after_cancel) == 1
    run_semantic_escalation(req, labeler)
    cached = json.loads(
        (Path(src.staged_review.staged_path) / "candidates.json").read_text()
    )["images"]
    assert set(cached) == {"f0.png", "f1.png"}


def test_a_different_pending_escalation_is_skipped_without_overwrite(tmp_path):
    """I2: a DIFFERENT staged result is protected; a resume is not blocked."""
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, prompt="ant"), labeler)
    ant_staged = src.staged_review.staged_path

    # A different prompt would DESTROY the staged 'ant' result: refuse.
    result = run_semantic_escalation(_request(tmp_path, src, prompt="beetle"), labeler)
    assert result.staged == []
    assert result.skipped and result.skipped[0][0] == src.name
    assert src.staged_review.staged_path == ant_staged
    assert Path(ant_staged).is_dir()

    # ... but re-issuing the SAME run is a resume and must NOT be refused.
    resumed = run_semantic_escalation(_request(tmp_path, src, prompt="ant"), labeler)
    assert resumed.staged == [src.name]


def test_sources_pending_replacement_lists_only_real_replacements(tmp_path):
    from hydra_suite.detectkit.jobs.semantic_escalation import (
        sources_pending_replacement,
    )

    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    assert sources_pending_replacement(_request(tmp_path, src, prompt="ant")) == []
    run_semantic_escalation(_request(tmp_path, src, prompt="ant"), labeler)
    # Same run -> a resume, nothing at risk.
    assert sources_pending_replacement(_request(tmp_path, src, prompt="ant")) == []
    # Different prompt -> the staged 'ant' result would be destroyed.
    assert sources_pending_replacement(_request(tmp_path, src, prompt="beetle")) == [
        src.name
    ]


def test_an_unreviewed_sam2_escalation_is_not_silently_destroyed(tmp_path):
    """I2 concretely: the GUI used to pass overwrite=True unconditionally."""
    from hydra_suite.detectkit.gui.models import StagedReview
    from hydra_suite.detectkit.jobs.semantic_escalation import (
        sources_pending_replacement,
    )

    src = _make_source(tmp_path, n_images=1)
    sam2_dir = tmp_path / "artifacts" / "pending_escalations" / "sam2-staged"
    (sam2_dir / "labels").mkdir(parents=True)
    (sam2_dir / "labels" / "f0.txt").write_text("0 0.1 0.1 0.2 0.2\n")
    src.staged_review = StagedReview(staged_path=str(sam2_dir), producer="sam2")
    req = _request(tmp_path, src, prompt="ant")
    assert sources_pending_replacement(req) == [src.name]
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    result = run_semantic_escalation(req, labeler)
    assert result.staged == []
    assert (sam2_dir / "labels" / "f0.txt").exists()


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
    assert src.staged_review is None


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
    from hydra_suite.detectkit.gui.models import StagedReview

    src.staged_review = StagedReview(staged_path=str(tmp_path), producer="sam2")
    with pytest.raises(ValueError, match="not a SAM3"):
        accept_pending_semantic_escalation(src, _Project(tmp_path, [src]), tmp_path)


def test_accept_refuses_when_the_staging_dir_is_gone(tmp_path):
    import shutil

    src = _make_source(tmp_path, n_images=1)
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, project=project), labeler)
    shutil.rmtree(src.staged_review.staged_path)
    with pytest.raises(RuntimeError, match="missing on disk"):
        accept_pending_semantic_escalation(src, project, tmp_path)


def test_labelled_frames_reads_every_geometry_level(tmp_path):
    from hydra_suite.detectkit.jobs.semantic_escalation import labelled_frames_for

    src = _make_source(tmp_path, n_images=3)
    labels = Path(src.path) / "labels"
    labels.joinpath("f0.txt").write_text("0 0.5 0.5 0.1 0.1\n")  # AABB
    labels.joinpath("f1.txt").write_text(
        "0 0.4 0.4 0.5 0.4 0.5 0.5 0.4 0.5\n"  # OBB quad
    )
    labels.joinpath("f2.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.25 0.2 0.15 0.25 0.08 0.2\n"  # 5-point polygon
    )
    frames = labelled_frames_for(src)
    assert len(frames) == 3
    for _path, records in frames:
        assert len(records) == 1
        assert records[0].points.shape[1] == 2


def test_labelled_frames_skips_empty_label_files(tmp_path):
    from hydra_suite.detectkit.jobs.semantic_escalation import labelled_frames_for

    src = _make_source(tmp_path, n_images=2)
    (Path(src.path) / "labels" / "f0.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert len(labelled_frames_for(src)) == 1


# --- I1: the prompt-failure denominator -------------------------------------


def test_frames_processed_counts_frames_not_instances(tmp_path):
    """I1: `labelled` counts INSTANCES, so it is the wrong denominator.

    The caller computed `frames = labelled + empty_images`. With 10 instances
    per frame that inflates the denominator ~10x and the prompt-failure rule
    can only fire on a total shutout. This pins the frame count itself, which
    is what the rule must divide by.
    """
    src = _make_source(tmp_path, n_images=4)
    labeler = ScriptedLabeler(
        [SemanticInstance(_blob(40 + 80 * i, 40 + 80 * i), 0.9) for i in range(10)]
    )
    result = run_semantic_escalation(_request(tmp_path, src), labeler)
    assert result.frames_processed == 4
    assert result.labelled == 40  # 10 instances x 4 frames -- NOT a frame count
    assert result.labelled + result.empty_images == 40  # the old, wrong figure


def test_prompt_failure_fires_on_a_majority_empty_run_with_many_instances(tmp_path):
    """The exact case the old caller's arithmetic could not detect."""
    from hydra_suite.detectkit.jobs.semantic_escalation import SemanticEscalationResult

    # 100 frames, 60 empty, the other 40 with 10 instances each.
    result = SemanticEscalationResult(
        labelled=400, empty_images=60, frames_processed=100
    )
    # The rule, given the run's own frame count, fires.
    assert is_prompt_failure(result) is True
    # Given the caller's old denominator (labelled + empty = 460), it does not.
    assert is_prompt_failure(result, result.labelled + result.empty_images) is False


# --- I9: a cancelled run reports itself as cancelled -------------------------


def test_a_cancelled_run_is_flagged_as_cancelled(tmp_path):
    src = _make_source(tmp_path, n_images=3)

    class Counting(ScriptedLabeler):
        calls = 0

        def label_image(self, *a, **k):
            Counting.calls += 1
            return super().label_image(*a, **k)

    labeler = Counting([SemanticInstance(_blob(200, 200), 0.9)])
    result = run_semantic_escalation(
        _request(tmp_path, src), labeler, should_stop=lambda: Counting.calls >= 1
    )
    assert result.cancelled is True
    assert result.frames_processed == 1  # got through one of three frames
    # A complete run is NOT flagged.
    src2 = _make_source(tmp_path, name="src2", n_images=1)
    done = run_semantic_escalation(
        _request(tmp_path, src2),
        ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)]),
    )
    assert done.cancelled is False


# --- I4: the cache floor makes downward re-thresholding honest ---------------


def test_candidates_are_cached_below_the_run_confidence(tmp_path):
    """I4: collecting at req.confidence silently truncated the cache.

    A run at 0.35 kept nothing below 0.35, so re-thresholding down to 0.20 --
    which the dialog, the results dialog and the user guide all promise is
    free and complete -- returned a truncated set.
    """
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler(
        [
            SemanticInstance(_blob(100, 100), 0.90),
            SemanticInstance(_blob(300, 300), 0.25),
        ]
    )
    run_semantic_escalation(_request(tmp_path, src, confidence=0.35), labeler)
    staged = Path(src.staged_review.staged_path)
    cache = json.loads((staged / "candidates.json").read_text())
    confs = sorted(c["c"] for c in cache["images"]["f0.png"]["candidates"])
    assert confs == [0.25, 0.9], "the 0.25 candidate must survive into the cache"
    # The staged labels still honour the RUN threshold.
    assert len((staged / "labels" / "f0.txt").read_text().strip().splitlines()) == 1
    # ... and re-thresholding downward now finds it.
    assert rethreshold_staged(src, confidence=0.20, merge_iou=0.5) == 2


def test_rethreshold_refuses_to_go_below_the_recorded_cache_floor(tmp_path):
    """An OLD staging dir records the higher floor it was collected at."""
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(100, 100), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, confidence=0.35), labeler)
    staged = Path(src.staged_review.staged_path)
    run_json = json.loads((staged / "run.json").read_text())
    assert run_json["confidence_floor"] == pytest.approx(0.05)
    run_json["confidence_floor"] = 0.35  # simulate a pre-I4 cache
    (staged / "run.json").write_text(json.dumps(run_json))
    with pytest.raises(ValueError, match="truncated"):
        rethreshold_staged(src, confidence=0.20, merge_iou=0.5)
    # At or above the recorded floor it is still allowed.
    assert rethreshold_staged(src, confidence=0.40, merge_iou=0.5) == 1


def test_confidence_alone_does_not_invalidate_the_candidate_cache(tmp_path):
    """Re-running at a new confidence must RESUME, not re-infer."""
    src = _make_source(tmp_path, n_images=2)

    class Counting(ScriptedLabeler):
        calls = 0

        def label_image(self, *a, **k):
            Counting.calls += 1
            return super().label_image(*a, **k)

    labeler = Counting([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, confidence=0.35), labeler)
    after_first = Counting.calls
    run_semantic_escalation(_request(tmp_path, src, confidence=0.60), labeler)
    assert Counting.calls == after_first


# --- I7: promotion image lookup ---------------------------------------------


def test_promotion_matches_nested_images_by_relative_path(tmp_path):
    """I7: `origin_images.rglob(f"{rel.stem}.*")` walked the WHOLE tree.

    Two nested subdirectories holding the same stem made the match
    nondeterministic (whichever rglob yielded first won), and every label
    paid a full-tree walk. This nested layout has that exact collision.
    """
    root = tmp_path / "sources" / "nested"
    for sub in ("a", "b"):
        (root / "images" / sub).mkdir(parents=True)
        (root / "labels" / sub).mkdir(parents=True)
    # Same stem in two subdirectories, with visibly different pixels.
    cv2.imwrite(
        str(root / "images" / "a" / "f0.png"), np.zeros((400, 400, 3), dtype=np.uint8)
    )
    cv2.imwrite(
        str(root / "images" / "b" / "f0.png"),
        np.full((400, 400, 3), 255, dtype=np.uint8),
    )
    (root / "classes.txt").write_text("object\n")
    src = OBBSource(path=str(root), name="nested", level="polygon")
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, project=project), labeler)

    sibling = accept_pending_semantic_escalation(src, project, tmp_path)
    sib = Path(sibling.path)
    assert (sib / "labels" / "a" / "f0.txt").exists()
    assert (sib / "labels" / "b" / "f0.txt").exists()
    # Each image must be the one from its OWN subdirectory, not whichever
    # rglob happened to yield first.
    assert cv2.imread(str(sib / "images" / "a" / "f0.png")).mean() == 0
    assert cv2.imread(str(sib / "images" / "b" / "f0.png")).mean() == 255


def test_a_staged_label_with_no_origin_image_is_skipped_not_orphaned(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    project = _Project(tmp_path, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, project=project), labeler)
    staged = Path(src.staged_review.staged_path)
    # A staged label whose image no longer exists in the origin.
    (staged / "labels" / "ghost.txt").write_text(
        (staged / "labels" / "f0.txt").read_text()
    )
    sibling = accept_pending_semantic_escalation(src, project, tmp_path)
    sib = Path(sibling.path)
    assert not (sib / "labels" / "ghost.txt").exists()
    assert (sib / "labels" / "f0.txt").exists()
    # No label in the sibling is without its image.
    for lbl in (sib / "labels").rglob("*.txt"):
        rel = lbl.relative_to(sib / "labels")
        assert (sib / "images" / rel.with_suffix(".png")).exists()


def test_unique_source_name_avoids_a_stale_on_disk_directory(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    project = _Project(tmp_path, [src])
    # A leftover directory from a source the user DELETED from the project.
    stale = tmp_path / "sources" / "src-sam3-ant"
    (stale / "labels").mkdir(parents=True)
    (stale / "labels" / "leftover.txt").write_text("0 0.1 0.1 0.2 0.2\n")
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, project=project), labeler)
    sibling = accept_pending_semantic_escalation(src, project, tmp_path)
    assert Path(sibling.path) != stale
    assert not (Path(sibling.path) / "labels" / "leftover.txt").exists()


# --- I6 / I8: reference-body chain and the non-decoding has-labels check -----


def test_median_body_px_measures_the_longest_side_of_existing_labels(tmp_path):
    from hydra_suite.detectkit.jobs.semantic_escalation import median_body_px_for

    src = _make_source(tmp_path, n_images=3)
    labels = Path(src.path) / "labels"
    # 400x400 frames; AABB w=0.10 -> 40 px, 0.20 -> 80 px, 0.30 -> 120 px.
    labels.joinpath("f0.txt").write_text("0 0.5 0.5 0.10 0.05\n")
    labels.joinpath("f1.txt").write_text("0 0.5 0.5 0.20 0.05\n")
    labels.joinpath("f2.txt").write_text("0 0.5 0.5 0.30 0.05\n")
    assert median_body_px_for([src]) == pytest.approx(80.0, abs=1.0)


def test_median_body_px_is_zero_with_no_labels(tmp_path):
    from hydra_suite.detectkit.jobs.semantic_escalation import median_body_px_for

    assert median_body_px_for([_make_source(tmp_path, n_images=2)]) == 0.0


def test_has_labelled_frames_never_decodes_an_image(tmp_path, monkeypatch):
    """I8: the dialog answered a yes/no by decoding every labelled image."""
    import hydra_suite.detectkit.jobs.semantic_escalation as mod

    src = _make_source(tmp_path, n_images=2)
    monkeypatch.setattr(
        mod.cv2,
        "imread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not decode")),
    )
    assert mod.has_labelled_frames(src) is False
    (Path(src.path) / "labels" / "f1.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert mod.has_labelled_frames(src) is True


# --- Complete-frame visual preview -------------------------------------------


def test_preview_runs_every_tile_of_one_random_frame_and_returns_overlay(
    tmp_path, monkeypatch
):
    import hydra_suite.detectkit.jobs.semantic_escalation as mod

    src = _make_source(tmp_path, n_images=2)

    class Counting(ScriptedLabeler):
        def __init__(self, instances):
            super().__init__(instances)
            self.shapes = []

        def label_image(self, image_bgr, *a, **k):
            self.shapes.append(image_bgr.shape[:2])
            return super().label_image(image_bgr, *a, **k)

    labeler = Counting([SemanticInstance(_blob(50, 50), 0.9)] * 3)
    monkeypatch.setattr(mod.random, "choice", lambda choices: choices[-1])
    (Path(src.path) / "labels" / "f1.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    progress = []
    res = mod.preview_random_frame(
        labeler,
        [src],
        "ant",
        reference_body_px=20.0,
        tile_fraction=0.10,  # 200 px tiles over a 400x400 frame
        progress=lambda done, total: progress.append((done, total)),
    )
    assert len(labeler.shapes) == res.tiles_per_frame
    assert len(labeler.shapes) > 1, "the complete image must run every tile"
    assert set(labeler.shapes) == {(200, 200)}
    assert res.tile_px == 200
    assert res.predictions
    assert len(res.ground_truth) == 1
    assert res.seconds > 0.0  # MEASURED, never a hardcoded figure
    assert res.image_path.name == "f1.png"
    assert progress[-1] == (res.tiles_per_frame, res.tiles_per_frame)


def test_complete_frame_preview_worker_can_be_cancelled_before_inference():
    from hydra_suite.detectkit.jobs.semantic_escalation import FramePreviewWorker

    worker = FramePreviewWorker([], "ant", "sam3", {}, labeler=object())
    assert worker.cancelled is False
    worker.cancel()
    assert worker.cancelled is True


# --- Fix wave regressions (blockers A-E) -------------------------------------


def test_resume_is_honoured_through_a_symlinked_project_dir(tmp_path):
    """BLOCKER A: `project_dir` arrives UNRESOLVED, staging paths do not.

    ``ensure_bundle_subdirectory`` -> ``bundle_paths`` resolves the project
    root, so a project reached through a symlink (macOS /tmp, a symlinked
    home or lab share) recorded a RESOLVED ``staged_path`` and then compared
    it against an UNRESOLVED target -- making the run refuse itself as a
    replacement of a different escalation. ``tmp_path`` is already resolved,
    which is exactly why no existing test caught this.
    """
    from hydra_suite.detectkit.jobs.semantic_escalation import (
        sources_pending_replacement,
    )

    real = tmp_path / "real_project"
    real.mkdir()
    link = tmp_path / "link_project"
    link.symlink_to(real, target_is_directory=True)

    src = _make_source(link, n_images=2)
    project = _Project(link, [src])
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    req = _request(link, src, project=project)

    run_semantic_escalation(req, labeler)
    assert src.staged_review is not None
    staged = Path(src.staged_review.staged_path)
    cached = set(json.loads((staged / "candidates.json").read_text())["images"])

    # The very same run must be a RESUME: nothing pending replacement, no
    # skip, and the candidate cache survives.
    assert sources_pending_replacement(req) == []
    resumed = run_semantic_escalation(req, labeler)
    assert resumed.skipped == []
    assert resumed.staged == [src.name]
    assert Path(src.staged_review.staged_path) == staged
    assert set(json.loads((staged / "candidates.json").read_text())["images"]) == cached


def test_a_run_below_the_grid_floor_stages_every_instance(tmp_path):
    """BLOCKER B: the cache floor must never sit ABOVE the run's own value.

    The dialog allows confidences down to 0.01. Collecting at the grid
    bottom (0.05) for a run at 0.02 dropped candidates in [0.02, 0.05)
    before they reached the cache, so they could never become labels: the
    run staged 1 instance where it had asked for 2.
    """
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler(
        [
            SemanticInstance(_blob(100, 100), 0.90),
            SemanticInstance(_blob(300, 300), 0.03),
        ]
    )
    result = run_semantic_escalation(_request(tmp_path, src, confidence=0.02), labeler)
    assert result.labelled == 2
    staged = Path(src.staged_review.staged_path)
    cache = json.loads((staged / "candidates.json").read_text())
    assert sorted(c["c"] for c in cache["images"]["f0.png"]["candidates"]) == [
        0.03,
        0.9,
    ]
    assert len((staged / "labels" / "f0.txt").read_text().strip().splitlines()) == 2
    # The RECORDED floor is the one actually used, so rethreshold_staged
    # refuses only what this cache genuinely cannot serve.
    run_json = json.loads((staged / "run.json").read_text())
    assert run_json["confidence_floor"] == pytest.approx(0.02)
    assert rethreshold_staged(src, confidence=0.02, merge_iou=0.5) == 2


def test_rethreshold_refusal_at_the_grid_floor_gives_reachable_advice(tmp_path):
    """C: "re-run to collect at a lower floor" was impossible advice.

    A re-run at the same confidence collects at the same floor, so the only
    thing that helps is re-running at a LOWER confidence -- which the
    message must say, and which the review dialog's minimum must reflect.
    """
    from hydra_suite.detectkit.jobs.semantic_escalation import rethreshold_floor_for

    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(100, 100), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, confidence=0.35), labeler)
    staged = Path(src.staged_review.staged_path)

    assert rethreshold_floor_for([src]) == pytest.approx(0.05)
    with pytest.raises(ValueError) as exc:
        rethreshold_staged(src, confidence=0.02, merge_iou=0.5)
    assert "0.02 or lower" in str(exc.value)

    # A pre-I4 cache is the OTHER case: re-running does lower the floor.
    run_json = json.loads((staged / "run.json").read_text())
    run_json["confidence_floor"] = 0.35
    (staged / "run.json").write_text(json.dumps(run_json))
    assert rethreshold_floor_for([src]) == pytest.approx(0.35)
    with pytest.raises(ValueError) as exc:
        rethreshold_staged(src, confidence=0.20, merge_iou=0.5)
    assert "0.05" in str(exc.value)


def test_primer_params_record_the_confidence_the_labels_were_written_at(tmp_path):
    """D: the review dialog prefills its re-threshold prompt from this.

    Without it a run at 0.60 was reported to the user as being staged at the
    0.35 default, and nothing else on disk records the label threshold.
    """
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler([SemanticInstance(_blob(100, 100), 0.9)])
    run_semantic_escalation(_request(tmp_path, src, confidence=0.60), labeler)
    params = src.staged_review.params
    assert params["confidence"] == pytest.approx(0.60)
    # The CACHE floor is a different number and both are kept.
    assert params["confidence_floor"] == pytest.approx(0.05)


def test_orphaned_counts_cached_frames_whose_origin_image_is_gone(tmp_path):
    """E: `result.orphaned` was declared and never assigned.

    A resume whose source lost an image still holds that frame in the
    candidate cache, so it would be written as a staged label with no image
    behind it -- exactly what promotion has to throw away. The count is what
    makes the GUI's orphan note renderable.
    """
    src = _make_source(tmp_path, n_images=2)
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    req = _request(tmp_path, src)
    first = run_semantic_escalation(req, labeler)
    assert first.orphaned == 0

    (Path(src.path) / "images" / "f1.png").unlink()
    resumed = run_semantic_escalation(req, labeler)
    assert resumed.orphaned == 1
    assert resumed.labelled == 1  # only f0 still has an image behind it


def _cached_images(staged: Path) -> set:
    """Keys present in the staging dir's candidate cache (absent file = none)."""
    path = staged / "candidates.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text())["images"])


class _CountingLabeler(ScriptedLabeler):
    """ScriptedLabeler that records how many tiles it was asked to label."""

    def __init__(self, instances):
        super().__init__(instances)
        self.calls = 0

    def label_image(self, image_bgr, prompt, **kw):
        self.calls += 1
        return super().label_image(image_bgr, prompt, **kw)


def test_a_mid_frame_cancel_never_caches_the_partial_frame(tmp_path):
    """F1: cancelling between tiles must not poison the candidate cache.

    Before the fix, collect_candidates returned the tiles it had managed and
    the job wrote that partial list under the frame's cache key as if the
    frame were complete. On resume, `if rel in cache["images"]: continue`
    trusted it forever: zero further inference, cancelled reported False,
    and "re-run to carry on" did nothing.
    """
    src = _make_source(tmp_path, n_images=1)
    # 400x400 frame, 100 px tiles, no overlap => 16 tiles.
    labeler = _CountingLabeler([SemanticInstance(_blob(50, 50), 0.9)])
    req = _request(tmp_path, src, reference_body_px=100.0, tile_fraction=1.0)
    req.tile_px = 100

    result = run_semantic_escalation(
        req, labeler, should_stop=lambda: labeler.calls >= 3
    )
    assert result.cancelled is True
    tiles_first = labeler.calls
    assert 0 < tiles_first < 16, "the frame must have been cut off mid-way"

    staged = Path(src.staged_review.staged_path)
    assert _cached_images(staged) == set(), "a partial frame must NOT be cached"
    assert result.frames_processed == 0

    # Resume: same request, same staging dir, no cancel. It must actually
    # redo the frame from tile zero rather than trusting a poisoned entry.
    labeler2 = _CountingLabeler([SemanticInstance(_blob(50, 50), 0.9)])
    result2 = run_semantic_escalation(req, labeler2)
    assert result2.cancelled is False
    assert labeler2.calls == 16, "resume did no inference: the cache was poisoned"
    assert _cached_images(staged) == {"f0.png"}
    assert result2.frames_processed == 1
    assert result2.labelled > 0


def test_a_completed_frame_is_still_cached_and_reused_on_resume(tmp_path):
    """The complement: F1 must not disable legitimate resume."""
    src = _make_source(tmp_path, n_images=2)
    labeler = _CountingLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    req = _request(tmp_path, src)
    # Frame 0 is one full-frame tile, so calls>=1 cancels BETWEEN images,
    # after frame 0 has completed.
    run_semantic_escalation(req, labeler, should_stop=lambda: labeler.calls >= 1)
    staged = Path(src.staged_review.staged_path)
    assert _cached_images(staged) == {"f0.png"}, "a COMPLETE frame must be cached"

    labeler2 = _CountingLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    run_semantic_escalation(req, labeler2)
    assert labeler2.calls == 1, "the completed frame must be reused, not redone"


def test_mixed_case_image_extensions_are_not_orphaned(tmp_path):
    """F5: `a.Jpg` passes the run scan but the promotion lookup missed it.

    `_origin_image_for` tried only `ext` and `ext.upper()`, so a file whose
    extension is neither all-lower nor all-upper was scanned, inferred (GPU
    time spent) and cached, then silently dropped as an orphan. Invisible on
    macOS; real data loss on the case-sensitive Linux lab shares.
    """
    from hydra_suite.detectkit.jobs.semantic_escalation import _origin_image_for

    images = tmp_path / "images"
    (images / "sub").mkdir(parents=True)
    for name in ("a.Jpg", "sub/b.PnG", "c.jpg", "d.JPG"):
        (images / name).write_bytes(b"x")
    for name in ("a.Jpg", "sub/b.PnG", "c.jpg", "d.JPG"):
        assert _origin_image_for(images, Path(name)) == images / name, name
    assert _origin_image_for(images, Path("missing.jpg")) is None


def test_mixed_case_frames_survive_the_whole_run(tmp_path):
    """End to end: scanned, inferred and PROMOTED, not counted as orphans.

    NOTE: this one only FAILS pre-fix on a case-sensitive filesystem (the
    Linux deployment target); macOS's case-insensitive FS masks it. The unit
    test above is the filesystem-independent regression guard.
    """
    src = _make_source(tmp_path, n_images=0)
    root = Path(src.path)
    cv2.imwrite(
        str(root / "images" / "frame.Png"), np.zeros((400, 400, 3), dtype=np.uint8)
    )
    labeler = ScriptedLabeler([SemanticInstance(_blob(200, 200), 0.9)])
    result = run_semantic_escalation(_request(tmp_path, src), labeler)
    assert result.orphaned == 0, "a mixed-case frame was dropped as an orphan"
    assert result.labelled == 1


def test_a_mid_run_failure_does_not_leave_a_pointer_to_a_deleted_dir(tmp_path):
    """F7: the stale pending_escalation pointer must die with its directory.

    The previous staged dir is removed up front but src.staged_review
    is only replaced after the source finishes. A crash in between (here the
    plan_for_frame ValueError at overlap 0.9) left the source pointing at a
    directory that no longer exists.
    """
    from hydra_suite.detectkit.gui.models import StagedReview

    src = _make_source(tmp_path, n_images=1)
    old_dir = tmp_path / "artifacts" / "pending_escalations" / "old-staged"
    (old_dir / "labels").mkdir(parents=True)
    src.staged_review = StagedReview(
        staged_path=str(old_dir),
        target_level="polygon",
        created_at="2026-01-01T00:00:00",
    )

    class _Exploding(ScriptedLabeler):
        def label_image(self, *a, **k):
            raise ValueError("tile plan exceeds the ceiling")

    req = _request(tmp_path, src, overwrite=True)
    with pytest.raises(ValueError):
        run_semantic_escalation(req, _Exploding([]))

    assert not old_dir.exists(), "the old staged dir should have been removed"
    assert (
        src.staged_review is None or Path(src.staged_review.staged_path).exists()
    ), "pending_escalation points at a deleted directory"


def test_calibration_worker_decodes_frames_itself_not_on_the_gui_thread(tmp_path):
    """F4: the decode belongs in the worker, behind the progress dialog.

    `_run_calibration` used to build `[... labelled_frames_for(s) ...]` with
    NO limit on the GUI thread, before the progress dialog existed --
    cv2.imread of every labelled image of every selected source. 200 x
    4512^2 frames is minutes of frozen UI with no feedback and no cancel.
    """
    import inspect

    from hydra_suite.detectkit.gui.dialogs import semantic_escalation_dialog as dlg
    from hydra_suite.detectkit.jobs.semantic_escalation import CalibrationWorker

    src = _make_source(tmp_path, n_images=2)
    root = Path(src.path)
    for i in range(2):
        (root / "labels" / f"f{i}.txt").write_text(
            "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n"
        )

    # The worker takes SOURCES and resolves the frames itself.
    seen = {}

    class _Recording(ScriptedLabeler):
        pass

    worker = CalibrationWorker(
        [src], "ant", "sam3", {"reference_body_px": 80.0}, labeler=_Recording([])
    )
    worker.result_ready.connect(lambda pts: seen.setdefault("points", pts))
    worker.execute()
    assert "points" in seen and seen["points"], "the worker must resolve frames itself"

    # And the GUI path no longer decodes anything before starting the worker.
    src_text = inspect.getsource(dlg.SemanticEscalationDialog._run_calibration)
    assert (
        "labelled_frames_for" not in src_text
    ), "the dialog is still decoding images on the GUI thread"
    # The progress dialog must be constructed before the worker.
    assert src_text.index("QProgressDialog(") < src_text.index("CalibrationWorker(")


def test_body_size_measurement_is_capped_project_wide_and_says_so(tmp_path):
    """F4, other half: `median_body_px_for` decoded 20 frames PER SOURCE.

    At dialog open, on the GUI thread, across every source in the project.
    The sample is now bounded globally; because that changes what "median of
    your labels" measured over, the cap is surfaced in the provenance string
    rather than applied silently.
    """
    from hydra_suite.detectkit.gui.escalation_actions import resolve_reference_body_px
    from hydra_suite.detectkit.jobs.semantic_escalation import (
        MEDIAN_BODY_TOTAL_FRAMES,
        measure_median_body_px,
    )

    sources = []
    for s in range(4):
        src = _make_source(tmp_path, name=f"s{s}", n_images=15)
        root = Path(src.path)
        for i in range(15):
            (root / "labels" / f"f{i}.txt").write_text(
                "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n"
            )
        sources.append(src)

    median, sampled, truncated = measure_median_body_px(sources)
    assert median > 0
    assert sampled <= MEDIAN_BODY_TOTAL_FRAMES, "the project-wide cap is not applied"
    assert truncated is True

    class _P:
        slice_training = None

    project = _P()
    project.sources = sources
    value, origin = resolve_reference_body_px(project)
    assert value == pytest.approx(median)
    assert "capped sample" in origin, "the cap must be surfaced, not silent"
    assert str(sampled) in origin


def test_a_single_over_budget_source_is_reported_as_truncated(tmp_path):
    """Adversarial re-review B1: `truncated` had a false negative.

    The flag was set only at the top of the NEXT loop iteration, so one
    source holding more frames than the whole budget -- the exact case the
    cap exists for -- returned ``truncated=False`` and a provenance string
    claiming an uncapped measurement.
    """
    from hydra_suite.detectkit.jobs.semantic_escalation import (
        MEDIAN_BODY_TOTAL_FRAMES,
        measure_median_body_px,
    )

    n = MEDIAN_BODY_TOTAL_FRAMES + 10
    src = _make_source(tmp_path, name="solo", n_images=n)
    root = Path(src.path)
    for i in range(n):
        (root / "labels" / f"f{i}.txt").write_text(
            "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n"
        )

    median, sampled, truncated = measure_median_body_px([src])
    assert median > 0
    assert sampled == MEDIAN_BODY_TOTAL_FRAMES
    assert truncated is True, "a single over-budget source reported an uncapped sample"


def test_the_area_band_gates_what_is_staged(tmp_path):
    """The calibrated size gate must reach inference, not stop at the frontier.

    Without this, calibration would pick an operating point under one rule
    and the 30-hour run would emit under another.
    """
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler(
        [
            SemanticInstance(_blob(100, 100, side=20.0), 0.9),  # a body
            SemanticInstance(_blob(300, 300, side=350.0), 0.9),  # arena chunk
            SemanticInstance(_blob(50, 50, side=3.0), 0.9),  # a leg
        ]
    )
    req = _request(tmp_path, src, confidence=0.1)
    req.area_min_px2 = 120.0
    req.area_max_px2 = 1400.0
    result = run_semantic_escalation(req, labeler)
    assert result.labelled == 1
    params = src.staged_review.params
    assert params["area_min_px2"] == 120.0
    assert params["area_max_px2"] == 1400.0


def test_rethreshold_keeps_the_bands_gate(tmp_path):
    """A re-threshold replays cached candidates; the gate must replay too."""
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler(
        [
            SemanticInstance(_blob(100, 100, side=20.0), 0.9),
            SemanticInstance(_blob(300, 300, side=350.0), 0.9),
        ]
    )
    req = _request(tmp_path, src, confidence=0.1)
    req.area_min_px2 = 120.0
    req.area_max_px2 = 1400.0
    run_semantic_escalation(req, labeler)
    assert rethreshold_staged(src, confidence=0.2, merge_iou=0.5) == 1


def test_no_band_stages_everything_as_before(tmp_path):
    src = _make_source(tmp_path, n_images=1)
    labeler = ScriptedLabeler(
        [
            SemanticInstance(_blob(100, 100, side=20.0), 0.9),
            SemanticInstance(_blob(300, 300, side=350.0), 0.9),
        ]
    )
    result = run_semantic_escalation(_request(tmp_path, src, confidence=0.1), labeler)
    assert result.labelled == 2
