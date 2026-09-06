"""Wiring test for `tools/sam3_parity/matcher_gate.py`.

No sam3, no GPU: a fake labeler emits a CURVED outline whose vertex mean
falls outside itself, which is the exact geometry the shipped matcher gets
wrong. The test therefore proves both that the gate runs end to end and that
it separates the two arms in the measured direction.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "sam3_parity"))

from matcher_gate import (  # noqa: E402
    _frozen_centroid,
    _frozen_contains,
    frozen_match_one_to_one,
    main,
)

from hydra_suite.core.inference.semantic.calibration import (  # noqa: E402
    _contains,
    match_one_to_one,
    representative_point,
)


def _crescent(cx, cy, r_in=20.0, r_out=34.0, n=64):
    ang = np.linspace(0.0, np.pi * 1.3, n)
    outer = np.stack([cx + r_out * np.cos(ang), cy + r_out * np.sin(ang)], axis=1)
    inner = np.stack(
        [cx + r_in * np.cos(ang[::-1]), cy + r_in * np.sin(ang[::-1])], axis=1
    )
    return np.concatenate([outer, inner]).astype(np.float32)


def test_frozen_arm_reproduces_the_defect_and_the_live_arm_does_not():
    """The frozen copy must stay broken -- it is the 'before' of the gate."""
    label = _crescent(120.0, 120.0)
    pred = _crescent(120.6, 120.4, r_out=34.2)
    assert not _frozen_contains(label, _frozen_centroid(pred))
    assert frozen_match_one_to_one([pred], [label]) == []
    assert _contains(label, representative_point(pred))
    assert match_one_to_one([pred], [label]) == [(0, 0)]


class _FakeLabeler:
    def __init__(self, detections):
        self._detections = detections

    @property
    def name(self):
        return "fake"

    def label_image(
        self, image_bgr, prompt, *, confidence_threshold=0.0, max_instances=0
    ):
        return list(self._detections)


def _write_fixture(tmp_path, poly):
    import cv2

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for name in ("f0_tile000.jpg", "f0_tile001.jpg"):
        cv2.imwrite(str(images_dir / name), np.zeros((300, 300, 3), dtype=np.uint8))
    flat = [float(v) for v in np.asarray(poly).reshape(-1)]
    coco = {
        "images": [
            {"id": 0, "file_name": "f0_tile000.jpg", "width": 300, "height": 300},
            {"id": 1, "file_name": "f0_tile001.jpg", "width": 300, "height": 300},
        ],
        "categories": [{"id": 1, "name": "ant"}],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1, "segmentation": [flat]},
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [flat]},
        ],
    }
    path = tmp_path / "_annotations.coco.json"
    path.write_text(json.dumps(coco))
    return images_dir, path


def test_gate_runs_end_to_end_and_scores_both_arms_on_one_prediction_cache(tmp_path):
    from hydra_suite.core.inference.semantic.base import SemanticInstance

    label = _crescent(120.0, 120.0)
    pred = _crescent(120.6, 120.4, r_out=34.2)
    images_dir, coco_path = _write_fixture(tmp_path, label)
    labeler = _FakeLabeler([SemanticInstance(polygon_px=pred, confidence=0.9)])

    out = tmp_path / "gate.json"
    cache = tmp_path / "cache.pkl"
    rc = main(
        [
            "--checkpoint-a",
            "ckpt_a",
            "--frames-dir",
            str(images_dir),
            "--coco-json",
            str(coco_path),
            "--prompt",
            "ant",
            "--reference-body-px",
            "30",
            "--seam-margin-px",
            "2",
            "--merge-iou",
            "0.5",
            "--confidences",
            "0.5",
            "--cache",
            str(cache),
            "--out",
            str(out),
        ],
        labeler_factory=lambda _c: labeler,
    )
    assert rc == 0
    result = json.loads(out.read_text())
    shipped = result["arms"]["a"]["shipped_vertex_mean"][0]
    fixed = result["arms"]["a"]["fixed_representative_point"][0]
    # Same predictions in both arms -- that is the whole design.
    assert shipped["n_predictions"] == fixed["n_predictions"] > 0
    assert shipped["n_labels"] == fixed["n_labels"] > 0
    # ... and only the scoring differs, in the measured direction.
    assert shipped["recall"] == 0.0
    assert fixed["recall"] == 1.0
    assert fixed["extras_per_tile"] < shipped["extras_per_tile"]
    # The 15.9 %-style statistic is recomputed on this run's own labels.
    geom = result["label_geometry"]["a"]
    assert geom["vertex_mean_outside"] == geom["n_labels"] > 0
    assert geom["representative_point_outside"] == 0
    assert cache.exists()


def test_score_only_reuses_the_cache_without_a_labeler(tmp_path):
    from hydra_suite.core.inference.semantic.base import SemanticInstance

    label = _crescent(120.0, 120.0)
    images_dir, coco_path = _write_fixture(tmp_path, label)
    labeler = _FakeLabeler(
        [
            SemanticInstance(
                polygon_px=_crescent(120.6, 120.4, r_out=34.2), confidence=0.9
            )
        ]
    )
    cache = tmp_path / "cache.pkl"
    base = [
        "--checkpoint-a",
        "ckpt_a",
        "--frames-dir",
        str(images_dir),
        "--coco-json",
        str(coco_path),
        "--prompt",
        "ant",
        "--reference-body-px",
        "30",
        "--seam-margin-px",
        "2",
        "--merge-iou",
        "0.5",
        "--confidences",
        "0.5",
        "--cache",
        str(cache),
    ]
    first = tmp_path / "a.json"
    main(base + ["--out", str(first)], labeler_factory=lambda _c: labeler)

    def _explode(_checkpoint):  # pragma: no cover - must never be called
        raise AssertionError("--score-only must not build a labeler")

    second = tmp_path / "b.json"
    main(base + ["--score-only", "--out", str(second)], labeler_factory=_explode)
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    assert a["arms"] == b["arms"]


def test_score_only_without_a_cache_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "--score-only",
                "--frames-dir",
                str(tmp_path),
                "--coco-json",
                str(tmp_path / "x.json"),
                "--prompt",
                "ant",
                "--reference-body-px",
                "30",
                "--seam-margin-px",
                "2",
                "--merge-iou",
                "0.5",
                "--out",
                str(tmp_path / "o.json"),
            ]
        )
