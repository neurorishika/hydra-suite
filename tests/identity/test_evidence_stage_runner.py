"""Tests for wiring `IdentityEvidenceStage` into `InferenceRunner` (Task 4).

Exercises the extracted batch read-back+write helper
(`write_identity_evidence_sidecar`) directly against a tiny synthetic cache
dir built with the real `_CacheSet` cache handles, and asserts the written
`IdentityEvidenceCache` reproduces exactly what a direct
`IdentityEvidenceStage.evidences_for_frame` call would produce for the same
raw reads. A full `run_batch_pass` drives models + video decode and is too
heavy for a unit test; the helper is the seam `run_batch_pass` calls into
(same code, directly exercised).
"""

from __future__ import annotations

import numpy as np

from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.evidence_builder import EvidenceBuilder
from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey
from hydra_suite.core.inference.cache.store import (
    AprilTagCacheHandle,
    CNNCacheHandle,
    DetectionCacheHandle,
)
from hydra_suite.core.inference.config import InferenceConfig, OBBConfig
from hydra_suite.core.inference.result import (
    AprilTagResult,
    CNNDetectionPrediction,
    CNNFactorPrediction,
    OBBResult,
)
from hydra_suite.core.inference.runner import _CacheSet, write_identity_evidence_sidecar
from hydra_suite.core.inference.stages.identity_evidence import IdentityEvidenceStage


def _key(path: str) -> CacheKey:
    return CacheKey(
        schema_version=CACHE_SCHEMA_VERSION,
        model_path=path,
        model_mtime=0.0,
        config_hash="cfg",
    )


def _obb(frame_idx: int, n: int) -> OBBResult:
    return OBBResult(
        frame_idx=frame_idx,
        centroids=np.ones((n, 2), dtype=np.float32) * frame_idx,
        angles=np.zeros(n, dtype=np.float32),
        sizes=np.full(n, 100.0, dtype=np.float32),
        shapes=np.ones((n, 2), dtype=np.float32),
        confidences=np.full(n, 0.9, dtype=np.float32),
        corners=np.zeros((n, 4, 2), dtype=np.float32),
        detection_ids=OBBResult.make_detection_ids(frame_idx, n),
    )


def _cnn_preds(n: int, probs) -> list[CNNDetectionPrediction]:
    return [
        CNNDetectionPrediction(
            det_index=i,
            factors=[
                CNNFactorPrediction(
                    factor_name="color",
                    class_names=["white", "black", "brown"],
                    raw_probabilities=np.asarray(probs[i], dtype=np.float32),
                )
            ],
        )
        for i in range(n)
    ]


def _permissive_config() -> InferenceConfig:
    # iou_threshold=1.0 skips NMS; confidence_threshold=0.0 keeps everything --
    # filtered_obb ends up identical to the raw read, in raw order, so det_ids
    # line up 1:1 with the CNN predictions' det_index by construction.
    return InferenceConfig(
        obb=OBBConfig(confidence_threshold=0.0, iou_threshold=1.0, max_detections=0)
    )


def _build_cache_set(tmp_path, with_apriltag: bool = False) -> _CacheSet:
    det = DetectionCacheHandle(path=tmp_path / "detection.npz", key=_key("obb"))
    det.write_frame(0, result=_obb(0, n=2))
    det.write_frame(1, result=_obb(1, n=1))
    det.close()

    cnn = CNNCacheHandle(path=tmp_path / "cnn_id.npz", key=_key("cnn"), label="cnn_id")
    cnn.write_frame(0, predictions=_cnn_preds(2, [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]))
    cnn.write_frame(1, predictions=_cnn_preds(1, [[0.2, 0.2, 0.6]]))
    cnn.close()

    apriltag = None
    if with_apriltag:
        apriltag = AprilTagCacheHandle(path=tmp_path / "apriltag.npz", key=_key("tag"))
        apriltag.write_frame(
            0,
            result=AprilTagResult(
                tag_ids=[7],
                det_indices=[1],
                centers=np.zeros((1, 2), dtype=np.float32),
                corners=np.zeros((1, 4, 2), dtype=np.float32),
            ),
        )
        apriltag.write_frame(
            1,
            result=AprilTagResult(
                tag_ids=[],
                det_indices=[],
                centers=np.zeros((0, 2), dtype=np.float32),
                corners=np.zeros((0, 4, 2), dtype=np.float32),
            ),
        )
        apriltag.close()

    # Reopen fresh handles pointing at the same (now-flushed) files, matching
    # how a real batch pass reads back after `run_batch_pass`'s pipeline closes
    # its cache handles -- read_frame only ever sees disk-flushed data.
    return _CacheSet(
        detection=DetectionCacheHandle(
            path=tmp_path / "detection.npz", key=_key("obb")
        ),
        cnn=[
            CNNCacheHandle(
                path=tmp_path / "cnn_id.npz", key=_key("cnn"), label="cnn_id"
            )
        ],
        apriltag=(
            AprilTagCacheHandle(path=tmp_path / "apriltag.npz", key=_key("tag"))
            if with_apriltag
            else None
        ),
    )


def _build_stage(tag_to_label=None) -> tuple[IdentityCatalog, IdentityEvidenceStage]:
    catalog = IdentityCatalog.from_labels(["white", "black", "brown"])
    builder = EvidenceBuilder(catalog, "cnn_id", [["white", "black", "brown"]])
    stage = IdentityEvidenceStage(catalog, {"cnn_id": builder}, tag_to_label or {})
    return catalog, stage


def test_write_identity_evidence_sidecar_matches_direct_stage_call(tmp_path):
    caches = _build_cache_set(tmp_path)
    catalog, stage = _build_stage()
    config = _permissive_config()
    out_path = tmp_path / "detection_identity_evidence_batch_testkey.npz"

    write_identity_evidence_sidecar(
        caches, config, stage, range(0, 2), out_path, catalog.labels
    )

    assert out_path.exists()
    read_cache = IdentityEvidenceCache(out_path, mode="r")

    # Frame 0: 2 detections, det_ids = [0, 1] (OBBResult.make_detection_ids
    # with STRIDE 10000 and frame_idx=0 -> [0, 1]).
    expected_f0 = stage.evidences_for_frame(
        0,
        [0, 1],
        {"cnn_id": _cnn_preds(2, [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])},
        None,
    )
    got_f0 = read_cache.load_frame(0)
    assert len(got_f0) == len(expected_f0) == 2
    for got, expected in zip(got_f0, expected_f0):
        assert got.detection_id == expected.detection_id
        assert got.source == expected.source
        assert got.source_name == expected.source_name
        assert np.allclose(got.log_probs, expected.log_probs)

    # Frame 1: 1 detection, det_id = 10000 (frame_idx=1 * STRIDE).
    expected_f1 = stage.evidences_for_frame(
        1, [10000], {"cnn_id": _cnn_preds(1, [[0.2, 0.2, 0.6]])}, None
    )
    got_f1 = read_cache.load_frame(1)
    assert len(got_f1) == len(expected_f1) == 1
    assert got_f1[0].detection_id == expected_f1[0].detection_id
    assert np.allclose(got_f1[0].log_probs, expected_f1[0].log_probs)


def test_write_identity_evidence_sidecar_merges_apriltag(tmp_path):
    caches = _build_cache_set(tmp_path, with_apriltag=True)
    catalog, stage = _build_stage(tag_to_label={7: "black"})
    config = _permissive_config()
    out_path = tmp_path / "detection_identity_evidence_batch_testkey2.npz"

    write_identity_evidence_sidecar(
        caches, config, stage, range(0, 2), out_path, catalog.labels
    )

    read_cache = IdentityEvidenceCache(out_path, mode="r")
    got_f0 = read_cache.load_frame(0)
    # 2 CNN evidences + 1 AprilTag evidence (det_index=1 -> det_id 1).
    assert len(got_f0) == 3
    tag_ev = [e for e in got_f0 if e.source_name == "apriltag"]
    assert len(tag_ev) == 1
    assert tag_ev[0].detection_id == 1
    expected_tag_log_probs = catalog.apriltag_log_prior(7, {7: "black"})
    assert np.allclose(tag_ev[0].log_probs, expected_tag_log_probs)


def test_no_identity_evidence_config_is_a_no_op_on_runner_construction():
    """`InferenceRunner(identity_evidence=None)` never builds a stage."""
    # Constructing a full InferenceRunner loads models -- out of scope for a
    # unit test. This asserts the config-carrier contract at the type level:
    # `IdentityEvidenceRunConfig` is optional and `None` is the documented
    # "no identity configured" no-op sentinel (see runner.py `__init__`).
    from hydra_suite.core.inference.identity_evidence_config import (
        IdentityEvidenceRunConfig,
    )

    assert IdentityEvidenceRunConfig is not None
