"""R9/R10: the identity sidecar is keyed on N (+ replay filters) and rebuilt on
cache reuse from the per-animal caches; candidate-filter replays read caches
under their WRITTEN filters and fail loudly only when they need a missing row.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hydra_suite.core.individual.identity.cache import IdentityEvidenceCache
from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec
from hydra_suite.core.inference.config import (
    CNNConfig,
    HeadTailConfig,
    InferenceConfig,
    OBBConfig,
    OBBDirectConfig,
)
from hydra_suite.core.inference.downstream_select import DownstreamCacheError
from hydra_suite.core.inference.identity_evidence_config import (
    IdentityEvidenceCNNPhaseConfig,
    IdentityEvidenceRunConfig,
)
from hydra_suite.core.inference.result import (
    CNNDetectionPrediction,
    CNNFactorPrediction,
    CNNResult,
)
from tests.test_pipeline_superset import _fake_ht, _obb

_NAMES = ["ant1", "ant2", "ant3"]
_FRAMES = [0, 1]
_N_DETS = 30
_SMALL = 1  # raw row 1 is size-filtered out -> non-contiguous raw indices


def _cfg(n, conf=0.0, headtail=False):
    return InferenceConfig(
        obb=OBBConfig(
            mode="direct",
            direct=OBBDirectConfig(model_path="/m.pt"),
            max_detections=n,
            confidence_threshold=conf,
            iou_threshold=1.0,
            min_object_size=5.0,
        ),
        headtail=HeadTailConfig(model_path="/ht.pt") if headtail else None,
        cnn_phases=(
            [] if headtail else [CNNConfig(label="colortag", model_path="/c.pt")]
        ),
    )


def _identity():
    spec = resolve_catalog_spec(
        [
            {
                "label": "colortag",
                "unique_identifier": True,
                "class_names_per_factor": [_NAMES],
            }
        ],
        [],
    )
    return IdentityEvidenceRunConfig(
        catalog_spec=spec,
        cnn_phases=(
            IdentityEvidenceCNNPhaseConfig(
                label="colortag", class_names_per_factor=[_NAMES]
            ),
        ),
    )


def _fake_cnn(frames, obbs, model, cfg, runtime, geometry, headtail_by_frame=None):
    # Probabilities are a function of the crop's x, so evidence identifies the
    # detection it was computed for.
    out = {}
    for o in obbs:
        preds = []
        for i in range(o.num_detections):
            x = float(o.centroids[i, 0])
            p = np.array([1.0 + (x % 7), 1.0 + (x % 5), 1.0 + (x % 3)], np.float32)
            preds.append(
                CNNDetectionPrediction(
                    det_index=i,
                    factors=[CNNFactorPrediction("factor_0", _NAMES, p / p.sum())],
                )
            )
        out[o.frame_idx] = CNNResult(label=cfg.label, predictions=preds)
    return out


def _models(cnn=True, headtail=False):
    return MagicMock(
        obb=MagicMock(),
        headtail=MagicMock() if headtail else None,
        cnn=[MagicMock()] if cnn else [],
        pose=None,
        apriltag=None,
        bgsub=None,
    )


def _build(tmp, cfg, identity=None, models=None):
    """Write caches for frames 0..1 exactly as a batch pass does."""
    from hydra_suite.core.inference.runner import InferenceRunner, _open_caches

    with (
        patch("hydra_suite.core.inference.runner._load_all_models") as ml,
        patch(
            "hydra_suite.core.inference.pipeline.run_obb",
            side_effect=lambda frames, *a, **k: [
                _obb(i, _N_DETS, small=_SMALL) for i in range(len(frames))
            ],
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_cnn_batch", side_effect=_fake_cnn
        ),
        patch(
            "hydra_suite.core.inference.pipeline.run_headtail_batch",
            side_effect=_fake_ht,
        ),
    ):
        ml.return_value = models or _models()
        runner = InferenceRunner(cfg, cache_dir=tmp, identity_evidence=identity)
        caches = _open_caches(cfg, tmp)
        runner._run_batch([np.zeros((64, 1600, 3), np.uint8)] * 2, _FRAMES, caches)
        caches.close()
        if identity is not None:
            runner._write_identity_evidence_batch(_FRAMES[0], _FRAMES[-1])
    return runner


def _replay_runner(tmp, cfg, identity=None, cache_filter_hash=None):
    from hydra_suite.core.inference.runner import InferenceRunner

    with patch("hydra_suite.core.inference.runner._load_all_models") as ml:
        ml.return_value = _models(cnn=False)
        return InferenceRunner(
            cfg,
            cache_dir=tmp,
            cache_only=True,
            identity_evidence=identity,
            cache_filter_hash=cache_filter_hash,
        )


def _evidence(path):
    cache = IdentityEvidenceCache(path, mode="r")
    return {
        f: sorted(
            (e.detection_id, e.source_name, tuple(np.round(e.log_probs, 6)))
            for e in cache.load_frame(f)
        )
        for f in _FRAMES
    }


# --- R9 (a): sidecar key ----------------------------------------------------


def test_sidecar_key_follows_n_and_filters(tmp_path):
    ident = _identity()

    def path(cfg):
        return _replay_runner(tmp_path, cfg, ident).identity_evidence_sidecar_path(
            "batch"
        )

    assert path(_cfg(3)) != path(_cfg(5))
    assert path(_cfg(3)) == path(_cfg(3))
    assert path(_cfg(3, conf=0.1)) != path(_cfg(3, conf=0.2))


# --- R9 (b): rebuild on reuse -----------------------------------------------


def test_reuse_at_new_n_rebuilds_sidecar_without_models(tmp_path):
    ident = _identity()
    built = tmp_path / "n10"
    _build(built, _cfg(10), ident)
    fresh = tmp_path / "fresh3"
    fresh_runner = _build(fresh, _cfg(3), ident)
    want = _evidence(fresh_runner.identity_evidence_sidecar_path("batch"))

    runner = _replay_runner(built, _cfg(3), ident)
    assert runner.caches_all_valid()
    path = runner.identity_evidence_sidecar_path("batch")
    assert not path.exists()  # written at N=10 -> N=3 key absent
    with (
        patch("hydra_suite.core.inference.pipeline.run_cnn_batch") as cnn,
        patch("hydra_suite.core.inference.runner.run_cnn") as cnn_rt,
    ):
        assert runner.ensure_identity_evidence_sidecar(0, 1) == path
        cnn.assert_not_called()
        cnn_rt.assert_not_called()
    assert path.exists()
    got = _evidence(path)
    assert got == want
    # Final N=3 set is raw [0, 2, 3] (raw 1 is size-filtered): evidence is for
    # those stable ids, so raw-vs-positional lookup is distinguishable.
    assert sorted({d for d, _s, _p in got[0]}) == [0, 2, 3]

    # Present -> no rebuild (and an explicit out_path is honoured).
    out = tmp_path / "elsewhere" / path.name
    assert runner.ensure_identity_evidence_sidecar(0, 1, out_path=out) == out
    assert _evidence(out) == want


# --- R10: candidate filters vs. the written superset ------------------------


def test_optimizer_detection_replay_at_looser_conf_needs_no_per_animal_cache(
    tmp_path,
):
    """The optimizer proposal loop reads only the detection cache: replaying at a
    looser confidence than the caches were written with never touches the
    per-animal caches, so it cannot raise."""
    from hydra_suite.core.inference.runner import _open_caches
    from hydra_suite.core.tracking.optimization.optimizer import (
        _filter_cached_detections,
    )

    _build(tmp_path, _cfg(10, conf=0.7, headtail=True), models=_models(False, True))
    loose = _cfg(10, conf=0.5, headtail=True)
    caches = _open_caches(loose, tmp_path, read_only=True)
    assert caches.detection.is_valid()
    det_filter = MagicMock(inference_config=loose)
    with patch(
        "hydra_suite.core.inference.runner._load_headtail_for_indices"
    ) as ht_load:
        out = _filter_cached_detections(det_filter, caches.detection, 0, None)
        ht_load.assert_not_called()
    assert out is not None


def _headtail_replay(tmp_path, conf):
    from hydra_suite.core.inference.cache.keys import replay_filter_hash

    # N=20 so the candidate final set reaches below the written conf (0.7).
    written = _cfg(20, conf=0.7, headtail=True)
    _build(tmp_path, written, models=_models(False, True))
    return _replay_runner(
        tmp_path,
        _cfg(20, conf=conf, headtail=True),
        cache_filter_hash=replay_filter_hash(written, None),
    )


def test_candidate_replay_stricter_than_written_filters_succeeds(tmp_path):
    runner = _headtail_replay(tmp_path, conf=0.8)
    assert runner.caches_all_valid()
    fr = runner.load_frame(0)
    assert fr.headtail.heading_hints.tolist() == [
        float(fr.obb.centroids[i, 0]) for i in range(fr.obb.num_detections)
    ]


def test_candidate_replay_looser_than_written_filters_raises_clearly(tmp_path):
    runner = _headtail_replay(tmp_path, conf=0.5)
    assert runner.caches_all_valid()
    with pytest.raises(DownstreamCacheError, match="recomputing for these filter"):
        runner.load_frame(0)


def test_cache_filter_hash_refused_for_writing_runner(tmp_path):
    from hydra_suite.core.inference.runner import InferenceRunner

    with patch("hydra_suite.core.inference.runner._load_all_models"):
        with pytest.raises(ValueError):
            InferenceRunner(_cfg(3), cache_dir=tmp_path, cache_filter_hash="x")
