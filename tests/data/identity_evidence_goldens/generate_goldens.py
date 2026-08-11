"""One-shot generation script for the committed identity-evidence goldens.

GENERATION HELPER -- NOT RUN BY PYTEST. This script was run ONCE, while
``IdentityEvidenceEmitter`` still existed, to freeze its output for the two
parity-test scenario families into the ``.npz`` files committed alongside it
in this directory. The parity tests (``tests/identity/test_evidence_builder_parity.py``,
``tests/identity/test_evidence_phase_basis_parity.py``) now load those
frozen goldens instead of instantiating the emitter, which has since been
deleted (Identity Phase 7 Task 4: clean-break retirement).

Kept for reproducibility/documentation only. Re-running it requires
resurrecting ``IdentityEvidenceEmitter`` (e.g. via ``git show`` at the commit
before its deletion) -- it will NOT run against the current source tree.

Usage (historical):
    PYTHONPATH=src python tests/data/identity_evidence_goldens/generate_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hydra_suite.core.individual.classification.cnn import ClassPrediction
from hydra_suite.core.individual.identity.calibration import CalibrationModel
from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.individual.identity.resolve import resolve_catalog_spec
from hydra_suite.core.tracking.identity.evidence_emitter import IdentityEvidenceEmitter

OUT_DIR = Path(__file__).parent


def _dummy_predictions(det_ids, n_factors):
    preds = []
    for slot, _det_id in enumerate(det_ids):
        preds.append(
            ClassPrediction(
                det_index=slot,
                factor_names=tuple(f"factor_{i}" for i in range(n_factors)),
                class_names=tuple(None for _ in range(n_factors)),
                confidences=tuple(0.0 for _ in range(n_factors)),
            )
        )
    return preds


def _remap_verbatim(log_probs, source_labels, identity_catalog):
    if identity_catalog is None:
        return np.asarray(log_probs, dtype=np.float64)
    arr = np.asarray(log_probs, dtype=np.float64)
    if source_labels is None:
        if len(arr) == identity_catalog.size:
            out = arr.copy()
            out -= np.logaddexp.reduce(out)
            return out
        return identity_catalog.known_uniform_log_prior()

    labels = tuple(str(label) for label in source_labels)
    if len(labels) != len(arr):
        return identity_catalog.known_uniform_log_prior()

    probs = np.exp(arr - np.max(arr))
    probs /= np.clip(probs.sum(), 1e-300, None)
    remapped = np.full(identity_catalog.size, 1e-300, dtype=np.float64)
    for src_idx, label in enumerate(labels):
        if not identity_catalog.contains(label):
            continue
        remapped[identity_catalog.index_of(label)] += float(probs[src_idx])
    remapped /= np.clip(remapped.sum(), 1e-300, None)
    return np.log(np.clip(remapped, 1e-300, None))


# ---------------------------------------------------------------------------
# test_evidence_builder_parity.py goldens
# ---------------------------------------------------------------------------


def _gen_builder_case(
    name: str,
    labels,
    catalog_known_labels,
    calibration,
    probs,
    det_ids,
    frame_idx: int,
    tmp_path: Path,
    source_name: str = "cnn0",
):
    catalog = IdentityCatalog.from_labels(catalog_known_labels)
    emitter = IdentityEvidenceEmitter(
        cache_path=str(tmp_path / f"{name}.npz"),
        source_name=source_name,
        class_labels_per_factor=labels,
        runtime_signature="cpu",
        calibration_signature="calsig" if calibration is not None else "",
        calibration=calibration,
    )
    assert emitter.catalog_labels == catalog.labels

    evidences = emitter.build_frame_evidences(
        frame_idx,
        _dummy_predictions(det_ids, len(labels)),
        posteriors=probs,
        detection_ids=det_ids,
    )

    log_probs = np.stack([e.log_probs for e in evidences])
    observed_mask = np.stack([e.observed_mask for e in evidences])
    detection_id = np.array([e.detection_id for e in evidences], dtype=np.int64)
    source_names = np.array([e.source_name for e in evidences])
    calibration_signatures = np.array([e.calibration_signature for e in evidences])

    out_path = OUT_DIR / f"builder_parity_{name}.npz"
    np.savez(
        out_path,
        log_probs=log_probs,
        observed_mask=observed_mask,
        detection_id=detection_id,
        source_names=source_names,
        calibration_signatures=calibration_signatures,
    )
    print(f"wrote {out_path}")


def generate_builder_parity_goldens(tmp_path: Path) -> None:
    _gen_builder_case(
        "single_factor",
        [["white", "black", "brown"]],
        ["white", "black", "brown"],
        None,
        [
            [np.array([0.7, 0.2, 0.1])],
            [np.array([0.1, 0.1, 0.8])],
        ],
        [10, 11],
        5,
        tmp_path,
    )

    _gen_builder_case(
        "multifactor_with_underscore",
        [["dark_red", "blue"], ["big", "small"]],
        ["dark_red_big", "dark_red_small", "blue_big", "blue_small"],
        None,
        [
            [np.array([0.7, 0.3]), np.array([0.6, 0.4])],
            [np.array([0.2, 0.8]), np.array([0.9, 0.1])],
        ],
        [20, 21],
        7,
        tmp_path,
    )

    _gen_builder_case(
        "with_calibration_temperature",
        [["dark_red", "blue"], ["big", "small"]],
        ["dark_red_big", "dark_red_small", "blue_big", "blue_small"],
        CalibrationModel(temperature=2.5),
        [
            [np.array([0.55, 0.45]), np.array([0.51, 0.49])],
            [np.array([0.9, 0.1]), np.array([0.3, 0.7])],
        ],
        [30, 31],
        9,
        tmp_path,
    )

    _gen_builder_case(
        "with_gapped_empty_factor",
        [["a", "b"], [], ["c", "d"]],
        ["a_c", "a_d", "b_c", "b_d"],
        None,
        [
            [np.array([0.6, 0.4]), np.array([0.3, 0.7])],
            [np.array([0.2, 0.8]), np.array([0.9, 0.1])],
        ],
        [40, 41],
        11,
        tmp_path,
    )

    _gen_builder_case(
        "with_colliding_composite_labels",
        [["a", "a_b"], ["b_c", "c"]],
        ["a_b_c", "a_c", "a_b_b_c"],
        None,
        [
            [np.array([0.6, 0.4]), np.array([0.7, 0.3])],
            [np.array([0.3, 0.7]), np.array([0.2, 0.8])],
        ],
        [50, 51],
        13,
        tmp_path,
    )


# ---------------------------------------------------------------------------
# test_evidence_phase_basis_parity.py goldens
# ---------------------------------------------------------------------------


def _old_global_log_probs(
    identity_catalog,
    source_name,
    class_labels_per_factor,
    raw_probs,
    tmp_path,
):
    emitter = IdentityEvidenceEmitter(
        cache_path=tmp_path / f"{source_name}_old.npz",
        source_name=source_name,
        class_labels_per_factor=class_labels_per_factor,
    )
    evidences = emitter.build_frame_evidences(
        frame_idx=0,
        predictions=_dummy_predictions([0], len(class_labels_per_factor)),
        posteriors=[raw_probs],
    )
    assert len(evidences) == 1
    return _remap_verbatim(
        evidences[0].log_probs, emitter.catalog_labels, identity_catalog
    )


def generate_phase_basis_goldens(tmp_path: Path) -> None:
    # Case 1: CNN + AprilTag, CNN labels disjoint from global catalog.
    cnn_classifiers = [
        {
            "label": "cnn_color",
            "unique_identifier": False,
            "class_names_per_factor": [["white", "black", "brown"]],
        }
    ]
    tag_identity_labels = ["antA", "antB"]
    catalog_spec = resolve_catalog_spec(cnn_classifiers, tag_identity_labels)
    identity_catalog = IdentityCatalog.from_spec(catalog_spec)
    assert not ({"white", "black", "brown"} & set(identity_catalog.labels))

    raw_probs = [np.array([0.7, 0.2, 0.1], dtype=np.float32)]
    old = _old_global_log_probs(
        identity_catalog,
        "cnn_color",
        [["white", "black", "brown"]],
        raw_probs,
        tmp_path,
    )
    out_path = OUT_DIR / "phase_basis_parity_cnn_apriltag.npz"
    np.savez(
        out_path, old_log_probs=old, catalog_labels=np.array(identity_catalog.labels)
    )
    print(f"wrote {out_path}")

    # Case 2: two CNN phases, each a proper subset of the union catalog.
    cnn_classifiers = [
        {
            "label": "cnn_p",
            "unique_identifier": True,
            "class_names_per_factor": [["p1", "p2"]],
        },
        {
            "label": "cnn_q",
            "unique_identifier": True,
            "class_names_per_factor": [["q1", "q2", "q3"]],
        },
    ]
    catalog_spec = resolve_catalog_spec(cnn_classifiers, [])
    identity_catalog = IdentityCatalog.from_spec(catalog_spec)
    assert set(identity_catalog.labels) == {"unknown", "p1", "p2", "q1", "q2", "q3"}

    raw_probs_p = [np.array([0.6, 0.4], dtype=np.float32)]
    raw_probs_q = [np.array([0.5, 0.3, 0.2], dtype=np.float32)]

    old_p = _old_global_log_probs(
        identity_catalog, "cnn_p", [["p1", "p2"]], raw_probs_p, tmp_path
    )
    old_q = _old_global_log_probs(
        identity_catalog, "cnn_q", [["q1", "q2", "q3"]], raw_probs_q, tmp_path
    )
    out_path = OUT_DIR / "phase_basis_parity_two_cnn_phase.npz"
    np.savez(
        out_path,
        old_log_probs_p=old_p,
        old_log_probs_q=old_q,
        catalog_labels=np.array(identity_catalog.labels),
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        generate_builder_parity_goldens(tmp_path)
        generate_phase_basis_goldens(tmp_path)
