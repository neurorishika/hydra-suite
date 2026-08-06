import json
from pathlib import Path

import hydra_suite.training.model_publish as mp
from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.canonical_meta import (
    read_canonical_meta,
    warn_on_geometry_mismatch,
)
from hydra_suite.training.contracts import TrainingRole


def test_stamp_round_trips(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    (tmp_path / "m.pt.canonical_meta.json").write_text(
        json.dumps(g.to_dict()), encoding="utf-8"
    )
    assert read_canonical_meta(model) == g


def test_unstamped_model_is_none(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    assert read_canonical_meta(model) is None
    assert (
        warn_on_geometry_mismatch(
            model, CanonicalGeometry.from_reference(20.0, 2.0, 1.3)
        )
        is None
    )


def test_mismatch_is_reported(tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"x")
    trained = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    (tmp_path / "m.pt.canonical_meta.json").write_text(
        json.dumps(trained.to_dict()), encoding="utf-8"
    )
    session = CanonicalGeometry.from_reference(20.0, 2.44, 2.0)
    msg = warn_on_geometry_mismatch(model, session)
    assert msg is not None and "margin" in msg


def test_publish_trained_model_stamps_and_round_trips(tmp_path, monkeypatch):
    """End-to-end: a real publish call site (canonical_geometry=<real geometry>)

    writes the sidecar AND the registry mirror, and read_canonical_meta reads
    the published artifact's stamp back byte-for-byte. A unit test of the
    reader alone cannot catch a caller that never supplies the geometry --
    this closes that loop the way tests/test_model_publish_slice_geometry.py
    closes it for slice_geometry.
    """
    monkeypatch.setattr(mp, "get_models_root", lambda: tmp_path)

    src = tmp_path / "weights.pt"
    src.write_bytes(b"fake-weights")
    geom = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)

    key, stored = mp.publish_trained_model(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        artifact_path=str(src),
        size="n",
        species="ant",
        model_info="cnn_identity",
        trained_from_run_id="r1",
        dataset_fingerprint="fp",
        base_model="",
        canonical_geometry=geom,
    )
    stored_path = Path(stored)

    assert read_canonical_meta(stored_path) == geom

    reg = mp.load_model_registry()
    assert reg["entries"][key]["canonical_geometry"] == geom.to_dict()
    sidecar_name = reg["entries"][key]["canonical_meta_sidecar"]
    assert (stored_path.parent / sidecar_name).exists()


def test_publish_trained_model_without_geometry_stays_unstamped(tmp_path, monkeypatch):
    """The default (no canonical_geometry passed) must not fabricate a stamp."""
    monkeypatch.setattr(mp, "get_models_root", lambda: tmp_path)

    src = tmp_path / "weights2.pt"
    src.write_bytes(b"x")

    key, stored = mp.publish_trained_model(
        role=TrainingRole.OBB_DIRECT,
        artifact_path=str(src),
        size="s",
        species="ant",
        model_info="obb",
        trained_from_run_id="r2",
        dataset_fingerprint="fp",
        base_model="yolo26s-obb.pt",
    )
    stored_path = Path(stored)

    assert read_canonical_meta(stored_path) is None
    assert not stored_path.with_suffix(
        stored_path.suffix + ".canonical_meta.json"
    ).exists()
    reg = mp.load_model_registry()
    assert "canonical_geometry" not in reg["entries"][key]
    assert "canonical_meta_sidecar" not in reg["entries"][key]
