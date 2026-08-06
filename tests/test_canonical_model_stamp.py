import json

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.inference.canonical_meta import (
    read_canonical_meta,
    warn_on_geometry_mismatch,
)


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
