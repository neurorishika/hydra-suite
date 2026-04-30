"""Tests for cross-root discovery of ClassKit classifier artifacts."""

from __future__ import annotations


def test_enumerate_discovers_classification_identity_roots(tmp_path, monkeypatch):
    from hydra_suite.training import model_publish

    roots = {
        "classification/identity": ["model_a.pth"],
        "classification/orientation": ["model_b.pth"],
        "tiny-classify/some_scheme": ["t.pth"],
        "custom-classify/multihead/scheme": ["c.pth"],
        "YOLO-classify/scheme": ["y.pt"],
        "YOLO-classify/multihead/scheme": ["m.multihead.json"],
    }
    models_root = tmp_path / "models"
    for rel, files in roots.items():
        (models_root / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            (models_root / rel / name).write_bytes(b"x")
    monkeypatch.setattr(model_publish, "get_models_root", lambda: models_root)

    entries = list(
        model_publish.enumerate_classifier_artifacts(roles=("cnn_identity",))
    )
    # Every directory above (except orientation) is valid for cnn_identity discovery.
    paths = {e["path"] for e in entries}
    assert "classification/identity/model_a.pth" in paths
    assert "tiny-classify/some_scheme/t.pth" in paths
    assert "custom-classify/multihead/scheme/c.pth" in paths
    assert "YOLO-classify/scheme/y.pt" in paths
    assert "YOLO-classify/multihead/scheme/m.multihead.json" in paths
    # Orientation entries are excluded from cnn_identity enumeration.
    assert not any("orientation" in p for p in paths)


def test_enumerate_filters_to_headtail_roots(tmp_path, monkeypatch):
    from hydra_suite.training import model_publish

    models_root = tmp_path / "models"
    (models_root / "classification/orientation").mkdir(parents=True)
    (models_root / "classification/orientation/ht.pth").write_bytes(b"x")
    (models_root / "tiny-classify/scheme").mkdir(parents=True)
    (models_root / "tiny-classify/scheme/t.pth").write_bytes(b"x")
    monkeypatch.setattr(model_publish, "get_models_root", lambda: models_root)

    entries = list(model_publish.enumerate_classifier_artifacts(roles=("head_tail",)))
    paths = {e["path"] for e in entries}
    # Both orientation-canonical + classkit publish roots are enumerated for head-tail;
    # the caller (panel) applies further validation via ClassifierBackend metadata.
    assert "classification/orientation/ht.pth" in paths
    assert "tiny-classify/scheme/t.pth" in paths


def test_enumerate_skips_factor_artifacts_when_manifest_exists(tmp_path, monkeypatch):
    import json

    from hydra_suite.training import model_publish

    models_root = tmp_path / "models"
    bundle_root = models_root / "custom-classify" / "multihead" / "scheme"
    bundle_root.mkdir(parents=True)
    (bundle_root / "factor_a.pth").write_bytes(b"a")
    (bundle_root / "factor_b.pth").write_bytes(b"b")
    (bundle_root / "bundle.multihead.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "classifier_multihead_bundle",
                "factor_names": ["a", "b"],
                "factor_models": [
                    {"factor": "a", "path": "factor_a.pth", "class_names": ["x"]},
                    {"factor": "b", "path": "factor_b.pth", "class_names": ["y"]},
                ],
                "input_size": [64, 64],
                "monochrome": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_publish, "get_models_root", lambda: models_root)

    entries = list(
        model_publish.enumerate_classifier_artifacts(roles=("cnn_identity",))
    )
    paths = {entry["path"] for entry in entries}
    assert "custom-classify/multihead/scheme/bundle.multihead.json" in paths
    assert "custom-classify/multihead/scheme/factor_a.pth" not in paths
    assert "custom-classify/multihead/scheme/factor_b.pth" not in paths
