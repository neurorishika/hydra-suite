from hydra_suite.core.inference.config import PoseViTPoseConfig


def test_auto_export_defaults_true_and_roundtrips():
    c = PoseViTPoseConfig(model_path="m.pt")
    assert c.auto_export is True
    from dataclasses import asdict

    d = asdict(c)
    assert d["auto_export"] is True
    c2 = PoseViTPoseConfig(**d)
    assert c2.auto_export is True
