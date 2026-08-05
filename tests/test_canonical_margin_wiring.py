"""The canonical margin must be settable, and identical on both entry points."""

from pathlib import Path

from hydra_suite.trackerkit.advanced_defaults import DEFAULT_ADVANCED_CONFIG
from hydra_suite.trackerkit.cli_config import (
    TrackerCliVideoProbe,
    load_tracker_cli_session,
)


def test_margin_has_one_default_table():
    assert "canonical_margin" in DEFAULT_ADVANCED_CONFIG
    assert "reference_aspect_ratio" in DEFAULT_ADVANCED_CONFIG


def test_both_builders_share_the_table():
    src = Path(__file__).resolve().parents[1] / "src" / "hydra_suite"
    for rel in ("trackerkit/cli_config.py", "trackerkit/gui/orchestrators/config.py"):
        text = (src / rel).read_text(encoding="utf-8")
        assert "DEFAULT_ADVANCED_CONFIG" in text, rel


def test_cli_honours_a_configured_margin(tmp_path):
    session = load_tracker_cli_session(
        str(tmp_path / "clip.mp4"),
        config_data={
            "file_path": str(tmp_path / "clip.mp4"),
            "fps": 30.0,
            "canonical_margin": 1.6,
        },
        video_probe=TrackerCliVideoProbe(
            fps=30.0, total_frames=60, width=640, height=480
        ),
    )
    assert session.params["ADVANCED_CONFIG"]["canonical_margin"] == 1.6


def test_dead_key_is_gone():
    src = Path(__file__).resolve().parents[1] / "src"
    hits = [
        p
        for p in src.rglob("*.py")
        if "yolo_headtail_canonical_margin" in p.read_text(encoding="utf-8")
    ]
    assert hits == [], f"dead margin key still read in {hits}"
