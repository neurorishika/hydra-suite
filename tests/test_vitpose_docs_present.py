from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(p):
    return (ROOT / p).read_text(encoding="utf-8").lower()


def test_runtime_integration_lists_vitpose_pose_stage():
    assert "vitpose_pose" in _read("docs/developer-guide/runtime-integration.md")


def test_posekit_guide_documents_vitpose():
    assert "vitpose" in _read("docs/user-guide/posekit.md")


def test_compute_runtimes_lists_vitpose():
    assert "vitpose" in _read("docs/user-guide/compute-runtimes.md")


def test_ui_components_posekit_lists_vitpose():
    assert "vitpose" in _read("docs/reference/ui-components-posekit.md")


def test_readme_mentions_vitpose():
    assert "vitpose" in _read("README.md")


def test_claude_md_mentions_vitpose():
    assert "vitpose" in _read("CLAUDE.md")
