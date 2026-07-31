import hashlib

import pytest

from hydra_suite.posekit.core.vitpose_checkpoints import (
    CATALOG,
    fetch_pinned,
    resolve_checkpoint,
)


def test_resolve_passthrough_local_path(tmp_path):
    f = tmp_path / "mine.pth"
    f.write_bytes(b"abc")
    assert resolve_checkpoint(str(f), tmp_path / "cache") == f


def test_fetch_pinned_verifies_sha(tmp_path, monkeypatch):
    payload = b"weights-bytes"
    good = hashlib.sha256(payload).hexdigest()
    src = tmp_path / "hf_cache_file.pth"
    src.write_bytes(payload)

    monkeypatch.setattr(
        "hydra_suite.posekit.core.vitpose_checkpoints.hf_hub_download",
        lambda repo_id, filename: str(src),
    )
    dest = tmp_path / "w.pth"
    out = fetch_pinned("repo/x", "w.pth", good, dest)
    assert out.read_bytes() == payload
    # wrong hash must raise and not leave a file behind
    with pytest.raises(ValueError):
        fetch_pinned("repo/x", "w.pth", "0" * 64, tmp_path / "bad.pth")
    assert not (tmp_path / "bad.pth").exists()


def test_catalog_has_coco_b_entry():
    assert "vitpose-b-coco" in CATALOG
    e = CATALOG["vitpose-b-coco"]
    assert e.variant == "B" and e.num_keypoints == 17
    for entry in CATALOG.values():
        assert len(entry.sha256) == 64 and entry.repo_id and entry.filename
