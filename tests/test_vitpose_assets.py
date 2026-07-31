import hashlib
from pathlib import Path

import pytest

from tools.vitpose.fetch_assets import ASSETS, AssetIntegrityError, verify


def test_verify_accepts_matching_sha256(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"hello vitpose")
    digest = hashlib.sha256(b"hello vitpose").hexdigest()
    verify(p, digest)  # must not raise


def test_verify_rejects_mismatched_sha256(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"tampered")
    with pytest.raises(AssetIntegrityError) as exc:
        verify(p, "0" * 64)
    assert "sha256 mismatch" in str(exc.value).lower()


def test_detections_asset_pins_the_real_file_not_the_dummy():
    """The LiteHrnet copy on GitHub is a 250KB dummy (1000 boxes, all score 0.99).
    The genuine file is 16,383,781 bytes. Pinning size+sha is what separates them."""
    a = ASSETS["coco_val2017_person_detections"]
    assert a.size == 16_383_781
    assert a.sha256 == (
        "53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59"
    )


def test_fetch_refuses_unpinned_asset_by_default(tmp_path: Path, monkeypatch):
    """An unpinned asset must fail loudly rather than silently skip
    verification -- silently-unverified is the failure mode this module exists
    to prevent."""
    from tools.vitpose import fetch_assets

    monkeypatch.setitem(
        fetch_assets.ASSETS,
        "_unpinned",
        fetch_assets.Asset("hf", "repo", "f.bin", "", 0),
    )
    with pytest.raises(AssetIntegrityError, match="no pinned sha256"):
        fetch_assets.fetch("_unpinned", tmp_path)
