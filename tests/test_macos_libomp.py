"""libomp de-duplication repairs links a previous run left dangling."""

from __future__ import annotations

from hydra_suite.runtime import macos_libomp as lo


def _site(prefix):
    site = prefix / "lib" / "python3.13" / "site-packages"
    site.mkdir(parents=True)
    return site


def test_dangling_link_is_restored_from_backup(tmp_path):
    torch_lib = _site(tmp_path) / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    link = torch_lib / "libomp.dylib"
    link.symlink_to(tmp_path / "lib" / "gone.dylib")
    (torch_lib / ("libomp.dylib" + lo.BACKUP_SUFFIX)).write_bytes(b"orig")
    lo.repair_dangling_links(tmp_path)
    assert not link.is_symlink() and link.read_bytes() == b"orig"


def test_dangling_link_without_backup_is_removed(tmp_path):
    sk = _site(tmp_path) / "sklearn" / ".dylibs"
    sk.mkdir(parents=True)
    link = sk / "libomp.dylib"
    link.symlink_to(tmp_path / "nowhere.dylib")
    lo.repair_dangling_links(tmp_path)
    assert not link.exists() and not link.is_symlink()
