import pytest

from hydra_suite.core.inference.sam2 import checkpoints as ck


def test_default_variant_is_in_catalog():
    assert ck.DEFAULT_VARIANT in ck.SAM2_VARIANTS
    assert "sam2.1-hiera-large" in ck.available_variants()


def test_unknown_variant_raises_named(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        ck.ensure_checkpoint("bogus", cache_dir=tmp_path)


def test_offline_uncached_raises_named(tmp_path):
    with pytest.raises(ValueError, match="not downloaded"):
        ck.ensure_checkpoint(
            ck.DEFAULT_VARIANT, allow_download=False, cache_dir=tmp_path
        )


def test_cached_checkpoint_returned_without_download(tmp_path, monkeypatch):
    variant = ck.DEFAULT_VARIANT
    dest = tmp_path / f"{variant}.pt"
    dest.write_bytes(b"fake")

    def _boom(*a, **k):
        raise AssertionError("should not download when cached")

    monkeypatch.setattr(ck, "hf_hub_download", _boom)
    assert ck.ensure_checkpoint(variant, cache_dir=tmp_path) == dest


def test_public_download_retries_anonymously_on_stale_token(monkeypatch):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    import hydra_suite.core.inference.sam2.checkpoints as ck

    calls = []

    def fake(repo_id, filename, token=None):
        calls.append(token)
        if token is None:
            response = httpx.Response(
                401, request=httpx.Request("HEAD", "https://hf.co")
            )
            raise HfHubHTTPError("401 Unauthorized", response=response)
        return "/tmp/ok.pt"

    monkeypatch.setattr(ck, "hf_hub_download", fake)
    assert ck._download("facebook/sam2.1-hiera-tiny", "x.pt") == "/tmp/ok.pt"
    assert calls == [None, False]
