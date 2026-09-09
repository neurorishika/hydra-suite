"""Cache identity is content-based: same bytes anywhere == same key."""

import json
import os
import shutil

from hydra_suite.core.inference import content_id
from hydra_suite.core.inference.cache.base import CACHE_SCHEMA_VERSION, CacheKey


def _touch_different_mtime(path):
    os.utime(path, (1, 1))


def test_file_content_id_is_path_and_mtime_independent(tmp_path):
    a = tmp_path / "here" / "model.pt"
    b = tmp_path / "somewhere" / "else" / "model.pt"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_bytes(b"weights" * 100)
    shutil.copy2(a, b)
    _touch_different_mtime(b)
    assert content_id.file_content_id(str(a)) == content_id.file_content_id(str(b))
    assert content_id.file_content_id(str(a)).startswith("sha256:")


def test_one_byte_change_changes_the_file_content_id(tmp_path):
    a = tmp_path / "m.pt"
    a.write_bytes(b"aaaa")
    first = content_id.file_content_id(str(a))
    a.write_bytes(b"aaab")
    assert content_id.file_content_id(str(a)) != first


def test_directory_content_id_is_path_independent(tmp_path):
    a = tmp_path / "run_a"
    b = tmp_path / "nested" / "run_b"
    for root in (a, b):
        root.mkdir(parents=True)
        (root / "best.ckpt").write_bytes(b"ckpt")
        (root / "training_config.json").write_text("{}")
    assert content_id.directory_content_id(str(a)) == content_id.directory_content_id(
        str(b)
    )
    assert content_id.directory_content_id(str(a)).startswith("dirsha256:")


def test_directory_content_id_ignores_local_runtime_artifacts(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "best.ckpt").write_bytes(b"ckpt")
    before = content_id.directory_content_id(str(root))
    engines = root / ".hydra-runtime-artifacts"
    engines.mkdir()
    (engines / "model.engine").write_bytes(b"host-specific")
    assert content_id.directory_content_id(str(root)) == before


def test_directory_content_id_notices_a_changed_member(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "best.ckpt").write_bytes(b"ckpt")
    before = content_id.directory_content_id(str(root))
    (root / "best.ckpt").write_bytes(b"ckpt2")
    assert content_id.directory_content_id(str(root)) != before


def test_missing_path_yields_empty_id(tmp_path):
    assert content_id.file_content_id(str(tmp_path / "nope.pt")) == ""
    assert content_id.file_content_id("") == ""


def test_two_different_missing_models_get_different_content_ids(tmp_path):
    """Fix M5: two configured-but-missing models must NOT collapse to the same
    "" id, or a cache written while model A was missing would spuriously
    validate for model B. file_content_id("") == "" for both is fine (that's
    the file-level primitive); model_content_id must NOT collapse them."""
    content_id.model_content_id.cache_clear()
    a = content_id.model_content_id(str(tmp_path / "missing_a.pt"))
    b = content_id.model_content_id(str(tmp_path / "missing_b.pt"))
    assert a != b
    assert a.startswith("missing:")
    assert b.startswith("missing:")


def test_unconfigured_model_path_is_still_empty():
    assert content_id.model_content_id("") == ""
    assert content_id.model_content_id(None) == ""


def _write_multihead_manifest(
    root, manifest_name="clf.multihead.json", head_bytes=b"head_a"
):
    (root / "clf_flat.pth").write_bytes(head_bytes)
    manifest = root / manifest_name
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "classifier_multihead_bundle",
                "factor_names": ["colour"],
                "factor_models": [
                    {"factor": "colour", "path": "clf_flat.pth", "class_names": ["a"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_multihead_manifest_content_id_notices_a_retrained_head(tmp_path):
    """Fix A1b + W4: retraining a head IN PLACE under the same filename
    changes zero bytes of the manifest JSON itself -- a v5 key built by
    hashing only the manifest (file_content_id) would be BLIND to the
    retrain and would wrongly validate a stale cache. model_content_id must
    fold every factor_models[].path head into the manifest's identity
    (backend.py:494-497 is the resolution algorithm this mirrors: base =
    manifest.parent; (base / entry["path"]).resolve()).

    Fix W4: deliberately NO `cache_clear()` call between the two
    `model_content_id` calls below. This is the actual bug fix wave 3's
    version of this test masked: `_content_id_for_hint` is `@lru_cache`-d,
    keyed on `_stat_hint(manifest)`. Retraining a SIBLING head file changes
    zero bytes/stat of the MANIFEST itself, so a version of `_stat_hint`
    that only stats the manifest returns the identical memo key before and
    after -- `model_content_id` would return the STALE cached composite
    without a manual `cache_clear()` ever being called, which is exactly
    what a real long-lived TrackerKit GUI session does (it never calls
    `cache_clear()` between ClassKit retraining a head in another process
    and the GUI's own next cache-key computation). Calling `cache_clear()`
    here would hide that regression entirely -- the fixed `_stat_hint` must
    make the SECOND `model_content_id` call itself observe the retrain via
    a changed memo key, with the cache warm the whole time."""
    content_id.model_content_id.cache_clear()
    manifest = _write_multihead_manifest(tmp_path)
    before = content_id.model_content_id(str(manifest))
    assert before.startswith("multihead:")
    # Minor fix (round-7): capture the manifest's OWN file_content_id BEFORE
    # the head rewrite, so the assertion below is a real before/after
    # comparison, not the tautological `x == x` the previous draft had
    # (`file_content_id(str(manifest)) == file_content_id(str(manifest))`,
    # which is trivially true regardless of whether the manifest bytes
    # actually changed and proves nothing).
    manifest_only_id_before = content_id.file_content_id(str(manifest))
    (tmp_path / "clf_flat.pth").write_bytes(b"retrained_head")
    after = content_id.model_content_id(str(manifest))  # NOTE: no cache_clear() here
    assert after != before
    # The manifest bytes alone are unchanged -- proves the composite is
    # actually reading the head, not just re-hashing the manifest file.
    assert content_id.file_content_id(str(manifest)) == manifest_only_id_before


def test_multihead_manifest_content_id_is_path_independent(tmp_path):
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    a.mkdir()
    b.mkdir()
    manifest_a = _write_multihead_manifest(a)
    manifest_b = _write_multihead_manifest(b)
    content_id.model_content_id.cache_clear()
    assert content_id.model_content_id(str(manifest_a)) == content_id.model_content_id(
        str(manifest_b)
    )


def test_video_signature_survives_a_touched_mtime(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"\x00" * (1 << 20))
    first = content_id.video_signature(str(v))
    _touch_different_mtime(v)
    assert content_id.video_signature(str(v)) == first


def test_video_signature_changes_when_content_changes(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"\x00" * (1 << 20))
    first = content_id.video_signature(str(v))
    v.write_bytes(b"\x01" * (1 << 20))
    assert content_id.video_signature(str(v)) != first


def test_video_signature_detects_a_tail_only_change(tmp_path):
    """A re-encode that keeps the head must still invalidate. 4 MiB never
    enters the seek branch (file <= _VIDEO_PROBE), so this alone would pass
    even with a head-only implementation — it is NOT sufficient coverage by
    itself; see the 8-16 MiB and >16 MiB cases below."""
    v = tmp_path / "clip.mp4"
    body = bytearray(b"\x00" * (4 << 20))
    v.write_bytes(bytes(body))
    first = content_id.video_signature(str(v))
    body[-16:] = b"\xff" * 16
    v.write_bytes(bytes(body))
    assert content_id.video_signature(str(v)) != first


def test_video_signature_detects_a_tail_only_change_between_head_and_full_probe(
    tmp_path, monkeypatch
):
    """12 MiB file: bigger than the 8 MiB head probe, smaller than 2x the
    probe, so the seek branch's overlap-with-head math is exercised."""
    monkeypatch.setattr(content_id, "_VIDEO_PROBE", 8 << 20)
    v = tmp_path / "clip.mp4"
    body = bytearray(b"\x00" * (12 << 20))
    v.write_bytes(bytes(body))
    first = content_id.video_signature(str(v))
    body[-16:] = b"\xff" * 16
    v.write_bytes(bytes(body))
    assert content_id.video_signature(str(v)) != first


def test_video_signature_detects_a_tail_only_change_above_2x_probe(
    tmp_path, monkeypatch
):
    """>16 MiB (>2x probe): head and tail windows are disjoint; a tail-only
    edit must still be caught even though head bytes are fully unchanged.
    Probe size is monkeypatched down so the test stays fast."""
    monkeypatch.setattr(content_id, "_VIDEO_PROBE", 1 << 20)  # 1 MiB probe
    v = tmp_path / "clip.mp4"
    body = bytearray(b"\x00" * (3 << 20))  # 3 MiB, > 2x the 1 MiB probe
    v.write_bytes(bytes(body))
    first = content_id.video_signature(str(v))
    body[-16:] = b"\xff" * 16
    v.write_bytes(bytes(body))
    assert content_id.video_signature(str(v)) != first


def test_video_signature_of_missing_file_is_empty(tmp_path):
    assert content_id.video_signature(str(tmp_path / "gone.mp4")) == ""
    assert content_id.video_signature(None) == ""


def test_model_content_id_is_memoized_per_process(tmp_path, monkeypatch):
    m = tmp_path / "m.pt"
    m.write_bytes(b"x" * 4096)
    content_id.model_content_id.cache_clear()
    calls = []
    real_open = open

    def counting_open(path, *args, **kwargs):
        calls.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", counting_open)
    first = content_id.model_content_id(str(m))
    second = content_id.model_content_id(str(m))
    assert first == second
    # Minor fix: `_stat_hint` (and therefore `file_content_id`, which opens
    # `Path(real)` where `real = os.path.realpath(path)`) opens the REALPATH,
    # not the caller's original string. On macOS, tmp_path lives under
    # /var, which is itself a symlink to /private/var, so
    # os.path.realpath(m) != str(m) -- comparing against str(m) here is
    # fragile and would undercount (or overcount, if some OTHER file happens
    # to share the same basename) on exactly that platform. Compare
    # realpaths on both sides.
    real_m = os.path.realpath(str(m))
    assert len([c for c in calls if os.path.realpath(c) == real_m]) == 1


def test_cache_key_has_no_mtime_field_and_new_string_form():
    key = CacheKey(
        schema_version=CACHE_SCHEMA_VERSION, model_id="sha256:ab", config_hash="cd"
    )
    assert not hasattr(key, "model_mtime")
    assert not hasattr(key, "model_path")
    assert key.as_string() == f"v{CACHE_SCHEMA_VERSION}|sha256:ab|cd"


def test_schema_version_is_five():
    assert CACHE_SCHEMA_VERSION == 5
