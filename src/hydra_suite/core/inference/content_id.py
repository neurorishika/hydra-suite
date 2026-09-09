"""Content-based identity for models and videos.

Cache keys used to be ``(path, mtime)``-based, which made every cache
machine-local: a cache computed on a compute box could never be reused on the
staging machine because the model lived at a different absolute path. These
primitives identify an artifact by what it CONTAINS, so a cache travels with
the job (see docs/superpowers/specs/2026-09-09-portable-tracking-jobs-design.md
section 7b).
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB
_VIDEO_PROBE = 8 << 20  # head and tail bytes hashed for a video signature

# Host-specific engine caches must never influence a portable identity.
# Fix V-minor: `.DS_Store` and `__pycache__` are host/OS/filesystem-visit
# noise, not model content -- a Finder window opened on a SLEAP run
# directory on the Mac (or a stray bytecode cache from any tooling that
# imports something inside it) changes `directory_content_id` and defeats
# an otherwise-valid pulled cache purely because of when/whether a Finder
# window happened to be opened, which has nothing to do with model
# portability.
_EXCLUDED_DIR_NAMES = {".hydra-runtime-artifacts", ".DS_Store", "__pycache__"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def file_content_id(path: str | os.PathLike[str] | None) -> str:
    """``"sha256:<hex>"`` of a file's bytes; ``""`` if absent or unreadable."""
    if not path:
        return ""
    p = Path(path)
    try:
        if not p.is_file():
            return ""
        return f"sha256:{_sha256_file(p)}"
    except OSError:
        return ""


def directory_content_id(path: str | os.PathLike[str] | None) -> str:
    """``"dirsha256:<hex>"`` over sorted ``(relpath, sha256)`` of every member.

    MEASURED COST OF THIS DEVIATION (round-8, on the real fixture SLEAP run
    dir ``pose/SLEAP/20260214-224154_unet_ant_single_instance``, 94.7 MB):
    whole-tree 82.6 ms vs artifacts.py-allowlist 69.8 ms -- a 12.8 ms delta,
    because ``best.ckpt`` alone is 94 MB and the extra ``.slp``/``.csv``
    members total 0.4 MB. Memoized once per process, so this is immaterial
    against ``PERF_TOLERANCE=1.25``. Round-8 review raised the allowlist as a
    PERF concern; it was measured and rejected on the numbers. Do not
    re-litigate without a new measurement.

    DEVIATION FROM SPEC §7b.2: the spec says to reuse the artifact-fingerprint
    file SET (``core/individual/pose/artifacts.py``'s selection). This hashes
    every file under the directory instead and excludes only
    ``.hydra-runtime-artifacts/`` by name. Deliberate, not an oversight: that
    directory is nearly a phantom exclusion — it exists only as an OSError
    fallback (``core/inference/runtime_artifacts.py:765-768``); real pose
    exports (ONNX/TensorRT/CoreML) are written as SIBLINGS of the pose run
    directory, not inside it, so they are not members of ``root`` at all and
    the "exclude runtime artifacts" tests below are largely tautological
    (nothing host-specific is actually under `root` to exclude in the common
    case). We accept the deviation because hashing the whole tree is simpler,
    strictly safer (it can only over-invalidate, never silently miss a real
    content change), and the fingerprint subset is itself an
    implementation detail of a different, unrelated consumer.

    KNOWN LIVE COST of this deviation, not theoretical (fix V5, exercised at
    Task 13's Step 8-ish real-run check): a SLEAP-directory model means
    hashing EVERY file under the training-run tree, including
    ``labels_*.slp`` and ``viz/*.png`` — non-model members that can change
    (a log, a lockfile written during inference) without the actual model
    weights changing, which over-invalidates a pulled cache for reasons
    unrelated to model identity, and costs a full-tree hash once per fresh
    process where the pre-v5 path-based key paid nothing. If this shows up
    as a real PERF-gate or false-invalidation problem (see Task 13's
    self-diagnosing SLEAP-directory check), the fallback is the narrower
    ``core/individual/pose/artifacts.py`` fingerprint selection this
    docstring deviates from — not implemented here, kept as an escape hatch.
    """
    if not path:
        return ""
    root = Path(path)
    try:
        if not root.is_dir():
            return ""
        parts: list[str] = []
        for child in sorted(root.rglob("*")):
            if not child.is_file():
                continue
            rel = child.relative_to(root)
            if any(part in _EXCLUDED_DIR_NAMES for part in rel.parts):
                continue
            parts.append(f"{rel.as_posix()}={_sha256_file(child)}")
        blob = "\n".join(parts).encode("utf-8")
        return f"dirsha256:{hashlib.sha256(blob).hexdigest()}"
    except OSError:
        return ""


def _stat_hint(path: str) -> tuple[object, ...]:
    """Cheap local identity used ONLY as a memoization key, never in a cache key.

    KNOWN STALENESS (not a regression — matches the old (path, mtime) behaviour):
    for a directory, ``st.st_mtime_ns`` is the directory ENTRY's own mtime,
    which does not change when a member file already inside it is overwritten
    in place (only when an entry is added/removed/renamed). So
    ``model_content_id`` on a directory can return a stale memoized value
    within one process if a pose-run directory's member is edited in place
    without an entry list change. The old path-based key had the identical
    blind spot (it never looked inside directories at all), so this is not a
    new failure mode, only carried forward. Cross-process/cross-run identity
    is unaffected: a fresh process always re-hashes the tree.

    Fix W4 (multihead memo re-opens the bug it was added to close): for a
    ``.multihead.json`` path, the hint is NOT just ``(realpath, size,
    mtime_ns)`` of the manifest file. Retraining a head (ClassKit, a SEPARATE
    process from any long-lived TrackerKit GUI holding this memo) rewrites a
    SIBLING ``*_flat*.pth`` file that ``factor_models[].path`` points at —
    zero bytes of the manifest itself change, so the manifest-only hint is
    identical before and after a retrain and ``lru_cache`` returns the SAME
    stale composite id it returned before, defeating the whole point of fix
    A1b's composite digest. `_stat_hint` therefore also stats every head the
    manifest currently references (the same resolution
    ``_multihead_manifest_content_id`` uses: ``base = manifest.parent;
    (base / entry["path"]).resolve()``) and folds each head's ``(size,
    mtime_ns)`` into the returned tuple, so retraining a head changes the
    MEMO KEY itself and the next call re-hashes rather than returning a
    cached value. `cnn_cache_key` (`cache/keys.py:361-365`) runs inside a
    potentially long-lived TrackerKit GUI process while ClassKit retrains
    heads in a different process — this is exactly the path that stayed
    stale before.
    """
    try:
        st = os.stat(path)
        real = os.path.realpath(path)
    except OSError:
        # NOT realpath: on an unstattable path we deliberately keep the caller's
        # string verbatim, so the memo key and the sentinel below agree on what
        # "this artifact" means even when the path cannot be canonicalized.
        # (Fix M-minor: the sentinel comment used to say "realpath", which
        # disagreed with this branch. `_content_id_for_hint` reads `hint[0]`,
        # which is realpath on the success branch and the raw string here.)
        return (str(path), -1, -1)
    base_hint: tuple[object, ...] = (real, st.st_size, st.st_mtime_ns)
    if real.lower().endswith(".multihead.json"):
        head_stats: list[tuple[int, int]] = []
        try:
            import json

            data = json.loads(Path(real).read_text(encoding="utf-8"))
            manifest_dir = Path(real).parent
            for entry in data.get("factor_models", []) or []:
                entry_path = entry.get("path") if isinstance(entry, dict) else None
                if not entry_path:
                    continue
                head = (manifest_dir / entry_path).resolve()
                try:
                    hst = os.stat(head)
                    head_stats.append((hst.st_size, hst.st_mtime_ns))
                except OSError:
                    head_stats.append((-1, -1))
        except (OSError, ValueError):
            # Unparseable manifest: fall through with base_hint only -- the
            # manifest-only hint is still correct, just not head-aware; this
            # matches _multihead_manifest_content_id's own (OSError, ValueError)
            # handling (fix Z6: it degrades to file_content_id(manifest) on an
            # unreadable/malformed manifest, not "" -- a non-empty id here is
            # still important so a manifest-only-but-valid file keeps a stable,
            # non-sentinel identity even when head resolution can't happen).
            head_stats = []
        base_hint = base_hint + tuple(sorted(head_stats))
    return base_hint


def _multihead_manifest_content_id(manifest_path: Path) -> str:
    """Composite digest of a ``.multihead.json`` manifest AND every head it
    references (fix A1b).

    ``file_content_id`` alone hashes only the manifest bytes -- a few hundred
    bytes of JSON with ``factor_models[].path`` entries pointing at sibling
    ``_flat.pth``/``_flat_1.pth`` checkpoints
    (``core/individual/classification/backend.py:494-497``: ``base =
    manifest_path.parent; factor_path = (base / entry["path"]).resolve()``).
    Retraining a head IN PLACE under the same filename changes zero bytes the
    manifest's own hash sees, so a v5 key built from the manifest alone is
    BLIND to that retrain and would wrongly reuse a stale cache. Fold each
    head's own content id into the manifest's.

    This corrects the design doc's restatement ("model_id of the selected
    checkpoint only, as today; sibling heads travel with it and are covered
    by discover_multihead_model_bundle at pack time") for THIS format:
    ``discover_multihead_model_bundle`` only recognises ``*.bundle.json``
    (``classkit/model_bundle.py:12,82,119``) and returns ``None`` for a bare
    ``.multihead.json`` path -- it covers nothing here, at pack time or
    otherwise (see Task 11 fix A1a). The "selected checkpoint only" identity
    is correct for the ``.bundle.json`` case; it is wrong for
    ``.multihead.json``, which is the format the real production identity
    classifier actually uses (``tools/equivalence/fixtures/configs/
    ant_cnn_identity.json:235``).
    """
    # Fix Z6: `_stat_hint` above catches `(OSError, ValueError)` and guards
    # the manifest actually being a dict; this function must be at least as
    # defensive, because unlike `_stat_hint` (a memo-key helper only), an
    # UNCAUGHT exception here propagates out of `model_content_id` ->
    # `cnn_cache_key` -> `_open_caches`, aborting the whole run BEFORE model
    # loading ever gets a chance to raise the proper `ClassifierFormatError`
    # for the same malformed manifest. A truncated/corrupted
    # `.multihead.json` raises `json.JSONDecodeError` (a `ValueError`), not
    # `OSError` -- catching only `OSError` lets it escape uncaught. A
    # syntactically valid but non-dict JSON document (e.g. a bare `[]` or a
    # number) makes `data.get(...)` raise `AttributeError`.
    import json

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return file_content_id(manifest_path)
    except (OSError, ValueError):
        return file_content_id(manifest_path)
    manifest_id = file_content_id(manifest_path)
    base = manifest_path.parent
    head_ids = []
    for entry in data.get("factor_models", []) or []:
        entry_path = entry.get("path") if isinstance(entry, dict) else None
        if not entry_path:
            continue
        head_ids.append(file_content_id((base / entry_path).resolve()))
    blob = "\n".join([manifest_id, *sorted(head_ids)]).encode("utf-8")
    return f"multihead:{hashlib.sha256(blob).hexdigest()}"


@lru_cache(maxsize=256)
def _content_id_for_hint(hint: tuple[object, ...]) -> str:
    # Fix W4: hint is (realpath, size, mtime_ns) for a plain file/directory,
    # or that plus a sorted tuple of (size, mtime_ns) per referenced head for
    # a .multihead.json manifest -- see _stat_hint. hint[0] is always the
    # realpath (or the raw unstattable string) regardless of length.
    real = hint[0]
    p = Path(real)
    if p.is_dir():
        dir_id = directory_content_id(real)
        if not dir_id:
            # Minor fix: directory_content_id() returns "" on ANY member
            # OSError (e.g. a permission-denied file partway through the
            # tree) for a directory that genuinely EXISTS -- silently
            # producing an empty content id for a real model directory.
            # Log loudly rather than let this degrade quietly and unnoticed.
            import logging

            logging.getLogger(__name__).warning(
                "directory_content_id() returned empty for an existing "
                "directory %s -- a member file likely raised OSError during "
                "the walk; the resulting content id is empty rather than "
                "reflecting the directory's real contents",
                real,
            )
        return dir_id
    if p.name.lower().endswith(".multihead.json"):
        multihead_id = _multihead_manifest_content_id(p)
        if multihead_id:
            return multihead_id
    id_ = file_content_id(real)
    if id_:
        return id_
    # MISSING-MODEL SENTINEL (fix M5): file_content_id("") is "" for BOTH "no
    # path configured" and "path configured but the file is gone". If two
    # DIFFERENT missing models both fell through to "", a cache written while
    # model A was missing would spuriously validate for model B — worse than
    # the old (path, mtime) key, which at least differed by path. Distinguish
    # missing-but-configured artifacts by hashing their path string
    # (`hint[0]`: realpath when the artifact was stattable, the caller's raw
    # string when it was not -- see `_stat_hint`) instead of their bytes, so
    # two different missing paths still diverge.
    #
    # WHY A PATH-DEPENDENT SENTINEL IS ACCEPTABLE IN A PORTABILITY-MOTIVATED
    # KEY (fix M-minor -- state this, do not silently rely on it):
    #   1. It is unreachable on any cache-HIT path. A missing CNN classifier is
    #      logged and SKIPPED (`core/inference/config.py:1227-1233` continues),
    #      so no CNNConfig -- and therefore no cache key -- ever carries it. A
    #      missing detection/pose/head-tail model fails model loading and the
    #      run dies before any cache is written. The sentinel can only appear
    #      in a key that is never compared against a cache that exists.
    #   2. The PRE-v5 key was equally path-bound for exactly this case:
    #      `keys.py:419-423` returned mtime 0.0 on OSError while still
    #      embedding the absolute path, so v5 is no worse here.
    #   3. Collapsing to "" would be STRICTLY WORSE than either: it would make
    #      two different missing models share one identity, so a cache written
    #      while model A was missing could validate for model B.
    if real and real != "None":
        return f"missing:{hashlib.sha256(real.encode('utf-8')).hexdigest()[:16]}"
    return ""


def model_content_id(path: str | os.PathLike[str] | None) -> str:
    """Content id of a model artifact (file or directory), memoized per process.

    Hashing a 100 MB checkpoint costs ~0.3 s; the optimizer and preview paths
    rebuild keys many times per session, so memoize on ``(realpath, size,
    mtime_ns)``. mtime therefore remains a LOCAL fast-path hint and never
    reaches the key itself.

    A configured-but-missing artifact does NOT collapse to ``""`` — see the
    missing-model sentinel in ``_content_id_for_hint``. An unconfigured
    (empty/None) path still returns ``""``.
    """
    if not path:
        return ""
    return _content_id_for_hint(_stat_hint(str(path)))


model_content_id.cache_clear = _content_id_for_hint.cache_clear  # type: ignore[attr-defined]


def video_signature(path: str | os.PathLike[str] | None) -> str:
    """``"{size}:{sha256(head 8 MiB || tail 8 MiB)[:32]}"``; ``""`` if absent.

    Size plus the first and last 8 MiB catches every realistic replacement (a
    re-encode, a trim, a different clip under the same name) for ~30 ms,
    without reading a 50 GB file. Symlinks are followed, as before.

    NOT memoized, unlike ``model_content_id``. ``optimizer_workers.py:343``
    calls this once per evaluation during autotune search, so an unmemoized
    16 MiB read per call is a real, deliberately-accepted cost: video files
    are large enough that even the ``(realpath, size, mtime_ns)`` memoization
    hint used for models is comparatively cheap to skip re-deriving, and the
    autotune loop already dominates wall-clock with model inference. If this
    ever shows up in a profile, add the same ``lru_cache``-on-stat-hint
    pattern as ``model_content_id`` — do not memoize on path alone.
    """
    if not path:
        return ""
    try:
        st = os.stat(path)  # follows symlinks
        size = st.st_size
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read(_VIDEO_PROBE))
            if size > _VIDEO_PROBE:
                handle.seek(max(size - _VIDEO_PROBE, _VIDEO_PROBE))
                digest.update(handle.read(_VIDEO_PROBE))
        return f"{size}:{digest.hexdigest()[:32]}"
    except OSError:
        return ""
