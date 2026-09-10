from __future__ import annotations

from dataclasses import dataclass

# Bumped any time the on-disk schema of any cached result type changes:
# - Adding/removing/renaming fields
# - Changing dtype or shape conventions
# - Changing whether the cache stores raw vs calibrated outputs
# v1 = legacy pre-redesign caches (DetectionCache, etc.)
# v2 = new pipeline (this redesign)
# v3 = bumped for bg-sub: every prior bgsub cache was produced under unseeded
#      random priming and keyed by a hash that ignored THRESHOLD_VALUE. Those
#      artifacts are unsound and must not be inherited.
# v4 = bumped for the identity-subsystem repair (2026-08-27,
#      docs/superpowers/plans/done/2026-08-27-identity-subsystem-repair.md): the
#      Layer-2 fit-policy-aware dispatch (Tasks 1-3) and the head-first crop
#      orientation fix (Task 4) change what the CNN and head/tail stages
#      produce from the SAME model_path/mtime/geometry inputs, so old caches
#      must be invalidated even though none of those fields changed.
# v5 = model and video identity became CONTENT-based rather than
#      (absolute path, mtime)-based, so a cache produced on a compute box is
#      reusable on the staging machine. ``model_path``+``model_mtime`` collapse
#      into a single ``model_id``. See the portable-jobs design, section 7b.
CACHE_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class CacheKey:
    """Identifies a cache file's compatibility with a current configuration.

    A cache is reusable iff: schema_version matches AND model_id matches AND
    config_hash matches.
    """

    schema_version: int  # CACHE_SCHEMA_VERSION at write time
    model_id: str  # content identity: "sha256:…", "dirsha256:…", "a|b", or a sentinel
    config_hash: str  # sha256 hex of model-affecting config fields; "" when none apply

    def as_string(self) -> str:
        return f"v{self.schema_version}|{self.model_id}|{self.config_hash}"

    def matches(self, other: "CacheKey") -> bool:
        return self.as_string() == other.as_string()
