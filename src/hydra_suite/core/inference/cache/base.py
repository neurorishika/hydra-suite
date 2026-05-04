from __future__ import annotations

from dataclasses import dataclass

# Bumped any time the on-disk schema of any cached result type changes:
# - Adding/removing/renaming fields
# - Changing dtype or shape conventions
# - Changing whether the cache stores raw vs calibrated outputs
# v1 = legacy pre-redesign caches (DetectionCache, CNNIdentityCache, etc.)
# v2 = new pipeline (this redesign)
CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CacheKey:
    """Identifies a cache file's compatibility with a current configuration.

    A cache is reusable iff: schema_version matches AND model_path matches AND
    model_mtime matches (within 1ms) AND config_hash matches.
    """

    schema_version: int  # CACHE_SCHEMA_VERSION at write time
    model_path: str  # primary model path (or "|"-joined for sequential)
    model_mtime: float  # os.path.getmtime of primary model; 0.0 if no model file
    config_hash: str  # sha256 hex of model-affecting config fields; "" when none apply

    def as_string(self) -> str:
        return (
            f"v{self.schema_version}|{self.model_path}"
            f"|{self.model_mtime:.6f}|{self.config_hash}"
        )

    def matches(self, other: "CacheKey") -> bool:
        return (
            self.schema_version == other.schema_version
            and self.model_path == other.model_path
            and abs(self.model_mtime - other.model_mtime) < 1e-3
            and self.config_hash == other.config_hash
        )
