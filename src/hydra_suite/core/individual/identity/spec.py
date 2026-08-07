"""Structured, serializable identity catalog domain.

The persisted, ordered set of known identities for a run. Each entry retains
its ``(factor, class)`` provenance so composite identities never need to be
decoded by splitting a "_"-joined string (the legacy landmine). The joined
``display_label`` is kept only as a backward-compatible presentation string.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    """One known identity and its structured factor provenance.

    Parameters
    ----------
    display_label:
        Backward-compatible presentation string. For a CNN composite this is
        ``"_".join(class for _, class in factors if class)``; for a tag/flat
        label it is the bare label.
    factors:
        Ordered ``(factor_name, class_name)`` pairs. Empty for a tag / flat
        single-label identity.
    source:
        ``"cnn"`` or ``"tag"``.
    """

    display_label: str
    factors: tuple[tuple[str, str], ...]
    source: str


@dataclass(frozen=True)
class IdentityCatalogSpec:
    """Ordered domain of known identities (the unknown slot is implicit at 0)."""

    entries: tuple[CatalogEntry, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        """Display labels in registration order (excluding the unknown slot)."""
        return tuple(e.display_label for e in self.entries)

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": [
                {
                    "display_label": e.display_label,
                    "factors": [list(pair) for pair in e.factors],
                    "source": e.source,
                }
                for e in self.entries
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IdentityCatalogSpec":
        entries = tuple(
            CatalogEntry(
                display_label=str(d["display_label"]),
                factors=tuple((str(f[0]), str(f[1])) for f in d.get("factors", [])),
                source=str(d.get("source", "cnn")),
            )
            for d in data.get("entries", [])
        )
        return cls(entries=entries)
