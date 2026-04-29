"""Typed dataclass schemas for ClassKit project, model, and active-learning configuration."""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Embedding model selection and compute configuration."""

    name: str = "vit_b_16"
    batch_size: int = 32
    device: str = "cuda"


@dataclass
class ALConfig:
    """Active-learning batch configuration, including acquisition strategy weights."""

    batch_size: int = 40
    strategy_weights: Dict[str, float] = field(  # noqa: DC01  (dataclass field)
        default_factory=lambda: {
            "uncertainty": 0.4,
            "diversity": 0.35,
            "representativeness": 0.15,
            "audit": 0.10,
        }
    )
    balance_mode: bool = (
        False  # noqa: DC01 — replace representative slot with label-balance scoring
    )


@dataclass
class Factor:
    """A single labeling axis with its allowed labels and optional keyboard shortcuts."""

    name: str
    labels: List[str]
    shortcut_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize this factor to a plain dict for JSON persistence."""
        return {
            "name": self.name,
            "labels": self.labels,
            "shortcut_keys": self.shortcut_keys,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Factor":
        """Reconstruct a Factor from a previously serialized dict."""
        return cls(
            name=d["name"],
            labels=d["labels"],
            shortcut_keys=d.get("shortcut_keys", []),
        )


@dataclass
class LabelingScheme:
    """Multi-factor compositional labeling scheme.

    Each Factor defines one labeling axis; composite labels are encoded by joining
    per-factor values with ``|``.
    """

    name: str
    factors: List[Factor]
    training_modes: List[str]
    description: str = ""

    def __post_init__(self) -> None:
        is_multi_factor = len(self.factors) > 1
        for factor in self.factors:
            normalized_labels: list[str] = []
            saw_unknown = False
            for raw_label in factor.labels:
                label = str(raw_label)
                if label.strip().lower() == "unknown":
                    if not saw_unknown:
                        normalized_labels.append("unknown")
                        saw_unknown = True
                    continue
                normalized_labels.append(label)
            if is_multi_factor and not saw_unknown:
                normalized_labels.append("unknown")
            factor.labels = normalized_labels

    @property
    def total_classes(self) -> int:
        """Total number of unique composite class combinations (product of per-factor counts)."""
        result = 1
        for f in self.factors:
            result *= len(f.labels)
        return result

    def encode_label(self, factor_values: List[str]) -> str:
        if len(factor_values) != len(self.factors):
            raise ValueError(
                f"Expected {len(self.factors)} factor values, got {len(factor_values)}"
            )
        return "|".join(factor_values)

    def decode_label(self, composite: str) -> List[str]:
        parts = composite.split("|")
        if len(parts) != len(self.factors):
            raise ValueError(
                f"Expected {len(self.factors)} parts in composite label, got {len(parts)}"
            )
        return parts

    def valid_encoded_labels(self) -> set[str]:
        """Return every persisted label value allowed by this scheme."""
        if not self.factors:
            return set()
        return {
            self.encode_label(list(values))
            for values in product(*(factor.labels for factor in self.factors))
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "factors": [f.to_dict() for f in self.factors],
            "training_modes": self.training_modes,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelingScheme":
        return cls(
            name=d["name"],
            factors=[Factor.from_dict(f) for f in d.get("factors", [])],
            training_modes=d.get("training_modes", []),
            description=d.get("description", ""),
        )


@dataclass
class ProjectConfig:
    name: str
    classes: List[str]
    root_dir: Path
    description: str = ""
    scheme: Optional[LabelingScheme] = None


@dataclass
class ClassKitConfig:
    project: ProjectConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    al: ALConfig = field(default_factory=ALConfig)  # noqa: DC01  (dataclass field)
    viz_umap_n_neighbors: int = 15  # noqa: DC01  (dataclass field)
    viz_umap_min_dist: float = 0.1  # noqa: DC01  (dataclass field)
    index_hnsw_m: int = 32  # noqa: DC01  (dataclass field)
