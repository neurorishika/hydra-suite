"""
Active learning batch acquisition strategies.

Implements the recipe:
- 40% uncertainty (highest entropy / smallest margin)
- 35% diversity (k-center / farthest-first)
- 15% representativeness (dense clusters with low label coverage)
- 10% audits (random + high-disagreement clusters)
"""

try:
    import numpy as np
except ImportError:
    np = None

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .density import select_diverse_samples


@dataclass
class BatchConfig:
    """Configuration for batch acquisition."""

    batch_size: int = 100
    uncertainty_fraction: float = 0.40
    diversity_fraction: float = 0.35
    representative_fraction: float = 0.15
    audit_fraction: float = 0.10  # noqa: DC01  (dataclass field)
    per_cluster_cap: Optional[int] = None  # Max samples per cluster
    min_per_class: int = (
        0  # noqa: DC01  (dataclass field) — minimum per class if imbalanced
    )
    balance_mode: bool = (
        False  # noqa: DC01 — replace representative slot with label-balance scoring
    )


class UncertaintySelector:
    """Select samples with highest uncertainty."""

    @staticmethod
    def entropy(probs: np.ndarray) -> np.ndarray:
        """
        Compute entropy of probability distributions.

        Args:
            probs: (N, num_classes) probabilities

        Returns:
            entropy: (N,) entropy values
        """
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        return -(probs * np.log(probs)).sum(axis=1)

    @staticmethod
    def margin(probs: np.ndarray) -> np.ndarray:
        """
        Compute margin between top two predictions.

        Args:
            probs: (N, num_classes) probabilities

        Returns:
            margin: (N,) margin values (smaller = more uncertain)
        """
        # Sort probabilities
        sorted_probs = np.sort(probs, axis=1)
        # Margin = difference between top 2
        return sorted_probs[:, -1] - sorted_probs[:, -2]

    def select(
        self,
        probs: np.ndarray,
        n_samples: int,
        unlabeled_mask: np.ndarray,
        method: str = "entropy",
    ) -> np.ndarray:
        """
        Select most uncertain samples.

        Args:
            probs: (N, num_classes) prediction probabilities
            n_samples: Number to select
            unlabeled_mask: (N,) boolean mask of unlabeled samples
            method: 'entropy' or 'margin'

        Returns:
            selected_indices: (n_samples,) indices
        """
        if method == "entropy":
            uncertainty = self.entropy(probs)
        elif method == "margin":
            uncertainty = -self.margin(probs)  # Negative so higher is more uncertain
        else:
            raise ValueError(f"Unknown method: {method}")

        # Only consider unlabeled
        uncertainty[~unlabeled_mask] = -np.inf

        # Select top
        selected = np.argsort(uncertainty)[-n_samples:][::-1]
        return selected


class RepresentativeSelector:
    """Select samples from dense, under-labeled clusters."""

    def select(
        self,
        embeddings: np.ndarray,
        cluster_assignments: np.ndarray,
        label_coverage: Dict[int, float],
        cluster_densities: np.ndarray,
        n_samples: int,
        unlabeled_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Select samples from dense clusters with low label coverage.

        Args:
            embeddings: (N, D)
            cluster_assignments: (N,) cluster IDs
            label_coverage: cluster_id -> fraction labeled
            cluster_densities: (K,) density per cluster (lower = denser)
            n_samples: Number to select
            unlabeled_mask: (N,) boolean mask

        Returns:
            selected_indices: (n_samples,) indices
        """
        # Score clusters: dense + low coverage
        n_clusters = len(cluster_densities)
        cluster_scores = np.zeros(n_clusters)

        for cluster_id in range(n_clusters):
            coverage = label_coverage.get(cluster_id, 0.0)
            density = cluster_densities[cluster_id]

            # Higher score = denser + less labeled
            # Use inverse density (smaller distance = higher density)
            if density > 0:
                cluster_scores[cluster_id] = (1.0 - coverage) / (density + 1e-6)

        # Select samples from top clusters
        selected = []
        top_clusters = np.argsort(cluster_scores)[::-1]

        for cluster_id in top_clusters:
            if len(selected) >= n_samples:
                break

            # Get unlabeled samples from this cluster
            mask = (cluster_assignments == cluster_id) & unlabeled_mask
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            # Take random subset from this cluster
            needed = n_samples - len(selected)
            take = min(len(indices), needed)
            selected.extend(np.random.choice(indices, take, replace=False))

        return np.array(selected[:n_samples])


class BalanceSelector:
    """Select samples that reduce label-distribution imbalance in the labeled set.

    Score = Σ_c  prob[i,c] / (labeled_count[c] + 1)

    Samples predicted (softly) to belong to underrepresented classes score higher.
    """

    def select(
        self,
        probs: np.ndarray,
        class_names: List[str],
        image_labels: List[Optional[str]],
        n_samples: int,
        unlabeled_mask: np.ndarray,
    ) -> np.ndarray:
        from collections import Counter

        labeled_count: Counter = Counter(
            lbl for lbl in image_labels if lbl is not None and lbl != ""
        )

        # Build per-class inverse-frequency weight vector (shape C)
        weights = np.array(
            [1.0 / (labeled_count.get(name, 0) + 1) for name in class_names],
            dtype=np.float64,
        )

        # Soft balance score: probs (N,C) @ weights (C,) → (N,)
        scores = probs.astype(np.float64) @ weights
        scores[~unlabeled_mask] = -np.inf

        n_avail = int(unlabeled_mask.sum())
        k = min(n_samples, n_avail)
        if k <= 0:
            return np.array([], dtype=int)

        selected = np.argsort(scores)[-k:][::-1]
        return selected


class AuditSelector:
    """Select audit samples for quality assurance."""

    def select(
        self,
        n_samples: int,
        unlabeled_mask: np.ndarray,
        cluster_assignments: Optional[np.ndarray] = None,
        cluster_disagreements: Optional[Dict[int, float]] = None,
        audit_fraction_random: float = 0.5,
    ) -> np.ndarray:
        """
        Select audit samples.

        Args:
            n_samples: Number to select
            unlabeled_mask: (N,) boolean mask
            cluster_assignments: (N,) cluster IDs (optional)
            cluster_disagreements: cluster_id -> disagreement rate (optional)
            audit_fraction_random: Fraction of audits that are random

        Returns:
            selected_indices: (n_samples,) indices
        """
        n_random = int(n_samples * audit_fraction_random)
        n_targeted = n_samples - n_random

        selected = []

        # Random audits
        unlabeled_indices = np.where(unlabeled_mask)[0]
        if len(unlabeled_indices) > 0:
            random_selected = np.random.choice(
                unlabeled_indices, min(n_random, len(unlabeled_indices)), replace=False
            )
            selected.extend(random_selected)

        # Targeted audits (high-disagreement clusters)
        if (
            cluster_assignments is not None
            and cluster_disagreements is not None
            and n_targeted > 0
        ):
            # Sort clusters by disagreement
            sorted_clusters = sorted(
                cluster_disagreements.items(), key=lambda x: x[1], reverse=True
            )

            for cluster_id, _ in sorted_clusters:
                if len(selected) >= n_samples:
                    break

                mask = (cluster_assignments == cluster_id) & unlabeled_mask
                indices = np.where(mask)[0]

                if len(indices) == 0:
                    continue

                needed = n_samples - len(selected)
                take = min(len(indices), needed)
                selected.extend(np.random.choice(indices, take, replace=False))

        return np.array(selected[:n_samples])


class BatchAcquisition:
    """Main batch acquisition orchestrator."""

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.uncertainty_selector = UncertaintySelector()
        self.representative_selector = RepresentativeSelector()
        self.balance_selector = BalanceSelector()
        self.audit_selector = AuditSelector()

    def select_batch(
        self,
        embeddings: np.ndarray,
        probs: np.ndarray,
        unlabeled_mask: np.ndarray,
        cluster_assignments: Optional[np.ndarray] = None,
        label_coverage: Optional[Dict[int, float]] = None,
        cluster_densities: Optional[np.ndarray] = None,
        cluster_disagreements: Optional[Dict[int, float]] = None,
        image_labels: Optional[List[Optional[str]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        """
        Select a batch using the full recipe.

        Args:
            embeddings: (N, D) embeddings
            probs: (N, num_classes) prediction probabilities
            unlabeled_mask: (N,) boolean mask
            cluster_assignments: (N,) cluster IDs (optional)
            label_coverage: cluster_id -> fraction labeled (optional)
            cluster_densities: (K,) density per cluster (optional)
            cluster_disagreements: cluster_id -> disagreement (optional)
            image_labels: per-sample label strings for balance mode (optional)
            class_names: model output class names aligned with probs columns (optional)

        Returns:
            selected_indices: (batch_size,) selected indices
            breakdown: Dictionary with indices per reason
        """
        cfg = self.config
        breakdown = {}

        # Calculate component sizes
        n_uncertainty = int(cfg.batch_size * cfg.uncertainty_fraction)
        n_diversity = int(cfg.batch_size * cfg.diversity_fraction)
        n_representative = int(cfg.batch_size * cfg.representative_fraction)
        n_audit = cfg.batch_size - (n_uncertainty + n_diversity + n_representative)

        selected_indices = set()

        # 1. Uncertainty
        if n_uncertainty > 0:
            uncertain = self.uncertainty_selector.select(
                probs, n_uncertainty, unlabeled_mask
            )
            breakdown["uncertainty"] = uncertain.tolist()
            selected_indices.update(uncertain)

        # 2. Diversity
        if n_diversity > 0:
            # Only consider unlabeled embeddings
            unlabeled_embs = embeddings[unlabeled_mask]
            unlabeled_indices = np.where(unlabeled_mask)[0]

            if len(unlabeled_embs) > 0:
                diverse_local = select_diverse_samples(unlabeled_embs, n_diversity)
                diverse = unlabeled_indices[diverse_local]
                breakdown["diversity"] = diverse.tolist()
                selected_indices.update(diverse)

        # 3. Representativeness / Balance
        if n_representative > 0:
            if (
                cfg.balance_mode
                and image_labels is not None
                and class_names is not None
            ):
                balance = self.balance_selector.select(
                    probs,
                    class_names,
                    image_labels,
                    n_representative,
                    unlabeled_mask,
                )
                breakdown["balance"] = balance.tolist()
                selected_indices.update(balance)
            elif cluster_assignments is not None:
                if label_coverage is None:
                    label_coverage = {}
                if cluster_densities is None:
                    from .density import compute_cluster_densities

                    cluster_densities = compute_cluster_densities(
                        embeddings, cluster_assignments
                    )

                representative = self.representative_selector.select(
                    embeddings,
                    cluster_assignments,
                    label_coverage,
                    cluster_densities,
                    n_representative,
                    unlabeled_mask,
                )
                breakdown["representative"] = representative.tolist()
                selected_indices.update(representative)

        # 4. Audits
        if n_audit > 0:
            audits = self.audit_selector.select(
                n_audit,
                unlabeled_mask,
                cluster_assignments,
                cluster_disagreements,
            )
            breakdown["audit"] = audits.tolist()
            selected_indices.update(audits)

        # Convert to array
        final_selected = np.array(list(selected_indices))

        # Apply per-cluster cap if specified
        if cfg.per_cluster_cap is not None and cluster_assignments is not None:
            final_selected = self._apply_cluster_cap(
                final_selected, cluster_assignments, cfg.per_cluster_cap
            )

        return final_selected, breakdown

    def _apply_cluster_cap(
        self, selected: np.ndarray, cluster_assignments: np.ndarray, cap: int
    ) -> np.ndarray:
        """Enforce maximum samples per cluster."""
        selected_clusters = cluster_assignments[selected]
        unique_clusters = np.unique(selected_clusters)

        capped = []
        for cluster_id in unique_clusters:
            mask = selected_clusters == cluster_id
            cluster_samples = selected[mask]

            if len(cluster_samples) > cap:
                # Random subsample
                cluster_samples = np.random.choice(cluster_samples, cap, replace=False)

            capped.extend(cluster_samples)

        return np.array(capped)
