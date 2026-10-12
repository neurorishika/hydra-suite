"""Density estimation utilities (moved to :mod:`hydra_suite.core.selection.density`).

Kept as a re-export so existing ClassKit imports keep working.
"""

from hydra_suite.core.selection.density import (  # noqa: F401
    compute_cluster_densities,
    compute_knn_density,
    select_diverse_samples,
)
