"""sklearn-based clustering (moved to :mod:`hydra_suite.core.selection.clustering`).

Kept as a re-export so existing ClassKit imports keep working.
"""

from hydra_suite.core.selection.clustering import (  # noqa: F401
    ClusterStats,
    SKLearnClusterer,
    _build_knn_fn,
)
