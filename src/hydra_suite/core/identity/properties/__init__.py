"""Properties caching and CSV export aggregation.

Correction 25 (Task 18a): IndividualPropertiesCache is re-exported with a
try/except guard so this package remains importable after the legacy cache
module is removed in Task 18.  New callers should use
hydra_suite.core.inference.cache.PoseCacheHandle instead.
"""

from hydra_suite.core.identity.properties.export import (
    POSE_SUMMARY_COLUMNS,
    augment_trajectories_with_pose_cache,
    augment_trajectories_with_pose_df,
    build_pose_keypoint_labels,
    build_pose_lookup_dataframe,
    merge_interpolated_apriltag_df,
    merge_interpolated_cnn_df,
    merge_interpolated_headtail_df,
    merge_interpolated_pose_df,
    pose_wide_columns_for_labels,
)

try:
    from hydra_suite.core.identity.properties.cache import (  # noqa: F401
        IndividualPropertiesCache,
    )
except ImportError:
    IndividualPropertiesCache = None  # type: ignore[assignment,misc]

__all__ = [
    "IndividualPropertiesCache",
    "POSE_SUMMARY_COLUMNS",
    "build_pose_keypoint_labels",
    "build_pose_lookup_dataframe",
    "augment_trajectories_with_pose_cache",
    "augment_trajectories_with_pose_df",
    "merge_interpolated_pose_df",
    "merge_interpolated_apriltag_df",
    "merge_interpolated_cnn_df",
    "merge_interpolated_headtail_df",
    "pose_wide_columns_for_labels",
]
