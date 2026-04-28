"""Global identity fragment solver.

Replaces the HMM-based offline decoder with:
1. PELT changepoint detection on per-trajectory CNN probability matrices.
2. Fragment building from detected changepoints.
3. Global MILP assignment: maximises spatial continuity + CNN/tag evidence
   with a confidence-weighted online-label prior and a margin threshold.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from hydra_suite.core.identity.catalog import IdentityCatalog

log = logging.getLogger(__name__)

_LABEL_COL = "IdentityAssignedLabel"
_CONF_COL = "IdentityAssignedConfidence"
_UNKNOWN_VALUES = frozenset({"", "unknown"})


def detect_identity_changepoints(
    df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> dict[Any, list[int]]:
    """Return {traj_id: [split_frame_indices]} using PELT on CNN prob matrix.

    A split_frame_index is the *exclusive end* of a segment (i.e., the first
    frame of the following segment), matching the convention returned by
    ``ruptures`` ``predict()``.  Equivalently, segment k spans
    [split_indices[k-1], split_indices[k]).
    Trajectories with no CNN evidence or fewer than min_fragment_frames*2
    rows are returned with no splits.
    """
    raise NotImplementedError


def build_fragments(
    df: pd.DataFrame,
    changepoints: dict[Any, list[int]],
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Return a fragments DataFrame with one row per (traj_id, segment).

    Columns: TrajectoryID, FragmentID, StartFrame, EndFrame,
    StartX, StartY, EndX, EndY, MeanCNNProbs (dict serialised as object),
    OnlineLabel, OnlineConfidence.
    """
    raise NotImplementedError


def solve_global_assignment(
    fragments_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Return fragments_df with an added AssignedLabel column.

    Uses MILP with:
    - Spatial continuity score between consecutive fragments of the same identity.
    - CNN evidence score from MeanCNNProbs.
    - Online label prior: online_prior_weight * OnlineConfidence bonus for the
      online label column.
    - Margin threshold: only re-assign when best_score - second_best > threshold;
      otherwise keep OnlineLabel.
    - Uniqueness: at most one fragment per identity per overlapping time window.
    """
    raise NotImplementedError


def apply_fragment_labels(
    df: pd.DataFrame,
    fragments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Write AssignedLabel from fragments back into trajectories.

    Updates IdentityAssignedLabel, IdentityAssignedConfidence,
    IdentityAssignedID, IdentityCommitted in the trajectory DataFrame.
    Rows not covered by any fragment are unchanged.
    Returns a copy.
    """
    raise NotImplementedError


def run_fragment_solver(
    trajectories_df: pd.DataFrame,
    catalog: IdentityCatalog,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """End-to-end fragment solver: detect -> build -> assign -> apply.

    Parameters
    ----------
    trajectories_df : post-augmentation trajectory DataFrame.
    catalog : IdentityCatalog for the run.
    params : optional overrides. Keys:
        CHANGEPOINT_PENALTY          float  default 3.0
        MIN_FRAGMENT_FRAMES          int    default 5
        FRAGMENT_CNN_WEIGHT          float  default 0.40
        FRAGMENT_SPATIAL_WEIGHT      float  default 0.35
        ONLINE_PRIOR_WEIGHT          float  default 0.25
        ASSIGNMENT_MARGIN_THRESHOLD  float  default 0.10
        MAX_VELOCITY_BREAK           float  default 50.0
        TAG_IDENTITY_LABELS          list   default []
        FRAGMENT_SOLVER_ILP_TIME_LIMIT float default 30.0
    """
    raise NotImplementedError
