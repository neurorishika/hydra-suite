"""Classifier-independent trust for per-frame identity evidence.

Why this exists
---------------
An identity classifier is only as good as the crop it is shown. When two
animals touch, when a detection is a merged blob or a fragment, or when the
pose model cannot find the body, the crop is not "one animal, seen clearly"
and the classifier's answer -- however confident -- is not evidence about
that animal. Averaging such frames with good ones lets a trajectory that
lives in a crowd (or a piece of debris tracked for the whole video) build up
as much "support" as an animal that was seen clearly many times, and it then
holds a label a real animal needs.

This module decides, from geometry alone (never from the classifier), which
evidence frames are trustworthy, and collapses runs of near-identical frames
into *distinct views* so a stationary object seen 10 000 times does not
outvote an animal seen clearly from 20 different positions.

Every threshold is in body-length (BL) units, and the body length and the
single-animal shape prior are calibrated per video from that video's own
ISOLATED detections (a whole-video median is dominated by touching animals
in crowded videos). Each check switches itself off when the data cannot
support it:

- **isolation**: the nearest other detection centre in the same frame is
  farther than ``IDENTITY_TRUST_ISOLATION_BL`` body lengths. Needs detection
  geometry (the detection cache); without it the whole layer is inactive.
- **single-animal shape**: OBB area within ``+-IDENTITY_TRUST_AREA_TOLERANCE``
  of the prior area and elongation at least
  ``IDENTITY_TRUST_MIN_ASPECT_FRACTION`` of the prior elongation. Disabled
  when fewer than ``IDENTITY_TRUST_MIN_PRIOR_DETECTIONS`` isolated
  detections exist to form the prior.
- **keypoint completeness**: at least ``IDENTITY_TRUST_MIN_KEYPOINT_FRACTION``
  of the pose keypoints have confidence ``>= IDENTITY_TRUST_KEYPOINT_CONF``.
  Disabled when the run has no pose columns.

There is deliberately no stationarity filter: a resting animal is still an
animal, and the distinct-view collapse already stops a resting animal (or a
static object) from accumulating weight by repetition.

Core: numpy / pandas / scipy only; no Qt, no app-layer imports.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_POSE_CONF_RE = re.compile(r"^PoseKpt_(.+)_Conf$")


@dataclass(frozen=True)
class TrustSettings:
    """All evidence-trust knobs, read from engine params (see ``from_params``).

    Distances are in body lengths, angles in degrees, fractions in [0, 1].
    """

    enabled: bool = False
    isolation_bl: float = 1.0
    prior_isolation_bl: float = 2.0
    area_tolerance: float = 0.35
    min_aspect_fraction: float = 0.65
    min_prior_detections: int = 200
    keypoint_conf: float = 0.5
    min_keypoint_fraction: float = 0.75
    view_move_bl: float = 0.5
    view_turn_deg: float = 30.0
    min_views: int = 1
    split_min_log_odds: float = 8.0
    split_min_views: int = 3
    split_min_confidence: float = 0.5

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None) -> "TrustSettings":
        p = params or {}
        d = cls()

        def _f(key: str, default: float) -> float:
            try:
                return float(p.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        def _i(key: str, default: int) -> int:
            try:
                return int(p.get(key, default))
            except (TypeError, ValueError):
                return int(default)

        return cls(
            enabled=bool(p.get("IDENTITY_EVIDENCE_TRUST", d.enabled)),
            isolation_bl=_f("IDENTITY_TRUST_ISOLATION_BL", d.isolation_bl),
            prior_isolation_bl=_f(
                "IDENTITY_TRUST_PRIOR_ISOLATION_BL", d.prior_isolation_bl
            ),
            area_tolerance=_f("IDENTITY_TRUST_AREA_TOLERANCE", d.area_tolerance),
            min_aspect_fraction=_f(
                "IDENTITY_TRUST_MIN_ASPECT_FRACTION", d.min_aspect_fraction
            ),
            min_prior_detections=max(
                1,
                _i("IDENTITY_TRUST_MIN_PRIOR_DETECTIONS", d.min_prior_detections),
            ),
            keypoint_conf=_f("IDENTITY_TRUST_KEYPOINT_CONF", d.keypoint_conf),
            min_keypoint_fraction=_f(
                "IDENTITY_TRUST_MIN_KEYPOINT_FRACTION", d.min_keypoint_fraction
            ),
            view_move_bl=_f("IDENTITY_TRUST_VIEW_MOVE_BL", d.view_move_bl),
            view_turn_deg=_f("IDENTITY_TRUST_VIEW_TURN_DEG", d.view_turn_deg),
            min_views=max(1, _i("IDENTITY_TRUST_MIN_VIEWS", d.min_views)),
            split_min_log_odds=_f(
                "IDENTITY_TRUST_SPLIT_MIN_LOG_ODDS", d.split_min_log_odds
            ),
            split_min_views=max(
                1, _i("IDENTITY_TRUST_SPLIT_MIN_VIEWS", d.split_min_views)
            ),
            split_min_confidence=_f(
                "IDENTITY_TRUST_SPLIT_MIN_CONFIDENCE", d.split_min_confidence
            ),
        )


@dataclass
class DetectionGeometry:
    """Per-detection OBB geometry for one video, aligned arrays."""

    detection_ids: np.ndarray  # int64
    frames: np.ndarray  # int64
    cx: np.ndarray
    cy: np.ndarray
    major: np.ndarray
    minor: np.ndarray
    angle: np.ndarray  # radians, axial (OBB orientation)

    def __len__(self) -> int:
        return int(len(self.detection_ids))

    @classmethod
    def from_arrays(
        cls,
        *,
        detection_ids,
        frames,
        centroids,
        corners,
    ) -> "DetectionGeometry":
        corners = np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2)
        e1 = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
        e2 = np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1)
        centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
        # Orientation of the longer edge (axial).
        vec = np.where(
            (e1 >= e2)[:, None],
            corners[:, 1] - corners[:, 0],
            corners[:, 2] - corners[:, 1],
        )
        angles = np.arctan2(vec[:, 1], vec[:, 0])
        return cls(
            detection_ids=np.asarray(detection_ids, dtype=np.int64),
            frames=np.asarray(frames, dtype=np.int64),
            cx=centroids[:, 0],
            cy=centroids[:, 1],
            major=np.maximum(e1, e2),
            minor=np.minimum(e1, e2),
            angle=angles.astype(np.float64),
        )


def load_detection_geometry(detection_cache_path) -> DetectionGeometry | None:
    """Read every detection's OBB geometry from a ``detection.npz`` cache.

    Returns ``None`` (never raises) when the path is missing, unreadable, or
    carries no detections -- the trust layer then reports itself inactive.
    """
    if not detection_cache_path:
        return None
    try:
        from hydra_suite.core.inference.cache import open_detection_cache_reader

        handle = open_detection_cache_reader(detection_cache_path)
    except Exception:
        log.warning(
            "evidence trust: cannot open detection cache %s",
            detection_cache_path,
            exc_info=True,
        )
        return None
    # Orientation is always derived from the OBB's long edge: a cached angle
    # may follow a w/h convention that jumps by 90 degrees when the box's
    # sides swap, which would fake a "turn" for the distinct-view collapse.
    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("detection_ids", "frame_indices", "centroids", "corners")
    }
    try:
        if not handle.is_valid():
            return None
        for arrays in handle.iter_arrays():
            if not all(k in arrays for k in parts):
                continue
            if len(arrays["detection_ids"]) == 0:
                continue
            for k in parts:
                parts[k].append(np.asarray(arrays[k]))
    except Exception:
        log.warning(
            "evidence trust: failed reading detection cache %s",
            detection_cache_path,
            exc_info=True,
        )
        return None
    finally:
        try:
            handle.close()
        except Exception:
            pass
    if not parts["detection_ids"]:
        return None
    return DetectionGeometry.from_arrays(
        detection_ids=np.concatenate(parts["detection_ids"]),
        frames=np.concatenate(parts["frame_indices"]),
        centroids=np.concatenate(parts["centroids"]),
        corners=np.concatenate(parts["corners"]),
    )


def nearest_neighbour_distance(frames: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Distance from each detection to the nearest OTHER detection in its frame.

    ``inf`` for a detection alone in its frame.
    """
    from scipy.spatial import cKDTree

    n = len(frames)
    out = np.full(n, np.inf, dtype=np.float64)
    if n < 2:
        return out
    order = np.argsort(frames, kind="stable")
    f_sorted = frames[order]
    bounds = np.flatnonzero(np.diff(f_sorted)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, n]
    pts = np.column_stack([x[order], y[order]])
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        if e - s == 2:
            d = float(np.hypot(*(pts[s] - pts[s + 1])))
            out[order[s]] = out[order[s + 1]] = d
            continue
        dd, _ = cKDTree(pts[s:e]).query(pts[s:e], k=2)
        out[order[s:e]] = dd[:, 1]
    return out


@dataclass
class TrustCalibration:
    """What the trust layer learned about one video (logged once)."""

    body_length: float
    body_length_all: float
    prior_area: float
    prior_aspect: float
    n_detections: int
    n_isolated: int
    isolation_active: bool
    shape_active: bool
    keypoints_active: bool
    keypoint_columns: tuple[str, ...] = ()


def calibrate(
    geom: DetectionGeometry, settings: TrustSettings, keypoint_columns=()
) -> tuple[TrustCalibration, np.ndarray]:
    """Per-video body length + single-animal shape prior from isolated detections.

    Returns the calibration and the per-detection nearest-neighbour distance
    (pixels), reused by :func:`build_evidence_trust`.
    """
    nn = nearest_neighbour_distance(geom.frames, geom.cx, geom.cy)
    valid = np.isfinite(geom.major) & (geom.major > 0) & (geom.minor > 0)
    bl_all = float(np.median(geom.major[valid])) if valid.any() else float("nan")
    isolated = valid & (nn > settings.prior_isolation_bl * bl_all)
    n_iso = int(isolated.sum())
    shape_active = n_iso >= settings.min_prior_detections
    if shape_active:
        bl = float(np.median(geom.major[isolated]))
        area = float(np.median(geom.major[isolated] * geom.minor[isolated]))
        aspect = float(
            np.median(geom.major[isolated] / np.clip(geom.minor[isolated], 1e-3, None))
        )
    else:
        bl, area, aspect = bl_all, float("nan"), float("nan")
    cal = TrustCalibration(
        body_length=bl,
        body_length_all=bl_all,
        prior_area=area,
        prior_aspect=aspect,
        n_detections=len(geom),
        n_isolated=n_iso,
        isolation_active=bool(math.isfinite(bl) and bl > 0),
        shape_active=bool(shape_active),
        keypoints_active=bool(keypoint_columns),
        keypoint_columns=tuple(keypoint_columns),
    )
    return cal, nn


def pose_conf_columns(df: pd.DataFrame) -> list[str]:
    """``PoseKpt_<name>_Conf`` columns that carry at least one value."""
    cols = [c for c in df.columns if _POSE_CONF_RE.match(str(c))]
    return [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().any()]


@dataclass
class EvidenceTrust:
    """Per-video trust result consumed by the fragment solver.

    ``trusted`` maps a trusted DetectionID to its ``(x, y, angle)`` (cache
    pixel units, axial radians), which is all the distinct-view collapse
    needs. DetectionIDs are unique per video, so this survives trajectory
    renumbering, PELT splits and per-arena solving.
    """

    settings: TrustSettings
    calibration: TrustCalibration | None
    trusted: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    n_candidate_rows: int = 0
    reason_inactive: str = ""

    @property
    def active(self) -> bool:
        return (
            self.settings.enabled
            and self.calibration is not None
            and self.calibration.isolation_active
            and not self.reason_inactive
        )

    def summary(self) -> dict[str, Any]:
        cal = self.calibration
        out: dict[str, Any] = {
            "active": self.active,
            "reason_inactive": self.reason_inactive,
            "trusted_rows": len(self.trusted),
            "candidate_rows": self.n_candidate_rows,
            "trusted_fraction": (
                len(self.trusted) / self.n_candidate_rows
                if self.n_candidate_rows
                else 0.0
            ),
        }
        if cal is not None:
            out.update(
                body_length_px=cal.body_length,
                body_length_all_px=cal.body_length_all,
                prior_area_px2=cal.prior_area,
                prior_aspect=cal.prior_aspect,
                n_isolated_for_prior=cal.n_isolated,
                checks={
                    "isolation": cal.isolation_active,
                    "shape": cal.shape_active,
                    "keypoints": cal.keypoints_active,
                },
            )
        return out


def build_evidence_trust(
    df: pd.DataFrame,
    params: Mapping[str, Any] | None,
    *,
    detection_cache_path=None,
    geometry: DetectionGeometry | None = None,
) -> EvidenceTrust:
    """Decide which trajectory rows carry trustworthy identity evidence.

    Run ONCE per video, over the whole trajectory frame (before any per-arena
    split), so the calibration is per video. ``geometry`` may be supplied
    directly (tests); otherwise it is read from ``detection_cache_path``.
    """
    settings = TrustSettings.from_params(params)
    if not settings.enabled:
        return EvidenceTrust(settings, None, reason_inactive="disabled")
    if df is None or df.empty or "DetectionID" not in df.columns:
        return EvidenceTrust(settings, None, reason_inactive="no detection-backed rows")
    if geometry is None:
        geometry = load_detection_geometry(detection_cache_path)
    if geometry is None or len(geometry) == 0:
        trust = EvidenceTrust(
            settings, None, reason_inactive="no detection geometry (detection cache)"
        )
        log.warning(
            "evidence trust: enabled but no detection geometry is available "
            "(detection cache %s) -- trust layer inactive, solver uses the "
            "untiered path.",
            detection_cache_path,
        )
        return trust

    kp_cols = pose_conf_columns(df)
    cal, nn = calibrate(geometry, settings, kp_cols)
    if not cal.isolation_active:
        return EvidenceTrust(settings, cal, reason_inactive="no valid body length")

    bl = cal.body_length
    det_ok = nn > settings.isolation_bl * bl
    if cal.shape_active:
        area_r = geometry.major * geometry.minor / cal.prior_area
        asp_r = geometry.major / np.clip(geometry.minor, 1e-3, None) / cal.prior_aspect
        # Elongation has no upper bound: a walking animal is long.
        det_ok &= (np.abs(area_r - 1.0) <= settings.area_tolerance) & (
            asp_r >= settings.min_aspect_fraction
        )
    pos = pd.Series(np.arange(len(geometry)), index=geometry.detection_ids)
    pos = pos[~pos.index.duplicated(keep="first")]

    rows = df[pd.notna(df["DetectionID"])]
    det_ids = rows["DetectionID"].astype(np.int64).to_numpy()
    gidx = pos.reindex(det_ids).to_numpy()
    has_geom = ~np.isnan(gidx)
    ok = np.zeros(len(rows), dtype=bool)
    gi = gidx[has_geom].astype(np.int64)
    ok[has_geom] = det_ok[gi]
    if cal.keypoints_active:
        conf = rows[list(kp_cols)].apply(pd.to_numeric, errors="coerce").to_numpy()
        frac = (conf >= settings.keypoint_conf).sum(axis=1) / max(1, conf.shape[1])
        ok &= frac >= settings.min_keypoint_fraction

    trusted: dict[int, tuple[float, float, float]] = {}
    sel = np.flatnonzero(ok)
    g_sel = gidx[sel].astype(np.int64)
    for det_id, g in zip(det_ids[sel], g_sel):
        trusted[int(det_id)] = (
            float(geometry.cx[g]),
            float(geometry.cy[g]),
            float(geometry.angle[g]),
        )
    trust = EvidenceTrust(settings, cal, trusted=trusted, n_candidate_rows=len(rows))
    s = trust.summary()
    log.info(
        "evidence trust: %d/%d detection-backed rows trusted (%.1f%%); body "
        "length %.1f px from %d isolated detections (all-detection median "
        "%.1f px); prior area %.0f px^2, aspect %.2f; checks: isolation=%s "
        "shape=%s keypoints=%s (%d keypoint columns)",
        s["trusted_rows"],
        s["candidate_rows"],
        100.0 * s["trusted_fraction"],
        cal.body_length,
        cal.n_isolated,
        cal.body_length_all,
        cal.prior_area,
        cal.prior_aspect,
        cal.isolation_active,
        cal.shape_active,
        cal.keypoints_active,
        len(cal.keypoint_columns),
    )
    if not cal.shape_active:
        log.info(
            "evidence trust: shape check disabled -- only %d isolated "
            "detections (< %d) to form a single-animal prior.",
            cal.n_isolated,
            settings.min_prior_detections,
        )
    return trust


def _axial_turn_deg(a: float, b: float) -> float:
    """Smallest angle between two axial (mod pi) orientations, in degrees."""
    d = (a - b) % math.pi
    return math.degrees(min(d, math.pi - d))


def collapse_views(
    frames: Sequence[int],
    xs: Sequence[float],
    ys: Sequence[float],
    angles: Sequence[float],
    body_length: float,
    settings: TrustSettings,
) -> list[list[int]]:
    """Group consecutive trusted frames into distinct views.

    A new view starts when the animal has moved at least
    ``view_move_bl`` body lengths from the current view's anchor or turned
    at least ``view_turn_deg`` (axially). Returns lists of positions into
    the input sequences, in frame order.
    """
    n = len(frames)
    if n == 0:
        return []
    order = np.argsort(np.asarray(frames), kind="stable")
    move = settings.view_move_bl * body_length
    views: list[list[int]] = []
    cur = [int(order[0])]
    ax, ay, aa = xs[order[0]], ys[order[0]], angles[order[0]]
    for k in order[1:]:
        k = int(k)
        moved = math.hypot(xs[k] - ax, ys[k] - ay) >= move
        turned = (
            math.isfinite(angles[k])
            and math.isfinite(aa)
            and _axial_turn_deg(angles[k], aa) >= settings.view_turn_deg
        )
        if moved or turned:
            views.append(cur)
            cur = [k]
            ax, ay, aa = xs[k], ys[k], angles[k]
        else:
            cur.append(k)
    views.append(cur)
    return views


@dataclass
class FragmentViews:
    """Distinct trusted views of one fragment."""

    view_frames: list[tuple[int, int]]  # (first, last) FrameID per view
    view_probs: np.ndarray  # (V, n_known) absolute posterior per view

    @property
    def n_views(self) -> int:
        return int(len(self.view_frames))

    def support(self, known_labels: Sequence[str]) -> dict[str, float]:
        if self.n_views == 0:
            return {}
        mean = self.view_probs.mean(axis=0)
        return {lbl: float(mean[i]) for i, lbl in enumerate(known_labels)}


def fragment_views(
    rows: pd.DataFrame,
    raw_sequence: Sequence[tuple[int, np.ndarray]],
    trust: EvidenceTrust,
) -> FragmentViews:
    """Trusted distinct views for one fragment's rows.

    ``raw_sequence`` is the fragment's RAW (unsmoothed) ``[(FrameID,
    catalog_log_probs)]`` evidence: smoothing would leak untrusted frames'
    evidence into trusted ones. A view's posterior is the mean absolute
    posterior (known labels only; mass on ``unknown`` stays out) of its
    frames.
    """
    cal = trust.calibration
    empty = FragmentViews([], np.zeros((0, 0)))
    if not trust.active or cal is None or not raw_sequence:
        return empty
    ev = {int(f): lp for f, lp in raw_sequence}
    sub = rows[pd.notna(rows["DetectionID"])]
    frames, xs, ys, angs, probs = [], [], [], [], []
    for f, d in zip(sub["FrameID"].to_numpy(), sub["DetectionID"].to_numpy()):
        geo = trust.trusted.get(int(d))
        lp = ev.get(int(f))
        if geo is None or lp is None:
            continue
        frames.append(int(f))
        xs.append(geo[0])
        ys.append(geo[1])
        angs.append(geo[2])
        p = np.exp(np.asarray(lp, dtype=np.float64) - np.logaddexp.reduce(lp))
        probs.append(p[1:])
    if not frames:
        return empty
    groups = collapse_views(frames, xs, ys, angs, cal.body_length, trust.settings)
    P = np.asarray(probs)
    view_frames = [(frames[g[0]], frames[g[-1]]) for g in groups]
    view_probs = np.stack([P[g].mean(axis=0) for g in groups])
    return FragmentViews(view_frames, view_probs)


def find_label_switches(
    views: FragmentViews,
    settings: TrustSettings,
    *,
    _depth: int = 0,
    _max_depth: int = 8,
) -> list[int]:
    """Cut frames where a trajectory's TRUSTED evidence switches label.

    A cut between views ``k-1`` and ``k`` is a candidate when the
    ``split_min_views`` views just before it favour one label ``A`` and the
    ``split_min_views`` views just after it a different label ``B``. Each
    side's span is then extended outward over every view that still prefers
    its own label to the other one, and the cut is accepted only if

    - both spans are confident: mean absolute posterior of ``A`` on the left
      span and of ``B`` on the right span ``>= split_min_confidence``, and
    - the two-label explanation beats the best single-label explanation of
      the joined span by at least ``split_min_log_odds`` nats of evidence::

          gain = sum_left log p_v(A) + sum_right log p_v(B) - max_L sum_span log p_v(L)

    (per-view probabilities floored at 1e-3, so one view cannot dominate).
    The best cut is placed in the middle of the (untrusted) gap between the
    two views -- never on a trusted frame -- and both sides are searched
    again, so several switches in one trajectory are all found. Returned
    values are inclusive segment ends (``split_trajectories_at_changepoints``
    convention).
    """
    V = views.n_views
    m = settings.split_min_views
    if V < 2 * m or _depth >= _max_depth:
        return []
    P = views.view_probs
    logp = np.log(np.clip(P, 1e-3, 1.0))
    best_gain, best_k = -np.inf, -1
    for k in range(m, V - m + 1):
        a = int(np.argmax(P[k - m : k].mean(axis=0)))
        b = int(np.argmax(P[k : k + m].mean(axis=0)))
        if a == b:
            continue
        lo = k - m
        while lo > 0 and P[lo - 1, a] >= P[lo - 1, b]:
            lo -= 1
        hi = k + m
        while hi < V and P[hi, b] >= P[hi, a]:
            hi += 1
        if (
            P[lo:k, a].mean() < settings.split_min_confidence
            or P[k:hi, b].mean() < settings.split_min_confidence
        ):
            continue
        single = float(logp[lo:hi].sum(axis=0).max())
        gain = float(logp[lo:k, a].sum() + logp[k:hi, b].sum() - single)
        if gain > best_gain:
            best_gain, best_k = gain, k
    if best_k < 0 or best_gain < settings.split_min_log_odds:
        return []
    left_end = views.view_frames[best_k - 1][1]
    right_start = views.view_frames[best_k][0]
    cut = int((left_end + right_start) // 2)
    cut = min(max(cut, left_end), right_start - 1)
    left = FragmentViews(views.view_frames[:best_k], P[:best_k])
    right = FragmentViews(views.view_frames[best_k:], P[best_k:])
    return sorted(
        find_label_switches(left, settings, _depth=_depth + 1)
        + [cut]
        + find_label_switches(right, settings, _depth=_depth + 1)
    )
