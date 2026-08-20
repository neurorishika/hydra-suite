"""End-to-end proof that arenas are tracked independently.

Tile one fixture clip into a 2x2 grid, declare each tile an arena, and require
each arena's trajectories to reproduce a run of that same tile *on its own*
exactly. Any cross-arena leak -- a shared per-frame local, a global identity
constraint, an ungrouped post-processing step -- shows up here as a mismatch.

The reference is the same video, not the untiled clip
-----------------------------------------------------
Tiling changes the frame geometry and every detector is sensitive to it, so
comparing a 2400px tiled run against a 1200px single-clip run would compare two
different detection problems -- and the only way to make that pass is a
positional tolerance wide enough to hide the leakage this file exists to find.

So the four reference runs use *the same tiled video*, each restricted to one
tile by an ROI (and therefore ``n_arenas == 1``). All five runs decode
byte-identical frames and detect on byte-identical geometry; the only
difference is how many arenas the tracking layer is told about. The expected
answer is then exact, and every comparison below is bit-for-bit on X/Y/Theta
with no tolerance at all.

Three fixture choices that the oracle's power depends on
--------------------------------------------------------
1. ``worm_bgsub``. The reference construction needs detections restricted to
   one tile, i.e. a working ROI gate. Background subtraction honours
   ``ROI_MASK`` (``core/inference/stages/bgsub.py`` intersects it with the
   foreground mask). The batched YOLO OBB path does **not**:
   ``core/tracking/worker.py`` calls ``InferenceRunner.load_frame`` with no
   mask, so ``filter_for_source`` gates on ``roi_mask=None`` and every
   detection in the frame survives. That is pre-existing ``main`` behaviour --
   no equivalence fixture sets ``roi_shapes``, so nothing covers it -- but it
   does mean a per-region reference run is only constructible on the bg-sub
   path today. Consequence, stated plainly: this oracle covers arena-blocked
   assignment, arena-gated respawn/bootstrap and arena-grouped
   post-processing, but NOT the per-arena identity decoder registry, because
   bg-sub runs cannot enable identity analysis at all ("Individual analysis
   requires YOLO OBB mode").

2. Arenas are exactly the tiles. bg-sub's lighting stabilisation computes its
   statistics over the ROI pixels, so the reference ROI must have the same
   pixel statistics as the full frame. Four identical tiles do; an arbitrary
   region does not.

3. A *mirror* tiling with a narrow gutter, not a plain repeat. Under
   ``np.tile`` this fixture's animals end up 1200px from any arena border, so
   every cross-arena pair is rejected on distance before arena membership is
   ever consulted. Reflecting each tile puts animals a few tens of pixels
   across the border instead -- inside the assignment gate -- so a cross-arena
   match is at least geometrically legal. Reflection also preserves each
   tile's pixel multiset, which the ROI-vs-full-frame comparison needs.

What this oracle demonstrably catches, and what it does not
----------------------------------------------------------
Verified by disabling one mechanism at a time and re-running (see the task
report). It catches:

* the free-detection bootstrap loop taking the first lost slot in *global*
  slot order (3 of the 5 tests fail);
* interpolated post-processing rows losing their ``arena_id``.

It does **not** catch, on this fixture, deleting the arena block from the cost
matrix or the arena gate from respawn: with all of them removed the output is
unchanged. That is not slack in the comparison -- it is exact -- it is that
those two gates never fire here. The Hungarian optimum never prefers a
cross-arena pair while each arena has at least as many slots as detections, and
respawn only sees an unassigned detection in the same circumstance. The regime
that would exercise them is an arena holding MORE detections than it has slots
-- which is unreachable in a valid reference run, because bg-sub's detection
cap is ``MAX_TARGETS`` itself, so a one-arena run can never see a surplus. The
same asymmetry is a defect in its own right (the 4-arena run's cap is
``n_arenas * animals_per_arena`` and is spent globally, so a crowded arena can
starve a quiet one); it is out of this file's scope and reported separately.
Cover those two gates with the assigner-level unit tests, not with this oracle.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

cv2 = pytest.importorskip("cv2")
pd = pytest.importorskip("pandas")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tools/equivalence/fixtures"
CLIP = FIXTURES / "clips/worm_bgsub.mp4"
CLIP_CONFIG = FIXTURES / "configs/worm_bgsub.json"

# 80 frames is not arbitrary: it is long enough for the bg-sub background model
# to settle, for tracks to be lost and respawned, and -- critically -- for the
# post-processing gap interpolation to fire. A shorter clip produces no
# interpolated rows, and the oracle then cannot see defects in that stage.
N_FRAMES = 80
# Per-arena animal budget. The fixture's own ``max_targets`` (10), used for both
# sides: the reference run gets ``max_targets=10``, the 4-arena run gets
# ``animals_per_arena=10`` and therefore ``MAX_TARGETS=40``.
ANIMALS_PER_ARENA = 10
N_ARENAS = 4
# (column, row) of each quadrant, in arena-id order.
QUADRANTS = ((0, 0), (1, 0), (0, 1), (1, 1))
# Assignment gates are widened to ~200 body lengths (~2500px, i.e. wider than
# the whole tiled frame) for BOTH sides of the comparison. This is what makes
# the oracle able to fail. At the fixture's own gate (~10 body lengths) every
# cross-arena pair is already rejected on distance alone, arenas being 1200px
# apart -- so the arena gates in the cost matrix and in respawn are dead weight
# and the oracle passes unchanged when you delete them (measured). Widening the
# distance gate makes a cross-arena match geometrically legal, leaving the
# arena gates as the only thing preventing it. It is applied identically to the
# multi-arena run and to every reference run, so it cannot mask a difference:
# it only removes an unrelated gate that was hiding one.
GATE_MULTIPLIER = 10.0
# Blank margin between tiles, in pixels. It has to clear the bg-sub morphology
# kernel (two objects a few px apart merge into ONE contour, which would make
# the 4-arena run detect something the reference runs cannot) while staying
# well inside the assignment gate above (~126px), so that a track in one arena
# and a detection in the next are close enough for a cross-arena match to be
# geometrically legal. 40px sits comfortably between the two.
GUTTER_PX = 40

pytestmark = pytest.mark.skipif(
    not CLIP.exists() or not CLIP_CONFIG.exists(),
    reason=(
        "equivalence fixtures not fetched -- run "
        "tools/equivalence/fixtures/fetch_fixtures.sh"
    ),
)

# Side outputs cost time and disk and prove nothing about arena independence.
_DISABLE = {
    "video_output_enabled": False,
    "enable_confidence_density_map": False,
    "enable_dataset_generation": False,
    "enable_individual_dataset": False,
    "enable_individual_image_save": False,
    "final_media_export_videos_enabled": False,
}


def _mirror_tile(frame: np.ndarray) -> np.ndarray:
    """Lay *frame* out as a 2x2 *mirror* tiling with a blank gutter.

    Each tile is a reflection of its neighbours across the shared seam, so
    every object near a tile edge acquires a near-mirror-image partner just
    across the arena border. A plain ``np.tile`` does not: the fixture's
    animals sit near the frame's top/left edges, whose copies land at the outer
    edges of the tiled frame, 1200px from any arena border -- and at that range
    the Kalman distance gate rejects every cross-arena pair on its own, leaving
    the arena gates unable to change any outcome.

    Reflection also preserves each tile's pixel multiset exactly, which the
    ROI-vs-full-frame comparison depends on wherever a bg-sub statistic is
    computed over ROI pixels.
    """
    h, w = frame.shape[:2]
    g = GUTTER_PX
    canvas = np.full((2 * h + g, 2 * w + g, frame.shape[2]), 255, frame.dtype)
    canvas[0:h, 0:w] = np.flip(frame, (0, 1))
    canvas[0:h, w + g :] = np.flip(frame, 0)
    canvas[h + g :, 0:w] = np.flip(frame, 1)
    canvas[h + g :, w + g :] = frame
    return canvas


def _tile_2x2(src: Path, dst: Path, n_frames: int) -> tuple[int, int, int]:
    """Write an ``n_frames``-long 2x2 mirror tiling of *src*; return (w,h,n)."""
    cap = cv2.VideoCapture(str(src))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(
            str(dst),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * 2 + GUTTER_PX, height * 2 + GUTTER_PX),
        )
        written = 0
        while written < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(_mirror_tile(frame))
            written += 1
        writer.release()
    finally:
        cap.release()
    return width, height, written


def _quadrant_shape(col: int, row: int, w: int, h: int, arena_id: int | None) -> dict:
    """One tile of the 2x2 tiling, as an ROI polygon.

    Arenas must be exactly the tiles, not some other partition. bg-sub's
    lighting stabilisation computes its statistics over the ROI pixels
    (``BackgroundModel.apply_lighting_stabilization``), so the reference run's
    ROI and the 4-arena run's ROI must have the same pixel statistics or the
    two runs detect differently and the comparison is invalid. Four identical
    tiles have exactly the statistics of one tile; any other region does not
    (measured: vertical-strip arenas shift detections by ~0.5px).
    """
    x0 = col * (w + GUTTER_PX)
    y0 = row * (h + GUTTER_PX)
    shape = {
        "type": "polygon",
        "mode": "include",
        "params": [
            [x0, y0],
            [x0 + w, y0],
            [x0 + w, y0 + h],
            [x0, y0 + h],
        ],
    }
    if arena_id is not None:
        shape["arena_id"] = arena_id
    return shape


def _run_tracking(
    video: Path,
    out_dir: Path,
    roi_shapes: list[dict],
    n_frames: int,
) -> str:
    """Run one headless tracking session into *out_dir*; return the CSV stem."""
    from hydra_suite.trackerkit.cli import run_tracking_cli

    out_dir.mkdir(parents=True, exist_ok=True)
    # The video is symlinked into out_dir so every derived artifact (detection
    # cache, CSVs) lands there and the runs cannot see each other's caches.
    link = out_dir / video.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(video.resolve())
    for stale in out_dir.glob(".inference_cache_*"):
        shutil.rmtree(stale, ignore_errors=True)

    with open(CLIP_CONFIG, encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg.update(_DISABLE)
    cfg["file_path"] = str(link)
    cfg["csv_path"] = str(out_dir / f"{link.stem}_tracking.csv")
    cfg["use_cached_detections"] = False
    cfg["start_frame"] = 0
    cfg["end_frame"] = n_frames - 1
    cfg["roi_shapes"] = roi_shapes
    # The solver auto-pick keys off the TOTAL slot count, so a 4-arena run and a
    # one-arena run would otherwise be free to choose different assignment
    # algorithms. Pin both explicitly (the fixture already does) so the oracle
    # compares tracking, not solver selection.
    cfg["enable_greedy_assignment"] = False
    cfg["enable_spatial_optimization"] = False
    cfg["max_distance_multiplier"] = GATE_MULTIPLIER
    cfg["kalman_max_velocity_multiplier"] = GATE_MULTIPLIER
    # Lighting stabilisation is the one bg-sub stage whose statistics are taken
    # over the ROI pixels rather than per-pixel, so it makes the detections
    # depend on the ROI's *shape* -- a percentile over one tile's pixels and
    # over four tiles' pixels can land between different samples and shift a
    # centroid by ~0.1px. Off, bg-sub is purely per-pixel plus an ROI
    # intersection and the reference construction is exact by construction
    # rather than by luck. Off on both sides, so it cannot mask a difference.
    cfg["enable_lighting_stabilization"] = False
    cfg["max_targets"] = ANIMALS_PER_ARENA
    if any("arena_id" in shape for shape in roi_shapes):
        cfg["animals_per_arena"] = ANIMALS_PER_ARENA

    config_path = out_dir / "oracle_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    assert (
        run_tracking_cli([str(link)], config_path=str(config_path)) == 0
    ), f"tracking run failed for {out_dir.name}"
    return link.stem


@pytest.fixture(scope="module")
def tiled_runs(tmp_path_factory):
    """Run the 4-arena tiling once plus one single-arena run per quadrant.

    Returns ``(multi_frames, ref_frames)`` where ``multi_frames`` maps CSV kind
    -> DataFrame for the 4-arena run and ``ref_frames[i]`` does the same for the
    reference run of quadrant ``i``.
    """
    base = tmp_path_factory.mktemp("arena_oracle")
    tiled = base / "worm_tiled.mp4"
    width, height, written = _tile_2x2(CLIP, tiled, N_FRAMES)
    assert written == N_FRAMES, f"tiled only {written}/{N_FRAMES} frames"

    arenas = [
        _quadrant_shape(col, row, width, height, idx)
        for idx, (col, row) in enumerate(QUADRANTS)
    ]
    stem = _run_tracking(tiled, base / "multi", arenas, written)

    def _load(run_dir: Path) -> dict:
        frames = {}
        for kind in ("forward", "backward", "final"):
            path = run_dir / f"{stem}_tracking_{kind}.csv"
            assert path.is_file(), f"missing {path}"
            frames[kind] = pd.read_csv(path)
            assert not frames[kind].empty, f"{path} has no rows"
        return frames

    multi = _load(base / "multi")
    refs = []
    for idx, (col, row) in enumerate(QUADRANTS):
        _run_tracking(
            tiled,
            base / f"ref{idx}",
            [_quadrant_shape(col, row, width, height, None)],
            written,
        )
        refs.append(_load(base / f"ref{idx}"))
    return multi, refs


def _slot_rows(df, slot: int):
    """One track slot's rows, frame-ordered."""
    return df[df["TrackID"] == slot].sort_values("FrameID").reset_index(drop=True)


def _canonical_trajectories(df) -> list:
    """Arena-order-independent canonical form of a set of trajectories.

    A trajectory becomes the tuple of its ``(frame, x, y, theta, state)`` rows;
    the set of trajectories becomes the sorted list of those tuples. Trajectory
    *ids* are deliberately not compared: they are globally unique, so the
    4-arena run necessarily numbers them differently. Everything that describes
    *which detections were linked into which track* is compared exactly.
    """
    out = []
    for _, group in df.groupby("TrajectoryID"):
        group = group.sort_values("FrameID")
        rows = []
        for frame, x, y, theta, state in zip(
            group["FrameID"],
            group["X"],
            group["Y"],
            group["Theta"],
            group["State"],
        ):
            rows.append(
                (
                    int(frame),
                    None if pd.isna(x) else float(x),
                    None if pd.isna(y) else float(y),
                    None if pd.isna(theta) else float(theta),
                    str(state),
                )
            )
        out.append(tuple(rows))
    return sorted(out, key=repr)


def test_arena_oracle_is_not_vacuous(tiled_runs):
    """Guard the oracle itself: an empty or degenerate run would pass anything."""
    multi, refs = tiled_runs
    forward = multi["forward"]
    assert "arena_id" in forward.columns
    assert set(forward["arena_id"].unique()) == set(range(N_ARENAS))
    for idx, ref in enumerate(refs):
        assert len(ref["forward"]) > 100, f"reference {idx} produced too few rows"
        assert (
            ref["forward"]["X"].notna().sum() > 100
        ), f"reference {idx} tracked nothing"
        assert ref["final"]["TrajectoryID"].nunique() >= 4

    # The final CSV must contain gap-interpolated rows (State == "occluded" with
    # a real position). Without them the post-processing interpolation stage is
    # never exercised and a defect there is invisible.
    final = multi["final"]
    interpolated = final[(final["State"] == "occluded") & final["X"].notna()]
    assert len(interpolated) > 0, "no interpolated rows -- interpolation untested"


def test_single_arena_reference_emits_no_arena_column(tiled_runs):
    """``arena_id`` is emitted only when ``n_arenas > 1``.

    The byte-identity gate compares column lists; an ``arena_id`` leaking into a
    single-arena run would break it on schema grounds alone.
    """
    _, refs = tiled_runs
    for idx, ref in enumerate(refs):
        for kind, frame in ref.items():
            assert (
                "arena_id" not in frame.columns
            ), f"reference {idx} {kind} CSV grew an arena_id column"


def _slot_history(df, slot: int) -> tuple:
    """One slot's whole life as a comparable tuple of per-frame rows."""
    rows = _slot_rows(df, slot)
    return tuple(
        (
            int(frame),
            None if pd.isna(x) else float(x),
            None if pd.isna(y) else float(y),
            None if pd.isna(theta) else float(theta),
            str(state),
        )
        for frame, x, y, theta, state in zip(
            rows["FrameID"], rows["X"], rows["Y"], rows["Theta"], rows["State"]
        )
    )


@pytest.mark.parametrize("kind", ["forward", "backward"])
def test_each_arena_reproduces_the_single_arena_run_slot_for_slot(tiled_runs, kind):
    """Frame-for-frame, bit-for-bit equality of every slot's life, per arena.

    Slots are laid out in contiguous per-arena blocks, so arena ``i`` owns
    global slots ``i * ANIMALS_PER_ARENA ...``; that block must live exactly
    the lives that the run which saw only that tile gave its own slots.

    Compared as a multiset rather than slot-index for slot-index: which of an
    arena's free slots bootstraps which new detection follows the frame's
    global detection ordering, so two runs can swap a pair of an arena's own
    slots. That is a relabelling inside one arena and carries no information
    across arenas; every position, angle and state still has to match exactly.
    """
    multi, refs = tiled_runs
    frame = multi[kind]
    for arena, ref in enumerate(refs):
        block = range(arena * ANIMALS_PER_ARENA, (arena + 1) * ANIMALS_PER_ARENA)
        # Sorted by ``repr`` because a row's X/Y/Theta are ``None`` when the
        # slot had no detection, and None is not orderable against a float.
        # repr round-trips floats exactly, so equal histories sort together.
        got = sorted((_slot_history(frame, slot) for slot in block), key=repr)
        want = sorted(
            (_slot_history(ref[kind], slot) for slot in range(ANIMALS_PER_ARENA)),
            key=repr,
        )
        assert (
            got == want
        ), f"{kind}: arena {arena}'s slots do not reproduce the single-arena run"
        assert (frame[frame["TrackID"].isin(block)]["arena_id"] == arena).all()


def test_each_arena_final_output_reproduces_the_single_arena_run(tiled_runs):
    """Post-processed trajectories: exact per-arena equality.

    This is the assertion that covers everything downstream of tracking --
    forward/backward resolution, relinking, identity post-processing and gap
    interpolation -- all of which must partition by arena.
    """
    multi, refs = tiled_runs
    final = multi["final"]
    for arena, ref in enumerate(refs):
        got = _canonical_trajectories(final[final["arena_id"] == arena])
        want = _canonical_trajectories(ref["final"])
        assert got == want, (
            f"arena {arena}'s post-processed trajectories differ from the "
            f"single-arena run ({len(got)} vs {len(want)} trajectories)"
        )


def test_no_output_row_loses_its_arena(tiled_runs):
    """Every emitted row carries an arena, and no trajectory spans two.

    Rows created downstream of tracking (gap interpolation reindexes a
    trajectory onto a contiguous frame range) must inherit their trajectory's
    arena; a NaN there silently drops the row out of every arena grouping.
    """
    multi, _ = tiled_runs
    for kind, frame in multi.items():
        assert frame["arena_id"].notna().all(), f"{kind}: rows with no arena_id"
        spans = frame.groupby("TrajectoryID")["arena_id"].nunique()
        assert (spans == 1).all(), (
            f"{kind}: trajectories spanning arenas: "
            f"{spans[spans > 1].index.tolist()}"
        )


def test_trajectory_ids_stay_globally_unique(tiled_runs):
    """Per-arena processing renumbers ids; the result must not collide."""
    multi, _ = tiled_runs
    final = multi["final"]
    per_arena = final.groupby("arena_id")["TrajectoryID"].nunique().sum()
    assert (
        final["TrajectoryID"].nunique() == per_arena
    ), "a TrajectoryID is reused by two arenas"
