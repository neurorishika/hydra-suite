"""Task 12 / Fix 1: confidence-density regions must be computed per arena.

`ENABLE_CONFIDENCE_DENSITY_MAP` defaults True, and the pre-fix pipeline
accumulated ONE global density volume over every arena, then turned it into
region flags that drive a hard `cost = 1e9` gate in the assignment loop. That
coupled arenas two independent ways, both reproduced below as tests:

1. **Bounding-box spillover** -- a crowd wholly inside arena 0 yields a region
   whose rectangular bbox reaches across the arena wall, so an arena-1
   detection gets flagged by arena 0's crowd.
2. **Global-max coupling** -- `smooth_and_binarize` thresholds at
   `threshold * max(whole volume)`, so a dense arena raises the bar for every
   other arena and can erase a region that arena would have had entirely on
   its own. This needs no adjacency at all.

The property under test is the one the whole feature is defined by: an arena's
density flags must be a pure function of that arena's own detections. Each test
therefore compares a JOINT run (all arenas' detections) against a SOLO run (the
arena's own detections only) through the SAME layout, and demands identical
flags for that arena.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hydra_suite.core.tracking.arenas import ArenaLayout
from hydra_suite.core.tracking.confidence.confidence_density import (
    DensityRegion,
    compute_density_map_from_cache,
    load_regions,
    save_regions,
)
from hydra_suite.core.tracking.confidence.density import get_density_region_flags

FRAME_W, FRAME_H = 256, 128
N_FRAMES = 12
# Arena 0 = left half, arena 1 = right half, separated by a blank gutter so no
# arena's pixels touch another's.
WALL_X = 128
GUTTER = 2


def _two_arena_layout(animals_per_arena: int = 4) -> ArenaLayout:
    labels = np.zeros((FRAME_H, FRAME_W), dtype=np.uint16)
    labels[:, : WALL_X - GUTTER] = 1
    labels[:, WALL_X + GUTTER :] = 2
    return ArenaLayout(
        n_arenas=2, animals_per_arena=animals_per_arena, label_image=labels
    )


def _cache(points_per_frame, size: float = 64.0):
    """Build a {frame: (meas, confs, sizes)} cache from ``(x, y, conf)`` points.

    Confidence 0 means maximally uncertain, i.e. a full-weight Gaussian -- the
    density map's strongest possible signal. Confidence 1.0 means weight 0: the
    detection contributes NOTHING to the volume, so on its own it can never
    form a region. That is what isolates bbox spillover below from the
    global-max effect.
    """
    cache = {}
    for f in range(N_FRAMES):
        pts = points_per_frame
        cache[f] = (
            np.array([[x, y, 0.0] for x, y, _ in pts], dtype=np.float32),
            np.array([c for _, _, c in pts], dtype=np.float32),
            np.full(len(pts), size, dtype=np.float32),
        )
    return cache


def _compute(cache, layout, **over):
    kwargs = dict(
        frame_h=FRAME_H,
        frame_w=FRAME_W,
        sigma_scale=1.0,
        temporal_sigma=1.0,
        threshold=0.3,
        downsample_factor=2,
        min_frame_duration=3,
        min_area_px=4,
    )
    kwargs.update(over)
    cdm, _ = compute_density_map_from_cache(
        detection_cache=cache, arena_layout=layout, **kwargs
    )
    return cdm.regions


def _flags(regions, points, layout, frame_idx=6):
    xy = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
    meas = [[p[0], p[1], 0.0] for p in points]
    meas_arena = layout.arena_of_points(xy, frame_size=(FRAME_W, FRAME_H))
    return get_density_region_flags(
        meas, regions, frame_idx=frame_idx, meas_arena=meas_arena
    )


# ---------------------------------------------------------------------------
# Coupling 1: bounding-box spillover across the arena wall
# ---------------------------------------------------------------------------

# A tight crowd hugging arena 0's inner wall. Its Gaussians (and the bbox of the
# resulting component) reach past WALL_X into arena 1.
# Sigma is ``sqrt(size)/2``, so size 400 gives 10px-wide Gaussians whose
# component (and bbox) reaches ~15px past the crowd -- across the wall.
SPILL_SIZE = 400.0
CROWD_A0 = [(x, 64.0, 0.0) for x in (108.0, 112.0, 116.0, 120.0, 124.0)]
# A lone arena-1 detection just across the wall, inside the spilled bbox. Its
# confidence is 1.0 -- weight 0 -- so it contributes nothing of its own and
# ANY flag it receives must have come from another arena's detections.
PROBE_A1 = (132.0, 64.0, 1.0)


def test_arena1_detection_is_not_flagged_by_arena0_crowd():
    layout = _two_arena_layout()
    joint = _compute(_cache(CROWD_A0 + [PROBE_A1], size=SPILL_SIZE), layout)
    solo = _compute(_cache([PROBE_A1], size=SPILL_SIZE), layout)

    assert not solo, "fixture broken: the zero-weight probe must form no region"
    joint_flag = _flags(joint, [PROBE_A1], layout)
    solo_flag = _flags(solo, [PROBE_A1], layout)
    assert joint_flag.tolist() == solo_flag.tolist(), (
        "arena 1's flag for its own detection changed because arena 0 got "
        f"crowded: joint={joint_flag.tolist()} solo={solo_flag.tolist()}"
    )


def test_every_region_is_tagged_with_exactly_one_arena():
    layout = _two_arena_layout()
    regions = _compute(_cache(CROWD_A0 + [PROBE_A1], size=SPILL_SIZE), layout)
    assert regions, "fixture produced no regions -- the test proves nothing"
    assert all(r.arena in (0, 1) for r in regions)
    # And the crowd's region really does belong to arena 0.
    assert any(r.arena == 0 for r in regions)


def test_arena_tag_blocks_a_spilled_bbox_from_flagging_the_neighbour():
    """Direct check of the tag semantics, independent of the volume masking.

    Even a hand-built region whose bbox spans BOTH arenas may only flag
    detections of its own arena.
    """
    layout = _two_arena_layout()
    spanning = DensityRegion("region-1", 0, N_FRAMES, (0, 0, FRAME_W, FRAME_H), arena=0)
    flags = _flags([spanning], [(60.0, 64.0), PROBE_A1[:2]], layout)
    assert flags.tolist() == [True, False]


# ---------------------------------------------------------------------------
# Coupling 2: global-max coupling (no adjacency required)
# ---------------------------------------------------------------------------

# Arena 0 gets a big pile of coincident detections; arena 1 gets a small,
# isolated cluster far away (opposite side of the frame). Pre-fix, arena 0's
# pile raises the global max so far that arena 1's own cluster falls below
# `threshold * global_max` and its region vanishes.
PILE_A0 = [(40.0 + 0.3 * i, 64.0 + 0.3 * i, 0.0) for i in range(24)]
CLUSTER_A1 = [(200.0, 64.0, 0.0), (204.0, 64.0, 0.0)]
PROBE_A1_FAR = (202.0, 64.0)


def test_arena1_own_region_survives_a_crowded_arena0():
    layout = _two_arena_layout(animals_per_arena=32)
    joint = _compute(_cache(PILE_A0 + CLUSTER_A1), layout)
    solo = _compute(_cache(CLUSTER_A1), layout)

    solo_flag = _flags(solo, [PROBE_A1_FAR], layout)
    assert solo_flag.tolist() == [
        True
    ], "fixture broken: arena 1's cluster must be flagged when alone"

    joint_flag = _flags(joint, [PROBE_A1_FAR], layout)
    assert joint_flag.tolist() == solo_flag.tolist(), (
        "arena 1's own region was erased by arena 0's crowd raising the "
        "global binarisation maximum"
    )


def test_arena1_regions_are_bit_identical_joint_vs_solo():
    """The strong form: arena 1's whole region set, not just one flag."""
    layout = _two_arena_layout(animals_per_arena=32)
    joint = _compute(_cache(PILE_A0 + CLUSTER_A1), layout)
    solo = _compute(_cache(CLUSTER_A1), layout)

    def _key(regions):
        return sorted(
            (r.frame_start, r.frame_end, r.pixel_bbox, r.arena)
            for r in regions
            if r.arena == 1
        )

    assert _key(solo), "fixture broken: arena 1 must have regions on its own"
    assert _key(joint) == _key(solo)


# ---------------------------------------------------------------------------
# Detections outside every arena
# ---------------------------------------------------------------------------


def test_detection_outside_every_arena_is_never_flagged_by_a_tagged_region():
    layout = _two_arena_layout()
    gutter_pt = (float(WALL_X), 64.0)
    assert (
        int(
            layout.arena_of_points(
                np.array([gutter_pt], dtype=np.float32),
                frame_size=(FRAME_W, FRAME_H),
            )[0]
        )
        == -1
    ), "fixture broken: the probe point must be outside every arena"

    everywhere = DensityRegion(
        "region-1", 0, N_FRAMES, (0, 0, FRAME_W, FRAME_H), arena=0
    )
    assert _flags([everywhere], [gutter_pt], layout).tolist() == [False]


def test_out_of_arena_detections_contribute_to_no_arena_volume():
    layout = _two_arena_layout()
    gutter_crowd = [(float(WALL_X) + dx, 64.0, 0.0) for dx in (-1.0, 0.0, 1.0)]
    with_gutter = _compute(_cache(CLUSTER_A1 + gutter_crowd), layout)
    without = _compute(_cache(CLUSTER_A1), layout)

    def _key(regions):
        return sorted(
            (r.frame_start, r.frame_end, r.pixel_bbox, r.arena) for r in regions
        )

    assert _key(with_gutter) == _key(without)


# ---------------------------------------------------------------------------
# Single-arena inertness (the merge-blocking property)
# ---------------------------------------------------------------------------


def _single_layouts():
    labels = np.ones((FRAME_H, FRAME_W), dtype=np.uint16)
    return [
        None,
        ArenaLayout(n_arenas=1, animals_per_arena=4, label_image=None),
        ArenaLayout(n_arenas=1, animals_per_arena=4, label_image=labels),
        # Multi-arena bookkeeping but no label image -> nothing to mask by.
        ArenaLayout(n_arenas=2, animals_per_arena=4, label_image=None),
    ]


@pytest.mark.parametrize("layout", _single_layouts())
def test_single_arena_layouts_reproduce_the_untagged_whole_frame_path(layout):
    cache = _cache(CROWD_A0 + [PROBE_A1], size=SPILL_SIZE)
    baseline = _compute(cache, None)
    got = _compute(cache, layout)
    assert [
        (r.label, r.frame_start, r.frame_end, r.pixel_bbox, r.arena) for r in got
    ] == [
        (r.label, r.frame_start, r.frame_end, r.pixel_bbox, r.arena) for r in baseline
    ]
    assert all(r.arena is None for r in got)


def test_untagged_regions_flag_regardless_of_meas_arena():
    """An untagged region keeps whole-frame semantics even when arena ids are
    supplied -- this is what makes an OLD sidecar replay unchanged."""
    layout = _two_arena_layout()
    untagged = DensityRegion("region-1", 0, N_FRAMES, (0, 0, FRAME_W, FRAME_H))
    assert _flags([untagged], [(60.0, 64.0), PROBE_A1[:2]], layout).tolist() == [
        True,
        True,
    ]


# ---------------------------------------------------------------------------
# Persisted sidecar format
# ---------------------------------------------------------------------------


def test_untagged_region_sidecar_is_byte_identical_to_the_pre_arena_format(tmp_path):
    path = tmp_path / "confidence_regions.json"
    save_regions([DensityRegion("region-1", 0, 5, (1, 2, 3, 4))], path)
    payload = json.loads(path.read_text())
    assert payload == [
        {
            "label": "region-1",
            "frame_start": 0,
            "frame_end": 5,
            "pixel_bbox": [1, 2, 3, 4],
        }
    ], "single-arena sidecars must not gain an 'arena' key"


def test_old_sidecar_without_arena_key_loads_as_untagged(tmp_path):
    path = tmp_path / "confidence_regions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "label": "region-1",
                    "frame_start": 0,
                    "frame_end": 5,
                    "pixel_bbox": [1, 2, 3, 4],
                }
            ]
        )
    )
    (region,) = load_regions(path)
    assert region.arena is None


def test_arena_tag_survives_the_sidecar_round_trip(tmp_path):
    """A backward pass reads regions the forward pass wrote; if the tag did not
    survive, the backward pass would silently fall back to whole-frame gating.
    """
    path = tmp_path / "confidence_regions.json"
    layout = _two_arena_layout()
    regions = _compute(_cache(CROWD_A0 + [PROBE_A1], size=SPILL_SIZE), layout)
    assert any(r.arena is not None for r in regions)
    save_regions(regions, path)
    loaded = load_regions(path)
    assert [r.arena for r in loaded] == [r.arena for r in regions]
    assert [r.pixel_bbox for r in loaded] == [r.pixel_bbox for r in regions]


# ---------------------------------------------------------------------------
# Volume masking (the second, independent barrier)
# ---------------------------------------------------------------------------


def test_no_region_bbox_reaches_into_another_arena():
    """Masking each arena's volume to its own pixels means a Gaussian tail that
    leaks over the wall cannot enlarge the region's bounding box past it.

    The arena tag already makes cross-arena flagging impossible; this pins the
    independent geometric barrier, so a future change that drops one of the two
    is still caught.
    """
    layout = _two_arena_layout()
    labels = layout.label_image
    regions = _compute(_cache(CROWD_A0 + [PROBE_A1], size=SPILL_SIZE), layout)
    assert regions, "fixture produced no regions -- the test proves nothing"
    for r in regions:
        x1, y1, x2, y2 = r.pixel_bbox
        patch = labels[y1 : y2 + 1, x1 : x2 + 1]
        foreign = set(np.unique(patch)) - {0, r.arena + 1}
        assert not foreign, (
            f"{r.label} (arena {r.arena}, bbox {r.pixel_bbox}) covers pixels "
            f"belonging to arena(s) {sorted(f - 1 for f in foreign)}"
        )
