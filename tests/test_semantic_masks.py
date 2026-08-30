import numpy as np

from hydra_suite.core.inference.masks import polygon_iou


def _square(x0, y0, side):
    return np.array(
        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
        dtype=np.float32,
    )


def test_identical_squares_iou_is_one():
    a = _square(10, 10, 20)
    assert polygon_iou(a, a.copy()) == 1.0


def test_disjoint_squares_iou_is_zero():
    assert polygon_iou(_square(0, 0, 10), _square(50, 50, 10)) == 0.0


def test_half_overlapping_squares_iou_is_one_third():
    # Two 20x20 squares offset by 10 in x: intersection 200, union 600.
    iou = polygon_iou(_square(0, 0, 20), _square(10, 0, 20))
    assert abs(iou - 1.0 / 3.0) < 0.02


def test_non_convex_polygons_do_not_overlap_in_their_concavity():
    # Two interlocking L/U shapes whose bounding boxes overlap heavily but
    # whose filled areas do not. A convex-hull IoU would report a large
    # overlap here; a rasterized one reports ~0.
    u_shape = np.array(
        [[0, 0], [30, 0], [30, 30], [20, 30], [20, 10], [10, 10], [10, 30], [0, 30]],
        dtype=np.float32,
    )
    plug = np.array([[12, 14], [18, 14], [18, 30], [12, 30]], dtype=np.float32)
    assert polygon_iou(u_shape, plug) == 0.0


def test_degenerate_polygon_iou_is_zero():
    two_points = np.array([[0, 0], [10, 10]], dtype=np.float32)
    assert polygon_iou(two_points, _square(0, 0, 10)) == 0.0
    assert polygon_iou(np.zeros((0, 2), dtype=np.float32), _square(0, 0, 10)) == 0.0


def test_disjoint_far_apart_polygons_short_circuit_without_allocating():
    """F3: two tiny polygons at opposite corners of a 4512^2 frame.

    Without a disjoint-bbox early-out this rasterizes two (17680)^2 uint8
    canvases (~625 MB, ~65 ms) to compute a guaranteed 0.0. Fails on time
    (and, on a constrained box, on memory) before the fix.
    """
    import time

    a = _square(0, 0, 20)
    b = _square(4492, 4492, 20)
    started = time.perf_counter()
    for _ in range(50):
        assert polygon_iou(a, b) == 0.0
        assert polygon_iou(b, a) == 0.0
    elapsed = time.perf_counter() - started
    # 100 calls; the unfixed path costs ~65 ms EACH (>6 s total).
    assert elapsed < 0.5, f"disjoint polygon_iou is rasterizing: {elapsed:.3f}s"


def test_early_out_preserves_overlap_and_exactifies_edge_touching():
    """Genuine overlaps are untouched; edge-touching becomes exactly 0.0.

    The one behaviour change: edge-touching squares previously scored
    0.00625 from the rasterizer's shared boundary sliver. The bbox test
    treats a shared edge as disjoint (zero-area intersection), which is the
    geometrically correct answer and strictly below any merge threshold, so
    no merge decision can flip.
    """
    assert polygon_iou(_square(0, 0, 20), _square(10, 0, 20)) > 0.0
    assert polygon_iou(_square(0, 0, 20), _square(20, 0, 20)) == 0.0
    # Bboxes overlap but shapes are far in one axis only.
    assert polygon_iou(_square(0, 0, 20), _square(0, 4000, 20)) == 0.0
