from hydra_suite.detectkit.jobs.sam2_prompts import (
    SourceBox,
    build_prompts,
    read_boxes_from_label,
)


def test_read_obb_label_to_pixel_box(tmp_path):
    p = tmp_path / "f.txt"
    # 9-field OBB: unit square from (0.1,0.1) to (0.3,0.3) in a 100x100 image
    p.write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")
    boxes = read_boxes_from_label(p, 100, 100)
    assert len(boxes) == 1
    assert boxes[0].aabb == (10.0, 10.0, 30.0, 30.0)
    assert boxes[0].center == (20.0, 20.0)


def test_read_aabb_label_to_pixel_box(tmp_path):
    p = tmp_path / "f.txt"
    # 5-field aabb: cx=0.5 cy=0.5 w=0.2 h=0.4 in 100x100 -> x[40,60] y[30,70]
    p.write_text("0 0.5 0.5 0.2 0.4\n")
    boxes = read_boxes_from_label(p, 100, 100)
    assert boxes[0].aabb == (40.0, 30.0, 60.0, 70.0)
    assert boxes[0].center == (50.0, 50.0)


def test_build_prompts_overlapping_neighbor_becomes_negative():
    a = SourceBox(aabb=(0, 0, 10, 10), center=(5, 5), polygon_px=[])
    b = SourceBox(aabb=(8, 8, 18, 18), center=(13, 13), polygon_px=[])  # overlaps a
    c = SourceBox(aabb=(50, 50, 60, 60), center=(55, 55), polygon_px=[])  # disjoint
    prompts = build_prompts([a, b, c])
    # prompt for a: box=a.aabb, positive=[a.center], negative=[b.center] (not c)
    assert prompts[0].box_xyxy == (0, 0, 10, 10)
    assert prompts[0].positive_points == [(5, 5)]
    assert prompts[0].negative_points == [(13, 13)]
    # prompt for c: no overlaps -> no negatives
    assert prompts[2].negative_points == []
