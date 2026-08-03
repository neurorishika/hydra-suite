"""Unit tests for the external-ViTPose-checkpoint probe tool."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tools.vitpose.external_ckpt.crops import crop_matrix, select_samples, warp_crop
from tools.vitpose.external_ckpt.model import (
    HEATMAP_PX,
    IMAGE_PX,
    build_external_vitpose,
    decode_default,
    infer_num_keypoints,
    load_external_checkpoint,
    preprocess,
)
from tools.vitpose.external_ckpt.skeleton import builtin_skeleton


def test_ant_skeleton_has_nine_named_keypoints():
    spec = builtin_skeleton("ant")
    assert spec.num_keypoints == 9
    assert spec.keypoint_names == [
        "A_R_T",
        "A_L_T",
        "A_R_M",
        "A_L_M",
        "Head_T",
        "Centroid",
        "Abd_T",
        "Abd_B",
        "Head_B",
    ]
    assert spec.skeleton_edges == [
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 8),
        (8, 5),
        (5, 6),
        (6, 7),
    ]


def test_fly_skeleton_has_twentynine_keypoints_and_legs():
    spec = builtin_skeleton("fly")
    assert spec.num_keypoints == 29
    assert spec.keypoint_names[:4] == [
        "headTop",
        "thoraxCenter",
        "abdomenTop",
        "abdomenCenter",
    ]
    assert "hindlegRight" in spec.keypoint_names
    assert len(spec.skeleton_edges) == 28


def test_skeleton_colors_are_bgr_reversed_from_config_rgb():
    # ant keypoint 0 (A_R_T) is RGB [148, 0, 211] in the mmpose config.
    spec = builtin_skeleton("ant")
    assert spec.keypoint_colors_bgr[0] == (211, 0, 148)
    assert len(spec.keypoint_colors_bgr) == spec.num_keypoints
    assert len(spec.edge_colors_bgr) == len(spec.skeleton_edges)


def test_edges_index_within_range():
    for species in ("ant", "fly"):
        spec = builtin_skeleton(species)
        for a, b in spec.skeleton_edges:
            assert 0 <= a < spec.num_keypoints
            assert 0 <= b < spec.num_keypoints


def test_unknown_species_rejected():
    with pytest.raises(ValueError, match="unknown species"):
        builtin_skeleton("beetle")


def _apply(matrix, x, y):
    v = np.array([x, y, 1.0], dtype=np.float64)
    return (float(matrix[0] @ v), float(matrix[1] @ v))


def test_crop_matrix_maps_center_to_output_center():
    for rotate in (False, True):
        m = crop_matrix(
            cx=100.0,
            cy=50.0,
            theta=1.234,
            side_px=80.0,
            out_px=256,
            rotate=rotate,
        )
        out = _apply(m, 100.0, 50.0)
        assert out == pytest.approx((128.0, 128.0), abs=1e-4)


def test_axis_mode_is_pure_scale_and_translate():
    m = crop_matrix(
        cx=100.0,
        cy=50.0,
        theta=2.0,
        side_px=80.0,
        out_px=256,
        rotate=False,
    )
    # Right edge of the source square maps to the right edge of the crop.
    assert _apply(m, 140.0, 50.0) == pytest.approx((256.0, 128.0), abs=1e-4)
    # Bottom edge maps to the bottom edge -- no rotation regardless of theta.
    assert _apply(m, 100.0, 90.0) == pytest.approx((128.0, 256.0), abs=1e-4)


@pytest.mark.parametrize("theta", [0.0, 1.0, 2.5, -0.7, math.pi])
def test_rotate_mode_puts_heading_at_top_of_crop(theta):
    cx, cy, side = 100.0, 50.0, 80.0
    m = crop_matrix(cx, cy, theta, side_px=side, out_px=256, rotate=True)
    # A point half a crop-width ahead along the heading...
    ahead_x = cx + (side / 2.0) * math.cos(theta)
    ahead_y = cy + (side / 2.0) * math.sin(theta)
    # ...must land at top-center of the crop.
    assert _apply(m, ahead_x, ahead_y) == pytest.approx((128.0, 0.0), abs=1e-3)


def test_warp_crop_returns_square_output_of_requested_size():
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    frame[40:60, 90:110] = 255
    m = crop_matrix(100.0, 50.0, 0.0, side_px=80.0, out_px=256, rotate=False)
    crop = warp_crop(frame, m, 256)
    assert crop.shape == (256, 256, 3)
    assert crop.dtype == np.uint8
    assert crop[128, 128].tolist() == [255, 255, 255]


def _write_csv(tmp_path, rows):
    header = "TrajectoryID,X,Y,Theta,FrameID,State\n"
    body = "".join(f"{t},{x},{y},{th},{f},{s}\n" for t, x, y, th, f, s in rows)
    p = tmp_path / "track.csv"
    p.write_text(header + body)
    return p


def test_select_samples_returns_requested_count_and_spreads_over_frames(tmp_path):
    rows = []
    for frame in range(0, 100):
        for track in range(3):
            rows.append((track, 10 * track, 20 + frame, 0.5, frame, "active"))
    csv = _write_csv(tmp_path, rows)
    samples = select_samples(csv, n=12)
    assert len(samples) == 12
    frames = [s.frame_id for s in samples]
    assert frames == sorted(frames)
    assert len(set(frames)) == 12
    # Spread across the whole range, not clustered at the start.
    assert min(frames) < 10 and max(frames) > 89


def test_select_samples_varies_track_ids(tmp_path):
    rows = []
    for frame in range(0, 60):
        for track in range(4):
            rows.append((track, 10 * track, 20, 0.1, frame, "active"))
    csv = _write_csv(tmp_path, rows)
    samples = select_samples(csv, n=8)
    assert len({s.track_id for s in samples}) > 1


def test_select_samples_ignores_non_active_rows(tmp_path):
    rows = [(0, 1, 2, 0.0, f, "tentative") for f in range(50)]
    rows += [(1, 3, 4, 0.0, f, "active") for f in range(50)]
    csv = _write_csv(tmp_path, rows)
    samples = select_samples(csv, n=5)
    assert all(s.track_id == 1 for s in samples)


def test_select_samples_is_deterministic(tmp_path):
    rows = []
    for frame in range(0, 80):
        for track in range(3):
            rows.append((track, 5 * track, 7, 0.3, frame, "active"))
    csv = _write_csv(tmp_path, rows)
    assert select_samples(csv, n=9) == select_samples(csv, n=9)


def test_select_samples_raises_when_no_active_rows(tmp_path):
    csv = _write_csv(tmp_path, [(0, 1, 2, 0.0, 1, "lost")])
    with pytest.raises(ValueError, match="no active"):
        select_samples(csv, n=4)


def test_model_is_built_for_256_square_input():
    model = build_external_vitpose(9)
    assert model.backbone.pos_embed.shape == (1, 257, 768)
    assert model.keypoint_head.final_layer.weight.shape == (9, 256, 1, 1)


def test_model_forward_produces_64x64_heatmaps():
    model = build_external_vitpose(9).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, IMAGE_PX, IMAGE_PX))
    assert out.shape == (2, 9, HEATMAP_PX, HEATMAP_PX)


def test_infer_num_keypoints_reads_final_layer():
    model = build_external_vitpose(29)
    assert infer_num_keypoints(model.state_dict()) == 29


def test_load_external_checkpoint_round_trips_strictly(tmp_path):
    model = build_external_vitpose(9)
    ckpt = tmp_path / "fake.pth"
    torch.save({"state_dict": model.state_dict()}, ckpt)
    loaded, k = load_external_checkpoint(ckpt)
    assert k == 9
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)


def test_load_external_checkpoint_rejects_unexpected_keys(tmp_path):
    model = build_external_vitpose(9)
    state = dict(model.state_dict())
    state["backbone.bogus_key"] = torch.zeros(1)
    ckpt = tmp_path / "bad.pth"
    torch.save(state, ckpt)
    with pytest.raises(RuntimeError, match="bogus_key"):
        load_external_checkpoint(ckpt)


def test_load_external_checkpoint_handles_mmpose_meta_with_numpy_scalars(tmp_path):
    # Real collaborator checkpoints are mmpose blobs shaped
    # {"meta": {...numpy scalars...}, "state_dict": ..., "optimizer": ...}.
    # weights_only=True rejects numpy scalars in `meta` unless the loader
    # allowlists them at import time -- this reproduces that shape.
    model = build_external_vitpose(9)
    ckpt = tmp_path / "mmpose_like.pth"
    torch.save(
        {
            "meta": {"epoch": np.float64(1.5)},
            "state_dict": model.state_dict(),
        },
        ckpt,
    )
    loaded, k = load_external_checkpoint(ckpt)
    assert k == 9
    assert isinstance(loaded, type(model))


def test_preprocess_normalizes_rgb_with_imagenet_stats():
    crop = np.zeros((IMAGE_PX, IMAGE_PX, 3), dtype=np.uint8)
    crop[:, :, 2] = 255  # pure red in BGR
    out = preprocess(crop)
    assert out.shape == (3, IMAGE_PX, IMAGE_PX)
    assert out.dtype == np.float32
    # R channel: (1.0 - 0.485) / 0.229
    assert out[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-4)
    # G channel: (0.0 - 0.456) / 0.224
    assert out[1, 0, 0] == pytest.approx((0.0 - 0.456) / 0.224, abs=1e-4)


def test_decode_default_recovers_peak_scaled_to_crop_pixels():
    hm = np.zeros((1, 1, HEATMAP_PX, HEATMAP_PX), dtype=np.float32)
    hm[0, 0, 20, 10] = 5.0  # (row=20, col=10)
    coords, conf = decode_default(hm)
    stride = IMAGE_PX / HEATMAP_PX
    assert coords.shape == (1, 1, 2)
    assert coords[0, 0] == pytest.approx([10 * stride, 20 * stride], abs=1e-4)
    assert conf[0, 0] == pytest.approx(5.0)


def test_decode_default_applies_quarter_pixel_offset_toward_brighter_neighbor():
    hm = np.zeros((1, 1, HEATMAP_PX, HEATMAP_PX), dtype=np.float32)
    hm[0, 0, 20, 10] = 5.0
    hm[0, 0, 20, 11] = 3.0  # brighter to the right than to the left
    hm[0, 0, 21, 10] = 2.0  # brighter below than above
    coords, _ = decode_default(hm)
    stride = IMAGE_PX / HEATMAP_PX
    assert coords[0, 0] == pytest.approx(
        [(10 + 0.25) * stride, (20 + 0.25) * stride], abs=1e-4
    )


def test_decode_default_skips_offset_at_border():
    hm = np.zeros((1, 1, HEATMAP_PX, HEATMAP_PX), dtype=np.float32)
    hm[0, 0, 0, 0] = 5.0
    coords, _ = decode_default(hm)
    assert coords[0, 0] == pytest.approx([0.0, 0.0], abs=1e-6)
