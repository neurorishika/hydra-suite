"""PoseGeometry: the per-checkpoint input/heatmap geometry value object."""

from __future__ import annotations

import pytest

from hydra_suite.core.identity.pose.vitpose.geometry import (
    DEFAULT_GEOMETRY,
    PoseGeometry,
)


def test_default_geometry_matches_the_historical_constants():
    # The whole plan rests on the default being unchanged.
    assert DEFAULT_GEOMETRY.image_size_wh == (192, 256)
    assert DEFAULT_GEOMETRY.heatmap_size_wh == (48, 64)


def test_heatmap_is_always_a_quarter_of_the_image():
    for wh in [(192, 256), (256, 256), (320, 320), (128, 192)]:
        g = PoseGeometry(wh)
        assert g.heatmap_size_wh == (wh[0] // 4, wh[1] // 4)


def test_patch_grid_is_image_over_sixteen_in_hw_order():
    assert PoseGeometry((192, 256)).patch_grid_hw == (16, 12)
    assert PoseGeometry((256, 256)).patch_grid_hw == (16, 16)


def test_num_tokens_includes_the_cls_slot():
    # Upstream ViTPose keeps the MAE cls slot, so pos_embed is patches + 1.
    assert PoseGeometry((192, 256)).num_tokens == 16 * 12 + 1 == 193
    assert PoseGeometry((256, 256)).num_tokens == 16 * 16 + 1 == 257


def test_aspect_is_width_over_height():
    assert PoseGeometry((192, 256)).aspect == pytest.approx(0.75)
    assert PoseGeometry((256, 256)).aspect == pytest.approx(1.0)


def test_serialization_round_trip_is_height_first():
    g = PoseGeometry((192, 256))
    assert g.to_hw() == [256, 192]
    assert PoseGeometry.from_hw([256, 192]) == g
    assert PoseGeometry.from_hw(g.to_hw()) == g


def test_from_hw_accepts_a_tuple_and_normalizes_to_tuple_field():
    g = PoseGeometry.from_hw((256, 256))
    assert isinstance(g.image_size_wh, tuple)
    assert g.image_size_wh == (256, 256)


def test_list_input_is_normalized_to_a_tuple_so_the_value_stays_hashable():
    g = PoseGeometry([256, 256])
    assert g.image_size_wh == (256, 256)
    assert hash(g) == hash(PoseGeometry((256, 256)))


@pytest.mark.parametrize(
    "bad", [(192, 250), (200, 256), (0, 256), (192, 0), (-32, 256)]
)
def test_dimensions_must_be_positive_multiples_of_thirty_two(bad):
    with pytest.raises(ValueError, match="multiple of 32|positive"):
        PoseGeometry(bad)


def test_error_message_names_the_offending_dimension():
    with pytest.raises(ValueError, match="height"):
        PoseGeometry((192, 250))
    with pytest.raises(ValueError, match="width"):
        PoseGeometry((200, 256))


def test_from_hw_rejects_wrong_length():
    with pytest.raises(ValueError, match="two"):
        PoseGeometry.from_hw([256])


def test_geometry_is_frozen():
    g = PoseGeometry((192, 256))
    with pytest.raises(Exception):  # noqa: B017
        g.image_size_wh = (256, 256)
