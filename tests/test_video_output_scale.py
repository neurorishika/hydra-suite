"""Output-scale knob for the final annotated video (``video_output_scale``)."""

import pytest

from hydra_suite.core.post.video_output_scale import (
    DEFAULT_VIDEO_OUTPUT_SCALE,
    resolve_video_output_scale,
    scaled_output_size,
    validate_video_output_scale,
)


def test_default_is_half():
    assert DEFAULT_VIDEO_OUTPUT_SCALE == 0.5


def test_missing_key_resolves_to_default():
    assert resolve_video_output_scale({}) == 0.5
    assert resolve_video_output_scale({"video_output_scale": None}) == 0.5


def test_valid_values_pass_through():
    assert resolve_video_output_scale({"video_output_scale": 1.0}) == 1.0
    assert resolve_video_output_scale({"video_output_scale": 0.1}) == 0.1
    assert resolve_video_output_scale({"video_output_scale": 1}) == 1.0


@pytest.mark.parametrize(
    "bad", [0.0, 0.05, 1.01, 2.0, -0.5, float("nan"), "x", "0.5", True, False]
)
def test_out_of_range_is_a_loud_error_not_a_clamp(bad):
    with pytest.raises(ValueError, match="video_output_scale"):
        validate_video_output_scale(bad)
    with pytest.raises(ValueError, match="video_output_scale"):
        resolve_video_output_scale({"video_output_scale": bad})


@pytest.mark.parametrize(
    "w,h,scale,expected",
    [
        (4512, 4512, 0.5, (2256, 2256)),
        (4512, 4512, 1.0, (4512, 4512)),
        (1920, 1080, 0.5, (960, 540)),
        (1920, 1080, 0.35, (672, 378)),  # 672.0 / 378.0
        (1001, 999, 1.0, (1000, 998)),  # odd source -> even, never larger
        (101, 77, 0.33, (32, 24)),  # 33.33 -> 33 -> 32 ; 25.41 -> 25 -> 24
        (10, 10, 0.1, (2, 2)),  # min 2
        (3, 3, 0.1, (2, 2)),
    ],
)
def test_scaled_output_size_is_even_and_rounded(w, h, scale, expected):
    out = scaled_output_size(w, h, scale)
    assert out == expected
    assert out[0] % 2 == 0 and out[1] % 2 == 0
