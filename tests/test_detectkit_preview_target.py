from hydra_suite.detectkit.gui.prediction_preview import preview_object_tile_fraction


def test_median_target_over_imgsz():
    # median([200,300,400]) = 300 -> 300/640 = 0.46875
    assert (
        abs(preview_object_tile_fraction([200.0, 300.0, 400.0], 0.15, 640) - 0.46875)
        < 1e-6
    )


def test_even_count_median():
    # median([200,400]) = 300 -> 0.46875
    assert abs(preview_object_tile_fraction([200.0, 400.0], 0.15, 640) - 0.46875) < 1e-6


def test_empty_target_sizes_falls_back():
    assert preview_object_tile_fraction([], 0.2, 640) == 0.2


def test_zero_imgsz_falls_back():
    assert preview_object_tile_fraction([300.0], 0.2, 0) == 0.2


def test_result_is_clamped():
    # 4000/640 = 6.25 -> clamped to 0.9
    assert preview_object_tile_fraction([4000.0], 0.15, 640) == 0.9
