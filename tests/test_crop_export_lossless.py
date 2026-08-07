"""F5: crop-dataset export must be lossless-only (PNG default, jpg rejected).

The crop-dataset exporter (``IndividualDatasetGenerator``) writes MODEL INPUT
crops consumed by downstream training. JPEG recompression introduces DCT
quantization loss that the uncompressed canonical crop used at inference
never sees, breaking train/infer consistency. This exporter must therefore
reject jpg/jpeg outright and default to a lossless format (png).
"""

import pytest

from hydra_suite.core.identity.dataset.generator import IndividualDatasetGenerator


def _base_params(**overrides):
    params = {
        "ENABLE_INDIVIDUAL_DATASET": True,
        "ENABLE_INDIVIDUAL_IMAGE_SAVE": True,
        "INDIVIDUAL_CROP_PADDING": 0.1,
        "INDIVIDUAL_DATASET_RUN_ID": "run1",
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 1.0,
        "ADVANCED_CONFIG": {
            "reference_aspect_ratio": 2.0,
            "canonical_margin": 1.3,
        },
    }
    params.update(overrides)
    return params


def test_crop_export_rejects_jpg(tmp_path):
    with pytest.raises(ValueError, match="lossless"):
        IndividualDatasetGenerator(
            params=_base_params(INDIVIDUAL_OUTPUT_FORMAT="jpg"),
            output_dir=str(tmp_path),
            video_name="test_video",
            dataset_name="ds",
        )


def test_crop_export_rejects_jpeg(tmp_path):
    with pytest.raises(ValueError, match="lossless"):
        IndividualDatasetGenerator(
            params=_base_params(INDIVIDUAL_OUTPUT_FORMAT="jpeg"),
            output_dir=str(tmp_path),
            video_name="test_video",
            dataset_name="ds",
        )


def test_crop_export_default_is_png(tmp_path):
    gen = IndividualDatasetGenerator(
        params=_base_params(),
        output_dir=str(tmp_path),
        video_name="test_video",
        dataset_name="ds",
    )
    assert gen.output_format == "png"
