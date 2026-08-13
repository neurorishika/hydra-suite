from hydra_suite.trackerkit.headless_tracking import build_tracking_csv_header


def test_header_always_has_confidence_columns():
    header = build_tracking_csv_header()
    for col in ("DetectionConfidence", "AssignmentConfidence", "PositionUncertainty"):
        assert col in header


def test_header_apriltags_appends_tag_columns():
    header = build_tracking_csv_header(identity_method="apriltags")
    assert header[-4:] == [
        "DetectedTagID",
        "DetectedTagLabel",
        "DetectedTagConf",
        "DetectedTagHamming",
    ]
