from hydra_suite.core.individual.identity import columns as C


def test_family_prefixes_are_provenance_explicit():
    assert C.REALTIME_LABEL == "IdentityRealtimeLabel"
    assert C.FINAL_LABEL == "IdentityFinalLabel"
    assert C.FINAL_SOURCE == "IdentityFinalSource"
    assert C.EVIDENCE_SOURCES == "IdentityEvidenceSources"
    assert C.UNIQUE_IDENTITY_KEY == "UniqueIdentityKey"
    # House style uses uppercase "ID" (cf. TrajectoryID, DetectionID, FrameID).
    assert C.REALTIME_ID == "IdentityRealtimeID"
    assert C.FINAL_ID == "IdentityFinalID"
    # PascalCase house style: no underscores inside a column name.
    for k, v in vars(C).items():
        if k.isupper() and isinstance(v, str) and v.startswith("Identity"):
            assert "_" not in v, v


def test_no_legacy_names_leak():
    legacy = {
        "IdentityAssignedLabel",
        "IdentityAssignedConfidence",
        "IdentityOfflineLabel",
        "IdentityPosteriorMargin",
    }
    allnames = {v for k, v in vars(C).items() if isinstance(v, str) and k.isupper()}
    assert allnames.isdisjoint(legacy)


def test_realtime_row_order_is_the_worker_contract():
    order = C.identity_realtime_columns()
    assert order == [
        C.REALTIME_ID,
        C.REALTIME_LABEL,
        C.REALTIME_CONFIDENCE,
        C.REALTIME_MARGIN,
        C.REALTIME_ENTROPY,
        C.REALTIME_COMMITTED,
        C.EVIDENCE_SOURCES,
        C.EVIDENCE_CONFLICT_FLAG,
        C.REALTIME_SLOTLOCK,
    ]


def test_final_source_vocabulary():
    assert C.IdentityFinalSource.OFFLINE == "offline"
    assert C.IdentityFinalSource.REALTIME == "realtime"
    assert C.IdentityFinalSource.TAG == "tag"
    # 2026-08-27 identity-final-consistency: explicit denial token
    assert C.IdentityFinalSource.NONE == "none"
