"""D11 -- the drift guard's SEVERITY: warn everywhere, refuse only against a
named comparison baseline.

The guard already lives in Qt-free ``core/inference/geometry_drift.py`` and
is already called from four headless sites. What D11 adds is the one
severity distinction the eval spec identified: a run that EXPLICITLY NAMES a
comparison baseline is asking for the comparison to be guaranteed, so a
divergence there is defensibly fatal. Everywhere else the precedent stands
verbatim -- warn, never refuse.

The judgment call these tests pin down: under a named baseline, UNREADABLE
refuses alongside MISMATCH (the guard cannot verify the very thing the run
named -- the 2026-09-06 shape exactly), while NO_STAMPED never refuses (or
every comparison against an older unstamped artifact would break). That is
also what keeps the deliberate NO_STAMPED/UNREADABLE distinction from being
collapsed: they now differ in OBSERVABLE behaviour, not only in a label.
"""

from __future__ import annotations

import logging

import pytest

from hydra_suite.core.inference.geometry_drift import (
    DriftStatus,
    GeometryDriftRefusal,
    compare_geometry_value,
    enforce_drift_verdicts,
    sidecar_drift_verdicts,
)

BASELINE = "ant_sam3_v1"


def _verdict(status: DriftStatus):
    """One verdict of each status, built through the real comparator."""
    if status is DriftStatus.MISMATCH:
        return compare_geometry_value("object_tile_fraction", 0.055, 0.10)
    if status is DriftStatus.MATCH:
        return compare_geometry_value("object_tile_fraction", 0.055, 0.055)
    if status is DriftStatus.NO_STAMPED:
        return compare_geometry_value("object_tile_fraction", None, 0.10)
    if status is DriftStatus.UNREADABLE:
        return compare_geometry_value("train_tile_px", [[1, 2, 3], "x"], 971)
    if status is DriftStatus.PREFILL:
        return compare_geometry_value("reference_body_px", 53.4, 0.0)
    if status is DriftStatus.WITHIN_SET:
        return compare_geometry_value("train_tile_px", [[971, 971], [1766, 1766]], 971)
    raise AssertionError(status)


class TestTheComparatorStillProducesEachStatus:
    """Guards the fixtures above; a silent status change would void D11."""

    @pytest.mark.parametrize("status", list(DriftStatus))
    def test_each_status_is_reachable(self, status):
        assert _verdict(status).status is status


class TestWarnIsStillTheDefaultEverywhere:
    @pytest.mark.parametrize("status", list(DriftStatus))
    def test_no_baseline_named_never_refuses(self, status, caplog):
        with caplog.at_level(logging.INFO):
            enforce_drift_verdicts(
                logging.getLogger("d11"), [_verdict(status)], comparison_baseline=None
            )

    def test_an_empty_baseline_string_is_not_a_named_baseline(self):
        """`comparison_baseline` defaults to "" all the way down the stack."""
        enforce_drift_verdicts(
            logging.getLogger("d11"),
            [_verdict(DriftStatus.MISMATCH)],
            comparison_baseline="",
        )

    def test_warnings_are_still_emitted_without_a_baseline(self, caplog):
        with caplog.at_level(logging.WARNING):
            enforce_drift_verdicts(
                logging.getLogger("d11"), [_verdict(DriftStatus.MISMATCH)]
            )
        assert any("Geometry drift" in r.message for r in caplog.records)


class TestRefuseOnlyAgainstANamedBaseline:
    def test_mismatch_refuses(self):
        with pytest.raises(GeometryDriftRefusal) as excinfo:
            enforce_drift_verdicts(
                logging.getLogger("d11"),
                [_verdict(DriftStatus.MISMATCH)],
                comparison_baseline=BASELINE,
            )
        assert BASELINE in str(excinfo.value)
        assert "object_tile_fraction" in str(excinfo.value)

    def test_unreadable_refuses_too(self):
        """UNREADABLE means the guard is DISABLED for that field."""
        with pytest.raises(GeometryDriftRefusal):
            enforce_drift_verdicts(
                logging.getLogger("d11"),
                [_verdict(DriftStatus.UNREADABLE)],
                comparison_baseline=BASELINE,
            )

    @pytest.mark.parametrize(
        "status",
        [
            DriftStatus.NO_STAMPED,
            DriftStatus.MATCH,
            DriftStatus.PREFILL,
            DriftStatus.WITHIN_SET,
        ],
    )
    def test_everything_else_still_only_warns(self, status):
        enforce_drift_verdicts(
            logging.getLogger("d11"),
            [_verdict(status)],
            comparison_baseline=BASELINE,
        )

    def test_no_stamped_and_unreadable_are_not_collapsed(self):
        """The distinction now has observable teeth, not just a label."""
        enforce_drift_verdicts(
            logging.getLogger("d11"),
            [_verdict(DriftStatus.NO_STAMPED)],
            comparison_baseline=BASELINE,
        )
        with pytest.raises(GeometryDriftRefusal):
            enforce_drift_verdicts(
                logging.getLogger("d11"),
                [_verdict(DriftStatus.UNREADABLE)],
                comparison_baseline=BASELINE,
            )

    def test_it_still_warns_on_the_way_to_refusing(self, caplog):
        with caplog.at_level(logging.WARNING):
            with pytest.raises(GeometryDriftRefusal):
                enforce_drift_verdicts(
                    logging.getLogger("d11"),
                    [_verdict(DriftStatus.MISMATCH)],
                    comparison_baseline=BASELINE,
                )
        assert any("Geometry drift" in r.message for r in caplog.records)

    def test_every_offending_field_is_named_not_just_the_first(self):
        verdicts = [
            _verdict(DriftStatus.MISMATCH),
            _verdict(DriftStatus.UNREADABLE),
            _verdict(DriftStatus.MATCH),
        ]
        with pytest.raises(GeometryDriftRefusal) as excinfo:
            enforce_drift_verdicts(
                logging.getLogger("d11"), verdicts, comparison_baseline=BASELINE
            )
        message = str(excinfo.value)
        assert "object_tile_fraction" in message and "train_tile_px" in message


class TestHeadlessReachability:
    def test_it_works_on_verdicts_built_from_a_sidecar_mapping(self):
        """The headless shape: sidecar dict in, refusal out. No Qt anywhere."""
        verdicts = sidecar_drift_verdicts(
            {"object_tile_fraction": 0.055, "train_tile_px": 971},
            {"object_tile_fraction": 0.10},
            baseline_label=BASELINE,
        )
        with pytest.raises(GeometryDriftRefusal):
            enforce_drift_verdicts(
                logging.getLogger("d11"), verdicts, comparison_baseline=BASELINE
            )

    def test_no_verdicts_at_all_is_silent(self):
        enforce_drift_verdicts(
            logging.getLogger("d11"), (), comparison_baseline=BASELINE
        )
