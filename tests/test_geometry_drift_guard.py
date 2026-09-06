"""Shared, Qt-free train/serve geometry-drift guard.

The guard exists because a SAM3 run trained at ``object_tile_fraction =
0.055`` (a ``training/contracts.py`` default) was compared against a
checkpoint served at 0.10 -- 1766 px tiles versus 971 px -- and the
divergence entered SILENTLY. These tests pin the extraction of the
previously GUI-only guard
(``detectkit/gui/dialogs/semantic_escalation_dialog.py``) into core, with
its warn-never-refuse semantics preserved verbatim.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hydra_suite.core.inference.geometry_drift import (
    DriftStatus,
    GeometryDriftVerdict,
    GeometrySource,
    compare_geometry_value,
    effective_geometry_log_fields,
    format_drift_warning,
    log_effective_geometry,
    sidecar_drift_verdicts,
)


class TestCompareGeometryValue:
    def test_match_within_float_noise(self):
        verdict = compare_geometry_value("object_tile_fraction", 0.1, 0.1)
        assert verdict.status is DriftStatus.MATCH
        assert not verdict.is_mismatch

    def test_mismatch_is_reported_but_never_raises(self):
        verdict = compare_geometry_value("object_tile_fraction", 0.10, 0.055)
        assert verdict.status is DriftStatus.MISMATCH
        assert verdict.is_mismatch
        assert verdict.stamped_value == pytest.approx(0.10)
        assert verdict.effective_value == pytest.approx(0.055)
        assert verdict.field == "object_tile_fraction"

    def test_absent_effective_requests_a_prefill(self):
        # The dialog prefills only when the project value is <= 0.
        verdict = compare_geometry_value("reference_body_px", 80.0, 0.0)
        assert verdict.status is DriftStatus.PREFILL
        assert verdict.should_prefill
        assert verdict.stamped_value == pytest.approx(80.0)

    def test_falsy_stamped_value_is_absent_not_a_mismatch_with_zero(self):
        # Verbatim dialog behaviour: ``if sidecar_body_px:`` -- a stamped 0.0
        # makes no claim, so it can never be a mismatch.
        for stamped in (0.0, None, ""):
            verdict = compare_geometry_value("reference_body_px", stamped, 80.0)
            assert verdict.status is DriftStatus.NO_STAMPED

    def test_unparseable_stamped_value_degrades_to_no_stamped(self):
        verdict = compare_geometry_value("reference_body_px", "not-a-number", 80.0)
        assert verdict.status is DriftStatus.NO_STAMPED

    def test_unparseable_effective_value_degrades_to_no_stamped(self):
        verdict = compare_geometry_value("reference_body_px", 80.0, object())
        assert verdict.status is DriftStatus.NO_STAMPED

    def test_tolerance_is_float_noise_equality_not_a_measurement(self):
        assert (
            compare_geometry_value("reference_body_px", 80.0, 80.0 + 1e-9).status
            is DriftStatus.MATCH
        )
        assert (
            compare_geometry_value("reference_body_px", 80.0, 80.0 + 1e-3).status
            is DriftStatus.MISMATCH
        )

    def test_baseline_label_is_carried_for_the_caller_to_render(self):
        verdict = compare_geometry_value(
            "object_tile_fraction", 0.10, 0.055, baseline_label="epoch_003.pt"
        )
        assert verdict.baseline_label == "epoch_003.pt"

    def test_verdict_is_immutable_and_carries_no_prose(self):
        verdict = compare_geometry_value("object_tile_fraction", 0.10, 0.055)
        assert isinstance(verdict, GeometryDriftVerdict)
        with pytest.raises(Exception):
            verdict.field = "other"  # type: ignore[misc]
        # A typed verdict, not a pre-formatted string a GUI would have to parse.
        assert not any(
            isinstance(value, str) and " " in value
            for value in (verdict.field, verdict.baseline_label or "")
        )


class TestSidecarVerdicts:
    def test_reads_every_guarded_field_from_a_sam3_sidecar(self):
        meta = {
            "reference_body_px": 80.0,
            "object_tile_fraction": 0.10,
            "train_tile_px": 800,
        }
        verdicts = sidecar_drift_verdicts(
            meta,
            {
                "reference_body_px": 80.0,
                "object_tile_fraction": 0.055,
                "train_tile_px": 1454,
            },
            baseline_label="baseline.pt",
        )
        by_field = {v.field: v for v in verdicts}
        assert by_field["reference_body_px"].status is DriftStatus.MATCH
        assert by_field["object_tile_fraction"].status is DriftStatus.MISMATCH
        assert by_field["train_tile_px"].status is DriftStatus.MISMATCH
        assert all(v.baseline_label == "baseline.pt" for v in verdicts)

    def test_absent_sidecar_yields_no_verdicts(self):
        assert sidecar_drift_verdicts(None, {"object_tile_fraction": 0.055}) == ()

    def test_the_2026_09_06_incident_is_now_loud(self):
        """0.055 trained vs 0.10 baseline: the guard must flag it."""
        verdicts = sidecar_drift_verdicts(
            {"object_tile_fraction": 0.10}, {"object_tile_fraction": 0.055}
        )
        assert [v.status for v in verdicts] == [DriftStatus.MISMATCH]


class TestProvenanceLogging:
    def test_source_and_value_are_both_logged(self, caplog):
        with caplog.at_level(logging.INFO):
            log_effective_geometry(
                logging.getLogger("t"),
                "SAM3 dataset build",
                {"object_tile_fraction": 0.055},
                {"object_tile_fraction": GeometrySource.CONTRACT_DEFAULT},
            )
        text = caplog.text
        assert "object_tile_fraction" in text
        assert "0.055" in text
        assert "contract default" in text

    def test_every_source_has_a_human_label(self):
        for source in GeometrySource:
            assert source.label
        assert {s.name for s in GeometrySource} >= {
            "EXPLICIT",
            "CALIBRATION_PROFILE",
            "CORPUS_DERIVED",
            "CONTRACT_DEFAULT",
        }

    def test_log_fields_are_renderable_without_a_logger(self):
        fields = effective_geometry_log_fields(
            {"object_tile_fraction": 0.055},
            {"object_tile_fraction": GeometrySource.EXPLICIT},
        )
        assert fields == [("object_tile_fraction", 0.055, GeometrySource.EXPLICIT)]

    def test_unknown_source_is_reported_as_unknown_never_dropped(self):
        fields = effective_geometry_log_fields({"overlap": 0.25}, {})
        assert fields[0][2] is GeometrySource.UNKNOWN

    def test_drift_warning_text_mentions_both_values_and_never_refuses(self):
        message = format_drift_warning(
            compare_geometry_value("object_tile_fraction", 0.10, 0.055)
        )
        assert "0.1" in message and "0.055" in message
        # warn-never-refuse: a deliberate re-scale is legitimate.
        assert "intentional" in message.lower()


class TestQtFree:
    def test_core_module_imports_no_qt_and_no_app_layer(self):
        import hydra_suite.core.inference.geometry_drift as module

        source = open(module.__file__, encoding="utf-8").read()
        assert "PyQt" not in source and "QtWidgets" not in source
        for app_layer in ("detectkit", "trackerkit", "posekit", "classkit"):
            assert f"hydra_suite.{app_layer}" not in source


class TestServingSideWiring:
    """``--sahi-profile`` / ``build_engine_params`` provenance + guard."""

    def _sidecar(self, tmp_path, geometry):
        import json

        model = tmp_path / "model.pt"
        model.write_bytes(b"stub")
        (tmp_path / "model.pt.slice_meta.json").write_text(
            json.dumps({"schema_version": 2, "training_geometry": geometry})
        )
        return model

    def test_overlay_logs_geometry_with_its_source(self, tmp_path, caplog):
        from hydra_suite.trackerkit.engine_params import _slice_profile_overlay

        model = self._sidecar(
            tmp_path,
            {"object_tile_fraction": 0.10, "reference_body_px": 80.0, "overlap": 0.2},
        )
        with caplog.at_level(logging.INFO):
            values = _slice_profile_overlay({"yolo_obb_mode": "direct"}, str(model))
        assert values is not None
        assert "SAHI serving geometry" in caplog.text
        assert "stamped training geometry" in caplog.text

    def test_overlay_warns_when_served_fraction_leaves_the_trained_one(
        self, tmp_path, caplog
    ):
        from hydra_suite.trackerkit.engine_params import _slice_profile_overlay

        model = self._sidecar(
            tmp_path,
            {"object_tile_fraction": 0.10, "reference_body_px": 80.0},
        )
        cfg = {
            "yolo_obb_mode": "direct",
            "slice_profile_id": "__custom__",
            "slice_profile_settings": {
                "object_tile_fraction": 0.055,
                "trained_body_px": 80.0,
                "overlap": 0.2,
                "geometry_mode": "auto_object",
                "slice_width": 0,
                "slice_height": 0,
                "enabled": True,
            },
        }
        with caplog.at_level(logging.WARNING):
            _slice_profile_overlay(cfg, str(model))
        assert "Geometry drift" in caplog.text
        assert "object_tile_fraction" in caplog.text

    def test_a_measured_profile_is_not_warned_about(self, tmp_path, caplog):
        """A calibrated profile differs from the training stamp BY DESIGN.

        Direct calibration sweeps the fraction multiplicatively around the
        trained value, so warning here would fire on the sanctioned workflow
        and teach users to ignore the warning the incident needs them to read.
        """
        import json

        from hydra_suite.trackerkit.engine_params import _slice_profile_overlay

        model = tmp_path / "model.pt"
        model.write_bytes(b"stub")
        (tmp_path / "model.pt.slice_meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "training_geometry": {
                        "object_tile_fraction": 0.10,
                        "reference_body_px": 80.0,
                    },
                    "primary_profile_id": "measured",
                    "profiles": [
                        {
                            "id": "measured",
                            "name": "measured",
                            "settings": {
                                "enabled": True,
                                "geometry_mode": "auto_object",
                                "object_tile_fraction": 0.15,
                                "overlap": 0.2,
                                "trained_body_px": 80.0,
                                "slice_width": 0,
                                "slice_height": 0,
                            },
                        }
                    ],
                }
            )
        )
        with caplog.at_level(logging.WARNING):
            values = _slice_profile_overlay({"yolo_obb_mode": "direct"}, str(model))
        assert values["resolution"] in ("requested", "primary")
        assert values["object_tile_fraction"] == pytest.approx(0.15)
        assert "Geometry drift" not in caplog.text

    def test_overlay_never_raises_on_a_corrupt_sidecar(self, tmp_path):
        from hydra_suite.trackerkit.engine_params import _slice_profile_overlay

        model = tmp_path / "model.pt"
        model.write_bytes(b"stub")
        (tmp_path / "model.pt.slice_meta.json").write_text("{not json")
        assert _slice_profile_overlay({"yolo_obb_mode": "direct"}, str(model)) is None

    def test_multi_scale_model_is_not_warned_about_by_construction(
        self, tmp_path, caplog
    ):
        from hydra_suite.trackerkit.engine_params import _slice_profile_overlay

        model = self._sidecar(
            tmp_path,
            {
                "object_tile_fraction": 0.10,
                "reference_body_px": 80.0,
                "imgsz": 640,
                "target_sizes": [200.0, 300.0, 400.0],
            },
        )
        with caplog.at_level(logging.WARNING):
            _slice_profile_overlay({"yolo_obb_mode": "direct"}, str(model))
        assert "Geometry drift" not in caplog.text


class TestSlicedBuilderWiring:
    def test_build_logs_geometry_and_flags_a_default_fraction(self, tmp_path, caplog):
        from hydra_suite.training.geometry_levels import GeometryLevel
        from hydra_suite.training.sliced_dataset import (
            SliceBuildParams,
            build_sliced_obb_dataset,
        )

        merged = tmp_path / "merged"
        (merged / "images" / "train").mkdir(parents=True)
        (merged / "labels" / "train").mkdir(parents=True)
        (merged / "data.yaml").write_text("names:\n  0: ant\n")
        with caplog.at_level(logging.INFO):
            build_sliced_obb_dataset(
                merged,
                tmp_path / "out",
                level=GeometryLevel.OBB,
                params=SliceBuildParams(),
            )
        assert "Sliced OBB dataset build geometry" in caplog.text
        assert "contract default" in caplog.text
        assert "corpus-derived" in caplog.text

    def test_a_named_baseline_makes_the_incident_loud(self, tmp_path, caplog):
        import json

        from hydra_suite.training.geometry_levels import GeometryLevel
        from hydra_suite.training.sliced_dataset import (
            SliceBuildParams,
            build_sliced_obb_dataset,
        )

        merged = tmp_path / "merged"
        (merged / "images" / "train").mkdir(parents=True)
        (merged / "labels" / "train").mkdir(parents=True)
        (merged / "data.yaml").write_text("names:\n  0: ant\n")
        baseline = tmp_path / "baseline.pt"
        baseline.write_bytes(b"stub")
        (tmp_path / "baseline.pt.slice_meta.json").write_text(
            json.dumps({"object_tile_fraction": 0.10})
        )
        with caplog.at_level(logging.WARNING):
            build_sliced_obb_dataset(
                merged,
                tmp_path / "out",
                level=GeometryLevel.OBB,
                params=SliceBuildParams(object_tile_fraction=0.055),
                baseline_model_path=baseline,
            )
        assert "Geometry drift" in caplog.text
        assert "0.055" in caplog.text

    def test_the_baseline_never_changes_what_is_built(self, tmp_path):
        """Warn-never-refuse also means never-adopt: a PREFILL is report-only."""
        import json

        from hydra_suite.training.geometry_levels import GeometryLevel
        from hydra_suite.training.sliced_dataset import (
            SliceBuildParams,
            build_sliced_obb_dataset,
        )

        def _build(out_name, baseline):
            merged = tmp_path / f"merged_{out_name}"
            (merged / "images" / "train").mkdir(parents=True)
            (merged / "labels" / "train").mkdir(parents=True)
            (merged / "data.yaml").write_text("names:\n  0: ant\n")
            return build_sliced_obb_dataset(
                merged,
                tmp_path / out_name,
                level=GeometryLevel.OBB,
                params=SliceBuildParams(reference_body_px=0.0),
                baseline_model_path=baseline,
            )

        baseline = tmp_path / "baseline.pt"
        baseline.write_bytes(b"stub")
        (tmp_path / "baseline.pt.slice_meta.json").write_text(
            json.dumps({"reference_body_px": 999.0, "object_tile_fraction": 0.10})
        )
        guarded = _build("guarded", baseline)
        plain = _build("plain", None)
        manifest_a = json.loads(
            (Path(guarded.dataset_dir) / "manifest.json").read_text()
        )
        manifest_b = json.loads((Path(plain.dataset_dir) / "manifest.json").read_text())
        assert manifest_a["slice_geometry"] == manifest_b["slice_geometry"]
        assert manifest_a["counts"] == manifest_b["counts"]


class TestSam3BuilderWiring:
    def test_the_contract_default_fraction_is_named_as_such(self, monkeypatch, caplog):
        """The 0.055 that caused the incident IS the contract default."""
        import dataclasses

        from hydra_suite.training.contracts import Sam3LoraParams

        default = next(
            f.default
            for f in dataclasses.fields(Sam3LoraParams)
            if f.name == "object_tile_fraction"
        )
        assert default == pytest.approx(0.055)

    def test_builder_accepts_a_named_comparison_baseline(self):
        import inspect

        from hydra_suite.training.sam3_lora.dataset_build import (
            build_sam3_coco_dataset,
        )

        params = inspect.signature(build_sam3_coco_dataset).parameters
        assert "baseline_model_key" in params
        assert params["baseline_model_key"].default is None

    def test_sam3_build_logs_provenance_and_flags_a_baseline_mismatch(
        self, tmp_path, caplog, monkeypatch
    ):
        """Exercise the real build-time block, not just the shared function."""
        import hydra_suite.training.sam3_lora.dataset_build as build_mod
        from tests.test_sam3_dataset_build import _params, _source

        monkeypatch.setattr(
            build_mod,
            "sidecar_for",
            lambda key: {"reference_body_px": 80.0, "object_tile_fraction": 0.10},
        )
        with caplog.at_level(logging.INFO):
            build_mod.build_sam3_coco_dataset(
                _source(tmp_path / "src", n_frames=2, size=512),
                tmp_path / "out",
                _params(object_tile_fraction=0.055),
                baseline_model_key="baseline-key",
            )
        assert "SAM3 dataset build geometry" in caplog.text
        assert "contract default" in caplog.text
        drift = [r for r in caplog.records if "Geometry drift" in r.getMessage()]
        assert any("object_tile_fraction" in r.getMessage() for r in drift)
        assert all(r.levelno == logging.WARNING for r in drift)
