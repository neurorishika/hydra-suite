"""D10 -- a semantic calibration result is persisted on the MODEL SIDECAR.

The hard constraint these tests exist to hold: ``semantic/sam3.py`` REFUSES
to serve on a malformed sidecar, and two real published checkpoints in the
wild carry a SCALAR ``train_tile_px: 971``. So the writer is additive
read-modify-write (never a reshape), and the back-compat reader ships in the
same commit: a sidecar with no calibration block must still load, still
serve, and still read its scalar tile stamp.
"""

from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.core.inference.geometry_drift import stamped_tile_px_set
from hydra_suite.core.inference.semantic.calibration_record import (
    SERVING_CALIBRATION_KEY,
    CalibrationOrigin,
    calibration_record_for_sidecar,
    resolve_serving_calibration,
    serving_calibration,
    write_serving_calibration,
)

#: A minimal sidecar in the shape ``publish.py`` really writes for the two
#: already-published checkpoints: a SCALAR ``train_tile_px``, plus ``imgsz``
#: whose absence makes ``_sidecar_for_checkpoint`` raise.
LEGACY_SIDECAR = {
    "imgsz": 1024,
    "prompt": "ant",
    "reference_body_px": 53.4,
    "object_tile_fraction": 0.055,
    "train_tile_px": 971,
    "lora_state_keys": ["a.weight"],
}

SAVED_CALIBRATION = {
    "created_at": "2026-09-07T00:00:00+00:00",
    "variant": "ant_sam3_v1",
    "prompt": "ant",
    "source_names": ["clip_a"],
    "parameters": {"object_tile_fraction": 0.055},
    "reason": "recall-first",
    "recommended_index": 1,
    "points": [{"threshold": 0.3}, {"threshold": 0.5}],
    "preview_artifact": "previews/calib_2026.npz",
}


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "ant_sam3_v1.pt.sam3_meta.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestBackCompatReader:
    def test_sidecar_without_the_block_makes_no_claim(self):
        assert serving_calibration(LEGACY_SIDECAR) is None
        assert serving_calibration(None) is None
        assert serving_calibration({}) is None

    def test_scalar_stamped_checkpoint_still_reads_its_tile_geometry(self):
        """Both published checkpoints carry ``train_tile_px: 971``."""
        assert stamped_tile_px_set(LEGACY_SIDECAR) == ((971.0, 971.0),)

    def test_a_malformed_block_is_ignored_not_raised(self):
        for bad in ("nope", 3, [], {"points": "not-a-list"}):
            assert serving_calibration({SERVING_CALIBRATION_KEY: bad}) is None


class TestWriterIsAdditive:
    def test_write_preserves_every_existing_key(self, tmp_path):
        path = _write(tmp_path, LEGACY_SIDECAR)
        write_serving_calibration(path, SAVED_CALIBRATION)
        after = json.loads(path.read_text(encoding="utf-8"))
        for key, value in LEGACY_SIDECAR.items():
            assert after[key] == value
        assert stamped_tile_px_set(after) == ((971.0, 971.0),)

    def test_sam3_still_accepts_the_sidecar_after_a_write(self, tmp_path):
        """The refuse-on-malformed-sidecar guard must still pass."""
        from hydra_suite.core.inference.semantic import sam3

        checkpoint = tmp_path / "ant_sam3_v1.pt"
        checkpoint.write_bytes(b"x")
        path = _write(tmp_path, LEGACY_SIDECAR)
        write_serving_calibration(path, SAVED_CALIBRATION)
        meta = sam3._sidecar_for_checkpoint(checkpoint)
        assert meta["imgsz"] == 1024
        assert serving_calibration(meta) is not None

    def test_the_record_drops_the_project_local_preview_path(self, tmp_path):
        path = _write(tmp_path, LEGACY_SIDECAR)
        write_serving_calibration(path, SAVED_CALIBRATION)
        record = serving_calibration(json.loads(path.read_text(encoding="utf-8")))
        assert "preview_artifact" not in record
        assert record["recommended_index"] == 1
        assert record["points"] == SAVED_CALIBRATION["points"]

    def test_record_builder_is_pure_and_drops_previews(self):
        record = calibration_record_for_sidecar(SAVED_CALIBRATION)
        assert "preview_artifact" not in record
        assert record["prompt"] == "ant"
        assert "preview_artifact" in SAVED_CALIBRATION  # input untouched

    def test_writing_to_a_missing_sidecar_is_a_no_op_returning_false(self, tmp_path):
        assert (
            write_serving_calibration(tmp_path / "absent.json", SAVED_CALIBRATION)
            is False
        )

    def test_a_corrupt_sidecar_is_never_overwritten(self, tmp_path):
        path = tmp_path / "broken.sam3_meta.json"
        path.write_text("{not json", encoding="utf-8")
        assert write_serving_calibration(path, SAVED_CALIBRATION) is False
        assert path.read_text(encoding="utf-8") == "{not json"


class TestResolution:
    def test_sidecar_wins_when_present(self, tmp_path):
        path = _write(tmp_path, LEGACY_SIDECAR)
        write_serving_calibration(path, SAVED_CALIBRATION)
        meta = json.loads(path.read_text(encoding="utf-8"))
        record, origin = resolve_serving_calibration(meta, {"prompt": "stale"})
        assert origin is CalibrationOrigin.SIDECAR
        assert record["prompt"] == "ant"

    def test_project_only_calibration_keeps_working_and_is_reported_legacy(self):
        record, origin = resolve_serving_calibration(LEGACY_SIDECAR, SAVED_CALIBRATION)
        assert origin is CalibrationOrigin.PROJECT_LEGACY
        # NOT migrated: the caller gets the project's own dict back verbatim,
        # including the preview path the sidecar shape deliberately drops.
        assert record is SAVED_CALIBRATION

    def test_nothing_anywhere_is_absent(self):
        record, origin = resolve_serving_calibration(LEGACY_SIDECAR, {})
        assert record is None
        assert origin is CalibrationOrigin.ABSENT


class TestSidecarPathResolution:
    def test_registry_sidecar_path_is_the_checkpoint_adjacent_file(
        self, tmp_path, monkeypatch
    ):
        """Serving reads ``<ckpt>.sam3_meta.json``; the writer must target it."""
        from hydra_suite.core.inference.semantic import checkpoints

        checkpoint = tmp_path / "ant_sam3_v1.pt"
        checkpoint.write_bytes(b"x")
        sidecar = _write(tmp_path, LEGACY_SIDECAR)
        registry = {
            "schema_version": 2,
            "entries": {
                "ant_sam3_v1": {
                    "usage_role": "semantic_sam3",
                    "stored_path": str(checkpoint),
                    "sidecar_path": str(sidecar),
                }
            },
        }
        monkeypatch.setattr(checkpoints, "_load_registry", lambda *a, **k: registry)
        resolved = checkpoints.sidecar_path_for("ant_sam3_v1")
        assert resolved == checkpoint.with_name(checkpoint.name + ".sam3_meta.json")
        assert checkpoints.sidecar_path_for("sam3") is None
        assert checkpoints.sidecar_path_for("unknown") is None


# Import lightness for this module is guarded by
# ``tests/test_core_import_is_light.py``, which owns the discriminating
# forbidden-prefix list (coremltools imports sklearn on dev machines, so a
# bare ``'sklearn' in sys.modules`` check cannot discriminate).
