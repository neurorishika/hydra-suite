"""Task 4: the whole scale SET is stamped, and it cannot disarm the guard.

The failure this file exists to prevent is not an error, it is a SILENCE:
``_as_geometry_value`` returned ``None`` for any sequence longer than two, so
a scale-set stamp read as ``NO_STAMPED`` -- the drift guard switched itself
off while every surface reported "nothing to check", and a model would ship
unguarded. A stamped value that cannot be represented must be LOUD.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from hydra_suite.core.inference.geometry_drift import (
    DriftStatus,
    compare_geometry_value,
    format_drift_warning,
    stamped_object_tile_fraction,
    stamped_tile_px_set,
)

# The two checkpoints published before this change carry a SCALAR
# ``train_tile_px``. Verified on the real sidecars; reproduced verbatim.
LEGACY_SIDECAR = {
    "base_variant": "sam3",
    "prompt": "ant",
    "train_tile_px": 971,
    "reference_body_px": 97.1,
    "object_tile_fraction": 0.1,
    "imgsz": 1008,
}


class TestUnrepresentableIsLoud:
    def test_a_scale_set_stamp_never_reads_as_no_stamped(self):
        """The regression that matters: a 4-element stamp must not go quiet."""
        verdict = compare_geometry_value(
            "train_tile_px", [971, 1766, 500, 400], [971, 971]
        )
        assert verdict.status is not DriftStatus.NO_STAMPED

    def test_a_flat_long_list_is_unreadable_not_absent(self):
        verdict = compare_geometry_value("train_tile_px", [971, 1766, 500], 971)
        assert verdict.status is DriftStatus.UNREADABLE
        assert "could not be read" in format_drift_warning(verdict)

    def test_absent_is_still_absent(self):
        """Only PRESENT-but-unrepresentable is loud; missing stays quiet."""
        for stamped in (None, 0.0, "", [], "not-a-number"):
            assert (
                compare_geometry_value("train_tile_px", stamped, 971).status
                is DriftStatus.NO_STAMPED
            )


class TestSetComparison:
    SET = [[500, 500], [971, 971]]

    def test_equal_sets_match(self):
        assert (
            compare_geometry_value(
                "train_tile_px", self.SET, [[500, 500], [971, 971]]
            ).status
            is DriftStatus.MATCH
        )

    def test_disjoint_sets_mismatch(self):
        assert (
            compare_geometry_value(
                "train_tile_px", self.SET, [[300, 300], [400, 400]]
            ).status
            is DriftStatus.MISMATCH
        )

    def test_a_strict_subset_is_within_set_not_a_match(self):
        """Serving fewer scales than were trained is its own verdict."""
        assert (
            compare_geometry_value("train_tile_px", self.SET, [[500, 500]]).status
            is DriftStatus.WITHIN_SET
        )

    def test_serving_one_member_of_the_stamped_set_is_neither(self):
        verdict = compare_geometry_value("train_tile_px", self.SET, 971)
        assert verdict.status is DriftStatus.WITHIN_SET
        assert not verdict.is_mismatch

    def test_serving_outside_the_stamped_set_is_a_mismatch(self):
        assert (
            compare_geometry_value("train_tile_px", self.SET, 1766).status
            is DriftStatus.MISMATCH
        )


class TestBackCompatReader:
    def test_scalar_stamped_legacy_checkpoints_still_load(self):
        assert stamped_tile_px_set(LEGACY_SIDECAR) == ((971.0, 971.0),)
        assert stamped_object_tile_fraction(LEGACY_SIDECAR) == pytest.approx(0.1)

    def test_pair_stamp_reads_as_one_scale(self):
        assert stamped_tile_px_set({"train_tile_px": [800, 600]}) == ((800.0, 600.0),)

    def test_set_stamp_reads_as_its_scales(self):
        meta = {"train_tile_px_set": [[500, 500], [971, 971]]}
        assert stamped_tile_px_set(meta) == ((500.0, 500.0), (971.0, 971.0))

    def test_multiscale_prefill_fraction_is_named_as_a_prefill(self):
        meta = {
            "prefill_object_tile_fraction": 0.1,
            "object_tile_fractions": [0.05, 0.2],
        }
        assert stamped_object_tile_fraction(meta) == pytest.approx(0.1)

    def test_no_claim_reads_as_none(self):
        assert stamped_tile_px_set({}) is None
        assert stamped_object_tile_fraction({}) is None


class TestPublishGuard:
    def test_a_scale_set_survives_the_publish_payload(self):
        from hydra_suite.training.contracts import Sam3LoraParams
        from hydra_suite.training.sam3_lora.publish import _request_payload

        payload = _request_payload(
            run_id="r1",
            adapters_path=pathlib.Path("a.pt"),
            base_checkpoint=pathlib.Path("b.pt"),
            build_manifest={
                "tile_px_set": [[500, 500], [971, 971]],
                "object_tile_fractions": [0.2, 0.1],
                "prefill_tile_px": [971, 971],
                "prefill_object_tile_fraction": 0.1,
                "scale_range_px": [500, 971],
                "full_frame_mix": True,
                "reference_body_px": 97.1,
            },
            params=Sam3LoraParams(prompt="ant"),
            source_fingerprint="fp",
            models_root=pathlib.Path("m"),
            attempt_id="att",
        )
        geometry = payload["build_manifest"]
        assert geometry["tile_px_set"] == [[500, 500], [971, 971]]
        assert geometry["object_tile_fractions"] == [0.2, 0.1]
        assert geometry["full_frame_mix"] is True
        # The named prefill goes through the same square-pair collapse the
        # legacy scalar does, so the published sidecar keeps the shape every
        # existing consumer (and both on-disk sidecars) already carry.
        assert geometry["prefill_tile_px"] == 971
        assert geometry["prefill_object_tile_fraction"] == 0.1
        # A collapsed scalar must never be invented from a set.
        assert "tile_px" not in geometry or geometry["tile_px"] is None
        assert "object_tile_fraction" not in geometry or (
            geometry["object_tile_fraction"] is None
        )

    def test_a_malformed_scale_set_raises_rather_than_collapsing(self):
        from hydra_suite.training.contracts import Sam3LoraParams
        from hydra_suite.training.sam3_lora.publish import _request_payload

        with pytest.raises(ValueError, match="tile_px_set"):
            _request_payload(
                run_id="r1",
                adapters_path=pathlib.Path("a.pt"),
                base_checkpoint=pathlib.Path("b.pt"),
                build_manifest={"tile_px_set": [500, 971]},
                params=Sam3LoraParams(prompt="ant"),
                source_fingerprint="fp",
                models_root=pathlib.Path("m"),
                attempt_id="att",
            )


class TestManifest:
    def _manifest(self, tmp_path, name, **kwargs):
        from hydra_suite.training.contracts import Sam3LoraParams, SplitConfig
        from hydra_suite.training.sam3_lora.dataset_build import build_sam3_coco_dataset
        from tests.test_sam3_multiscale_gate import write_corpus

        out = tmp_path / name
        build_sam3_coco_dataset(
            str(write_corpus(tmp_path / f"src_{name}")),
            str(out),
            Sam3LoraParams(prompt="ant", **kwargs),
            seed=42,
            split=SplitConfig(),
        )
        return json.loads((out / "build_manifest.json").read_text())

    def test_multiscale_manifest_stamps_the_whole_set(self, tmp_path):
        manifest = self._manifest(
            tmp_path, "ms", object_tile_fractions=(0.055, 0.02), full_frame_mix=True
        )
        assert len(manifest["tile_px_set"]) == 2
        assert manifest["object_tile_fractions"] == [0.055, 0.02]
        assert manifest["full_frame_mix"] is True
        assert manifest["scale_range_px"] == [
            min(min(pair) for pair in manifest["tile_px_set"]),
            max(max(pair) for pair in manifest["tile_px_set"]),
        ]
        counts = manifest["scale_counts"]["train"]
        assert set(counts) == {
            *(f"tile:{w}x{h}" for w, h in manifest["tile_px_set"]),
            "full",
        }
        assert all(bucket["tiles"] > 0 for bucket in counts.values())

    def test_multiscale_manifest_omits_the_bare_collapsed_scalars(self, tmp_path):
        manifest = self._manifest(tmp_path, "ms2", object_tile_fractions=(0.055, 0.02))
        # A median under a measurement's name is how a scale set silently
        # becomes "the training tile size" downstream.
        assert "tile_px" not in manifest
        assert "object_tile_fraction" not in manifest
        assert manifest["prefill_object_tile_fraction"] > 0
        assert len(manifest["prefill_tile_px"]) == 2


class TestPublishedSidecar:
    """`publish_sam3_model` must succeed on BOTH shapes, unchanged on one."""

    def _publish(self, tmp_path, manifest):
        import torch

        from hydra_suite.training.contracts import Sam3LoraParams
        from tests.test_sam3_publish import publish_sam3_model

        torch.save({"detector.qkv.weight": torch.randn(4, 4)}, tmp_path / "base.pt")
        torch.save(
            {"qkv.lora_A": torch.randn(2, 4), "qkv.lora_B": torch.randn(4, 2)},
            tmp_path / "adapters.pt",
        )
        _, artifact = publish_sam3_model(
            run_id="r1",
            adapters_path=tmp_path / "adapters.pt",
            base_checkpoint=tmp_path / "base.pt",
            build_manifest=manifest,
            params=Sam3LoraParams(prompt="ant", rank=2, alpha=4),
            source_fingerprint="fp1",
            models_root=tmp_path / "models",
        )
        return json.loads(pathlib.Path(str(artifact) + ".sam3_meta.json").read_text())

    def test_single_scale_sidecar_is_unchanged(self, tmp_path):
        meta = self._publish(
            tmp_path,
            {"tile_px": 971, "reference_body_px": 97.1, "object_tile_fraction": 0.1},
        )
        assert meta["train_tile_px"] == 971
        assert meta["object_tile_fraction"] == 0.1
        assert "train_tile_px_set" not in meta
        # And it reads back through the same reader the legacy artifacts use.
        assert stamped_tile_px_set(meta) == ((971.0, 971.0),)

    def test_multiscale_sidecar_carries_the_set_and_no_bare_scalar(self, tmp_path):
        meta = self._publish(
            tmp_path,
            {
                "tile_px_set": [[500, 500], [971, 971]],
                "object_tile_fractions": [0.2, 0.1],
                "prefill_tile_px": [971, 971],
                "prefill_object_tile_fraction": 0.1,
                "scale_range_px": [500, 971],
                "full_frame_mix": False,
                "reference_body_px": 97.1,
            },
        )
        assert meta["train_tile_px_set"] == [[500, 500], [971, 971]]
        assert "object_tile_fraction" not in meta
        assert "train_tile_px" not in meta
        assert meta["prefill_object_tile_fraction"] == 0.1
        # Shape note, pinned so it is a decision rather than an accident: the
        # child copies `prefill_tile_px` through verbatim, so a pair stays a
        # pair here, while the PARENT path collapses a square pair to the
        # legacy scalar first (asserted in TestPublishGuard). Both shapes are
        # square-equivalent and both read back identically -- which is the
        # property that actually matters.
        assert meta["prefill_train_tile_px"] == [971, 971]
        assert stamped_tile_px_set(meta) == ((500.0, 500.0), (971.0, 971.0))
        assert stamped_object_tile_fraction(meta) == pytest.approx(0.1)
