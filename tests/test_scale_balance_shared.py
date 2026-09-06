"""Task 5 -- the framework-agnostic half of scale-grouped batching.

`scale_group_for_path`, `scale_group_weights` and `ScaleGroupedBatchSampler`
shipped inside `training/ultralytics_scale_balance.py`, but nothing in them is
Ultralytics-specific. They now live in `training/scale_balance.py` so the SAM3
sidecar can group batches too; the Ultralytics module keeps ONLY the installer
(which is genuinely framework-specific) and re-exports the rest so no caller
moves.

The SAM3 half deliberately does NOT parse filenames: `dataset_build` writes
`scale_group` as a field on the COCO image record (D19), so grouping reads
data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydra_suite.training import scale_balance as sb
from hydra_suite.training import ultralytics_scale_balance as usb


def test_the_pure_functions_live_in_the_shared_module():
    for name in (
        "scale_group_for_path",
        "scale_group_weights",
        "ScaleGroupedBatchSampler",
    ):
        assert hasattr(sb, name), name
    assert sb.__name__ == "hydra_suite.training.scale_balance"


def test_the_shared_module_pulls_in_no_framework():
    """It must be importable inside the slim `sam3-lora` sidecar env."""
    source = __import__("inspect").getsource(sb)
    imports = [
        line.strip()
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    ]
    for line in imports:
        assert not any(
            heavy in line
            for heavy in ("ultralytics", "torch", "cv2", "numpy", "hydra_suite")
        ), line


def test_ultralytics_still_re_exports_them():
    assert usb.scale_group_for_path is sb.scale_group_for_path
    assert usb.scale_group_weights is sb.scale_group_weights
    assert usb.ScaleGroupedBatchSampler is sb.ScaleGroupedBatchSampler


def test_grouping_still_behaves(tmp_path):
    assert sb.scale_group_for_path("a_t640x640_0001.jpg") == "tile:640x640"
    assert sb.scale_group_for_path("a_full.jpg") == "full"
    assert sb.scale_group_for_path("nonsense.jpg") == "other"


# ---------------------------------------------------------------------------
# SAM3 adapter
# ---------------------------------------------------------------------------


def _descriptor(dl, index: int, group: str):
    return dl.TileDescriptor(
        image_id=index,
        image_path=f"/nonexistent/{group}_{index}.jpg",
        positive_prompt="ant",
        negative_prompts=(),
        instances=(),
        width=8,
        height=8,
        scale_group=group,
    )


def test_tile_descriptor_carries_the_scale_group():
    from hydra_suite.training.sam3_lora import dataloader as dl

    assert _descriptor(dl, 1, "tile:363x363").scale_group == "tile:363x363"
    # Defaulted, so hand-built single-scale descriptors stay valid.
    assert (
        dl.TileDescriptor(
            image_id=1,
            image_path="x.jpg",
            positive_prompt="ant",
            negative_prompts=(),
            instances=(),
        ).scale_group
        == ""
    )


def test_grouped_epoch_batches_are_scale_homogeneous(monkeypatch):
    """The reason grouping matters here: the collator pads to the batch's max
    instance count, so a coarse tile in a fine batch inflates peak VRAM."""
    from hydra_suite.training.sam3_lora import dataloader as dl

    groups = ["tile:727x727"] * 5 + ["tile:363x363"] * 4 + ["full"] * 3
    descriptors = [_descriptor(dl, i, g) for i, g in enumerate(groups)]

    # Collate to the descriptor list itself so the test needs no images.
    monkeypatch.setattr(dl, "_default_transform", lambda: None)
    monkeypatch.setattr(dl, "load_datapoints", lambda d, t: [d])
    monkeypatch.setattr(dl, "collate_datapoints", lambda pending: list(pending))

    batches = list(
        dl.collate_epoch_batches(descriptors, 2, seed=7, group_by_scale=True)
    )
    seen = []
    for batch in batches:
        assert len({d.scale_group for d in batch}) == 1, "heterogeneous batch"
        seen.extend(d.image_id for d in batch)
    assert sorted(seen) == list(range(len(descriptors))), "every tile once per epoch"


def test_ungrouped_is_still_the_default(monkeypatch):
    """The single-scale arm must take literally today's path."""
    from hydra_suite.training.sam3_lora import dataloader as dl

    descriptors = [
        _descriptor(dl, i, g)
        for i, g in enumerate(["tile:727x727"] * 4 + ["tile:363x363"] * 4)
    ]
    monkeypatch.setattr(dl, "_default_transform", lambda: None)
    monkeypatch.setattr(dl, "load_datapoints", lambda d, t: [d])
    monkeypatch.setattr(dl, "collate_datapoints", lambda pending: list(pending))

    default = [
        [d.image_id for d in batch]
        for batch in dl.collate_epoch_batches(descriptors, 3, seed=3)
    ]
    explicit_off = [
        [d.image_id for d in batch]
        for batch in dl.collate_epoch_batches(
            descriptors, 3, seed=3, group_by_scale=False
        )
    ]
    assert default == explicit_off


def test_grouped_batch_count_matches_what_is_yielded(monkeypatch):
    from hydra_suite.training.sam3_lora import dataloader as dl

    descriptors = [
        _descriptor(dl, i, g)
        for i, g in enumerate(["tile:727x727"] * 5 + ["tile:363x363"] * 4)
    ]
    monkeypatch.setattr(dl, "_default_transform", lambda: None)
    monkeypatch.setattr(dl, "load_datapoints", lambda d, t: [d])
    monkeypatch.setattr(dl, "collate_datapoints", lambda pending: list(pending))

    predicted = dl.grouped_batch_count(descriptors, 2)
    actual = len(
        list(dl.collate_epoch_batches(descriptors, 2, seed=1, group_by_scale=True))
    )
    assert predicted == actual


def test_unparseable_group_is_counted_and_warned(caplog):
    """R8: a descriptor with no scale group must never be silently folded in."""
    from hydra_suite.training.sam3_lora import dataloader as dl

    descriptors = [
        _descriptor(dl, 0, "tile:727x727"),
        dl.TileDescriptor(
            image_id=1,
            image_path="x.jpg",
            positive_prompt="ant",
            negative_prompts=(),
            instances=(),
        ),
    ]
    with caplog.at_level("WARNING"):
        summary = dl.scale_group_summary(descriptors)
    assert summary["ungrouped"] == 1
    assert any("ungrouped" in rec.message for rec in caplog.records)


def test_realised_stamp_is_written_on_both_arms(tmp_path):
    from hydra_suite.training.sam3_lora import dataloader as dl

    for applied in (True, False):
        run_dir = tmp_path / f"run_{applied}"
        path = dl.write_sam3_scale_grouping_stamp(
            run_dir,
            requested=True,
            applied=applied,
            reason="" if applied else "dataset carries no scale groups",
            group_counts={"tile:727x727": 5},
        )
        assert path is not None
        payload = json.loads(path.read_text())
        assert payload["requested"]["scale_grouped_batching"] is True
        assert payload["applied"]["scale_grouped_batching"] is applied


def test_stamp_write_failure_is_logged_not_swallowed(tmp_path, caplog, monkeypatch):
    """Finding 3: an OSError writing the grouping stamp must not vanish
    silently -- a training run should not die over it, but it must be
    visible in the logs."""
    from hydra_suite.training.sam3_lora import dataloader as dl

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _boom)
    with caplog.at_level("WARNING"):
        result = dl.write_sam3_scale_grouping_stamp(
            run_dir,
            requested=True,
            applied=True,
            reason="",
            group_counts={"tile:727x727": 5},
        )
    assert result is None
    assert any("scale-grouping stamp" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


def test_sam3_never_grows_a_distributed_launch():
    """SAM3 does not inherit R7's DDP hazard today -- it is single-process,
    single-GPU by construction. If that EVER changes, the ungrouped fallback
    must raise, not warn. This tripwire fails the moment it changes."""
    from pathlib import Path as _P

    import hydra_suite.training.sam3_lora as pkg

    banned = (
        "DistributedSampler",
        "torch.distributed",
        "init_process_group",
        "world_size",
        "local_rank",
        "torchrun",
    )
    root = _P(pkg.__file__).parent
    hits = [
        f"{path.name}:{token}"
        for path in sorted(root.rglob("*.py"))
        for token in banned
        if token in path.read_text(encoding="utf-8")
    ]
    assert not hits, (
        "SAM3 gained a distributed launch; scale-grouped batching must now "
        f"RAISE rather than silently fall back to ungrouped: {hits}"
    )


@pytest.mark.parametrize(
    "counts,env,expected",
    [
        ({"tile:727x727": 4, "ungrouped": 0}, None, (True, True)),
        ({"tile:727x727": 4, "ungrouped": 0}, "0", (False, False)),
        ({"tile:727x727": 4, "ungrouped": 0}, "false", (False, False)),
        ({"tile:727x727": 4, "ungrouped": 0}, "1", (True, True)),
        # Single-scale build: requested, but there is nothing to group by.
        ({"ungrouped": 9}, None, (True, False)),
        ({}, None, (True, False)),
    ],
)
def test_scale_grouping_decision(counts, env, expected):
    """The three-line decision `run_training` makes, as a pure function.

    `run_training` itself needs CUDA, so the decision is extracted rather than
    left inline and untested: whether a run is grouped is exactly the thing the
    realised stamp exists to make non-confusable.
    """
    from hydra_suite.training.sam3_lora import dataloader as dl

    requested, applied, reason = dl.scale_grouping_decision(counts, env)
    assert (requested, applied) == expected
    assert bool(reason) is not applied
