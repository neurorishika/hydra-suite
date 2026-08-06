"""A shipped config key that nothing reads is a lie to the operator."""

import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "hydra_suite"

# Keys that no longer exist under global canonicalization.
RETIRED = {
    "identity_crop_size_multiplier",
    "identity_crop_min_size",
    "identity_crop_max_size",
}

# Shipped default.json keys with no consumer anywhere in src/, confirmed dead
# by an earlier review. Each is a separate decision about tracking behaviour
# (not a crop-geometry regression) and is tracked as an operator backlog item
# rather than deleted outright.
KNOWN_UNWIRED = {
    # Sibling keys "dataset_class_name" and "dataset_diversity_window" are
    # wired through the dataset panel; "dataset_confidence_threshold" is
    # shipped (and overridden in ooceraea_biroi.json) but never read by the
    # dataset-generation code path or the panel that builds its params.
    "dataset_confidence_threshold",
    # No pose-precompute code path reads a batch size from config at all;
    # the value is shipped but has no consumer.
    "pose_precompute_batch_size",
}


def test_retired_crop_keys_are_gone():
    defaults = json.loads(
        (SRC / "resources/configs/default.json").read_text(encoding="utf-8")
    )
    assert RETIRED.isdisjoint(defaults.keys())


def test_no_shipped_key_is_unreferenced():
    defaults = json.loads(
        (SRC / "resources/configs/default.json").read_text(encoding="utf-8")
    )
    sources = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py"))
    unreferenced = [
        k for k in defaults if k not in sources and k.upper() not in sources
    ]
    assert set(unreferenced) == KNOWN_UNWIRED, (
        f"unreferenced keys not accounted for in KNOWN_UNWIRED: "
        f"{set(unreferenced) - KNOWN_UNWIRED}; "
        f"KNOWN_UNWIRED entries no longer unreferenced (safe to remove from "
        f"the allow-list): {KNOWN_UNWIRED - set(unreferenced)}"
    )
