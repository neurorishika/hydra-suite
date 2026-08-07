"""The four detection-derived canvas helpers are retired.

Canvas dimensions are a property of the project, not of a detection, so a
function that computes them from corners has no meaning under global
canonicalization. These greps stop them creeping back via a shim.
"""

from pathlib import Path

import pytest

RETIRED = [
    "compute_crop_dimensions",
    "compute_native_crop_dimensions",
    "compute_native_scale_affine",
    "compute_alignment_affine",
]


@pytest.mark.parametrize("name", RETIRED)
def test_no_source_file_references_the_retired_api(name):
    src = Path(__file__).resolve().parents[1] / "src" / "hydra_suite"
    hits = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if name in p.read_text(encoding="utf-8")
    ]
    assert hits == [], f"{name} still referenced in {hits}"


@pytest.mark.parametrize("name", RETIRED)
def test_not_importable_from_the_package(name):
    import hydra_suite.core.canonicalization as canon

    assert not hasattr(canon, name)
