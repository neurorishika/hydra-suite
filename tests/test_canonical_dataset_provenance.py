import json

from hydra_suite.core.canonicalization.geometry import CanonicalGeometry
from hydra_suite.core.individual.dataset.naming import read_canonical_provenance


def test_provenance_round_trips(tmp_path):
    g = CanonicalGeometry.from_reference(20.0, 2.44, 1.5)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "parameters": {
                    "canonical": {
                        **g.to_dict(),
                        "clipped_count": 3,
                        "worst_overflow_ratio": 1.08,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert read_canonical_provenance(tmp_path) == g


def test_missing_provenance_is_none_not_a_guess(tmp_path):
    assert read_canonical_provenance(tmp_path) is None


def test_legacy_metadata_without_the_block_is_none(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"parameters": {"padding_fraction": 0.1}}), encoding="utf-8"
    )
    assert read_canonical_provenance(tmp_path) is None
