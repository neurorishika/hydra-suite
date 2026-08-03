from hydra_suite.detectkit.gui.models import OBBSource
from hydra_suite.training.dataset_builders import eligible_sources


def test_unreviewed_source_excluded_with_message():
    ok = OBBSource(name="orig", level="obb", reviewed=True)
    pending = OBBSource(name="orig_seg", level="polygon", reviewed=False)
    kept, messages = eligible_sources([ok, pending])
    assert [s.name for s in kept] == ["orig"]
    assert any("orig_seg" in m and "unreviewed" in m.lower() for m in messages)


def test_all_reviewed_sources_kept():
    a = OBBSource(name="a", reviewed=True)
    b = OBBSource(name="b", reviewed=True)
    kept, messages = eligible_sources([a, b])
    assert len(kept) == 2 and messages == []
