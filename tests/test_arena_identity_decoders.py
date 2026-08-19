import numpy as np
import pytest

from hydra_suite.core.individual.identity.catalog import IdentityCatalog
from hydra_suite.core.tracking.identity.decoder_registry import ArenaDecoderRegistry


@pytest.fixture
def catalog():
    return IdentityCatalog.from_labels(["antA", "antB"])


@pytest.fixture
def params():
    return {"IDENTITY_ONLINE_COMMIT_THRESHOLD": 0.9}


def test_single_arena_registry_creates_one_decoder(catalog, params):
    reg = ArenaDecoderRegistry(catalog, params, np.zeros(4, dtype=np.int32))
    assert reg.n_decoders == 1


def test_registry_creates_one_decoder_per_arena(catalog, params):
    slot_arena = np.repeat([0, 1, 2], 2).astype(np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    assert reg.n_decoders == 3


def test_slot_routes_to_its_arena_decoder(catalog, params):
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    assert reg.decoder_for_slot(0) is reg.decoder_for_slot(1)
    assert reg.decoder_for_slot(2) is reg.decoder_for_slot(3)
    assert reg.decoder_for_slot(0) is not reg.decoder_for_slot(2)


def test_update_frame_partitions_evidence_by_arena(catalog, params):
    """The whole point: each decoder enforces uniqueness over ITS arena only,
    so 'antA' can be assigned once in arena 0 and again in arena 1.

    NOTE: adapted from the task brief's sketch, which mocked
    ``update_frame(frame_idx, evidence_dict)`` (a 2-arg simplification). The
    real ``OnlineIdentityDecoder.update_frame`` -- and every worker.py call
    site -- uses ``update_frame(frame_idx, visible_slots, slot_evidences)``
    (3 positional args: a slots list *and* a slot->evidence dict). The
    registry must forward that exact shape unchanged per arena, so this test
    asserts against the real signature instead of the brief's simplified
    mock.
    """
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    seen = {}

    for arena, dec in reg.decoders.items():
        dec.update_frame = lambda frame_idx, visible_slots, slot_evidences, a=arena: (
            seen.__setitem__(a, (sorted(visible_slots), dict(slot_evidences))),
            [],
        )[1]

    reg.update_frame(0, [0, 1, 2, 3], {0: "e0", 1: "e1", 2: "e2", 3: "e3"})
    assert seen[0] == (
        [0, 1],
        {0: "e0", 1: "e1"},
    ), "arena 0 decoder must see only arena 0 slots/evidence"
    assert seen[1] == (
        [2, 3],
        {2: "e2", 3: "e3"},
    ), "arena 1 decoder must see only arena 1 slots/evidence"


def test_decay_absent_slots_partitions_by_arena(catalog, params):
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    calls = {}

    for arena, dec in reg.decoders.items():
        dec.decay_absent_slot_beliefs = lambda slots, a=arena: calls.__setitem__(
            a, list(slots)
        )

    reg.decay_absent_slot_beliefs([0, 2, 3])
    assert calls[0] == [0]
    assert calls[1] == [2, 3]


def test_get_slot_log_posteriors_merges_across_arenas(catalog, params):
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    for arena, dec in reg.decoders.items():
        dec.get_slot_log_posteriors = lambda slots, a=arena: {s: a for s in slots}
    assert reg.get_slot_log_posteriors([0, 1, 2, 3]) == {0: 0, 1: 0, 2: 1, 3: 1}


def test_catalog_attribute_is_shared_across_all_arena_decoders(catalog, params):
    """worker.py reads ``_identity_online_decoder._catalog`` directly (e.g.
    for ``apriltag_log_prior``/``cnn_log_prior``); the registry must expose
    the same attribute, and it must be the identical arena-invariant catalog
    object handed to every per-arena decoder -- never a per-arena copy."""
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    assert reg._catalog is catalog
    for dec in reg.decoders.values():
        assert dec._catalog is catalog


def test_clear_slot_routes_to_owning_arena_decoder(catalog, params):
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    calls = {}

    for arena, dec in reg.decoders.items():
        dec.clear_slot = lambda slot_index, reason="", respawn_frame_idx=None, a=arena: calls.setdefault(
            a, []
        ).append(
            (slot_index, reason, respawn_frame_idx)
        )

    reg.clear_slot(2, reason="respawn at frame 5", respawn_frame_idx=5)
    assert calls == {1: [(2, "respawn at frame 5", 5)]}


def test_get_belief_routes_to_owning_arena_decoder(catalog, params):
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    sentinel = object()

    reg.decoders[1].get_belief = lambda slot_index: (
        sentinel if slot_index == 3 else None
    )

    assert reg.get_belief(3) is sentinel
    assert reg.get_belief(2) is None


def test_all_active_slots_merges_across_arenas(catalog, params):
    slot_arena = np.array([0, 0, 1, 1], dtype=np.int32)
    reg = ArenaDecoderRegistry(catalog, params, slot_arena)
    reg.decoders[0].all_active_slots = lambda: [1, 0]
    reg.decoders[1].all_active_slots = lambda: [3]
    assert reg.all_active_slots() == [0, 1, 3]
