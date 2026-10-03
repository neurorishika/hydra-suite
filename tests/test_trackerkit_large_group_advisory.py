"""Tracking's large-group advisory follows the configured total slot count."""

from types import SimpleNamespace

import pytest

from hydra_suite.trackerkit.gui.main_window import MainWindow


@pytest.mark.parametrize(
    ("per_arena", "arena_count", "visible"),
    [(100, 1, False), (200, 1, False), (100, 2, False), (101, 2, True), (67, 3, True)],
)
def test_large_group_advisory_uses_total_animals(per_arena, arena_count, visible):
    shown = []
    total_labels = []
    window = SimpleNamespace(
        _setup_panel=SimpleNamespace(
            spin_max_targets=SimpleNamespace(value=lambda: per_arena),
            lbl_animals_per_arena_total=SimpleNamespace(setText=total_labels.append),
        ),
        _tracking_panel=SimpleNamespace(
            lbl_large_group_warning=SimpleNamespace(setVisible=shown.append)
        ),
        config=SimpleNamespace(animals_per_arena=None),
        roi_shapes=[{"arena_id": i, "mode": "include"} for i in range(arena_count)],
    )

    MainWindow._update_animals_per_arena_total_label(window)

    assert shown == [visible]
    assert window.config.animals_per_arena == per_arena
    assert total_labels == [
        (
            f"= {per_arena * arena_count} total across {arena_count} arenas"
            if arena_count > 1
            else ""
        )
    ]
