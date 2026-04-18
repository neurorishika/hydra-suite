from __future__ import annotations

from collections import Counter, defaultdict
from math import floor
from pathlib import Path
from typing import Hashable, Sequence


def _normalize_split_strategy(strategy: str | None) -> str:
    normalized = str(strategy or "stratified").strip().lower()
    return normalized if normalized in {"stratified", "random"} else "stratified"


def _normalize_groups(
    groups: Sequence[Hashable] | None,
    *,
    n_items: int,
) -> list[Hashable] | None:
    if groups is None:
        return None
    normalized = list(groups)
    if len(normalized) != n_items:
        raise ValueError("groups must have the same length as labels")
    return normalized


def _resolve_label_expansion_destination(
    src_label: Hashable,
    mapping: dict[str, str],
    *,
    known_labels: Sequence[Hashable] | None = None,
) -> tuple[bool, str | None]:
    """Resolve one label-expansion mapping row to a canonical destination label."""

    src_name = str(src_label).strip()
    src_name_ci = src_name.lower()
    dst_name = mapping.get(src_name)
    if dst_name is None:
        for key, value in mapping.items():
            if str(key).strip().lower() == src_name_ci:
                dst_name = value
                break
    if dst_name is None:
        return False, None

    dst_name_clean = str(dst_name).strip()
    if not dst_name_clean:
        return True, None

    if known_labels is None:
        return True, dst_name_clean

    known_by_ci = {
        str(label).strip().lower(): str(label).strip() for label in known_labels
    }
    return True, known_by_ci.get(dst_name_clean.lower())


def build_label_expansion_records(
    labels: Sequence[Hashable],
    *,
    label_expansion: dict[str, dict[str, str]] | None = None,
    groups: Sequence[Hashable] | None = None,
    known_labels: Sequence[Hashable] | None = None,
) -> list[dict[str, object]]:
    """Return original-plus-expanded sample records for split planning.

    Expanded samples retain their source group so grouped holdouts keep related
    original and mirrored samples together when grouping is enabled.
    """

    labels_list = [str(label).strip() for label in labels]
    normalized_groups = _normalize_groups(groups, n_items=len(labels_list))
    normalized_expansion = dict(label_expansion or {})

    records: list[dict[str, object]] = []
    for index, label_name in enumerate(labels_list):
        group = normalized_groups[index] if normalized_groups is not None else None
        records.append(
            {
                "source_index": index,
                "label": label_name,
                "group": group,
                "is_expanded": False,
                "axis": "",
            }
        )
        for axis, mapping in normalized_expansion.items():
            matched_rule, dst_name = _resolve_label_expansion_destination(
                label_name,
                mapping,
                known_labels=known_labels,
            )
            if not matched_rule or dst_name is None:
                continue
            records.append(
                {
                    "source_index": index,
                    "label": dst_name,
                    "group": group,
                    "is_expanded": True,
                    "axis": str(axis or "").strip().lower(),
                }
            )

    return records


def build_label_expansion_split_key(
    source_path: str | Path,
    axis: str,
    destination_label: Hashable,
) -> str:
    """Return a stable key for one preplanned expanded sample."""

    return "::".join(
        [
            str(Path(source_path).resolve()),
            str(axis or "").strip().lower(),
            str(destination_label).strip().lower(),
        ]
    )


def _build_group_entries(
    labels: Sequence[Hashable],
    groups: Sequence[Hashable],
) -> list[dict[str, object]]:
    labels_list = list(labels)
    indices_by_group: dict[Hashable, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        indices_by_group[group].append(index)

    entries: list[dict[str, object]] = []
    for group in sorted(indices_by_group, key=str):
        indices = list(indices_by_group[group])
        label_counts = Counter(labels_list[index] for index in indices)
        dominant_label = sorted(
            label_counts.items(),
            key=lambda item: (-item[1], str(item[0])),
        )[0][0]
        entries.append(
            {
                "group": group,
                "indices": indices,
                "label_counts": dict(label_counts),
                "dominant_label": dominant_label,
                "size": len(indices),
            }
        )
    return entries


def _assign_groups_to_split(
    group_entries: list[dict[str, object]],
    split_by_index: list[str],
    *,
    split_name: str,
    target_allocations: dict[Hashable, int],
) -> None:
    remaining = {label: int(count) for label, count in target_allocations.items()}
    desired_total = sum(remaining.values())
    assigned_total = 0
    while group_entries and assigned_total < desired_total:
        best_pos: int | None = None
        best_key: tuple[int, int, int, int, str] | None = None
        for position, entry in enumerate(group_entries):
            remaining_after_assignment = sum(
                int(candidate["size"])
                for idx, candidate in enumerate(group_entries)
                if idx != position
            )
            if remaining_after_assignment <= 0:
                continue

            label_counts = entry["label_counts"]
            gain = sum(
                min(int(remaining.get(label, 0)), int(count))
                for label, count in label_counts.items()
            )
            overshoot = sum(
                max(0, int(count) - int(remaining.get(label, 0)))
                for label, count in label_counts.items()
            )
            key = (
                -int(gain),
                max(0, assigned_total + int(entry["size"]) - desired_total),
                int(overshoot),
                int(entry["size"]),
                int(entry.get("shuffle_rank", 0)),
                str(entry["group"]),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_pos = position

        if best_pos is None:
            break

        entry = group_entries.pop(best_pos)
        for index in entry["indices"]:
            split_by_index[int(index)] = split_name
        for label, count in entry["label_counts"].items():
            remaining[label] = max(0, int(remaining.get(label, 0)) - int(count))
        assigned_total += int(entry["size"])


def _assign_random_groups_to_split(
    group_entries: list[dict[str, object]],
    split_by_index: list[str],
    *,
    split_name: str,
    desired_total: int,
) -> int:
    """Assign shuffled groups to a holdout split without exhausting train data."""

    assigned_total = 0
    while group_entries and assigned_total < desired_total:
        remaining_needed = desired_total - assigned_total
        best_pos: int | None = None
        best_key: tuple[int, int, int, str] | None = None
        for position, entry in enumerate(group_entries):
            group_size = int(entry["size"])
            remaining_after_assignment = sum(
                int(candidate["size"])
                for idx, candidate in enumerate(group_entries)
                if idx != position
            )
            if remaining_after_assignment <= 0:
                continue

            key = (
                0 if group_size <= remaining_needed else 1,
                abs(group_size - remaining_needed),
                int(entry.get("shuffle_rank", 0)),
                str(entry["group"]),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_pos = position

        if best_pos is None:
            break

        entry = group_entries.pop(best_pos)
        for index in entry["indices"]:
            split_by_index[int(index)] = split_name
        assigned_total += int(entry["size"])

    return assigned_total


def build_grouped_stratified_splits(
    labels: Sequence[Hashable],
    groups: Sequence[Hashable],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> list[str]:
    """Return deterministic train/val/test split labels while keeping groups intact."""

    import numpy as np

    n_items = len(labels)
    if n_items == 0:
        return []

    val_fraction = min(max(float(val_fraction), 0.0), 1.0)
    test_fraction = min(max(float(test_fraction), 0.0), 1.0)

    labels_list = list(labels)
    groups_list = list(groups)
    counts_by_label = dict(Counter(labels_list))
    test_allocations = _allocate_holdout_counts(
        counts_by_label,
        fraction=test_fraction,
        total_items=n_items,
    )
    val_allocations = _allocate_holdout_counts(
        counts_by_label,
        fraction=val_fraction,
        total_items=n_items,
        preallocated=test_allocations,
    )

    group_entries = _build_group_entries(labels_list, groups_list)
    rng = np.random.default_rng(seed)
    rng.shuffle(group_entries)
    for rank, entry in enumerate(group_entries):
        entry["shuffle_rank"] = rank

    split_by_index = ["train"] * n_items
    _assign_groups_to_split(
        group_entries,
        split_by_index,
        split_name="test",
        target_allocations=test_allocations,
    )
    _assign_groups_to_split(
        group_entries,
        split_by_index,
        split_name="val",
        target_allocations=val_allocations,
    )

    if val_fraction > 0.0 and "val" not in split_by_index and n_items > 1:
        train_counts = Counter(
            labels_list[index]
            for index, split in enumerate(split_by_index)
            if split == "train"
        )
        entries_by_group = _build_group_entries(labels_list, groups_list)
        for entry in entries_by_group:
            if any(split_by_index[index] != "train" for index in entry["indices"]):
                continue
            if any(
                int(train_counts.get(label, 0)) > int(count)
                for label, count in entry["label_counts"].items()
            ):
                for index in entry["indices"]:
                    split_by_index[int(index)] = "val"
                return split_by_index

    return split_by_index


def build_grouped_random_splits(
    labels: Sequence[Hashable],
    groups: Sequence[Hashable],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> list[str]:
    """Return deterministic random train/val/test split labels while keeping groups intact."""

    import numpy as np

    n_items = len(labels)
    if n_items == 0:
        return []

    val_fraction = min(max(float(val_fraction), 0.0), 1.0)
    test_fraction = min(max(float(test_fraction), 0.0), 1.0)

    desired_test = int(round(float(n_items) * test_fraction))
    desired_val = int(round(float(n_items) * val_fraction))
    if test_fraction > 0.0 and desired_test <= 0 and n_items > 1:
        desired_test = 1
    desired_test = min(max(0, desired_test), max(0, n_items - 1))

    remaining_after_test = n_items - desired_test
    if val_fraction > 0.0 and desired_val <= 0 and remaining_after_test > 1:
        desired_val = 1
    desired_val = min(max(0, desired_val), max(0, remaining_after_test - 1))

    group_entries = _build_group_entries(labels, groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(group_entries)
    for rank, entry in enumerate(group_entries):
        entry["shuffle_rank"] = rank

    split_by_index = ["train"] * n_items
    _assign_random_groups_to_split(
        group_entries,
        split_by_index,
        split_name="test",
        desired_total=desired_test,
    )
    _assign_random_groups_to_split(
        group_entries,
        split_by_index,
        split_name="val",
        desired_total=desired_val,
    )

    return split_by_index


def _allocate_holdout_counts(
    counts_by_label: dict[Hashable, int],
    *,
    fraction: float,
    total_items: int,
    preallocated: dict[Hashable, int] | None = None,
) -> dict[Hashable, int]:
    """Allocate a stratified holdout count per class.

    The allocator preserves at least one train sample per class whenever
    possible and uses largest-remainder apportionment so the overall holdout
    count stays close to the requested fraction.
    """

    if total_items <= 0 or fraction <= 0.0:
        return {label: 0 for label in counts_by_label}

    preallocated = preallocated or {}
    capacities: dict[Hashable, int] = {}
    for label, count in counts_by_label.items():
        already = int(preallocated.get(label, 0))
        remaining = max(0, int(count) - already)
        capacities[label] = max(0, remaining - 1)

    desired_total = min(
        int(round(float(total_items) * float(fraction))),
        sum(capacities.values()),
    )
    if desired_total <= 0:
        return {label: 0 for label in counts_by_label}

    allocations: dict[Hashable, int] = {}
    remainders: list[tuple[float, int, str, Hashable]] = []
    for label, count in counts_by_label.items():
        raw = float(count) * float(fraction)
        base = min(int(floor(raw)), capacities[label])
        allocations[label] = base
        remainders.append((raw - floor(raw), int(count), str(label), label))

    allocated_total = sum(allocations.values())
    for _remainder, _count, _label_text, label in sorted(
        remainders,
        key=lambda item: (-item[0], -item[1], item[2]),
    ):
        if allocated_total >= desired_total:
            break
        if allocations[label] >= capacities[label]:
            continue
        allocations[label] += 1
        allocated_total += 1

    return allocations


def build_stratified_splits(
    labels: Sequence[Hashable],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> list[str]:
    """Return deterministic stratified train/val/test split labels.

    The split preserves class proportions as closely as practical while keeping
    at least one training example per class whenever a class has multiple
    samples.
    """

    import numpy as np

    n_items = len(labels)
    if n_items == 0:
        return []

    val_fraction = min(max(float(val_fraction), 0.0), 1.0)
    test_fraction = min(max(float(test_fraction), 0.0), 1.0)

    labels_list = list(labels)
    counts_by_label = dict(Counter(labels_list))
    test_allocations = _allocate_holdout_counts(
        counts_by_label,
        fraction=test_fraction,
        total_items=n_items,
    )
    val_allocations = _allocate_holdout_counts(
        counts_by_label,
        fraction=val_fraction,
        total_items=n_items,
        preallocated=test_allocations,
    )

    indices_by_label: dict[Hashable, list[int]] = defaultdict(list)
    for index, label in enumerate(labels_list):
        indices_by_label[label].append(index)

    rng = np.random.default_rng(seed)
    split_by_index = ["train"] * n_items
    for label in sorted(indices_by_label, key=str):
        indices = list(indices_by_label[label])
        rng.shuffle(indices)

        n_test = min(len(indices), int(test_allocations.get(label, 0)))
        n_val = min(
            max(0, len(indices) - n_test),
            int(val_allocations.get(label, 0)),
        )

        for index in indices[:n_test]:
            split_by_index[index] = "test"
        for index in indices[n_test : n_test + n_val]:
            split_by_index[index] = "val"

    if val_fraction > 0.0 and "val" not in split_by_index and n_items > 1:
        train_counts = Counter(
            labels_list[index]
            for index, split in enumerate(split_by_index)
            if split == "train"
        )
        for label in sorted(
            counts_by_label, key=lambda item: (-counts_by_label[item], str(item))
        ):
            if train_counts.get(label, 0) > 1:
                for index, split in enumerate(split_by_index):
                    if split == "train" and labels_list[index] == label:
                        split_by_index[index] = "val"
                        return split_by_index

    return split_by_index


def build_random_splits(
    labels: Sequence[Hashable],
    *,
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> list[str]:
    """Return deterministic random train/val/test split labels."""

    import numpy as np

    n_items = len(labels)
    if n_items == 0:
        return []

    val_fraction = min(max(float(val_fraction), 0.0), 1.0)
    test_fraction = min(max(float(test_fraction), 0.0), 1.0)

    desired_test = int(round(float(n_items) * test_fraction))
    desired_val = int(round(float(n_items) * val_fraction))

    if test_fraction > 0.0 and desired_test <= 0 and n_items > 1:
        desired_test = 1
    desired_test = min(max(0, desired_test), max(0, n_items - 1))

    remaining_after_test = n_items - desired_test
    if val_fraction > 0.0 and desired_val <= 0 and remaining_after_test > 1:
        desired_val = 1
    desired_val = min(max(0, desired_val), max(0, remaining_after_test - 1))

    indices = list(range(n_items))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_by_index = ["train"] * n_items
    for index in indices[:desired_test]:
        split_by_index[index] = "test"
    for index in indices[desired_test : desired_test + desired_val]:
        split_by_index[index] = "val"

    return split_by_index


def build_dataset_splits(
    labels: Sequence[Hashable],
    *,
    strategy: str = "stratified",
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
    groups: Sequence[Hashable] | None = None,
) -> list[str]:
    """Return deterministic split labels using the requested strategy."""

    normalized_strategy = _normalize_split_strategy(strategy)
    normalized_groups = _normalize_groups(groups, n_items=len(labels))
    if normalized_groups is not None:
        if normalized_strategy == "random":
            return build_grouped_random_splits(
                labels,
                normalized_groups,
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed,
            )
        return build_grouped_stratified_splits(
            labels,
            normalized_groups,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    if normalized_strategy == "random":
        return build_random_splits(
            labels,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    return build_stratified_splits(
        labels,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )


def _split_counts(splits: Sequence[str]) -> dict[str, int]:
    counts = Counter(str(split or "train") for split in splits)
    return {
        "train": int(counts.get("train", 0)),
        "val": int(counts.get("val", 0)),
        "test": int(counts.get("test", 0)),
    }


def _requested_holdout_names(
    *,
    val_fraction: float,
    test_fraction: float,
) -> list[str]:
    requested: list[str] = []
    if float(test_fraction) > 0.0:
        requested.append("test")
    if float(val_fraction) > 0.0:
        requested.append("val")
    return requested


def build_training_dataset_splits(
    labels: Sequence[Hashable],
    *,
    strategy: str = "stratified",
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
    groups: Sequence[Hashable] | None = None,
    min_train_items: int = 2,
) -> tuple[list[str], bool]:
    """Return training splits, falling back to item-level splitting if grouping is too coarse."""

    grouped_splits = build_dataset_splits(
        labels,
        strategy=strategy,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        groups=groups,
    )
    if groups is None:
        return grouped_splits, False

    requested_holdouts = _requested_holdout_names(
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    grouped_counts = _split_counts(grouped_splits)
    grouped_missing = sum(
        1 for split_name in requested_holdouts if grouped_counts.get(split_name, 0) <= 0
    )
    grouped_train_ok = grouped_counts.get("train", 0) >= int(min_train_items)

    item_splits = build_dataset_splits(
        labels,
        strategy=strategy,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        groups=None,
    )
    item_counts = _split_counts(item_splits)
    item_missing = sum(
        1 for split_name in requested_holdouts if item_counts.get(split_name, 0) <= 0
    )
    item_train_ok = item_counts.get("train", 0) >= int(min_train_items)

    if item_train_ok and (item_missing < grouped_missing or not grouped_train_ok):
        return item_splits, True

    return grouped_splits, False
