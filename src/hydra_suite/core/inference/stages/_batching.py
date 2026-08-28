from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


def predict_in_chunks(
    items: Sequence[_T],
    batch_size: int,
    predict: Callable[[Sequence[_T]], list[_R]],
) -> list[_R]:
    """Run an order-preserving backend forward in configured-size chunks."""
    if not items:
        return []
    chunk_size = max(1, int(batch_size))
    if len(items) <= chunk_size:
        return predict(items)

    out: list[_R] = []
    for start in range(0, len(items), chunk_size):
        out.extend(predict(items[start : start + chunk_size]))
    return out
