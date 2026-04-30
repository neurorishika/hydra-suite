"""Colorblind-safe palette and category-to-color mapping utilities."""

from __future__ import annotations

import math
from typing import Any, Iterable

from PySide6.QtGui import QColor

# Curated high-contrast palette with colorblind-safe primaries first.
# Order matters: early colors are maximally distinct for small class counts.
_COLORBLIND_PALETTE_HEX = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#661100",
    "#6699CC",
    "#AA4466",
    "#4477AA",
    "#228833",
    "#CCBB44",
    "#EE6677",
    "#BBBBBB",
]


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _normalize_category(value: Any) -> str | None:
    if _is_missing_value(value):
        return None
    return str(value).strip()


def _sort_key(category: str) -> tuple[int, float | str]:
    try:
        return (0, float(category))
    except Exception:
        return (1, category.lower())


def build_category_color_map(
    values: Iterable[Any],
    *,
    category_order: Iterable[Any] | None = None,
) -> dict[str, QColor]:
    """Build deterministic category->QColor mapping from observed values."""
    categories = {c for c in (_normalize_category(v) for v in values) if c is not None}

    ordered: list[str] = []
    if category_order is not None:
        seen: set[str] = set()
        for value in category_order:
            category = _normalize_category(value)
            if category is None or category in seen or category not in categories:
                continue
            ordered.append(category)
            seen.add(category)
        remaining = [category for category in categories if category not in seen]
        ordered.extend(sorted(remaining, key=_sort_key))
    else:
        ordered = sorted(categories, key=_sort_key)

    color_map: dict[str, QColor] = {}
    n_palette = len(_COLORBLIND_PALETTE_HEX)

    for idx, category in enumerate(ordered):
        if idx < n_palette:
            color_map[category] = QColor(_COLORBLIND_PALETTE_HEX[idx])
            continue

        # Fallback for very large category counts: deterministic hue wheel.
        hue = int((idx * 137.50776405) % 360)
        color_map[category] = QColor.fromHsv(hue, 170, 220)

    return color_map


def color_for_value(
    value: Any,
    color_map: dict[str, QColor],
    default: QColor | None = None,
) -> QColor:
    """Resolve a QColor for a value using normalized-category lookup."""
    category = _normalize_category(value)
    if category is None:
        return default or QColor(100, 100, 255)
    return color_map.get(category, default or QColor(100, 100, 255))


def best_text_color(background: QColor) -> QColor:
    """Return black or white text color for maximal contrast on a background."""

    def _linear(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r = _linear(float(background.red()))
    g = _linear(float(background.green()))
    b = _linear(float(background.blue()))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    contrast_white = 1.05 / (luminance + 0.05)
    contrast_black = (luminance + 0.05) / 0.05
    return QColor("#FFFFFF") if contrast_white >= contrast_black else QColor("#000000")


def to_hex(color: QColor) -> str:
    return color.name(QColor.HexRgb)
