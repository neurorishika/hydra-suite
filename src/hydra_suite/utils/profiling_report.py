"""Render a span snapshot as log lines.

Kept out of both ``profiling.py`` (which stays a stdlib-only primitive) and
``TrackingProfiler`` (which would otherwise grow a renderer alongside its
lifecycle role).

Percentages are OF THE PARENT, never of a global total: at depth>=2 summed
span time legitimately exceeds wall-clock when threads overlap, so a
"% of run" column would be a lie. Off-thread subtrees carry a ``concurrent``
marker so a subtree that is 43% of its thread but 4% of the pass reads as both.
"""

from __future__ import annotations

SPAN_TREE_HEADER = "  {:<38} {:>9} {:>7} {:>8} {:>9} {:>9}".format(
    "SPAN", "total", "% par", "n", "ms/call", "max ms"
)


def render_tree_lines(snapshot: dict, main_thread: str) -> list[str]:
    """Indented tree, children sorted by ``total_s`` descending."""
    lines: list[str] = []
    _render(snapshot, snapshot["total_s"], 0, main_thread, lines)
    return lines


def _render(
    node: dict, parent_total: float, depth: int, main_thread: str, out: list[str]
) -> None:
    if depth > 0:
        pct = (node["total_s"] / parent_total * 100.0) if parent_total > 0 else 0.0
        n = max(node["n_calls"], 1)
        label = ("  " * depth) + node["name"]
        suffix = ""
        if node.get("units"):
            suffix += f" | {node['total_s'] / node['units'] * 1000:.2f} ms/u"
        if node.get("thread") and node["thread"] != main_thread:
            suffix += f" | concurrent ({node['thread']})"
        out.append(
            "  {:<38} {:>8.2f}s {:>6.1f}% {:>8d} {:>8.2f} {:>8.2f}{}".format(
                label[:38],
                node["total_s"],
                pct,
                node["n_calls"],
                node["total_s"] / n * 1000,
                node["max_s"] * 1000,
                suffix,
            )
        )
    for child in sorted(
        node.get("children", []), key=lambda c: c["total_s"], reverse=True
    ):
        _render(child, node["total_s"], depth + 1, main_thread, out)
