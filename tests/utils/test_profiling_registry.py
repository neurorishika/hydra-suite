import re
from pathlib import Path

import pytest

import hydra_suite
from hydra_suite.utils import profiling_names
from hydra_suite.utils.profiling import SpanRecorder
from hydra_suite.utils.profiling_names import spanned

SRC = Path(hydra_suite.__file__).parent


def _constants() -> dict[str, str]:
    return {
        k: v
        for k, v in vars(profiling_names).items()
        if k.isupper() and isinstance(v, str)
    }


def test_decorator_records_a_span():
    rec = SpanRecorder()

    @spanned("decorated")
    def work():
        return 42

    with rec.armed():
        assert work() == 42
    assert rec.snapshot()["children"][0]["name"] == "decorated"


def test_decorator_preserves_name_and_doc():
    @spanned("x")
    def work():
        """docstring."""

    assert work.__name__ == "work"
    assert work.__doc__ == "docstring."


def test_decorator_is_transparent_when_disarmed():
    @spanned("y")
    def work(a, b=2):
        return a + b

    assert work(1, b=3) == 4


@pytest.mark.xfail(
    reason="arms once instrumentation lands (Task 12 removes this marker)",
    strict=False,
)
def test_every_constant_is_used_somewhere_in_src():
    """A refactor that drops a span must fail a test, not go silent.

    Matches the ATTRIBUTE REFERENCE (``N.CNN`` / ``profiling_names.CNN``), not
    the bare name. A raw substring check is vacuous: ``CNN`` occurs in 79 files
    via ``CNNModel``, ``POSE`` in 29 via ``ENABLE_POSE_EXTRACTOR``, ``WARP``
    via ``WARP_BATCH``, ``WRITE`` via ``CACHE_WRITE``. Every one of those spans
    could be deleted wholesale and a substring test would stay green.
    """
    sources = [
        p.read_text() for p in SRC.rglob("*.py") if p.name != "profiling_names.py"
    ]
    blob = "\n".join(sources)
    unused = [
        k
        for k in _constants()
        if not re.search(rf"(?:\bN|profiling_names)\.{k}\b", blob)
    ]
    assert not unused, f"span names declared but never placed: {unused}"


def test_span_call_sites_use_constants_not_literals():
    """Enforces the registry rule: no duplicated string literals."""
    bad: list[str] = []
    pattern = re.compile(r"(?:^|[^\w.])(?:span|spanned)\(\s*([\"'])")
    for p in SRC.rglob("*.py"):
        if p.name in {"profiling.py", "profiling_names.py", "profiling_process.py"}:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pattern.search(line):
                bad.append(f"{p.relative_to(SRC)}:{i}: {line.strip()}")
    assert (
        not bad
    ), "span() called with a string literal; use a NAMES constant:\n" + "\n".join(bad)
