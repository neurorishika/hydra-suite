"""Bounded human-readable diagnostics for process and GUI boundaries."""

from __future__ import annotations

MAX_TERMINAL_TEXT_BYTES = 32 * 1024
_TRUNCATED_TEXT_SUFFIX = b"\n[message truncated]"
_MAX_TERMINAL_PAYLOAD_DEPTH = 16


def _safe_exception_text(error: BaseException, *, include_exception_type: bool) -> str:
    """Describe an exception without invoking its arbitrary ``__str__``."""

    exception_type = type(error).__name__
    try:
        raw_args = error.args
    except BaseException:  # noqa: B036 - diagnostic formatting must not fail
        raw_args = ()
    if type(raw_args) is not tuple:
        raw_args = ()

    safe_args: list[str] = []
    for arg in raw_args[:8]:
        if type(arg) is str:
            safe_args.append(arg[:MAX_TERMINAL_TEXT_BYTES])
        elif type(arg) is int:
            safe_args.append(repr(arg) if arg.bit_length() <= 8192 else "<large int>")
        elif type(arg) in (float, bool, type(None)):
            safe_args.append(repr(arg))
        else:
            safe_args.append(f"<{type(arg).__name__}>")
    if not safe_args:
        return exception_type
    detail = ", ".join(safe_args)
    if not include_exception_type:
        return detail
    return f"{exception_type}: {detail}"


def bounded_terminal_text(
    value: object,
    *,
    max_bytes: int = MAX_TERMINAL_TEXT_BYTES,
    include_exception_type: bool = True,
) -> str:
    """Return a bounded valid-UTF-8 diagnostic without arbitrary formatting."""

    if max_bytes < len(_TRUNCATED_TEXT_SUFFIX):
        raise ValueError("terminal text cap is smaller than the truncation suffix")
    if isinstance(value, BaseException):
        text = _safe_exception_text(
            value, include_exception_type=include_exception_type
        )
    elif type(value) is str:
        text = value
    elif type(value) is int:
        text = repr(value) if value.bit_length() <= 8192 else "<large int>"
    elif type(value) in (float, bool, type(None)):
        text = repr(value)
    else:
        text = f"<{type(value).__name__}>"

    # Slice before encoding so even a very large caller-owned string creates
    # only a bounded temporary. Decode on every path to remove lone surrogates
    # before the value crosses a Qt or JSON boundary.
    candidate = text[:max_bytes]
    encoded = candidate.encode("utf-8", errors="replace")
    truncated = len(text) > len(candidate) or len(encoded) > max_bytes
    if not truncated:
        return encoded.decode("utf-8")
    retained = encoded[: max_bytes - len(_TRUNCATED_TEXT_SUFFIX)].decode(
        "utf-8", errors="ignore"
    )
    return retained + _TRUNCATED_TEXT_SUFFIX.decode("ascii")


def sanitize_terminal_text_fields(value: object, *, _depth: int = 0) -> object:
    """Bound every string leaf in a small internal terminal-result payload.

    Exact built-in containers are updated in place where possible, avoiding a
    second full payload copy before Qt queues the result. Unexpected custom
    containers remain opaque rather than invoking user-controlled iteration.
    """

    if isinstance(value, BaseException):
        return bounded_terminal_text(value)
    if type(value) is str:
        return bounded_terminal_text(value)
    if _depth >= _MAX_TERMINAL_PAYLOAD_DEPTH:
        return "<terminal payload nesting omitted>"
    if type(value) is list:
        for index in range(len(value)):
            value[index] = sanitize_terminal_text_fields(
                value[index], _depth=_depth + 1
            )
        return value
    if type(value) is dict:
        for key in value:
            value[key] = sanitize_terminal_text_fields(value[key], _depth=_depth + 1)
        return value
    if type(value) is tuple:
        return tuple(
            sanitize_terminal_text_fields(item, _depth=_depth + 1) for item in value
        )
    return value
