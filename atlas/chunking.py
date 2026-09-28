"""Split documents that exceed a single slot's context into overlapping parts at natural breaks."""

from collections.abc import Awaitable, Callable

TokenCounter = Callable[[str], Awaitable[int]]

_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def find_break(text: str, lo: int, hi: int) -> int:
    """Best cut position in text[lo:hi], preferring paragraph > line > sentence > word breaks.

    Only the second half of the window is searched so parts don't end up tiny.
    """
    if hi >= len(text):
        return len(text)
    floor = lo + (hi - lo) // 2
    for sep in _BREAKS:
        idx = text.rfind(sep, floor, hi)
        if idx != -1:
            return idx + len(sep)
    return hi


def _overlap_start(text: str, start: int, end: int, overlap_chars: int) -> int:
    if overlap_chars <= 0:
        return end
    target = max(start + 1, end - overlap_chars)
    nl = text.find("\n", target, end)
    if nl != -1:
        return nl + 1
    sp = text.find(" ", target, end)
    return sp + 1 if sp != -1 else target


async def plan_parts(text: str, count_tokens: TokenCounter, budget: int, overlap_tokens: int = 0) -> list[tuple[int, int]]:
    """Return (start, end) character spans, each of at most `budget` tokens."""
    if budget < 64:
        raise ValueError("token budget per part is too small")
    total = await count_tokens(text)
    if total <= budget:
        return [(0, len(text))]

    chars_per_token = len(text) / max(total, 1)
    overlap_chars = int(min(overlap_tokens, budget // 4) * chars_per_token)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        window = int(budget * chars_per_token * 0.97)
        while True:
            end = find_break(text, start, start + max(window, 1))
            n = await count_tokens(text[start:end])
            if n <= budget:
                break
            window = int(window * budget / n * 0.95)
            if window < 16:
                raise ValueError("could not split document into parts that fit the token budget")
        spans.append((start, end))
        if end >= len(text):
            break
        start = _overlap_start(text, start, end, overlap_chars)
    return spans
