import pytest

from atlas.chunking import find_break, plan_parts


async def count_words(text: str) -> int:
    return len(text.split())


def test_find_break_prefers_paragraphs():
    text = "aaaa bbbb. cccc dddd.\n\neeee. ffff gggg"
    assert text[: find_break(text, 0, 34)].endswith("\n\n")


def test_find_break_ignores_breaks_in_first_half():
    text = "aa\n\nbbbbbbbbbbbbbbbb cccccccccccccccc dddd"
    assert find_break(text, 0, 40) > 20


def test_find_break_end_of_text():
    assert find_break("short", 0, 100) == 5


async def test_small_text_is_one_part():
    assert await plan_parts("one two three", count_words, budget=100) == [(0, 13)]


@pytest.mark.parametrize("overlap", [0, 20])
async def test_parts_cover_text_and_respect_budget(overlap):
    paragraphs = [" ".join(f"w{p}_{i}" for i in range(37)) for p in range(40)]
    text = "\n\n".join(paragraphs)
    budget = 150
    spans = await plan_parts(text, count_words, budget=budget, overlap_tokens=overlap)
    assert len(spans) > 1
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s, e), nxt in zip(spans, spans[1:] + [None]):
        assert await count_words(text[s:e]) <= budget
        if nxt:
            assert nxt[0] <= e, "parts must not leave gaps"
            assert nxt[0] > s, "parts must advance"
    if overlap:
        assert any(nxt[0] < e for (s, e), nxt in zip(spans, spans[1:]))


async def test_text_without_breaks_still_splits():
    text = "x" * 5000

    async def count_chars(t: str) -> int:
        return len(t)

    spans = await plan_parts(text, count_chars, budget=700)
    assert all(e - s <= 700 for s, e in spans)
    assert spans[-1][1] == len(text)
