import pytest

from atlas import prompts
from atlas.prompts import DOC_SENTINEL, QUESTION_SENTINEL, ThinkSplitter


def test_split_rendered():
    rendered = f"<s>[SYS]be nice[/SYS][USER]{DOC_SENTINEL}{QUESTION_SENTINEL}[/USER][ASSISTANT]"
    lay = prompts.split_rendered(rendered)
    assert lay.head == "<s>[SYS]be nice[/SYS][USER]"
    assert lay.mid == ""
    assert lay.tail == "[/USER][ASSISTANT]"
    assert not lay.thinking_open


def test_split_rendered_rejects_mangled_templates():
    with pytest.raises(prompts.TemplateError):
        prompts.split_rendered(f"{DOC_SENTINEL} only")
    with pytest.raises(prompts.TemplateError):
        prompts.split_rendered(f"{QUESTION_SENTINEL}{DOC_SENTINEL}")
    with pytest.raises(prompts.TemplateError):
        prompts.split_rendered(f"{DOC_SENTINEL}{DOC_SENTINEL}{QUESTION_SENTINEL}")


def test_thinking_open():
    assert prompts.Layout("", "", "<|im_start|>assistant\n<think>\n").thinking_open
    assert not prompts.Layout("", "", "<|im_start|>assistant\n<think>\n\n</think>\n\n").thinking_open


def feed_all(splitter: ThinkSplitter, chunks: list[str]) -> tuple[str, str]:
    out = {"answer": "", "reasoning": ""}
    for c in chunks:
        for kind, piece in splitter.feed(c):
            out[kind] += piece
    for kind, piece in splitter.flush():
        out[kind] += piece
    return out["answer"], out["reasoning"]


def test_think_splitter_handles_tags_split_across_chunks():
    text = "<think>step one, step two</think>The answer is 42."
    for size in range(1, len(text) + 1):
        chunks = [text[i:i + size] for i in range(0, len(text), size)]
        answer, reasoning = feed_all(ThinkSplitter(), chunks)
        assert answer == "The answer is 42."
        assert reasoning == "step one, step two"


def test_think_splitter_starting_inside_reasoning():
    answer, reasoning = feed_all(ThinkSplitter(start_in_reasoning=True), ["hmm", "m</th", "ink>ok"])
    assert (answer, reasoning) == ("ok", "hmmm")


def test_think_splitter_keeps_lone_angle_brackets():
    answer, _ = feed_all(ThinkSplitter(), ["a <", "b> c <t", "able>"])
    assert answer == "a <b> c <table>"


def test_strip_reasoning():
    assert prompts.strip_reasoning("<think>x</think> yes") == "yes"
    assert prompts.strip_reasoning("<think>never closed") == ""


@pytest.mark.parametrize("text,expected", [
    ("NO_RELEVANT_INFORMATION", True),
    ("  **NO_RELEVANT_INFORMATION**", True),
    ("no_relevant_information.", True),
    ("NO_RELEVANT_INFORMATION\nThe document is about fleets.", True),
    ("The CISO is Dr. Brandt. NO_RELEVANT_INFORMATION", False),
    ("Revenue was EUR 48.2 million.", False),
])
def test_is_no_info(text, expected):
    assert prompts.is_no_info(text) is expected


@pytest.mark.parametrize("text,answer,rating", [
    ("Revenue was 48M.\nCOVERAGE: full", "Revenue was 48M.", "full"),
    ("Partly.\n\n**COVERAGE:** partial", "Partly.", "partial"),
    ("Nothing here.\ncoverage: NONE.", "Nothing here.", "none"),
    ("COVERAGE: full is mentioned\nlater COVERAGE: partial", "COVERAGE: full is mentioned\nlater", "partial"),
    ("No rating at all.", "No rating at all.", None),
])
def test_split_coverage(text, answer, rating):
    assert prompts.split_coverage(text) == (answer, rating)


def test_document_block_escapes_name():
    block = prompts.document_block('a "quoted"\nname', 1, 3, "body")
    assert block.startswith('<document name="a \'quoted\' name" part="2 of 3">\nbody\n')
