"""Prompt layout.

A cached document prompt must be a strict token prefix of every query prompt built on it.
We therefore render the model's own chat template once with two sentinels in the user turn

    [system: <prompt>] [user: <DOC><QUESTION>] [assistant generation prompt]

and cut the rendered text at the sentinels into head / mid / tail:

    prefix (persisted per document part) = head + document block + mid
    suffix (appended per query)          = question block + tail

Template text is tokenized with special-token parsing; document text, file names, questions
and model output are tokenized as plain text, so content cannot inject control tokens.
"""

import re
from dataclasses import dataclass

DOC_SENTINEL = "[[ATLAS-DOCUMENT-SLOT-5c1e]]"
QUESTION_SENTINEL = "[[ATLAS-QUESTION-SLOT-5c1e]]"
TURN_SENTINEL = "[[ATLAS-TURN-{}-5c1e]]"
NO_INFO = "NO_RELEVANT_INFORMATION"

# Bump whenever the structure of the persisted prefix changes, so stored KV caches get rebuilt.
PREFIX_VERSION = "1"


class TemplateError(RuntimeError):
    pass


@dataclass(frozen=True)
class Layout:
    head: str
    mid: str
    tail: str

    @property
    def thinking_open(self) -> bool:
        """True if the generation prompt already opened a reasoning block (model starts inside <think>)."""
        return self.tail.rfind("<think>") > self.tail.rfind("</think>")


def split_rendered(rendered: str) -> Layout:
    if rendered.count(DOC_SENTINEL) != 1 or rendered.count(QUESTION_SENTINEL) != 1:
        raise TemplateError("chat template did not preserve the content sentinels exactly once")
    i = rendered.index(DOC_SENTINEL)
    j = rendered.index(QUESTION_SENTINEL)
    if j < i:
        raise TemplateError("chat template reordered the user content")
    return Layout(rendered[:i], rendered[i + len(DOC_SENTINEL): j], rendered[j + len(QUESTION_SENTINEL):])


def layout_messages(system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": DOC_SENTINEL + QUESTION_SENTINEL},
    ]


# --- chat ------------------------------------------------------------------------------
# A question in a conversation is sent as a real chat: the cached document opens the first user
# turn, followed by the earlier questions and answers and the new question:
#
#   [system] [user: <DOC> q1] [assistant: a1] [user: q2] … [user: question] [generation prompt]
#
# The template is rendered with one sentinel per turn, so the template's own text (tokenized with
# special tokens) and the turns' content (tokenized as plain text) are kept apart.


@dataclass(frozen=True)
class ChatRender:
    head: str  # template text before the document (or before the first turn without one)
    glue: tuple[str, ...]  # template text before each turn, then the generation prompt (n + 1 pieces)

    @property
    def tail(self) -> str:
        return self.glue[-1]


def chat_messages(system_prompt: str, n_turns: int, with_doc: bool) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for i in range(n_turns):
        content = TURN_SENTINEL.format(i)
        if i == 0 and with_doc:
            content = DOC_SENTINEL + content
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": content})
    return messages


def split_chat(rendered: str, n_turns: int, with_doc: bool) -> ChatRender:
    marks = ([DOC_SENTINEL] if with_doc else []) + [TURN_SENTINEL.format(i) for i in range(n_turns)]
    pieces, pos = [], 0
    for mark in marks:
        if rendered.count(mark) != 1 or rendered.index(mark) < pos:
            raise TemplateError("chat template did not keep the conversation's turns in order")
        i = rendered.index(mark)
        pieces.append(rendered[pos:i])
        pos = i + len(mark)
    pieces.append(rendered[pos:])
    if with_doc:
        return ChatRender(pieces[0], tuple(pieces[1:]))
    return ChatRender(pieces[0], ("", *pieces[1:]))


def history_turns(history: list[tuple[str, str]]) -> list[str]:
    """Earlier (question, answer) pairs as alternating user / assistant contents."""
    return [text for question, answer in history for text in (question, answer)]


def _attr(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', "'").replace("\n", " ")


def document_block(name: str, part_idx: int, n_parts: int, text: str) -> str:
    part = f' part="{part_idx + 1} of {n_parts}"' if n_parts > 1 else ""
    return f'<document name="{_attr(name)}"{part}>\n{text}\n</document>\n\n'


# --- visual prefill ----------------------------------------------------------------------
# A visual document is sent as a multimodal prompt string: page images take the place of media
# markers. The marker is random per llama-server process, so prefixes are stored with a
# placeholder and the current marker is substituted when a prompt is sent.

MEDIA_PLACEHOLDER = "<<ATLAS-MEDIA-5c1e>>"


def defuse(text: str) -> str:
    """Multimodal prompt strings are tokenized with special-token parsing: keep user text (names,
    questions) from forming control tokens such as <|im_end|> or [INST]."""
    return text.replace("<", "<\u200b").replace("[", "[\u200b")


def visual_document_block(name: str, part_idx: int, n_parts: int, page_numbers: list[int]) -> str:
    part = f' part="{part_idx + 1} of {n_parts}"' if n_parts > 1 else ""
    pages = "".join(f"[Page {n}]\n{MEDIA_PLACEHOLDER}\n\n" for n in page_numbers)
    return f'<document name="{defuse(_attr(name))}"{part}>\n{pages}</document>\n\n'


def single_question_block(question: str) -> str:
    return (
        "Answer the question using only the document above, in the language of the question. "
        "If the document does not contain the answer, say so clearly.\n\n"
        f"Question: {question}"
    )


def map_question_block(question: str, rate_coverage: bool = False) -> str:
    """Per-document question for the map phase.

    Chosen by benchmark (2B model; 5 multi-document questions x 3 runs, facts found in the final
    answer): plain answers that are all synthesized found 24/27 facts. With rate_coverage every
    answer ends with a self-rating that the relevance filter uses to drop "none" answers: 23/27,
    23% faster. Showing the ratings to the synthesis instead of filtering scored 17/27, and an
    early "reply NO_RELEVANT_INFORMATION" exit dropped 20/33 relevant documents on compound
    questions, so neither is used. Rewording the rating prompt moved misses from 3/33 to 14/33:
    re-run a benchmark before changing any of these texts.
    """
    head = (
        f"Question: {question}\n\n"
        "Quote the passages of the document above that are relevant to the question, then answer "
        "based on those quotes. The question may have several parts and this document may cover "
        "only some of them"
    )
    if rate_coverage:
        return head + (
            ". End your reply with one final line: COVERAGE: full, COVERAGE: partial or "
            "COVERAGE: none, stating how much of the question this document answers."
        )
    return head + ": answer the parts it covers and briefly state which parts it does not cover."


_COVERAGE_RE = re.compile(r"[*_`]*COVERAGE[*_`]*\s*:\s*[*_`]*\s*(full|partial|none)\b[*_`.]*", re.I)


def split_coverage(answer: str) -> tuple[str, str | None]:
    """Remove the trailing coverage rating from a map answer; returns (answer, rating or None)."""
    matches = list(_COVERAGE_RE.finditer(answer))
    if not matches:
        return answer.strip(), None
    last = matches[-1]
    return (answer[: last.start()] + answer[last.end():]).strip(), last.group(1).lower()


def findings_block(findings: list[tuple[str, str]]) -> str:
    """findings: (label, text) pairs; label is e.g. '[3] Source: report.pdf (part 2 of 4)'."""
    return "Findings:\n\n" + "\n\n".join(f"{label}\n{text.strip()}" for label, text in findings) + "\n\n"


def synthesis_question_block(question: str) -> str:
    return f"Question: {question}"


def history_block(history: list[tuple[str, str]]) -> str:
    """Earlier turns as text inside the question, for chat templates that cannot render them as turns."""
    lines = [f"User: {q.strip()}\n\nAssistant: {a.strip()}" for q, a in history]
    return "Conversation so far:\n\n" + "\n\n".join(lines) + "\n\nNew message:\n"


_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)


def strip_reasoning(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def is_no_info(answer: str) -> bool:
    """True if the model's first words declare the document irrelevant."""
    return answer.strip().lstrip("*`'\"").upper().startswith(NO_INFO)


class ThinkSplitter:
    """Incrementally routes streamed text into ('reasoning', s) or ('answer', s) pieces."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, start_in_reasoning: bool = False):
        self.in_reasoning = start_in_reasoning
        self._buf = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buf += text
        out: list[tuple[str, str]] = []
        while self._buf:
            tag = self.CLOSE if self.in_reasoning else self.OPEN
            idx = self._buf.find(tag)
            if idx >= 0:
                if idx:
                    out.append(self._piece(self._buf[:idx]))
                self._buf = self._buf[idx + len(tag):]
                self.in_reasoning = not self.in_reasoning
                continue
            # keep a possible partial tag at the end of the buffer
            keep = 0
            for k in range(min(len(tag) - 1, len(self._buf)), 0, -1):
                if tag.startswith(self._buf[-k:]):
                    keep = k
                    break
            emit = self._buf[: len(self._buf) - keep]
            if emit:
                out.append(self._piece(emit))
            self._buf = self._buf[len(self._buf) - keep:]
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        out = [self._piece(self._buf)] if self._buf else []
        self._buf = ""
        return out

    def _piece(self, s: str) -> tuple[str, str]:
        return ("reasoning" if self.in_reasoning else "answer", s)
