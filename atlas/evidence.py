"""Evidence: where in the document an answer's quotes come from, for quick verification.

The model is asked to quote the passages it answers from. Those quotes are located in the
document's text, mapped to pages, and for PDFs to boxes on the rendered page. Matching works on a
"squashed" form of both texts (letters and digits only, case-folded), so it tolerates what text
extraction does to whitespace, line breaks, hyphenation and split words ("identifie d"). A quote
the model shortened with "…" is matched segment by segment, and a quote with a few words changed
still matches approximately (score < 1). A quote that cannot be found is reported as such: the
model may have paraphrased it, or made it up.
"""

import bisect
import re
import threading
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

MIN_QUOTE_WORDS = 4
MIN_QUOTE_CHARS = 12  # squashed (letters and digits)
MAX_QUOTE_CHARS = 2000
MAX_QUOTES = 40
MIN_SCORE = 0.5  # share of the quote found in one place; below this the quote counts as not found
SCATTERED = 0.8  # share of a quote's words on one page for "words found apart" (tables, forms)
RESTATED = 0.6  # share of an unfound quote's words taken from the question: the model quoted the question
GRAM = 12  # characters per anchor when matching approximately
MAX_HITS_PER_GRAM = 64

# written by extract._pdf before each page's text; in merged documents a file's first page is
# preceded by "[File: name]" lines, which belong to that page
PAGE_MARKER = re.compile(r"^(?:\[File: .+\]\n)*\[Page (\d+)\]$", re.M)
_MARKER_IN_QUOTE = re.compile(r"\[(?:File: [^\]\n]+|Page \d+)\]")
_ALNUM = re.compile(r"[^\W_]+")
_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|…|\[\s*(?:\.{3}|…)\s*\]|\(\s*(?:\.{3}|…)\s*\))\s*")
_LINE = r"(?:[^%s\n]|\n(?!\s*\n))*?"  # quoted text: any line breaks but a blank line
# one pattern, so quotation marks pair up left to right: a German „…“ closes with the mark that
# opens an English “…”, and a short "term" must not leave its closing mark to open the next quote
_QUOTED = re.compile('"(%s)"|“(%s)”|„(%s)[“”]|«\\s?(%s)\\s?»' % (_LINE % '"', _LINE % "”", _LINE % "“”", _LINE % "»"))
_BLOCKQUOTE = re.compile(r"(?:^[ \t]*>[^\n]*(?:\n|$))+", re.M)


def squash(text: str) -> str:
    return "".join(m.group().casefold() for m in _ALNUM.finditer(text))


def _words(text: str) -> list[str]:
    """Words of three letters or more and all numbers, case-folded (what a quote is compared by)."""
    return [w.casefold() for w in _ALNUM.findall(text) if len(w) >= 3 or w.isdigit()]


def restates(quote: str, question: str) -> bool:
    """The quote mostly repeats the question in other words (compared by the first five letters, so
    "sortiere" and "sortieren" match): the model quoted the request, not the document."""
    words = [w[:5] for w in _words(quote)]
    asked = {w[:5] for w in _words(question)}
    hits = sum(1 for w in words if w in asked)
    return hits >= 2 and hits / max(len(words), 1) >= RESTATED


def _strip_quote(text: str) -> str:
    text = re.sub(r"^[\s>*_`\-•]+|[\s*_`]+$", "", text)
    for a, b in (('"', '"'), ("“", "”"), ("„", "“"), ("„", "”"), ("«", "»")):
        if text.startswith(a) and text.endswith(b) and len(text) > 2:
            text = text[1:-1]
    return text.strip()


def quotes(answer: str, question: str = "") -> list[str]:
    """The passages an answer quotes: text in quotation marks and blockquotes, long enough to be a
    passage rather than a term, and not taken from the question."""
    found: list[str] = []
    rest = answer
    for m in _BLOCKQUOTE.finditer(answer):
        lines = [re.sub(r"^[ \t]*>[ \t]?", "", line) for line in m.group().splitlines()]
        found.append(_strip_quote(" ".join(line.strip() for line in lines if line.strip())))
        rest = rest.replace(m.group(), "\n")
    found += [next(g for g in m.groups() if g is not None) for m in _QUOTED.finditer(rest)]

    asked = squash(question)
    out: list[str] = []
    seen: list[str] = []
    for q in found:
        q = " ".join(q.split())
        s = squash(q)
        if len(q) > MAX_QUOTE_CHARS or len(q.split()) < MIN_QUOTE_WORDS or len(s) < MIN_QUOTE_CHARS or (asked and s in asked):
            continue
        if any(s in other for other in seen):
            continue  # the same passage again, or a part of one already listed
        seen.append(s)
        out.append(q)
        if len(out) >= MAX_QUOTES:
            break
    return out


@dataclass
class Match:
    start: int  # character offsets in the original text
    end: int
    score: float  # 1.0: every letter of the quote found in order; less: approximate


class Squashed:
    """A text reduced to its letters and digits, with the offset of each in the original."""

    def __init__(self, text: str, skip: list[tuple[int, int]] = ()):
        chars: list[str] = []
        index = array("i")
        skip = sorted(skip)
        k = 0
        for m in _ALNUM.finditer(text):
            while k < len(skip) and skip[k][1] <= m.start():
                k += 1
            if k < len(skip) and skip[k][0] <= m.start() < skip[k][1]:
                continue
            word = m.group()
            folded = word.casefold()
            chars.append(folded)
            if len(folded) == len(word):
                index.extend(range(m.start(), m.end()))
            else:  # "ß" -> "ss": map every folded character to its source character
                for i, ch in enumerate(word):
                    index.extend([m.start() + i] * len(ch.casefold()))
        self.s = "".join(chars)
        self.index = index
        self.n_text = len(text)

    def to_squashed(self, offset: int) -> int:
        return bisect.bisect_left(self.index, offset)

    def to_text(self, lo: int, hi: int) -> tuple[int, int]:
        """Squashed range [lo, hi) as an original-text range."""
        if not self.s:
            return 0, 0
        return self.index[min(lo, len(self.s) - 1)], self.index[min(hi, len(self.s)) - 1] + 1

    def locate(self, quote: str, lo: int = 0, hi: int | None = None) -> Match | None:
        """Where `quote` is in the text; searched within the original-text range [lo, hi)."""
        a = self.to_squashed(lo)
        b = len(self.s) if hi is None else self.to_squashed(hi)
        segments = [s for s in (squash(p) for p in _ELLIPSIS.split(quote)) if len(s) >= 8]
        if not segments:
            return None
        found = [(self._find(seg, a, b), len(seg)) for seg in segments]
        hits = [(m, n) for m, n in found if m]
        if not hits:
            return None
        total = sum(n for _, n in found)
        # segments of one quote must appear in order and near each other; otherwise keep the best
        ordered = all(x[0][0] <= y[0][0] for x, y in zip(hits, hits[1:]))
        spread = hits[-1][0][1] - hits[0][0][0]
        if len(hits) > 1 and ordered and spread <= 3 * total + 2000:
            start, end = hits[0][0][0], hits[-1][0][1]
            score = sum(m[2] * n for m, n in hits) / total
        else:
            (start, end, s), n = max(hits, key=lambda h: h[0][2] * h[1])
            score = s * n / total
        if score < MIN_SCORE:
            return None
        t0, t1 = self.to_text(start, end)
        return Match(t0, t1, round(score, 3))

    def _find(self, q: str, a: int, b: int) -> tuple[int, int, float] | None:
        """(start, end, score) of squashed `q` in self.s[a:b]."""
        s = self.s
        pos = s.find(q, a, b)
        if pos >= 0:
            return pos, pos + len(q), 1.0
        k = GRAM if len(q) >= 3 * GRAM else max(6, len(q) // 3)
        if len(q) < k:
            return None
        step = max(1, k // 2)
        anchors: list[tuple[int, int]] = []  # (offset of the quote in the text, index in the quote)
        for i in list(range(0, len(q) - k + 1, step)) + [len(q) - k]:
            gram = q[i:i + k]
            p = s.find(gram, a, b)
            hits = 0
            while p >= 0 and hits < MAX_HITS_PER_GRAM:
                anchors.append((p - i, i))
                hits += 1
                p = s.find(gram, p + 1, b)
        if not anchors:
            return None
        anchors.sort()
        tolerance = max(8, len(q) // 6)  # words the model left out or added shift later anchors
        best: tuple[float, int, int] | None = None
        j = 0
        for i in range(len(anchors)):
            while anchors[i][0] - anchors[j][0] > tolerance:
                j += 1
            window = anchors[j:i + 1]
            covered = _coverage([idx for _, idx in window], k, len(q))
            if best is None or covered > best[0]:
                best = (covered, j, i + 1)
        covered, j, e = best
        window = anchors[j:e]
        start = min(off + idx for off, idx in window)
        end = max(off + idx for off, idx in window) + k
        return start, end, covered / len(q)


def _coverage(starts: list[int], k: int, n: int) -> int:
    """Characters of the quote covered by anchors of length k starting at `starts`."""
    total, reach = 0, -1
    for s in sorted(set(starts)):
        e = min(s + k, n)
        if e > reach:
            total += e - max(s, reach)
            reach = e
    return total


class DocText:
    """A document's extracted text, prepared for locating quotes and mapping them to pages."""

    def __init__(self, text: str):
        self.text = text
        markers = list(PAGE_MARKER.finditer(text))
        self.page_starts = [m.start() for m in markers]
        self.marker_ends = [m.end() for m in markers]
        self.page_numbers = [int(m.group(1)) for m in markers]
        self.squashed = Squashed(text, [(m.start(), m.end()) for m in markers])

    def page_at(self, offset: int) -> int | None:
        i = bisect.bisect_right(self.page_starts, offset) - 1
        return self.page_numbers[i] if i >= 0 else None

    def page_range(self, page: int) -> tuple[int, int] | None:
        """Text offsets of one page (after its marker), or None if the page has no text."""
        if page not in self.page_numbers:
            return None
        i = self.page_numbers.index(page)
        start = min(self.marker_ends[i] + 1, len(self.text))
        end = self.page_starts[i + 1] if i + 1 < len(self.page_starts) else len(self.text)
        return start, end

    def pages_span(self, first: int, last: int) -> tuple[int, int] | None:
        """Text offsets covering pages first..last (1-based, inclusive)."""
        inside = [i for i, n in enumerate(self.page_numbers) if first <= n <= last]
        if not inside:
            return None
        end = self.page_starts[inside[-1] + 1] if inside[-1] + 1 < len(self.page_starts) else len(self.text)
        return self.page_starts[inside[0]], end

    def find(self, quote: str, prefer: tuple[int, int] | None = None) -> dict:
        """Locate one quote; `prefer` is the range of the document part the answer came from."""
        match, in_part = None, False
        shown, quote = quote, _MARKER_IN_QUOTE.sub(" ", quote).strip() or quote  # a quote may copy "[Page 6]" lines
        if prefer:
            match = self.squashed.locate(quote, *prefer)
            in_part = match is not None
        if match is None or match.score < 0.9:
            anywhere = self.squashed.locate(quote)
            if anywhere and (match is None or anywhere.score > match.score + 0.05):
                match, in_part = anywhere, bool(prefer) and prefer[0] <= anywhere.start < prefer[1]
        if match is None:
            return {"quote": shown, "found": False, **self._scattered(quote, prefer)}
        start, end = match.start, match.end
        while start > 0 and self.text[start - 1].isalnum():  # an approximate match may start mid-word
            start -= 1
        while end < len(self.text) and self.text[end].isalnum():
            end += 1
        head, tail = re.match(r"[^\w\s]*", quote).group(), re.search(r"[^\w\s]*$", quote).group()
        if head and self.text[:start].endswith(head):  # punctuation the quote starts or ends with
            start -= len(head)
        if tail and self.text.startswith(tail, end):
            end += len(tail)
        return {"quote": shown, "found": True, "start": start, "end": end, "score": match.score,
                "page": self.page_at(start), "page_end": self.page_at(max(start, end - 1)), "in_part": in_part}

    def _page_words(self) -> dict[int | None, set[str]]:
        """Each page's words (the whole text as one "page" if it has no page markers)."""
        if not hasattr(self, "_pw"):
            if self.page_numbers:
                self._pw = {n: set(_words(self.text[a:b])) for n in self.page_numbers
                            if (span := self.page_range(n)) for a, b in [span]}
            else:
                self._pw = {None: set(_words(self.text))}
        return self._pw

    def _scattered(self, quote: str, prefer: tuple[int, int] | None) -> dict:
        """A quote not found as one passage whose words are nearly all on one page: typical for tables
        and forms, where the text layer stores cells in another order than they appear on the page."""
        words = _words(quote)
        if len(words) < 3:
            return {}
        pages = self._page_words()
        candidates = [pages]
        if prefer and self.page_numbers:  # the pages of the part the answer came from first
            first, last = self.page_at(prefer[0]), self.page_at(max(prefer[0], prefer[1] - 1))
            if near := {n: w for n, w in pages.items() if first and last and first <= n <= last}:
                candidates.insert(0, near)
        for group in candidates:
            page, have = max(((n, sum(1 for w in words if w in ws)) for n, ws in group.items()), key=lambda x: x[1])
            if have / len(words) >= SCATTERED:
                return {"scattered": True, "coverage": round(have / len(words), 2), "page": page, "page_end": page}
        return {}

    def evidence(self, answer: str, question: str = "", prefer: tuple[int, int] | None = None) -> list[dict]:
        out = []
        for q in quotes(answer, question):
            hit = self.find(q, prefer)
            if not hit["found"] and not hit.get("scattered") and question and restates(q, question):
                continue  # the model put (a rewording of) the question in quotation marks
            out.append(hit)
        return out


class TextCache:
    """Recently used documents' prepared texts (building one takes ~0.1 s per 300k characters)."""

    def __init__(self, size: int = 16):
        self.size = size
        self._items: OrderedDict[tuple[str, float], DocText] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: Path) -> DocText:
        key = (str(path), path.stat().st_mtime)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        doc = DocText(path.read_text(encoding="utf-8"))
        with self._lock:
            self._items[key] = doc
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return doc


def page_boxes(pdf_data: bytes, page: int, passage: str) -> dict:
    """Boxes of `passage` on a PDF page, as fractions of the rendered page (x0, y0, x1, y1 from the
    top left). The passage is located in pdfium's own text of the page, so the character boxes fit."""
    import pypdfium2 as pdfium

    from .pages import PDFIUM
    with PDFIUM:
        pdf = pdfium.PdfDocument(pdf_data)
        try:
            if not 1 <= page <= len(pdf):
                raise ValueError("no such page")
            p = pdf[page - 1]
            textpage = p.get_textpage()
            try:
                text = textpage.get_text_range()
                match = Squashed(text).locate(passage)
                rects = []
                if match:
                    n = textpage.count_rects(match.start, match.end - match.start)
                    rects = [textpage.get_rect(i) for i in range(n)]
                left, bottom, right, top = p.get_cropbox()
                rotation = p.get_rotation()
            finally:
                textpage.close()
                p.close()
        finally:
            pdf.close()
    w, h = right - left, top - bottom
    boxes = []
    for l, b, r, t in rects:
        u0, v0, u1, v1 = (l - left) / w, (top - t) / h, (r - left) / w, (top - b) / h
        boxes.append([round(x, 4) for x in _rotate(u0, v0, u1, v1, rotation)])
    return {"boxes": boxes, "score": match.score if match else 0.0}


def _rotate(u0: float, v0: float, u1: float, v1: float, rotation: int) -> tuple[float, float, float, float]:
    """A box in unrotated page fractions, as it appears on the page rendered with its /Rotate."""
    if rotation == 90:
        return 1 - v1, u0, 1 - v0, u1
    if rotation == 180:
        return 1 - u1, 1 - v1, 1 - u0, 1 - v0
    if rotation == 270:
        return v0, 1 - u1, v1, 1 - u0
    return u0, v0, u1, v1
