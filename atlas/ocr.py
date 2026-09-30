"""OCR for documents without a text layer (scans, pictures of text).

The model reads such documents as page images. To check its answers, their quotes are located in
text read off the same pages by an OCR engine that has nothing to do with the model: that text
becomes the document's text (text.txt, with "[Page n]" lines, used to locate quotes and never shown
to the model), and where every recognized piece of text is on its page is kept in ocr.json, so a
located quote can be marked on the page image.

Engine: RapidOCR (PP-OCRv6 models on onnxruntime, CPU). Chosen by measurement on 30 scanned pages
of German company standards (tables, prose, sideways margin notes): it read 98.7 % of the words at
least three of four engine setups agreed on and produced 66 words no other setup read, against
96.4-97.5 % and 550-640 for Tesseract 5 (fast and best models), which dropped umlauts and turned
table rules into words. Its orientation classifier turned short upright lines upside down
("≤ 600" read as "009"), so it is not used: lines are read as they are, and sideways text (a box
taller than wide) is read turned both ways, keeping the more confident reading.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from . import pages as page_images

log = logging.getLogger(__name__)

OCR_DPI = 300  # scans are usually 300 dpi; rendering lower loses small print
MAX_IMAGE_SIDE = 5000
SIDEWAYS = 1.5  # a box this much taller than wide holds text running up or down the page
OCR_FILE = "ocr.json"
# The PP-OCR model also reads Chinese, and on Latin pages writes a few symbols as look-alike
# characters: table dashes as 一, check boxes as 口
LOOKALIKES = str.maketrans({"一": "–", "口": "□", "丨": "|"})


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import rapidocr  # noqa: F401
    except ImportError:
        return False
    return True


class Recognizer:
    """RapidOCR, created on first use. Returns [(text, (x0, y0, x1, y1) in pixels, turn)] where
    turn is 0 for horizontal text, 90 for text running down the page and -90 for text running up."""

    def __init__(self, threads: int):
        self.threads = threads
        self._engine = None

    def _get(self):
        if self._engine is None:
            from rapidocr import RapidOCR
            self._engine = RapidOCR(params={
                "Global.log_level": "error", "Global.use_cls": False,
                "EngineConfig.onnxruntime.intra_op_num_threads": self.threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })
        return self._engine

    def __call__(self, image) -> list[tuple[str, tuple[float, float, float, float], int]]:
        import numpy as np
        engine = self._get()
        # options given to a call stay set for later calls: always pass all of them
        found = engine(np.array(image), use_det=True, use_cls=False, use_rec=True)
        out = []
        for box, text, _ in zip(found.boxes if found.txts else [], found.txts or [], found.scores or []):
            xs, ys = [float(p[0]) for p in box], [float(p[1]) for p in box]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            turn = 0
            if y1 - y0 > SIDEWAYS * (x1 - x0):
                crop = image.crop((int(x0) - 4, int(y0) - 4, int(x1) + 4, int(y1) + 4))
                best = None
                for angle in (90, -90):  # PIL turns counterclockwise: 90 straightens text running down
                    r = engine(np.array(crop.rotate(angle, expand=True, fillcolor="white")),
                               use_det=False, use_cls=False, use_rec=True)
                    if r.txts and (best is None or r.scores[0] > best[1]):
                        best = (r.txts[0], float(r.scores[0]), angle)
                if best is None:
                    continue
                text, _, turn = best
            text = " ".join(str(text).split())
            if text:
                out.append((text, (x0, y0, x1, y1), turn))
        return out


def page_images_for_ocr(original: Path):
    """The pages of a PDF rendered at OCR_DPI, or the image itself (PIL images, one at a time)."""
    ext = original.suffix.lower()
    if ext in page_images.IMAGE_EXTENSIONS:
        img = page_images._open_image(original.read_bytes()).convert("RGB")
        if max(img.size) > MAX_IMAGE_SIDE:
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        yield img
        return
    import pypdfium2 as pdfium
    with page_images.PDFIUM:
        pdf = pdfium.PdfDocument(original.read_bytes())
        n = len(pdf)
    try:
        for i in range(n):
            with page_images.PDFIUM:  # page by page, so rendering elsewhere is not blocked for long
                image = page_images.render_page(pdf, i, OCR_DPI / 72).convert("RGB")
            yield image
    finally:
        with page_images.PDFIUM:
            pdf.close()


def latin_lookalikes(found: list) -> list:
    """Symbols read as Chinese look-alikes, put back on a page that is (almost) all Latin script."""
    text = "".join(f[0] for f in found)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    if not cjk or cjk > 0.2 * len(text):
        return found
    return [(t.translate(LOOKALIKES), box, turn) for t, box, turn in found]


def rows(found: list) -> list[list]:
    """Horizontal text in reading order: boxes whose middles are level form a row (a line, or a
    table row with its cells left to right); sideways text follows as rows of its own."""
    flat = sorted((f for f in found if not f[2]), key=lambda f: (f[1][1] + f[1][3]) / 2)
    out: list[list] = []
    for f in flat:
        mid, height = (f[1][1] + f[1][3]) / 2, f[1][3] - f[1][1]
        if out:
            last = out[-1]
            row_mid = sum((g[1][1] + g[1][3]) / 2 for g in last) / len(last)
            row_height = sum(g[1][3] - g[1][1] for g in last) / len(last)
            if abs(mid - row_mid) < 0.5 * min(height, row_height):
                last.append(f)
                continue
        out.append([f])
    for row in out:
        row.sort(key=lambda f: f[1][0])
    return out + [[f] for f in sorted((f for f in found if f[2]), key=lambda f: (f[1][0], f[1][1]))]


def read_document(original: Path, recognize, on_page=None) -> tuple[str, dict]:
    """OCR every page. Returns the text ("[Page n]" before each page that has text, like extracted
    PDF text) and the layout: for each page, [start, end, x0, y0, x1, y1, turn] per recognized piece,
    with character offsets into the text and the box as fractions of the page."""
    parts: list[str] = []
    pos = 0
    layout = []
    for n, image in enumerate(page_images_for_ocr(original), 1):
        w, h = image.size
        boxes = []
        page_rows = rows(latin_lookalikes(recognize(image)))
        if page_rows:
            head = ("\n\n" if parts else "") + f"[Page {n}]\n"
            parts.append(head)
            pos += len(head)
            for r, row in enumerate(page_rows):
                if r:
                    parts.append("\n")
                    pos += 1
                for c, (text, (x0, y0, x1, y1), turn) in enumerate(row):
                    if c:
                        parts.append(" ")
                        pos += 1
                    boxes.append([pos, pos + len(text), round(x0 / w, 4), round(y0 / h, 4),
                                  round(x1 / w, 4), round(y1 / h, 4), turn])
                    parts.append(text)
                    pos += len(text)
        layout.append({"n": n, "boxes": boxes})
        if on_page:
            on_page(n)
    return "".join(parts), {"engine": engine_name(), "dpi": OCR_DPI, "pages": layout}


def engine_name() -> str:
    try:
        from importlib.metadata import version
        return f"RapidOCR {version('rapidocr')} (PP-OCRv6)"
    except Exception:
        return "RapidOCR"


def boxes(layout: dict, page: int, start: int, end: int) -> list[list[float]]:
    """Where the text between two offsets is on a page (fractions of the page). A piece only partly
    inside is cut in proportion to its characters, along the direction its text runs."""
    out = []
    for s, e, x0, y0, x1, y1, turn in next((p["boxes"] for p in layout["pages"] if p["n"] == page), []):
        if e <= start or s >= end:
            continue
        f0, f1 = (max(start, s) - s) / (e - s), (min(end, e) - s) / (e - s)
        if turn == 90:  # runs down the page
            out.append([x0, y0 + f0 * (y1 - y0), x1, y0 + f1 * (y1 - y0)])
        elif turn == -90:  # runs up the page
            out.append([x0, y1 - f1 * (y1 - y0), x1, y1 - f0 * (y1 - y0)])
        else:
            out.append([x0 + f0 * (x1 - x0), y0, x0 + f1 * (x1 - x0), y1])
    return [[round(v, 4) for v in b] for b in out]


def load(doc_dir: Path) -> dict | None:
    try:
        return json.loads((doc_dir / OCR_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class OcrWorker:
    """Reads documents that have no text (visual documents: scans, images) one at a time in the
    background. A document can be asked while it waits: only locating quotes needs the text."""

    def __init__(self, store, settings, recognizer=None):
        self.store = store
        self.settings = settings
        self.enabled = settings.ocr and (recognizer is not None or available())
        self.recognizer = recognizer or Recognizer(settings.ocr_threads)
        self.state: dict[str, dict] = {}  # doc id -> {"state": queued|running|failed, "done", "total", "error"}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def needs_ocr(self, doc) -> bool:
        return (self.enabled and doc.n_chars == 0 and page_images.supports_visual(doc.name)
                and not (self.settings.docs_dir / doc.id / OCR_FILE).exists())

    def enqueue(self, doc_id: str) -> None:
        doc = self.store.get_document(doc_id)
        if doc is None or not self.needs_ocr(doc) or self.state.get(doc_id, {}).get("state") in ("queued", "running"):
            return
        self.state[doc_id] = {"state": "queued", "done": 0, "total": doc.n_pages or 0}
        self._queue.put_nowait(doc_id)

    def start(self) -> None:
        if not self.enabled:
            return
        for doc in self.store.list_documents():  # documents added before OCR existed, or left unread
            self.enqueue(doc.id)
        self._task = asyncio.create_task(self._run(), name="ocr")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def status(self, doc) -> dict | None:
        """What the library shows: done (with the engine), queued, running (pages), failed, or None."""
        if doc.id in self.state:
            return dict(self.state[doc.id])
        if (self.settings.docs_dir / doc.id / OCR_FILE).exists():
            return {"state": "done"}
        return None

    async def _run(self) -> None:
        while True:
            doc_id = await self._queue.get()
            try:
                await self._read(doc_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.exception("OCR of %s failed", doc_id)
                self.state[doc_id] = {"state": "failed", "error": str(e) or type(e).__name__}

    async def _read(self, doc_id: str) -> None:
        doc = self.store.get_document(doc_id)
        doc_dir = self.settings.docs_dir / doc_id
        original = next(doc_dir.glob("original*"), None)
        if doc is None or original is None or not self.needs_ocr(doc):
            self.state.pop(doc_id, None)
            return
        state = self.state[doc_id] = {"state": "running", "done": 0, "total": doc.n_pages or 0}

        def on_page(n: int) -> None:
            state["done"] = n

        log.info("OCR of %s (%s): %d page(s)", doc.name, doc_id, doc.n_pages)
        text, layout = await asyncio.to_thread(read_document, original, self.recognizer, on_page)
        if self.store.get_document(doc_id) is None or not doc_dir.is_dir():
            self.state.pop(doc_id, None)  # deleted meanwhile
            return
        tmp = doc_dir / "text.txt.tmp"
        tmp.write_text(text, encoding="utf-8")
        (doc_dir / OCR_FILE).write_text(json.dumps(layout), encoding="utf-8")
        os.replace(tmp, doc_dir / "text.txt")
        self.store.update_document(doc_id, n_chars=len(text))
        self.state.pop(doc_id, None)
        log.info("OCR of %s: %d characters", doc.name, len(text))
