"""PDF preparation: read a PDF's pages, text and chapter outline, render thumbnails, cut shards.

Where to cut is decided in the browser (by chapters, token budget, page count or ranges); this
module only reports what a PDF contains and builds the shards it is asked for.
"""

import io
import re
import shutil
import time
import zipfile
from pathlib import Path

TEXT_PAGE_MIN_CHARS = 20  # fewer extracted characters: the page is a scan, a picture or blank
WORKSPACE_TTL_S = 24 * 3600
_WS = re.compile(r"^[0-9a-f]{32}$")


class PdfToolError(ValueError):
    pass


def _reader(data: bytes):
    from pypdf import PdfReader
    try:
        return PdfReader(io.BytesIO(data))
    except Exception as e:  # pypdf raises many error types for damaged files
        raise PdfToolError(f"cannot read PDF: {e}") from e


def analyze(data: bytes) -> dict:
    """Per-page text and size, and the outline (bookmarks) with 1-based page numbers."""
    reader = _reader(data)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        box = page.mediabox
        pages.append({"n": i, "text": text, "chars": len(text), "has_text": len(text) >= TEXT_PAGE_MIN_CHARS,
                      "w": round(float(box.width)), "h": round(float(box.height))})
    outline: list[dict] = []

    def walk(items, level: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, level + 1)
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                continue
            title = re.sub(r"\s+", " ", str(getattr(item, "title", "") or "")).strip()
            if title and page >= 1:
                outline.append({"title": title[:200], "page": page, "level": level})

    try:
        walk(reader.outline, 1)
    except Exception:
        outline = []  # a broken outline must not break the analysis
    return {"n_pages": len(pages), "pages": pages, "outline": outline}


def thumbnail(data: bytes, page: int, width: int) -> bytes:
    import pypdfium2 as pdfium

    from .pages import PDFIUM, render_page
    with PDFIUM:  # pdfium is not thread-safe
        pdf = pdfium.PdfDocument(data)
        try:
            if not 1 <= page <= len(pdf):
                raise PdfToolError("no such page")
            w, _ = pdf.get_page_size(page - 1)
            image = render_page(pdf, page - 1, max(0.05, width / max(w, 1)))
        finally:
            pdf.close()
    buf = io.BytesIO()
    image.save(buf, "PNG", optimize=True)
    return buf.getvalue()


# Page keys not copied into shards. Link annotations point to other pages; copying a page follows
# them, and through the link graph of a cross-referenced PDF (rule books, manuals) pulls most of the
# document into every shard: 54 MB and 48 s for a 67-page shard of a 4,417-page PDF, instead of
# 0.7 MB and 2 s. Links out of a shard cannot work anyway, and prefill does not use annotations.
SHARD_EXCLUDED_KEYS = ("/Annots", "/B")


def build_shard(source, pages: list[int]) -> bytes:
    """A new PDF of the given 1-based pages, in the given order. `source`: PDF bytes or a PdfReader
    (pass one reader for several shards of the same PDF, so it is parsed only once)."""
    from pypdf import PdfWriter
    reader = _reader(source) if isinstance(source, bytes) else source
    if not pages or any(not 1 <= p <= len(reader.pages) for p in pages):
        raise PdfToolError("shard pages out of range")
    writer = PdfWriter()
    for p in pages:
        writer.add_page(reader.pages[p - 1], excluded_keys=SHARD_EXCLUDED_KEYS)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def reader(data: bytes):
    return _reader(data)


FILENAME_CHARS = 180  # document names are at most 200 characters (with suffix and counter)


def shard_filename(name: str, taken: set[str]) -> str:
    base = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", " ", name).strip().removesuffix(".pdf").strip() or "shard"
    if len(base) > FILENAME_CHARS:  # bookmark titles can be long: cut at a word, keep it readable
        base = base[:FILENAME_CHARS].rsplit(" ", 1)[0].rstrip(" -–,;:") + "…"
    candidate, k = f"{base}.pdf", 2
    while candidate.lower() in taken:
        candidate, k = f"{base} ({k}).pdf", k + 1
    taken.add(candidate.lower())
    return candidate


def build_zip(data: bytes, shards: list[tuple[str, list[int]]]) -> bytes:
    buf = io.BytesIO()
    taken: set[str] = set()
    source = _reader(data)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, pages in shards:
            z.writestr(shard_filename(name, taken), build_shard(source, pages))
    return buf.getvalue()


def workspace(root: Path, ws_id: str) -> Path:
    if not _WS.match(ws_id or ""):
        raise PdfToolError("invalid workspace")
    folder = root / ws_id
    if not (folder / "source.pdf").is_file():
        raise FileNotFoundError(ws_id)
    return folder


def sweep(root: Path) -> None:
    """Remove workspaces of PDFs analyzed more than a day ago."""
    if not root.is_dir():
        return
    now = time.time()
    for folder in root.iterdir():
        if folder.is_dir() and now - folder.stat().st_mtime > WORKSPACE_TTL_S:
            shutil.rmtree(folder, ignore_errors=True)
