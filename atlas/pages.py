"""Page images for visual prefill: PDF pages rendered to PNG, uploaded images normalized to PNG.

Queries must send exactly the bytes that were prefilled: llama-server identifies a cached image by
a hash of its file bytes. Pages are therefore rendered once per resolution and kept on disk.
"""

import io
import os
import shutil
import threading
from pathlib import Path, PurePath

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
VISUAL_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}
MAX_PAGES = 2000
MAX_IMAGE_SIDE = 4096  # uploaded images larger than this are scaled down first


# pdfium is not thread-safe: every use from worker threads must hold this lock, or the process crashes.
PDFIUM = threading.Lock()


class PageError(ValueError):
    pass


def supports_visual(filename: str) -> bool:
    return PurePath(filename).suffix.lower() in VISUAL_EXTENSIONS


def is_image(filename: str) -> bool:
    return PurePath(filename).suffix.lower() in IMAGE_EXTENSIONS


def page_count(filename: str, data: bytes) -> int:
    """Number of pages a document would have in visual mode (0 if it cannot be shown as pages)."""
    ext = PurePath(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        _open_image(data)
        return 1
    if ext == ".pdf":
        import pypdfium2 as pdfium
        try:
            with PDFIUM:
                pdf = pdfium.PdfDocument(data)
                try:
                    return len(pdf)
                finally:
                    pdf.close()  # explicitly, under the lock, not later by the garbage collector
        except Exception as e:  # pdfium raises its own error types
            raise PageError(f"cannot open PDF: {e}") from e
    return 0


def _open_image(data: bytes):
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise PageError(f"cannot read image: {e}") from e
    return img


def _png(img) -> bytes:
    if img.mode not in ("RGB", "L"):
        from PIL import Image
        background = Image.new("RGB", img.size, "white")  # flatten transparency onto white paper
        background.paste(img, mask=img.convert("RGBA").split()[-1])
        img = background
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def render_page(pdf, index: int, scale: float):
    """Render one page to a PIL image that no longer depends on pdfium memory. Hold PDFIUM."""
    page = pdf[index]
    try:
        bitmap = page.render(scale=scale)
        try:
            return bitmap.to_pil().copy()
        finally:
            bitmap.close()
    finally:
        page.close()


def page_dir(doc_dir: Path, dpi: int) -> Path:
    return doc_dir / f"pages-{dpi}"


def list_pages(folder: Path) -> list[Path]:
    return sorted(folder.glob("page-*.png"))


def ensure_pages(original: Path, doc_dir: Path, dpi: int) -> list[Path]:
    """Render the document's pages at `dpi` unless already done; returns the page files in order."""
    dest = page_dir(doc_dir, dpi)
    if (dest / ".complete").exists():
        return list_pages(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        data = original.read_bytes()
        ext = original.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            img = _open_image(data)
            if max(img.size) > MAX_IMAGE_SIDE:
                img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            (tmp / "page-0001.png").write_bytes(_png(img))
        elif ext == ".pdf":
            import pypdfium2 as pdfium
            with PDFIUM:
                try:
                    pdf = pdfium.PdfDocument(data)
                except Exception as e:
                    raise PageError(f"cannot open PDF: {e}") from e
                n = len(pdf)
            if n > MAX_PAGES:
                raise PageError(f"{n} pages; visual prefill supports at most {MAX_PAGES}")
            try:
                for i in range(n):
                    with PDFIUM:  # page by page, so thumbnails elsewhere are not blocked for long
                        image = render_page(pdf, i, dpi / 72)
                    (tmp / f"page-{i + 1:04d}.png").write_bytes(_png(image))
            finally:
                with PDFIUM:
                    pdf.close()
        else:
            raise PageError(f"visual prefill supports PDFs and images, not '{ext}'")
        (tmp / ".complete").touch()
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return list_pages(dest)
