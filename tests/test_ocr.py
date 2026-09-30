"""OCR of documents without a text layer, and page citations in answers from page images."""

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from atlas import evidence, ocr

from .helpers import query, wait_for
from .test_visual import image_pdf, ready, upload


class FakeRecognizer:
    """Stands in for RapidOCR: page n shows "Page n: pump pressure table", a table row of two cells,
    and (on page 1) a note running up the margin."""

    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        n = self.calls
        w, h = image.size
        found = [(f"Page {n}: pump pressure table", (0.1 * w, 0.1 * h, 0.6 * w, 0.13 * h), 0),
                 ("Flow m/s", (0.1 * w, 0.3 * h, 0.3 * w, 0.32 * h), 0),
                 ("3,30", (0.5 * w, 0.301 * h, 0.6 * w, 0.321 * h), 0)]
        if n == 1:
            found.append(("Copying this document is not permitted", (0.02 * w, 0.2 * h, 0.04 * w, 0.8 * h), -90))
        return found


@pytest.mark.parametrize("atlas", [{"ocr": True}], indirect=True)
async def test_scans_are_read_by_ocr_and_answers_cite_their_pages(atlas, fake):
    fake.vision = True
    await atlas.app.state.engine.refresh()
    worker = atlas.app.state.ocr
    worker.recognizer = FakeRecognizer()
    doc = (await upload(atlas, "scan.pdf", image_pdf(2), mode="visual"))["document"]
    assert doc["mode"] == "visual" and not doc["has_text"]
    docs = await wait_for(atlas, lambda ds: ds[0]["ocr"] == {"state": "done"} and ds[0]["status"] == "ready")
    assert docs[0]["has_text"] and worker.recognizer.calls == 2

    # the OCR text: pages in order, table cells on one line, the sideways note after the page
    text = (await atlas.get(f"/api/documents/{doc['id']}/text")).text
    assert text == ("[Page 1]\nPage 1: pump pressure table\nFlow m/s 3,30\nCopying this document is not permitted"
                    "\n\n[Page 2]\nPage 2: pump pressure table\nFlow m/s 3,30")

    # the answer from the page images quotes page 2; the quote is found in the OCR text there
    events = await query(atlas, "Please quote the pressure table.", [doc["id"]])
    (ev,) = next(e for e in events if e["type"] == "target" and e["status"] == "done")["evidence"]
    assert ev["found"] and ev["page"] == ev["cited_page"] == 2 and ev["ocr"] is True and ev["score"] == 1.0

    # marked on the page from the positions OCR kept (a part of a line is cut in proportion)
    boxes = (await atlas.get(f"/api/documents/{doc['id']}/boxes/2", params={"start": ev["start"], "end": ev["end"]})).json()
    assert boxes == {"boxes": [[0.1, 0.1, 0.6, 0.13]], "score": 1.0}
    flow = text.index("3,30", text.index("[Page 2]"))
    half = (await atlas.get(f"/api/documents/{doc['id']}/boxes/2", params={"start": flow, "end": flow + 4})).json()
    assert half["boxes"] == [[0.5, 0.301, 0.6, 0.321]]
    note = text.index("not permitted")
    up = (await atlas.get(f"/api/documents/{doc['id']}/boxes/1", params={"start": note, "end": note + 13})).json()
    (x0, y0, x1, y1), = up["boxes"]
    assert (x0, y0, x1) == (0.02, 0.2, 0.04) and 0.40 < y1 < 0.41  # text running up ends at the top

    # a document with text is not read again
    assert worker.needs_ocr(atlas.app.state.store.get_document(doc["id"])) is False


def test_rows_join_table_cells_and_keep_sideways_text_apart():
    found = [("b", (50, 100, 60, 120), 0), ("a", (10, 102, 20, 121), 0), ("next line", (10, 140, 90, 160), 0),
             ("margin", (0, 0, 5, 300), 90)]
    assert [[f[0] for f in row] for row in ocr.rows(found)] == [["a", "b"], ["next line"], ["margin"]]


def test_chinese_look_alikes_are_put_back_on_latin_pages():
    box = (0, 0, 1, 1)
    latin = [("M42 一 1370", box, 0), ("口 Hinweis: Änderungen", box, 0)]
    assert [t for t, _, _ in ocr.latin_lookalikes(latin)] == ["M42 – 1370", "□ Hinweis: Änderungen"]
    chinese = [("一二三", box, 0), ("口", box, 0)]
    assert ocr.latin_lookalikes(chinese) == chinese


def test_page_citations_are_read_from_answers():
    answer = ('Laut Norm "Die Auswuchtdrehzahl ist die maximale Betriebsdrehzahl" (p. 3) und '
              '„Wellenenden mit Paßfedernut sind auszufüllen“ (S. 4–5), auch "Table 10: Mechanische Eigenschaften '
              'für M6 bis M39 (Page 8)" und "S235JR ist der Werkstoff für Rahmen" (S235JR).\n\n'
              '> Kompensation der Unwuchten in 2 Ausgleichsebenen.\n[p. 6]')
    assert evidence.cited_quotes(answer) == [
        ("Kompensation der Unwuchten in 2 Ausgleichsebenen.", (6, 6)),
        ("Die Auswuchtdrehzahl ist die maximale Betriebsdrehzahl", (3, 3)),
        ("Wellenenden mit Paßfedernut sind auszufüllen", (4, 5)),
        ("Table 10: Mechanische Eigenschaften für M6 bis M39", (8, 8)),
        ("S235JR ist der Werkstoff für Rahmen", None),  # a steel grade in brackets is not a page
    ]
    doc = evidence.DocText("[Page 1]\nKeep the valve closed during maintenance.\n\n[Page 2]\nOther text.\n\n"
                           "[Page 3]\nKeep the valve closed during maintenance.")
    on3, said2, invented, beyond = (doc.evidence(f'"{q}" (p. {p}).') for q, p in (
        ("Keep the valve closed during maintenance", 3), ("Keep the valve closed during maintenance", 2),
        ("the pump must be replaced after every flood", 2), ("Keep the valve closed during maintenance", 99)))
    assert on3[0]["page"] == on3[0]["cited_page"] == 3  # a repeated passage: the one on the cited page
    assert said2[0]["page"] == 1 and said2[0]["cited_page"] == 2  # found, but not where the answer said
    assert invented[0] == {"quote": "the pump must be replaced after every flood", "found": False,
                           "cited_page": 2, "cited_page_end": 2}
    assert "cited_page" not in beyond[0]  # the document has no page 99


@pytest.mark.skipif(not ocr.available(), reason="rapidocr is not installed")
def test_rapidocr_reads_a_rendered_page(tmp_path):
    page = Image.new("RGB", (1654, 2339), "white")  # A4 at 200 dpi
    draw = ImageDraw.Draw(page)
    dejavu = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")  # Pillow's own font has no umlauts
    font = ImageFont.truetype(str(dejavu), 44) if dejavu.exists() else ImageFont.load_default(size=44)
    draw.text((150, 200), "Change the oil every 500 hours.", fill="black", font=font)
    draw.text((150, 320), "Größe", fill="black", font=font)
    draw.text((700, 320), "M42", fill="black", font=font)
    buf = io.BytesIO()
    page.save(buf, "PDF", resolution=200)
    (tmp_path / "original.pdf").write_bytes(buf.getvalue())
    text, layout = ocr.read_document(tmp_path / "original.pdf", ocr.Recognizer(threads=2))
    assert text.startswith("[Page 1]\nChange the oil every 500 hours")
    assert ("Größe M42" if dejavu.exists() else "M42") in text
    assert layout["pages"][0]["boxes"] and layout["engine"].startswith("RapidOCR")
