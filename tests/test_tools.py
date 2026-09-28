"""PDF tools (analysis, shards) and the token estimator."""

import io
import zipfile

from pypdf import PdfReader, PdfWriter

from .helpers import wait_for
from .test_visual import image_pdf, text_pdf


def book(chapters: dict[str, list[str]]) -> bytes:
    """A PDF with one page per text and a bookmark at the start of every chapter."""
    writer = PdfWriter()
    for title, pages in chapters.items():
        start = len(writer.pages)
        for text in pages:
            writer.append(PdfReader(io.BytesIO(text_pdf(text))))
        writer.add_outline_item(title, start)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


BOOK = book({
    "1 Introduction": ["The gearbox manual covers the GMD drive system.", "Scope and safety notes for operators."],
    "2 Maintenance": ["Change the oil every 500 hours of operation.", "Replace the bearing seals every 2 years."],
    "3 Appendix": ["Spare part numbers and supplier addresses."],
})


async def analyze(client, **kw) -> dict:
    r = await client.post("/api/tools/pdf", **kw)
    assert r.status_code == 200, r.text
    return r.json()


async def test_pdf_is_analyzed_page_by_page_with_its_chapters(atlas, fake):
    a = await analyze(atlas, files={"file": ("manual.pdf", BOOK)})
    assert a["name"] == "manual.pdf" and a["n_pages"] == 5 and a["exact"] is True
    assert [p["n"] for p in a["pages"]] == [1, 2, 3, 4, 5]
    assert all(p["has_text"] and p["tokens"] > 20 for p in a["pages"])
    assert a["total_tokens"] == sum(p["tokens"] for p in a["pages"])
    assert [(o["title"], o["page"], o["level"]) for o in a["outline"]] == [
        ("1 Introduction", 1, 1), ("2 Maintenance", 3, 1), ("3 Appendix", 5, 1)]
    assert a["part_tokens"] == 4096 - 2048 and a["shard_overhead_tokens"] > 0

    thumb = await atlas.get(f"/api/tools/pdf/{a['id']}/thumb/1?width=120")
    assert thumb.status_code == 200 and thumb.content[:4] == b"\x89PNG"
    assert (await atlas.get(f"/api/tools/pdf/{a['id']}/thumb/9")).status_code == 404

    scan = await analyze(atlas, files={"file": ("scan.pdf", image_pdf(2))})
    assert [p["has_text"] for p in scan["pages"]] == [False, False]


async def test_shards_become_library_documents(atlas, fake):
    a = await analyze(atlas, files={"file": ("manual.pdf", BOOK)})
    coll = (await atlas.post("/api/collections", json={"name": "Manual"})).json()
    body = {"shards": [{"name": "Introduction", "pages": [1, 2]}, {"name": "Maintenance", "pages": [3, 4]}],
            "collection_id": coll["id"], "mode": "text"}
    r = await atlas.post(f"/api/tools/pdf/{a['id']}/shards", json=body)
    assert r.status_code == 200, r.text
    names = [x["document"]["name"] for x in r.json()["results"]]
    assert names == ["Introduction.pdf", "Maintenance.pdf"]
    docs = await wait_for(atlas, lambda ds: len(ds) == 2 and all(d["status"] == "ready" for d in ds))
    assert {d["collection_id"] for d in docs} == {coll["id"]}
    maintenance = next(d for d in docs if d["name"] == "Maintenance.pdf")
    text = (await atlas.get(f"/api/documents/{maintenance['id']}/text")).text
    assert "oil every 500 hours" in text and "gearbox manual" not in text

    z = await atlas.post(f"/api/tools/pdf/{a['id']}/zip", json={"shards": body["shards"]})
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(z.content)) as archive:
        assert archive.namelist() == ["Introduction.pdf", "Maintenance.pdf"]
        assert len(PdfReader(io.BytesIO(archive.read("Maintenance.pdf"))).pages) == 2

    # a library PDF can be analyzed again from its original file
    original = await atlas.get(f"/api/documents/{maintenance['id']}/original")
    assert original.status_code == 200 and original.content[:5] == b"%PDF-"
    again = await analyze(atlas, data={"doc_id": maintenance["id"]})
    assert again["n_pages"] == 2 and again["doc_id"] == maintenance["id"]

    bad = await atlas.post(f"/api/tools/pdf/{a['id']}/shards", json={"shards": [{"name": "x", "pages": [9]}]})
    assert "out of range" in bad.json()["results"][0]["error"]
    assert (await atlas.delete(f"/api/tools/pdf/{a['id']}")).status_code == 200
    assert (await atlas.get(f"/api/tools/pdf/{a['id']}/thumb/1")).status_code == 404


async def test_token_estimator(atlas, fake):
    r = await atlas.post("/api/tools/estimate", data={"text": "The gearbox failed in April."})
    e = r.json()
    assert e["exact"] and e["tokens"] == len("The gearbox failed in April.")  # the fake tokenizes characters
    assert e["parts"] == 1 and e["part_tokens"] == 2048
    r = await atlas.post("/api/tools/estimate", files={"file": ("manual.pdf", BOOK)})
    e = r.json()
    assert e["pages"] == 5 and e["tokens"] > 100
    big = await atlas.post("/api/tools/estimate", data={"text": "x" * 5000})
    assert big.json()["parts"] == 3


async def test_thumbnails_can_be_rendered_concurrently(atlas):
    """pdfium is not thread-safe: parallel thumbnail requests must not crash the process."""
    import asyncio
    a = await analyze(atlas, files={"file": ("manual.pdf", BOOK)})
    rs = await asyncio.gather(*(atlas.get(f"/api/tools/pdf/{a['id']}/thumb/{n}?width={w}")
                                for n in range(1, 6) for w in (100, 140, 180)))
    assert all(r.status_code == 200 for r in rs)


def test_shards_do_not_drag_in_linked_pages():
    """Link annotations to other pages must not pull the rest of the PDF into a shard."""
    from pypdf.annotations import Link
    from pypdf.generic import Fit

    from atlas import pdftools
    writer = PdfWriter()
    for i in range(40):
        writer.append(PdfReader(io.BytesIO(text_pdf(f"Page {i}: " + "rule text " * 200))))
    for i in range(40):  # every page links to every tenth page, like a cross-referenced rule book
        for target in range(0, 40, 10):
            writer.add_annotation(i, Link(rect=(50, 50, 100, 60), target_page_index=target, fit=Fit.fit()))
    buf = io.BytesIO()
    writer.write(buf)
    source = buf.getvalue()
    shard = pdftools.build_shard(source, [5, 6])
    assert len(PdfReader(io.BytesIO(shard)).pages) == 2
    assert len(shard) < len(source) / 10, (len(shard), len(source))  # was 15% of the source before the fix
