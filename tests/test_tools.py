"""PDF tools (analysis, shards) and the token estimator."""

import io
import json
import zipfile

from pypdf import PdfReader, PdfWriter

from atlas import evidence, prompts

from .helpers import query, wait_for
from .test_visual import image_pdf, ready, text_pdf


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


async def test_long_shard_names_are_shortened_not_rejected(atlas):
    """Bookmark titles can be long (a rule book had 250-character shard names)."""
    a = await analyze(atlas, files={"file": ("manual.pdf", BOOK)})
    long = "RINA RULES interactive version - 1 January 2026 – SubArticle - 2.5 Assignment of a Dual Class " + "word " * 40
    r = await atlas.post(f"/api/tools/pdf/{a['id']}/shards", json={"shards": [{"name": long, "pages": [1]}]})
    assert r.status_code == 200, r.text
    name = r.json()["results"][0]["document"]["name"]
    assert len(name) <= 200 and name.endswith("….pdf") and name.startswith("RINA RULES interactive version")

    r = await atlas.post(f"/api/tools/pdf/{a['id']}/shards", json={"shards": [{"name": "ok", "pages": [1]}, {"name": "", "pages": [2]}]})
    assert r.status_code == 422 and r.json()["detail"].startswith("shards › #2 › name:"), r.json()


async def test_pdfs_merge_into_one_document_read_in_one_context(atlas, fake):
    fake.vision = True
    await atlas.app.state.engine.refresh()
    r = await atlas.post("/api/documents", files={"files": ("pump.pdf", text_pdf("Pump data sheet: the pump flow is 3,30 m/s."))},
                         data={"mode": "text"})
    pump = r.json()["results"][0]["document"]
    await ready(atlas, pump["id"])

    # uploads and a library PDF, in the order given
    uploads = [("files", ("manual.pdf", BOOK)), ("files", ("scan.pdf", image_pdf(1)))]
    r = await atlas.post("/api/tools/pdf/merge", files=uploads,
                         data={"order": json.dumps([{"file": 0}, {"doc": pump["id"]}, {"file": 1}])})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["name"] == "manual + pump + scan.pdf" and m["n_pages"] == 7
    assert m["files"] == [{"name": "manual.pdf", "page": 1, "pages": 5}, {"name": "pump.pdf", "page": 6, "pages": 1},
                          {"name": "scan.pdf", "page": 7, "pages": 1}]
    # a bookmark per file, with the file's own bookmarks below it
    assert [(o["title"], o["page"], o["level"]) for o in m["outline"]] == [
        ("manual", 1, 1), ("1 Introduction", 1, 2), ("2 Maintenance", 3, 2), ("3 Appendix", 5, 2),
        ("pump", 6, 1), ("scan", 7, 1)]
    pdf = await atlas.get(f"/api/tools/pdf/{m['id']}/pdf")
    assert pdf.status_code == 200 and len(PdfReader(io.BytesIO(pdf.content)).pages) == 7
    assert "manual%20%2B%20pump%20%2B%20scan.pdf" in pdf.headers["content-disposition"]

    # added as one document: a line before each file's first page (the scan has no text)
    r = await atlas.post(f"/api/tools/pdf/{m['id']}/shards",
                         json={"shards": [{"name": m["name"], "pages": list(range(1, 8))}], "mode": "text"})
    doc = r.json()["results"][0]["document"]
    await ready(atlas, doc["id"])
    text = (await atlas.get(f"/api/documents/{doc['id']}/text")).text
    assert text.startswith("[File: manual.pdf]\n[Page 1]\n") and "\n\n[File: pump.pdf]\n[Page 6]\n" in text
    assert text.endswith("\n\n[File: scan.pdf]")
    # the model gets the file lines; quotes are located on the merged document's pages
    events = await query(atlas, "Please quote the pump flow.", [doc["id"]])
    (ev,) = next(e for e in events if e["type"] == "target" and e["status"] == "done")["evidence"]
    assert ev["found"] and ev["page"] == 6

    # a shard across two files: file lines at its own page numbers; prefilled visually, the page
    # images get them too
    r = await atlas.post(f"/api/tools/pdf/{m['id']}/shards",
                         json={"shards": [{"name": "tail", "pages": [5, 6]}], "mode": "visual"})
    tail = r.json()["results"][0]["document"]
    await ready(atlas, tail["id"])
    text = (await atlas.get(f"/api/documents/{tail['id']}/text")).text
    assert text.startswith("[File: manual.pdf]\n[Page 1]\n") and "[File: pump.pdf]\n[Page 2]\n" in text
    (part,) = atlas.app.state.store.get_parts(tail["id"])
    assert "[File: manual.pdf]\n[Page 1]\n" in part.prefix_text and "[File: pump.pdf]\n[Page 2]\n" in part.prefix_text

    # errors
    one = await atlas.post("/api/tools/pdf/merge", files=uploads[:1], data={"order": json.dumps([{"file": 0}])})
    assert one.status_code == 400 and "at least two" in one.json()["detail"]
    bad = await atlas.post("/api/tools/pdf/merge", files=uploads[:1], data={"order": json.dumps([{"file": 0}, {"file": 3}])})
    assert bad.status_code == 400


def test_parts_of_merged_documents_name_their_file():
    files = [{"name": "a.pdf", "page": 1}, {"name": "b.pdf", "page": 3}]
    text = prompts.mark_files("[Page 1]\nalpha\n\n[Page 2]\nbeta\n\n[Page 3]\ngamma", files)
    assert text == "[File: a.pdf]\n[Page 1]\nalpha\n\n[Page 2]\nbeta\n\n[File: b.pdf]\n[Page 3]\ngamma"
    # a part that begins inside a file repeats its line; one that begins with a file line does not
    assert prompts.file_context(text, text.index("[Page 2]")) == "[File: a.pdf]\n"
    assert prompts.file_context(text, text.index("[File: b.pdf]") - 2) == ""
    block = prompts.visual_document_block("m.pdf", 1, 2, [2, 3], files)
    assert block.index("[File: a.pdf]\n[Page 2]") < block.index("[File: b.pdf]\n[Page 3]")
    # a file line belongs to the page after it, and quotes that copy the lines are still found
    doc = evidence.DocText(prompts.mark_files("[Page 1]\nThe pump is inspected yearly.\n\n[Page 2]\nValve V2 is closed "
                                              "before the pump starts.", [{"name": "a.pdf", "page": 1}, {"name": "b.pdf", "page": 2}]))
    hit = doc.find("[File: b.pdf] [Page 2] Valve V2 is closed before the pump starts.")
    assert hit["found"] and hit["score"] == 1.0 and hit["page"] == hit["page_end"] == 2
    assert hit["quote"].startswith("[File: b.pdf]") and doc.page_range(2)[0] == doc.text.index("Valve")
    # documents that are not merged are unchanged
    assert prompts.visual_document_block("m.pdf", 0, 1, [1]) == prompts.visual_document_block("m.pdf", 0, 1, [1], [])
    assert prompts.mark_files("[Page 1]\nalpha", []) == "[Page 1]\nalpha"
